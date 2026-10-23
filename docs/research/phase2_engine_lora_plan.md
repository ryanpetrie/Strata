# Phase 2.2 - engine support for per-request LoRA application

**Status:** plan, written 2026-10-21. It implements roadmap item
[2.2](decision_policy_roadmap.md#22-engine-support-for-per-request-lora-application-literature-doc-3b).
Companions: the measured record
[shared_backbone_decision_policy.md](shared_backbone_decision_policy.md) (the LOGITS path, the VRAM
figures, the concurrency fix in §13) and the literature framing
[decision_policy_literature_and_gaps.md §3.B](decision_policy_literature_and_gaps.md).

**What is measured and what is not.** Everything engine-side in this document is **unconfirmed static
analysis**: the code was read, nothing was built, no model was loaded, no engine was run (the GPU is
held by the running server; only one instance may run). The numbers this plan quotes about the engine's
own behaviour come from the companion doc and from `strata-unsloth-ud-iq4_xs.log` on this machine, and
are marked where they appear. The adapter-size arithmetic in §3 is arithmetic on the model's own
geometry (`include/strata/core/geometry.hpp`), not a measurement of an adapter that does not exist yet.

---

## 0. The answer in one page

| Question | Answer |
|---|---|
| Where does the adapter live? | In the engine process, beside the base weights, in device memory, uploaded at start-up from a new `--lora <dir>` flag. The same pattern the engine already uses for the control vector (`--control-vector-scaled`, `src/program/generate.cpp:1837-1918`) and the MTP draft layer (`--mtp`, 835 MiB of VRAM per this machine's log, `strata-unsloth-ud-iq4_xs.log:13`). |
| How is it turned on for one request? | A `lora=<id>` key on the request line, in the same `key=value` slot the sampling keys use (`src/program/generate.cpp:12001-12041`). Only `LOGITS` requests carry it; `GEN` never can. |
| Which path does it touch? | Three forward paths, all of them: the prompt read (`src/prefill/prefill.cpp`), the decode/verify window (`src/core/verify.cpp`), and the plain single-token path (`src/core/session.cpp:804`). The answer position's logits are produced by the **verify** path, not the prefill path (§1.4), so the verify path is the one that cannot be skipped. |
| How is the base path protected? | The adapter's delta is a separate low-rank add, gated by a **device-side flag** the way the control vector's is (`include/strata/kernels/cvec.hpp:28`, `src/kernels/cuda/cvec.cu:137-150`). With the flag off the kernel writes nothing at all, so the base output bytes are untouched. The flag is only ever set on a `LOGITS` request. |
| What about the conversation cache? | The adapter state joins the cache key exactly the way the control vector's does (`include/strata/core/conversation_cache.hpp:251-257`, `src/program/generate.cpp:12353`, `12592-12597`). A request whose adapter state differs from the live session's drops the live state and the checkpoints instead of resuming them. |
| Can the routed experts carry a LoRA? | **No, on this card.** A per-expert adapter at rank 16 needs ~7.55 GB of fp16 weights against ~58 MiB of headroom above the engine's 1024 MiB warning threshold (§3.3). Attention-only (and optionally the shared expert) is the only MoE-compatible option, which is why roadmap item 2.4 exists as an escalation and not as a default. |
| What is the biggest risk? | Not the kernel: the **number of branches** in the verify path (`src/core/verify.cpp:1343-1377`, `1504-1521`). A delta applied at the wrong one of them is silently wrong, not loudly wrong. §4 step 3 is where the plan spends its risk budget. |
| What would make this item infeasible? | §6: if the engine's own output is not reproducible request-to-request, the exit criterion as the roadmap words it ("byte-identical") cannot be tested at all, and the criterion has to be restated before any of this is built. |

---

## 1. Where the code is

### 1.1 Weight loading

| What | Where |
|---|---|
| The engine does **not** load the GGUF at run time for the dense weights. It loads a "pack": a text index plus binary blobs, produced by `tools/pack_index.py`. | `include/strata/core/weights.hpp:1-27` (the header states the rule: the loader "does not know what any tensor MEANS"), `src/core/weights.cpp:93` (`WeightTable::pool_bytes`), `:176` (`WeightTable::load`), `:525` (`load_hc_q8`) |
| The GGUF reader is header-only: `GgufFile`, `TensorInfo`, `GgufModel`, `read_geometry`. | `include/strata/artifact/gguf_reader.hpp:260`, `:378`, `:546`, `:658`; the CLI wrapper is `src/artifact/gguf_reader.cpp:7` |
| The per-layer tensor **names and shapes** (the table every layer is checked against). | `include/strata/core/family_layout.hpp:154-218` (`layer_tensors`), `:222-234` (`global_tensors`), `:240-283` (`check_gguf_layout`) |
| Per-layer name resolution at run time (`get("attn_qkv.weight")` -> `blk.<N>.attn_qkv.weight`). | `include/strata/core/layout.hpp:47-58` (`LayerView`), `src/core/layout.cpp` |
| The model's geometry: 48 layers, `full_attention_interval` 4, `n_embd` 2560, `n_head` 24, `head_dim` 256, `n_expert` 512, `n_expert_used` 10, `n_ff` 640, `hc` 4 / `hc_lr` 320. | `include/strata/core/geometry.hpp:23-68` |
| Which layers are full attention: `layer % qsa_interval == qsa_interval - 1` -> layers 3, 7, ..., 47 (**12 layers**); the other **36** are GDN (linear attention). | `include/strata/core/geometry.hpp:66-67`, `:75-77` |
| The MoE expert tensors: `ffn_gate_exps.weight {N, FF, NE}`, `ffn_up_exps.weight {N, FF, NE}`, `ffn_down_exps.weight {FF, N, NE}`; the shared expert: `ffn_gate_shexp.weight {N, FS}`, `ffn_up_shexp.weight {N, FS}`, `ffn_down_shexp.weight {FS, N}`. | `include/strata/core/family_layout.hpp:160-167` |
| The per-tensor quantization forms in the published files (`blk.N.ffn_*_exps.weight` and friends). | `data/gguf_fingerprints.json:280-345` |

### 1.2 The forward pass - a layer's projections

| What | Where |
|---|---|
| GDN layer (36 of them): the fused QKV projection `attn_qkv.weight {N, 10240}`, the gate `attn_gate.weight {N, 6144}`, and the output projection `ssm_out.weight {6144, N}`. | `src/core/layer.cpp:251` (`gdn_layer`), the QKV GEMV at `:276` |
| Full-attention layer (12 of them): `attn_q.weight {N, 12288}` (query **and** output gate), `attn_k.weight {N, 512}`, `attn_v.weight {N, 512}`, `attn_output.weight {6144, N}`. | `src/core/layer.cpp:1130` (`qsa_layer`), the Q GEMV at `:1240`, the output projection at `:1303-1307` |
| The router: `ffn_gate_inp.weight {N, 512}` (BF16), then softmax over all experts, top-k, renormalise. | `src/core/layer.cpp:405` (`moe_route`), the BF16 projection at `:417` |
| The shared expert (always on): `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`, plus the scalar gate `ffn_gate_inp_shexp.weight`. | `src/core/layer.cpp:468` (`moe_shared`) |
| The routed experts: gathered and applied through the expert cache. | `src/core/layer.cpp:556` (`moe_layer`), `src/core/expert_source.cpp`, `src/core/expert_cache.cpp`, `src/kernels/cuda/s2_expert_grouped.cu`, `src/kernels/cuda/native_mmvq.cu` |
| The whole layer, both kinds, plus the MoE: `block_layer`, `block_layer_pre`, `block_layer_post`. | `src/core/layer.cpp:1633`, `:1459`, `:1608` |
| The final norm and the vocabulary projection. | `src/core/layer.cpp:1377` (`lm_head_mix`), `:1398` (`lm_head`) |

### 1.3 The forward pass - the three paths the runtime uses

The engine has **three** places a layer's projections are computed. All three need the adapter, and
they are not interchangeable.

| Path | File | When it runs |
|---|---|---|
| The prompt read (a GEMM over T tokens) | `src/prefill/prefill.cpp`: GDN at `:3200` (`attn_qkv`), QSA at `:3226-3228` (`attn_k`, `attn_v`, `attn_q`), `attn_output` at `:3450`, the router at `:3462`, the shared expert at `:3525-3534`. The projection helpers are `native_proj` (`:2418`) and `bf16_proj` (`:2426`). | The bulk of every prompt. The prompt is read in segments (`src/program/generate.cpp:13701-13798`); a segment longer than `--short-read` (default 64, `:753`) goes through this path (`windows_ok`, `:12736-12742`). |
| The verify window (a multi-row MMVQ over the window's tokens) | `src/core/verify.cpp`: `Verifier::prelaunch` at `:2663`; GDN at `:1325-1400` (`mm(wqkv, ...)` at `:1361` and `:1377`); QSA at `:1460-1530` (`native_mmvq(wq->native_type, ...)`, e.g. `:1511`). | **Every decode token**, and the last prompt token. The running config has `--spec 4 --mtp <dir>` (`strata-unsloth-ud-iq4_xs.json:14-17`), so this is the hot path. |
| The plain single-token path | `src/core/session.cpp:804` (`session_token`) -> `src/core/layer.cpp:1633` (`block_layer`). | The CLI's `generate` mode and the non-graph fallback. |

### 1.4 Where the LOGITS row is produced (this is the fact that decides the design)

* The prompt is read up to token `n-1` **exclusive**: the segment cuts end with `n - 1`
  (`src/program/generate.cpp:13701`), and `read_part` calls `sp.run(ids.data()+q, r-q, q, e)`
  (`:13144-13147`), so the prefill path covers tokens `[0, n-1)`.
* The **last** prompt token is left to the first decode window: `first_window` is set at
  `src/program/generate.cpp:13847` and the window's size is set at `:14550-14551`
  (`T = merge_T > 0 ? merge_T : 1`).
* The `L`/`Z` rows are read out of the verifier's logits at that window
  (`src/program/generate.cpp:14772-14805`, `ver.copy_logits(merged ? T - 1 : 0, ...)`).

**Consequence (static analysis):** the answer position's distribution is produced by
`src/core/verify.cpp`, not by `src/prefill/prefill.cpp`. The prefill path produces the *context* -
the K/V and the recurrent state the answer position attends to. A LoRA applied to only one of the two
would give a distribution the model was never trained to produce. **Both paths must carry the
adapter**, and the conversation cache must keep the two populations apart.

### 1.5 The LOGITS and GEN commands

| What | Where |
|---|---|
| The `LOGITS <k> <max_new> <id,id,...>` parser; it rewrites the line into a `GEN` line and keeps `req_logits_k`. | `src/program/generate.cpp:11933-11960` |
| The `key=value` request keys (`temperature`, `top_p`, `cvec`, `ckpt`, `pin`, `logit_bias`, `img_keys`, ...) - the ids start at the first token **without** an `=`. | `src/program/generate.cpp:11999-12042` |
| `req_cvec` - the per-request switch for a loaded side-weight. The template this plan copies. | `src/program/generate.cpp:11978`, `:12012`, `:12194`, `:12597` |
| The `L`/`Z` rows and the `Z` normalizer. | `src/program/generate.cpp:14768-14805` |
| The other serve commands (`BSTOP`, `BYIELD`, `VRAM`, `BGEN`, `QUIT`, `SAVE`, `RESTORE`). | `src/program/generate.cpp:11620`, `:11625`, `:11631`, `:11646`, `:11666`, `:11710` |
| The start-up flag parser and the usage text. | `src/program/generate.cpp:2576-2606` (the control vector's flags), `:932-936` |
| The `INFO` line (the server parses it into `self.info`). | `src/program/generate.cpp:10735-10741` |

### 1.6 The conversation cache, the session identity, the FIFO

| What | Where |
|---|---|
| The cache's key today is (token prefix, image identity, **control-vector state**). | `include/strata/core/conversation_cache.hpp:251-266` (`best`), `:166` (`SavedConversation::cvec`), `:374-399` (`drop_superseded`), `:405-420` (`parked_prefix_tokens`), `:422-431` (`put`) |
| The live session's "what was this read with" state (`cvec_cached`). | `src/program/generate.cpp:9047`, `:10977`, `:12259`, `:12273`, `:12306`, `:12353`, `:12592-12597`, `:13178`, `:15120` |
| The batch slots carry the same flag. | `src/program/generate.cpp:12275`, `:12306`, `:13178`, `:15120` |
| The snapshot's view of it. | `include/strata/core/conversation_snapshot.hpp:81-85` (`ConversationView`), `src/core/conversation_state.cpp:312`, `:338` |
| The session **file** binds to the loaded control vector through a digest. | `include/strata/core/conversation_file.hpp:71-83` (`SessionConfig::cvec`), `:47-59` (`SessionIdentityBuilder`), `src/core/conversation_file.cpp:691`, `:770-776`, `:896`, `:1009`; the digest is built at `src/program/generate.cpp:1902-1910` and put in the config at `:9108` |
| The server's FIFO: one engine turn at a time. | `serve/server.py:3539` (`self.fifo`), `:3688-3716` (`_inference_fifo`), `:3719-3776` (`_decision_turn`) |
| The decision question takes the turn **once**, and the trie walk runs inside it. | `serve/server.py:4642-4669` |

### 1.7 The server side

| What | Where |
|---|---|
| `Engine.logits` - writes the `LOGITS` line and reads `Z`/`L`/`T`/`DONE`. | `serve/server.py:2144-2296`; the line is written at `:2168-2169` |
| `Engine.generate` - writes the `GEN` line. | `serve/server.py:2298-...`; the line is written at `:2311-2314` |
| The sampling keys' wire format (the `key=value` string). | `serve/server.py:1347-1385` |
| The `INFO` line is parsed into `self.info`. | `serve/server.py:1036-1039` |
| The decision question: the prompt, the engine turn, the trie walk. | `serve/server.py:4620-4669` |
| The extra row the trie walk asks for. | `serve/server.py:4483-4504` (`_logits_row`), `:4510-...` (`_systemone_trie`) |
| The first-token path's own module (top-k, the trie, the readings). | `serve/logits_first_token.py:37` (`LOGITS_TOP_K = 1024`), `:112` (`trie_plan`), `:213` (`trie_probs`), `:382` (`first_token_probs`) |
| The mock engine used by the tests. | `serve/server.py:317-369` |
| The wire-format test that pins the `LOGITS` line on both sides. | `serve/test_logits_first_token.py:651-696` (`EngineWireFormat`) |

---

## 2. The design

### 2.1 The adapter file: two formats, one converter

**On disk (what the trainer writes):** PEFT's `adapter_model.safetensors` plus `adapter_config.json`
(rank, alpha, target modules). Roadmap item 2.3 produces exactly this, so the training side needs no
conversion and no engine knowledge.

**In the engine (what the engine reads):** a directory with a text index and one binary blob, exactly
the shape of the base pack:

```
<adapter-dir>/index.txt     # a flat, fixed-width manifest
<adapter-dir>/lora.bin      # the fp16 tensors, in the order the index names
```

`index.txt` (one row per tensor, the pack's own convention: a flat, name-keyed manifest):

```
# strata lora index v1  --  generated by tools/lora_pack.py; do not hand-edit
# name rank alpha ne0 ne1 offset bytes
blk.3.attn_q.weight.lora_a  16  16.0     16  2560     0   81920
blk.3.attn_q.weight.lora_b  16  16.0  12288    16  81920  393216
```

The engine pairs `*.lora_a` with `*.lora_b` by the base name and checks the shapes against
`layer_tensors()` (§3.5). `ne0`/`ne1` are the tensor's shape in the engine's order (dim 0
contiguous), so `lora_a` is `[rank, in]` and `lora_b` is `[out, rank]`.

**Why the engine does not read safetensors.** The engine's own rule is on the record:
`include/strata/core/weights.hpp:25-27` - *"WHY AN INDEX FILE AND NOT A JSON PARSER: a JSON parser
in the engine would be a new, unaudited component whose failure mode is a wrong byte offset - which
decodes to a plausible weight and produces plausible logits. Python wrote the manifest and already
parses it correctly."* A safetensors header is JSON, so the same argument applies. The conversion
belongs in Python, where the manifest is already written. Python already has safetensors header
readers in this repo (`tools/ple_fp8_pack.py:25`, `tools/test_mtp_fetch.py:29`).

**The converter:** `tools/lora_pack.py`. It maps PEFT's names onto the engine's names:

| PEFT / HF name | The engine's tensor | Layer kind |
|---|---|---|
| `...self_attn.q_proj.weight` | `blk.N.attn_q.weight` | QSA only |
| `...self_attn.k_proj.weight` | `blk.N.attn_k.weight` | QSA only |
| `...self_attn.v_proj.weight` | `blk.N.attn_v.weight` | QSA only |
| `...self_attn.o_proj.weight` | `blk.N.attn_output.weight` | QSA only |
| `...experts.{i}.gate_proj.weight` / `up_proj` / `down_proj` (2D) | `blk.N.ffn_{gate,up,down}_exps.weight` | every layer |
| `...experts.gate_up_proj.weight` / `down_proj.weight` (3D) | same, fused | every layer |
| `...mlp.gate_proj` / `up_proj` / `down_proj` (the shared expert) | `blk.N.ffn_{gate,up,down}_shexp.weight` | every layer |

The HF names for the QSA projections are confirmed in this repo by the MTP inventory
(`bench/results/2026-10-07-community-rtx-5080-docker-wsl2/mtp-inventory.md:31-35`:
`mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`, BF16, `12288x2560`, `512x2560`, `512x2560`,
`2560x6144`). The HF names for the **GDN** layers are **not found in this repo** - see §7, question 1.

**The orientation trap (must be right, and is easy to get wrong).** The GGUF convention is "dim 0
contiguous" (`include/strata/core/family_layout.hpp:12`), so the engine's `W` is `[out, in]` with
`out` contiguous and a LoRA's delta is `y += s * B @ (A @ x)`. Hugging Face keeps the LAST axis
contiguous, so **BOTH matrices are transposed on the way in**: PEFT's `A` is `[r, in]` (row-major,
`in` contiguous) and the file holds `Aᵀ` as `[in, r]`; PEFT's `B` is `[out, r]` and the file holds
`Bᵀ` as `[r, out]`. The index's `ne0` is the contiguous axis, which is what makes the file rows
`ne0=rank ne1=in` for A and `ne0=out ne1=rank` for B - one convention, the one the index example and
§2.4 agree on (rank contiguous in A's file, out contiguous in B's).

The first draft of this section said only `B` was transposed and that `A` "maps directly". The
spike measured what that costs (`phase2_lora_spike.md` §3): the delta has the **same magnitude** as
the right answer and is uncorrelated with it (norms 27175 and 24515 against 24344; cosines −0.103
and 0.015), with no error anywhere. Both orientations are now pinned by a test:
`src/core/lora_pack_test.cpp` (what the engine reads) and `tools/test_lora_pack.py` (what the
converter writes) assert the same twelve fp16 values.

### 2.2 Loading and residency

* A new start-up flag `--lora <dir>`, parsed next to the control vector's flags
  (`src/program/generate.cpp:2576-2606`), with the usage text at `:932-936`.
* The adapter is uploaded **before the first request**, so the CUDA graphs that the verifier captures
  (`src/core/graph.cpp`, `src/core/verify.cpp:2663`) contain the adapter's kernel from the start.
  There is **no runtime load command**: a runtime load would need a graph re-capture, which is a
  much larger risk than the feature is worth.
* The uploaded tables live in their own device allocation, sized from the index (a few tens of MiB,
  §3). They are separate from the base weights' arena: the base weights are never written.
* The device-side state mirrors the control vector's exactly (`include/strata/kernels/cvec.hpp:25-37`):
  a per-request `int` flag in device memory, read by the kernel; a host-side `bool` mirror;
  `lora_set_enabled(bool)` synchronizes the device when it changes.
* The digest that binds a session file to the loaded adapter is computed the way the control
  vector's is (`src/program/generate.cpp:1902-1910`, `SessionIdentityBuilder`): the adapter's bytes,
  its rank, its alpha, its target set and the scaling. It goes into `SessionConfig` next to `cvec`
  (`include/strata/core/conversation_file.hpp:80`).

**New files (proposed):**

| Path | What |
|---|---|
| `include/strata/kernels/lora.hpp` | The device tables (the A/B pointers, the rank, the alpha), the per-request flag, `lora_apply`. Mirrors `include/strata/kernels/cvec.hpp`. |
| `src/kernels/cuda/lora.cu` | The CUDA implementation: the upload, the flag, and the delta kernel. Mirrors `src/kernels/cuda/cvec.cu`. |
| `include/strata/core/lora_pack.hpp`, `src/core/lora_pack.cpp` | The host-side reader for `index.txt` + `lora.bin`, and the name/shape checks against `layer_tensors()`. |
| `tools/lora_pack.py` | The converter from PEFT's safetensors to the engine's directory. |
| `src/kernels/lora_parity.cpp` | The parity test (the repo's kernel-test convention, e.g. `src/kernels/cvec_parity.cpp`). |

`src/kernels/cuda/lora.cu` must be added to the `strata_kernels` library in `CMakeLists.txt` (the
list is at `CMakeLists.txt:366-413`; `cvec.cu` is at `:403`).

### 2.3 Enabling and disabling it per request

The wire format follows the control vector's, which is the one precedent in this engine:

```
LOGITS 1024 32 lora=1 11,22,33
```

* `lora=1` selects the loaded adapter. `lora=0` (or the key absent) means "the base model".
* The key goes **before** the ids, because the engine's key parser stops at the first token without
  an `=` (`src/program/generate.cpp:12008-12009`) and the ids are parsed from there.
* The engine parses `lora=` in the same key loop as `cvec=` (`:12012`), into a `req_lora` that
  defaults to off.
* **A `GEN` line can carry `lora=` too, and the engine must refuse it** if it does - or, better,
  the server must never write it. The engine-side guarantee that matters is this: the adapter is
  applied **only when `req_logits_k > 0`**, i.e. only on a line that was a `LOGITS` line. A `GEN`
  line - from the chat path, from the batch path, from the CLI - can therefore never turn it on,
  whatever it carries. This is the structural guarantee behind "the GEN path is byte-identical",
  and it does not depend on the server behaving.

**The server side:**

* The engine's `INFO` line gains `lora=<dir>` (or `lora=none`), so the server knows an adapter is
  resident (`serve/server.py:1036-1039` parses `INFO` into `self.info`).
* The launch config gains a key under `systemone`: `"systemone": {"enabled": true, "lora": true}`
  (the config is read at `serve/server.py:8670-8686`; the flags are declared at `:3530-3532`).
* A `/v1/systemone` request may carry an optional `"lora": true|false` of its own; absent, the
  config's setting decides. This is what lets the exit-criterion test (§5.1) ask the **same**
  server for the decision with and without the adapter, in one run, with one model in memory.
* `Engine.logits` (`serve/server.py:2144`) takes an `adapter` argument and writes the key when it is
  set. The adapter id is threaded through `_logits_row` (`:4483`) and `_systemone_trie` (`:4510`) so
  **the trie walk's extra rows carry the same adapter** - they are the same question's work and
  share the engine turn (`:4642-4669`).

### 2.4 How the forward pass branches

The delta is a separate low-rank add, computed from the same activation the base projection reads:

```
y_base = dequant(W) @ x_q            # today's path, unchanged
y      = y_base + s * (B @ (A @ x))  # the adapter's path, new
```

with `s = alpha / r` (PEFT's default scaling; the value comes from the adapter's config and is
carried in the index).

**The hook points (one per projection per path):**

| Path | Hook |
|---|---|
| Prefill (`src/prefill/prefill.cpp`) | After the `native_proj` / `bf16_proj` call for the targeted tensor, add `s * B @ (A @ x)` into the same float output buffer. The activation is `m.mixed_h` (the layer's half-read activation) or the projection's own input (`m.attn_h`, `m.y_h`, `m.sh_h`). |
| Verify (`src/core/verify.cpp`) | After the projection's branch joins, add the delta into the same float output buffer (`qkv + tb*C`, `qfull_ + ...`, `Kc`, `Vc`, `bo`, ...). |
| Single token (`src/core/layer.cpp`) | After the `gemv_quantized` / `s_gemv_q8k_split` call, into `out` / `b.q_full` / `b.attn32`. |

**The activation the LoRA reads.** The base path quantizes the activation to Q8_0 or Q8_K (the
per-tensor contract in `include/strata/core/weights.hpp:80-86`); the LoRA's `A` is not a quantized
base weight and wants the float activation. The float activation is already in a register or a
float buffer at each hook point (`xm` in the verify path, `m.mixed_h` in the prefill path), so the
LoRA can read it directly - **no extra quantization pass is needed**, and the LoRA's arithmetic is
in fp32 (the closest to the training-time bf16/fp16 adapter, and the most accurate).

**The flag, and the byte-identity.** In its naive form the delta is one launch per targeted
projection, and every launch begins with:

```
if (*on == 0) return;    // the flag: this request runs the base model
```

With the flag off the kernel **writes nothing**. The base output buffer is therefore bit-identical
to a run with no adapter at all - not "numerically equal", *bit-identical*, because no arithmetic
touches the buffer. This is the same guarantee the control vector gives and states for itself
(`include/strata/kernels/cvec.hpp:13-16`: *"With the flag off the result is bit-identical to the
engine without a vector"*).

**The latency cost.** The FLOPs are negligible; the **launches** are not. A full attention adapter is
up to 4 targeted projections on 12 layers plus 3 on 36 layers = 156 projections, so up to 156 extra
kernel launches per token if each is separate. At the engine's per-launch cost that is a
millisecond-scale tax on a ~30 ms/token decode (the log's `18.8 tok/s` to `30.9 tok/s` decode rates,
`strata-unsloth-ud-iq4_xs.log:70`, `:79`). **The delta must be batched per layer** - one kernel per
layer that does all of that layer's projections' `A` products and `B` products - not one per
projection. This is a design constraint on the kernel, not an optimization; §4 step 3 carries it.

### 2.5 The conversation cache and the FIFO

**The cache key must carry the adapter state.** The cache's match test today is
`if (e.cvec != cvec) continue;` (`include/strata/core/conversation_cache.hpp:257`). The adapter's
state joins it:

* `SavedConversation` gains an adapter field next to `cvec` (`:166`).
* `ConversationCache::best`, `drop_superseded` and `parked_prefix_tokens` take it (`:251`, `:374`,
  `:405`).
* `ConversationView` gains it (`include/strata/core/conversation_snapshot.hpp:85`).
* The live session's `cvec_cached` becomes a pair (adapter id + control-vector state), and the
  "the live state was read with something else" branch (`src/program/generate.cpp:12592-12596`)
  drops the live state and the checkpoints instead of resuming them - **exactly** what it already
  does when the control vector's state changes.
* The batch slots carry it too (`:12275`, `:12306`).

**Why this is not optional.** The decision request resumes the prompt from the cache. If the cached
K/V was computed **without** the adapter and the answer position is computed **with** it, the model
reads a context it was never trained to read. The two populations have to be kept apart, and the
control vector's machinery is the existing, tested example of how.

**The FIFO.** The decision path takes the engine's turn once per question and holds it across the
trie walk (`serve/server.py:4642-4669`, `:3719-3776`). So no other request can change the adapter
state between the base request and the walk's extra rows. Good.

**The one gap: `--batch`.** With batch slots, the chat path does **not** hold the FIFO
(`serve/server.py:3731-3734`, and the companion doc §13.4), so the adapter state could change
between two requests on the same engine, and the slots' states would mix. **The feature must be
refused with `--batch`** (the running config has no `--batch`, `strata-unsloth-ud-iq4_xs.json:3-39`).
Making it per-slot is a separate, larger change and is out of scope for 2.2.

**The MTP drafter.** The draft layer is a separate model (`--mtp <dir>`, 835 MiB, log line 13). It
proposes tokens from the **base** distribution; the main model verifies them against the
**adapted** distribution, so the acceptance rate falls when the adapter is on. For the LOGITS path
this is a small cost (`max_new` is 1 for a trie-walk row and 32 for the base request,
`serve/server.py:4493`, `:4643`). It must be measured, not assumed.

### 2.6 The guarantee that the GEN path is byte-identical

Four independent mechanisms, in order of strength:

1. **Structural.** The adapter is applied only when `req_logits_k > 0`, i.e. only on a line that
   was a `LOGITS` line. A `GEN` line cannot turn it on.
2. **The flag.** The delta kernel writes nothing when the flag is off (`§2.4`).
3. **The cache.** A `GEN` request resumes only state that was read with the same adapter state
   (`§2.5`), so it never mixes a base-model context with an adapted one, or the reverse.
4. **Empirical.** The exit-criterion test (`§5`) compares a chat turn's token ids across two engine
   runs, with and without the adapter loaded.

Mechanisms 1-3 are structural and hold by construction; mechanism 4 is the measurement that shows
them. The order matters: if 4 fails while 1-3 hold, the failure is in something else (the noise floor
of §6), and that is a finding, not a contradiction.

---

## 3. The MoE problem

### 3.1 Why an attention-only LoRA avoids the router entirely

The router (`ffn_gate_inp.weight {N, 512}`, `src/core/layer.cpp:405-420`) selects **10 of 512**
experts per token (`include/strata/core/geometry.hpp:54`). A LoRA on a routed expert has to be
applied to whichever experts the router picked, for each token, and the set changes per token. That
is the whole difficulty: the adapter's effect is entangled with the router's decision.

A LoRA on an **attention** projection has no such problem. The attention projections are dense and
always applied; the delta is a fixed function of the activation. There is no selection, no gather,
no scatter, and no dependence on which experts fired. **That is why roadmap item 2.3's
"attention-only" is the right first move, and it is a property of the architecture, not a
simplification.**

### 3.2 Why the shared expert is a separate item (roadmap 2.4)

The shared expert (`ffn_gate_shexp.weight {N, 640}`, `ffn_up_shexp.weight {N, 640}`,
`ffn_down_shexp.weight {640, N}`, `include/strata/core/family_layout.hpp:162-164`) is **dense and
always on** - one set of weights per layer, applied to every token. A LoRA on it is therefore
exactly as easy as a LoRA on an attention projection: no router, no expert selection, no gather.

It is a *separate* roadmap item (2.4) because it is an **escalation, not a default**: the roadmap
asks for it only "if 2.3 plateaus", and its exit criterion is a measured improvement over 2.3, with
"if none, revert to attention-only and note it". The engine-side cost of adding it later is small
(three more projections on 48 layers, one more branch in the same two hooks), so the sequencing is a
research decision, not an engineering constraint. **2.2 should build the hook so that the shared
expert can be added later without touching the kernel** - i.e. the hook takes a table of
(projection, A, B) pairs, and the shared expert is three more rows.

### 3.3 The numbers: what fits on this card

The adapter's size is `(rank x in + out x rank)` parameters per targeted projection, at 2 bytes
(fp16). The geometry is `include/strata/core/geometry.hpp:23-68`: `n_embd` 2560, `n_head` 24,
`head_dim` 256, `n_ff` 640, `n_expert` 512, 48 layers, 12 QSA + 36 GDN.

| Adapter | Parameters at rank 16 | fp16 size | Fits above the 1024 MiB threshold? |
|---|---:|---:|---|
| QSA attention only (12 layers x q,k,v,o) | 5.70 M | **11.4 MB** | yes - ~20 % of the 58 MiB headroom |
| All attention (QSA + GDN's `attn_qkv`, `attn_gate`, `ssm_out`) | 23.10 M | **46.2 MB** | marginal - ~80 % of the headroom |
| All attention + the shared expert (48 layers x 3) | 30.47 M | **60.9 MB** | **no** - exceeds the 58 MiB headroom |
| Routed experts, per-expert (2D, vLLM's default) | 3.775 G | **7.55 GB** | **no** - 130x the headroom |
| Routed experts, shared-outer (`enable_moe_shared_loras`) | 761 M | **1.52 GB** | **no** - 26x the headroom |

The headroom: the minimum free VRAM during the 235-item run was **1082 MiB** and **1206 MiB at
load** (companion doc §10.8, measured), against the engine's **1024 MiB** warning threshold on
Windows (`src/program/generate.cpp:1412` `kVramSafeMib` = 1024 under `_WIN32`, 512 elsewhere; warned
at `:11687-11694`). So the headroom above the threshold is **58 MiB during a run** and **182 MiB at
load**.

**Conclusion.** On this card, the only MoE-compatible adapter is **QSA attention-only at a modest
rank** (rank 16 -> 11.4 MB; rank 32 -> 22.8 MB; rank 64 -> 45.6 MB). The routed experts are out of
the question at any rank that matters. This is a hard number, and it settles roadmap item 2.5
("maybe: selected-expert LoRA") before any of it is built.

**A consequence for the training side (roadmap 2.3).** PEFT's standard
`target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]` names the **QSA** layers' projections
only. The 36 GDN layers' projections have different module names (not found in this repo - §7,
question 1). So "attention-only" for this model means **12 of 48 layers** unless the training
recipe also targets the GDN modules. The adapter's effect is then roughly a quarter of what the
phrase suggests. This must be settled before 2.3's training recipe is written, because it decides
the adapter's capacity.

### 3.4 vLLM's mixed-MoE LoRA format, and how it maps here

vLLM's `enable_mixed_moe_lora_format` (see `vllm/config/lora.py` and `vllm/docs/features/lora.md`
in vLLM) forces the universal 2D MoE LoRA wrapper (`FusedMoEWithLoRA`) regardless of the base
model's `is_3d_moe_weight`, so that **2D (per-expert, megatron-style)** and **3D (fused,
peft-style)** adapters can be served from one deployment. The caller declares the layout per
adapter with `is_3d_lora_weight`:

| Layout | `is_3d_lora_weight` | Keys |
|---|---|---|
| 2D, per-expert | `false` | `...experts.{idx}.gate_proj.lora_A.weight`, `...experts.{idx}.up_proj.lora_A.weight`, `...experts.{idx}.down_proj.lora_A.weight` - one set per expert |
| 3D, fused | `true` | `...experts.gate_up_proj.lora_A.weight`, `...experts.down_proj.lora_A.weight` - one tensor stacking all experts on the leading dim |

**The mapping onto this engine's names:**

| vLLM / PEFT name | This engine | Per-expert or shared |
|---|---|---|
| `...experts.{i}.gate_proj` | `blk.N.ffn_gate_exps.weight`, expert `i` | 2D |
| `...experts.{i}.up_proj` | `blk.N.ffn_up_exps.weight`, expert `i` | 2D |
| `...experts.{i}.down_proj` | `blk.N.ffn_down_exps.weight`, expert `i` | 2D |
| `...experts.gate_up_proj` | `blk.N.ffn_{gate,up}_exps.weight`, stacked | 3D |
| `...experts.down_proj` | `blk.N.ffn_down_exps.weight`, stacked | 3D |
| `...mlp.gate_proj` / `up_proj` / `down_proj` | `blk.N.ffn_{gate,up,down}_shexp.weight` | the shared expert - always one copy |

**What this engine should do differently, and why.** vLLM's own documentation warns that under
`--enable-mixed-moe-lora-format` the engine *trusts* the declared `is_3d_lora_weight`, does not
inspect the checkpoint, and **"a wrong declaration will load weights into the wrong stacked buffers
and silently produce garbage outputs, with no error at load time"**. That is exactly the failure
mode this repo's own rules exist to prevent (`include/strata/core/weights.hpp:20-27`, and
`include/strata/core/layout.hpp:1-15`, whose whole purpose is "the layer where a pack and a kernel
disagree"). **This engine must infer the layout from the file, never from a declaration**, and must
refuse an adapter it cannot check.

### 3.5 What happens when the adapter names a tensor the model does not have

The rule: **the adapter is rejected at load time, with the offending name in the message. Never a
partial load, never a warning, never a silent skip.**

The engine already knows everything needed to decide:

| The adapter names | The engine knows | The answer |
|---|---|---|
| `blk.0.attn_q.weight` (layer 0 is GDN) | `layer_kind(g, 0) == Gdn` (`include/strata/core/family_layout.hpp:141-144`, with `is_qsa_layer` at `include/strata/core/geometry.hpp:75-77`) | **refuse**: "layer 0 is a GDN layer and has no `attn_q.weight`; the full-attention layers are 3, 7, ..., 47" |
| `blk.3.attn_q.weight` with the wrong shape | the shape from `layer_tensors()` (`include/strata/core/family_layout.hpp:196-199`) | **refuse**: "`blk.3.attn_q.weight`: the adapter's A is [16, 2560] and B is [12288, 16]; the model's tensor is [2560, 12288]" |
| `blk.3.ffn_gate_exps.weight` (a routed expert) | the expert count and the FF width | **refuse with the size**: "a routed-expert adapter over all 48 layers needs 7.55 GB at rank 16 (157 MB for one layer); this card has 58 MiB above the engine's threshold" |
| a tensor that is not in `layer_tensors()` at all | `LayerView::get` returns null for it, and `check_layer` (`include/strata/core/layout.hpp:66`) checks the shapes at load time. The table is **not** complete for this family (`layout_table_complete` is true only for `qwen35moe`, `include/strata/core/family_layout.hpp:237`), so the loader must reject any name it cannot resolve. | **refuse**: "the adapter names `blk.3.foo.weight`, which this model does not have" |
| a rank above the compiled maximum | the engine's own limit | **refuse**: "rank 256 is above this engine's maximum (64)" |

The adapter's presence in the index is not a silent no-op. The whole adapter is refused, the engine
starts without it, and the `INFO` line says `lora=none` so the server knows the decision path must
not ask for it.

---

## 4. The plan, in reviewable steps

Each step is separately reviewable and separately testable. The exit criterion is §5.

### Step 0 - The feasibility spike (do this first; it is cheap and it can kill the item)

**Measured 2026-10-21: [phase2_lora_spike.md](phase2_lora_spike.md)** (the script is
`tools/decision_policy/lora_spike.py`). The outcome: the generated token ids **are** byte-identical
across repeats and across engine instances; the answer position's `L`/`Z` rows are **not** - so
the third row of the table below is the one that applies to the row, and this plan's §5.1 A4 has
to become a tolerance. What the plan should change before step 1 is listed in the report's §5.4.

**What it measures.** Two things, both of which the rest of the plan depends on and neither of
which is known. (Two more measurements belong to the spike and are written up in §6.1: the adapter's
real VRAM cost, and the number of projection call sites the delta has to cover.)

* **S0.1 - the engine's run-to-run determinism.** Is the same request's output the same bytes every
  time? Without this, the exit criterion ("byte-identical") cannot be tested at all.
* **S0.2 - the adapter's arithmetic.** Does the intended `A`/`B` orientation reproduce a host
  reference (numpy or torch) on random tensors?

**How.** S0.1 needs the GPU free for about two minutes (the running server holds it; only one
instance may run). Start the engine with the launch config's own arguments and speak the line
protocol on stdin/stdout:

```
GEN 16 ckpt=0 <ids>          # the same 200-token prompt, three times, with an unrelated request between
LOGITS 1024 16 ckpt=0 <ids>   # the same prompt again; compare the L/Z lines
```

`ckpt=0` is the engine's own one-shot mode (`src/program/generate.cpp:11983`): no checkpoint at the
turn boundary, no session parked, so each request reads the prompt from token 0 - the two runs are
comparable. Compare the `T` lines byte for byte, and the `L`/`Z` lines byte for byte.

S0.2 needs no GPU: a host reference for `y += s * B @ (A @ x)` with the transposed-`B` convention,
checked against numpy on random tensors.

**Files touched:** none. The spike is a script under `tools/decision_policy/` (say
`tools/decision_policy/lora_spike.py`) and its report.

**Risk:** none (it changes nothing).

**What the result means.**

| S0.1 result | Consequence |
|---|---|
| Identical bytes across repeats | The exit criterion as the roadmap words it is testable. Proceed. |
| Identical for `LOGITS`, different for `GEN` | The chat path has a run-to-run wobble (the companion doc §10.8 already records a ~0.6 Capability-point spread from the cache's layout). **Restate the exit criterion** to "the answer position's logits row is byte-identical", which is the property 2.2 actually changes, and note the restatement in the roadmap. |
| Different bytes even for `LOGITS` | The engine is not deterministic run to run; the exit criterion must become a **tolerance** ("the base model's logits agree to within X"), and the whole "byte-identical" framing of the roadmap item has to be renegotiated with the maintainer before any code is written. |

**S0.1 is also the measurement the companion doc's VRAM re-check needs**: the free VRAM at the start
of each request is already logged (`src/program/generate.cpp:11680-11696`), and
`tools/decision_policy/vram_headroom.py` reads it.

### Step 1 - The file format and the loader (host side; no GPU, no model)

**Status: DONE (2026-10-22).** Built: `tools/lora_pack.py` (the converter), `include/strata/core/lora_pack.hpp` +
`src/core/lora_pack.cpp` (the reader, the refusals, the host reference and the session digest), and two tests -
`src/core/lora_pack_test.cpp` (registered as `lora_pack_test` under `STRATA_BUILD_CONVERSATION_TESTS`) and
`tools/test_lora_pack.py`. Both pass. What that establishes, and what it does not:

* the layout is pinned from **both** ends: the same twelve fp16 values are asserted by the converter's test and by
  the reader's, so the orientation cannot drift between them, and `--validate` reads a written adapter back and
  compares it with the source **exactly** (it refuses a source that fp16 cannot hold);
* every refusal of §3.5 is a case in the reader's test: the wrong shape (showing the adapter's A, its B and the
  model's tensor), a name the model does not have, a QSA projection on a GDN layer (naming the layers that have
  one), a routed-expert tensor (with both of its sizes, reproducing §3.3's arithmetic from the geometry), a rank
  above the engine's maximum, half a pair, a stale blob, overlapping rows, a 1-D tensor, and a pack's
  `index.txt` handed in as an adapter's;
* the loader reads the model's own layout table (`layer_tensors`/`global_tensors`), so an adapter is checked
  against the model it will run on, not against a list written twice;
* **what it does not do**: it does not convert a full-model adapter yet. The converter maps the QSA projections
  and the shared expert, and **refuses every other module by name**, because the 36 GDN layers' Hugging Face
  module names are still open question 1 of §7 (the training recipe decides whether "attention-only" is 12 layers
  or 48). The refusal is the planned behaviour, not a gap in the loader: a guessed name would be a silently wrong
  adapter, which is the one failure this format cannot detect downstream.

**Files:** `tools/lora_pack.py` (the converter), `include/strata/core/lora_pack.hpp` + `src/core/lora_pack.cpp`
(the reader), `src/core/lora_pack_test.cpp` and `tools/test_lora_pack.py` (the two tests).

**Format, as built.** `index.txt` starts with `# strata lora index v1` (a pack's index has another header and is
refused by it), then `# name rank alpha ne0 ne1 offset bytes`, then one row per matrix. `lora.bin` holds fp16,
little-endian, in the index's order, and the loader requires it to be **exactly** the bytes the rows describe (a
leftover blob from another adapter is refused). `alpha` is PEFT's `lora_alpha` and the scaling is `alpha / rank`;
`use_rslora` is refused because its scaling is `alpha / sqrt(rank)`.

**The digest.** `lora_digest` hashes the rows (name, shape, rank, alpha), the scaling and the adapter's bytes.
Step 6 puts it in `SessionConfig` beside the control vector's digest; the test pins that one byte of weight or one
digit of alpha changes it.

**Risk.** The orientation (§2.1). A transposed `B` produces plausible numbers. The mitigation is the round-trip
test: the converter writes a known adapter, the reader reads it, and the host reference reproduces the same
product.

**Test.** A host test that (a) round-trips an adapter, (b) rejects a wrong-shaped A/B with the tensor's name in
the message, (c) rejects a tensor the model does not have, and (d) rejects a routed-expert tensor with the size in
the message. No GPU, no model - it runs in the ordinary test suite (`python -m pytest` / the CMake test targets,
`CMakeLists.txt:121-133`).

### Step 2 - The device tables, the flag, and the delta kernel (CUDA + HIP)

**Files:** `include/strata/kernels/lora.hpp` (new), `src/kernels/cuda/lora.cu` (new),
`CMakeLists.txt:366-413` (add the source), `src/kernels/lora_parity.cpp` (new).

**Work.** The upload (fp16 -> the device), the per-request flag (a device `int`, a host mirror, a
synchronizing setter), and the delta kernel: for one layer's set of targeted projections, compute
`A @ x` for each, then `B @ t` for each, and add `s *` the result into the base output.

**Risk.** The flag's semantics: it must be read on the device, so the kernel is graph-safe (the
control vector's is: `include/strata/kernels/cvec.hpp:27-28`, `src/kernels/cuda/cvec.cu:137-150`).

**Test.** `src/kernels/lora_parity.cpp`, mirroring `src/kernels/cvec_parity.cpp`:
with the flag **off**, the output buffer is **bit-identical** to the input (the kernel writes
nothing); with the flag on, the result matches a host reference to fp32 tolerance; the flag change
is synchronized across devices.

### Step 3 - The hook in the verify path (the highest-risk step)

**Files:** `src/core/verify.cpp` (the GDN branch `:1325-1400`, the QSA branch `:1460-1530`), and the
mirror `sycl/src/core/verify.cpp` **only if** the feature is to be built for SYCL (see the scope note
in §4.9).

**Work.** One delta-add per targeted projection, placed **after the branch joins** so that every
optimization variant (`side_proj` at `:1343`, `br` at `:1367`, `qbr` at `:1509`, `qb` at `:1504`,
`batch_rec_` at `:1379`) is covered by a single call site.

**Risk.** **This is the step where a mistake is silent.** Missing one branch produces a wrong
distribution, not a crash. The mitigation is a parity test that compares the adapted row against a
host reference for a fixed synthetic input, run with the branch switches **on and off**
(`STRATA_DF_BRANCH`, `STRATA_SIDE_MIXER`, `STRATA_DEC_BATCH`, `STRATA_DF_QB_Q4`). The engine's own
convention for this kind of check is a `_parity.cpp` target (`src/kernels/verify_parity.cpp`,
`src/kernels/verify_batch_parity.cpp`).

**Test.** The parity target above, plus the end-to-end check in step 8.

### Step 4 - The hook in the prefill path and the single-token path

**Files:** `src/prefill/prefill.cpp` (the GDN branch `:3189-3213`, the QSA branch `:3214-...`, the
`attn_output` projection `:3450`, the shared expert `:3525-3534`), `src/core/layer.cpp`
(`gdn_layer` `:251`, `qsa_layer` `:1130`).

**Work.** The same delta-add after each targeted projection.

**Risk.** Lower than step 3: the prefill path has far fewer branches. But the prefill path is where
the **context** is computed, so a missing hook there gives a distribution that is adapted at the
answer position and base-model everywhere else - exactly the inconsistency §1.4 warns about.

**Test.** The same parity target, extended to the prefill's GEMM shape (T tokens at once).

### Step 5 - The engine's request plumbing

**Files:** `src/program/generate.cpp`: the flag parser (`:2576-2606`), the usage text (`:932-936`),
the key parser (`:12001-12041`), the `want_lora` computation next to `want_cvec` (`:12194`), the
switch next to `cvec_set_enabled` (`:12597`), the `INFO` line (`:10735-10741`).

**Work.** `--lora <dir>`; the `lora=` request key; the flag set/cleared between requests; the
`INFO` line's `lora=` field; **the refusal when `req_logits_k == 0` and `lora=` was asked for** (a
`GEN` line must never apply the adapter - the engine refuses it rather than ignoring it).

**Risk.** Low. The main risk is the engine's graph capture: the adapter's kernel must be in the
captured graph from the start, which is why the adapter is loaded at start-up and never at run time.

**Test.** The engine's own line-protocol test (the repo has `src/program/message_boundary_test.cpp`
and `src/program/conv_cache_test.cpp` for this kind of thing).

### Step 6 - The conversation cache and the session identity

**Files:** `include/strata/core/conversation_cache.hpp` (`:166`, `:251`, `:374`, `:405`, `:422`),
`include/strata/core/conversation_snapshot.hpp` (`:81-85`),
`src/core/conversation_state.cpp` (`:312`, `:338`),
`include/strata/core/conversation_file.hpp` (`:71-83`), `src/core/conversation_file.cpp`
(`:691`, `:770-776`, `:896`, `:1009`), `src/program/generate.cpp` (`:9047`, `:9108`, `:12259`,
`:12353`, `:12592-12597`, and the batch slots' `cvec` fields).

**Work.** The adapter state becomes part of the cache key and of the session file's identity.

**Risk.** **The highest correctness risk in the whole item.** A missed key field means a decision
request resumes a base-model context (or a chat turn resumes an adapted one) and produces a
distribution the model was never trained to produce - silently. The control vector's fields are
the checklist: `SavedConversation::cvec`, `best`, `drop_superseded`, `parked_prefix_tokens`, `put`,
`ConversationView::cvec`, `ConversationState`'s copy, the session file's `cvec` field and its
fingerprint, the live `cvec_cached`, the batch slots' `cvec`, and the "the state was read with
something else" branch.

**Test.** `src/core/conversation_cache_test.cpp` already exercises the control vector's key; the
adapter's key gets the same treatment (a cached conversation read with a different adapter state
must not match).

### Step 7 - The server side

**Files:** `serve/server.py`: `Engine.logits` (`:2144`, the line at `:2168`), `_logits_row` (`:4483`),
`_systemone_trie` (`:4510`), the decision question (`:4642-4669`), the config reading
(`:8670-8686`), the flags (`:3530-3532`), the `INFO` parsing (`:1036`).

**Work.** The `systemone.lora` config key; the optional per-request `"lora"` field; the `lora=` key
on the `LOGITS` line; the adapter id threaded through the trie walk.

**Risk.** Low, but the wire format is a known trap: the ids must come last, and the key must come
before them (`src/program/generate.cpp:12008-12009`). The roadmap's own history has this exact
bug: the engine once answered `ERR expected: LOGITS <k 1..1024> <max_new> <id,id,...>` because the
server wrote `LOGITS <k> <ids>` (companion doc §10.5), and every decision silently fell back to the
decode path.

**Test.** `serve/test_logits_first_token.py::EngineWireFormat` (`:651-696`) gains a case that pins
the line with and without the adapter, on both sides (the server's writer and the engine's parser
grammar). The mock engine (`serve/server.py:317-369`) is extended to record the adapter id it was
asked for.

### Step 8 - The exit criterion and the VRAM re-measurement

See §5. This is the step that closes the item.

### 4.9 Backends (a scope decision, stated plainly)

The repo's rule is "build every backend a file touches (CUDA, HIP, SYCL)". The LoRA kernel lives in
`src/kernels/cuda/`, which is compiled for **both** CUDA and HIP (HIP shares the sources through
`include/strata/hip_compat/`, `CMakeLists.txt:66-104`), so the CUDA and HIP builds both cover it.

**SYCL and Metal are separate source trees** (`sycl/src/...`, `metal/src/engine/...`; the SYCL build
is a separate CMake project, `CMakeLists.txt:14-20`, `sycl/CMakeLists.txt`), and this plan does not
touch them. The consequence is that **`--lora` does not exist in the SYCL and Metal builds** and
those engines refuse it as an unknown argument (the CUDA/HIP parser's rule:
`src/program/generate.cpp:2661-2667` - *"An unknown flag is an ERROR and not a warning"*). That is
the honest position for a first implementation: the feature is CUDA/HIP-only, and the other backends
are unchanged (their default paths are byte-identical, which is what the rule is about). Porting it
to SYCL and Metal is a follow-up, and it should be scoped as its own item rather than smuggled into
2.2.

### 4.10 What 2.2 does **not** do

* It does not build a multi-adapter registry. One adapter, loaded at start-up, selected per request.
  vLLM's `max_loras`, `LoRARequest` and `/v1/load_lora_adapter` are a different scale of feature.
* It does not support batch slots (`--batch`) - refused, §2.5.
* It does not support the routed experts - refused with the size in the message, §3.3.
* It does not fuse the adapter into the base weights. The base weights are never written.
* It does not change the default path: with no `--lora`, no adapter is uploaded, no flag buffer
  exists, the graphs are exactly today's, and the `INFO` line says `lora=none`.

---

## 5. The exit criterion, concretely

**The roadmap's words:** *"a unit/integration test showing chat output is byte-identical with and
without the adapter loaded, while a decision request uses the adapter."*

### 5.1 The test

**Where it lives:** `tools/decision_policy/lora_parity.py` (a script, because it drives the real
engine over its real pipe), plus its report in `tools/decision_policy/` in the companion doc's
§10 style.

**What it runs.** Two engine runs, **sequential**, one model in memory at a time:

| Run | Engine arguments | What it does |
|---|---|---|
| A | the launch config's arguments, **without** `--lora` | 1) one chat turn (`POST /v1/chat/completions`), record the token ids and the text; 2) one decision request (`POST /v1/systemone`), record the answer and the distribution |
| B | the same arguments, **with** `--lora <dir>` | 1) the same chat turn, record the token ids; 2) the same decision request **with** the adapter (`"lora": true`); 3) the same decision request **without** the adapter (`"lora": false`, §2.3) |

**What it asserts.**

| # | Assertion | Why it is the right property |
|---|---|---|
| A1 | **the chat turn's token ids in run B are byte-identical to run A's** | the roadmap's exit criterion, literally |
| A2 | the chat turn's **text** is byte-identical (the same assertion on the decoded bytes) | the same thing at the API level, where a consumer sees it |
| A3 | the decision's distribution **with** the adapter differs from the **without**-adapter distribution in run B | the adapter is actually applied - otherwise the feature is a no-op that passes A1 |
| A4 | the decision's distribution **without** the adapter in run B is identical to run A's - compared on the engine's raw `L`/`Z` rows (the ground truth) or, if the API is the only view, as exact equality of the returned numbers | the adapter's *presence* does not perturb the base path |
| A5 | the free VRAM at the start of every request stays **above 1024 MiB** (the engine's own warning threshold, `src/program/generate.cpp:1412`) | the roadmap's VRAM criterion (companion doc §10.8) |

A3 must be a **statistical** difference, not a bit difference: the adapter's effect on one
distribution is small by construction. The right test is that the top-1 token's probability moves
by more than the run-to-run spread measured in step 0, and that the KL between the two
distributions exceeds the spread. The companion doc §10.8 already gives the spread (a median
confidence movement of 0.004 and at most 0.11 across two runs of the same 235 items).

### 5.2 How it runs without a second GPU or a second model in memory

* **One GPU.** Both runs are on the same card, sequentially. The second run starts after the first
  engine has exited.
* **One model.** Run B loads the same base weights as run A, plus an adapter of ~11-23 MB. It never
  holds two copies of the backbone.
* **One engine at a time.** Run A's engine exits before run B starts. The roadmap's blocker #1
  ("GPU exclusivity") is respected: the measurement needs the card free, so it happens in a window
  where the running server is stopped, not alongside it.
* **No PyTorch.** The adapter for this test is a **synthetic** one: a deterministic pseudo-random
  `A`/`B` at rank 16 over the QSA projections, written by `tools/lora_pack.py` from a fixed seed.
  It is not a trained adapter and makes no quality claim. A trained adapter is 2.3's business.

### 5.3 Why the test is meaningful, and what would make it vacuous

The test is meaningful because A1 and A4 are **not** trivially true: the adapter's device
allocation, its flag, its cache-key field and its graph node all exist in run B, and any of them
going wrong (a stale flag, a mixed cache entry, a changed graph) shows up as a difference.

It would be vacuous if the adapter were never actually applied - which is exactly what A3 rules
out. A1 alone proves nothing; A1 + A3 together prove that the adapter is applied on the decision
path and not on the chat path.

### 5.4 The engine-level unit test that runs without the GPU

`src/kernels/lora_parity.cpp` (step 2): with the flag off, the delta kernel leaves the output
buffer **bit-identical**. That is the mechanical form of "the GEN path is byte-identical with and
without an adapter loaded", and it runs in the ordinary test suite.

---

## 6. The feasibility spike, in full

Step 0 above is the spike. This section states what it measures, what would make the item
infeasible, and the fallback question the roadmap asks about.

### 6.1 What it measures

| # | Measurement | Method | Decision it drives |
|---|---|---|---|
| M1 | **Is the engine's output reproducible request to request?** | the same `GEN`/`LOGITS` request with `ckpt=0`, three times, with an unrelated request in between; compare the bytes | whether the exit criterion can be worded as "byte-identical" at all |
| M2 | **Does the adapter's arithmetic round-trip?** | a host reference vs the converter + reader, on random tensors, including the transposed `B` | whether the orientation is right before any kernel exists |
| M3 | **How much VRAM does the adapter cost, really?** | the index's own sizes, and the free VRAM at the start of a request from the engine log (`src/program/generate.cpp:11680-11696`, read by `tools/decision_policy/vram_headroom.py`) | whether the 1024 MiB threshold survives (§3.3) |
| M4 | **How many projection call sites are there really?** | count the branches in `src/core/verify.cpp:1325-1400` and `:1460-1530` and in `src/prefill/prefill.cpp:3189-3213`, `:3214-...` | how big step 3 really is (this is the item's main cost) |

**M1 is the one that can make the item infeasible as specified.** If the engine's output is not
reproducible, then "byte-identical with and without the adapter loaded" is not a property any
implementation can have, and the criterion has to be renegotiated with the maintainer *before* the
work starts, not discovered at the end.

**M2 can make it infeasible too**, but only for the "transpose `B`" detail, which is fixable.

**M3 and M4 are sizing, not gates**: M3's numbers are already computed in §3.3 from the model's
geometry; M4 is a count.

### 6.2 The fallback, and whether the constraint is satisfiable another way

The roadmap's blocker #3 says: *"If it proves infeasible, the fallback is a separate, smaller
process for decisions - which violates the 'no second model in memory' constraint and should be
rejected."*

**The fallback is rejected.** A second process that answers decisions needs the backbone to answer
them; the backbone needs ~24.1 GiB and the card has 24.6 GiB with the running instance on it
(companion doc §1, §2.3). A second *smaller* model is a different research item (a trained policy
model), not this one, and it does not deliver the self-distillation the roadmap wants (the target
is the backbone's own thinking distribution, §3.C of the literature doc).

**But the constraint is satisfiable another way, and the evidence is in the engine today.** The
engine already holds a **second set of weights in the same process** and switches between them per
request:

* the **MTP draft layer** (`--mtp <dir>`), 835 MiB of VRAM on this machine
  (`strata-unsloth-ud-iq4_xs.log:13`: *"strata mtp: draft layer loaded, 835 MiB of VRAM (experts 675,
  dense 111)"*), with its own experts and dense weights, applied only when the request wants
  drafts;
* the **control vector** (`--control-vector-scaled`, `src/program/generate.cpp:1837-1918`), a
  per-request side-weight with a per-request switch (`cvec=0|1`, `:11978`, `:12012`) that is
  **part of the conversation cache's key** (`:12353`, `include/strata/core/conversation_cache.hpp:257`)
  and **part of the session file's identity** (`:9108`,
  `include/strata/core/conversation_file.hpp:80`).

So "a small extra set of weights resident beside the base model, applied per request, kept out of
the base model's way" is a **proven pattern in this engine**, not a new assumption. The LoRA is
36-76x smaller than the draft layer (11-23 MB against 835 MB). The roadmap's design line -
*"adapter loaded alongside base weights (small VRAM overhead); LOGITS command applies delta; GEN
command does not"* - is a direct copy of the control vector's design, which the engine has been
running.

**The honest statement of the risk.** The pattern is proven; the *kernel* is new. The thing that
could still fail is not "can a second set of weights fit" but "can the delta be applied at every
projection call site without a silent miss" (step 3). That is why step 3 is where the risk budget
goes, and why M4 counts the call sites before the work starts.

**If step 3 turns out to be infeasible** (too many branches to cover reliably), the honest outcome
is: **do not start 2.3**. The roadmap says the fallback should be rejected; this plan agrees and
adds that the correct response to an infeasible 2.2 is to stop, not to ship a second process.

---

## 7. The open questions

Each of these could not be resolved from the code in this repository.

| # | Question | Why it matters | What would resolve it |
|---|---|---|---|
| 1 | **What are the Hugging Face module names for the 36 GDN layers' projections?** The repo names them in GGUF (`attn_qkv`, `attn_gate`, `ssm_out`, `include/strata/core/family_layout.hpp:185-187`) but has no HF names for them. The QSA names are confirmed (`self_attn.{q,k,v,o}_proj`, `bench/results/2026-10-07-community-rtx-5080-docker-wsl2/mtp-inventory.md:31-35`). | It decides whether "attention-only" means 12 layers or 48 (§3.3), and therefore the adapter's capacity and the training recipe for 2.3. | The base checkpoint's `config.json` + `model.safetensors.index.json` for `Qwen3.8-Flash-Next` (the same files `tools/nvfp4_verify_gguf.py:44` and `tools/ple_fp8_pack.py:50` already read), or the upstream PEFT model's `adapter_config.json`'s `target_modules`. |
| 2 | **Does the engine's `attn_q.weight` really carry the output gate?** The layout says `{N, 2*n_head*head_dim}` (`include/strata/core/family_layout.hpp:196`) and `include/strata/core/layer.hpp:375-378` describes `attn * sigmoid(gate)`, and the MTP inventory's `q_proj` is 12288-wide, matching. But the *split* of the 12288 columns between query and gate is not written down in this repo. | A LoRA on the HF `q_proj` covers both halves; the engine's two halves may need different treatment (the gate half's delta would go through a sigmoid). | The upstream model's `config.json` (`q_lora_rank`, the head count, the gate flag), or a dump of the 12288 columns' norms. |
| 3 | **Is the engine's `LOGITS` answer row really the last prompt token's row in every configuration?** The reading of `src/program/generate.cpp:13701`, `:14550-14551` and `:14772` says yes for the default path; the merged-window path (`STRATA_ADMIT_MERGE_FIRST`, `:13707-13728`) changes which rows are in the first window. | It decides whether the adapter has to be applied to the *whole* first window or only to its last row. | A run with `STRATA_ADMIT_MERGE_FIRST=1` and the debug dump (`STRATA_DUMP_FIRST_LOGITS`, `:14757-14766`), comparing the row against the single-token case. |
| 4 | **What is the engine's per-launch overhead, and what does a per-layer batched delta cost?** | It decides whether the adapter is a 2 % or a 20 % tax on the decision latency (§2.4). | A measurement on the card, with and without the adapter, on the same prompt. The engine's own stage timing (`docs/DETAILS.md:417`) gives the per-stage breakdown. |
| 5 | **What does the MTP drafter's acceptance rate do when the main model runs with the adapter?** | The draft layer proposes from the base distribution; the main model verifies against the adapted one. | A measured draft acceptance rate with and without the adapter (the log already reports `drafts accepted N of M` per request, `strata-unsloth-ud-iq4_xs.log:70`). |
| 6 | **Is `strata`'s own `--serve` command list documented anywhere the plan should update?** The commands are dispatched at `src/program/generate.cpp:11620-11965`; `docs/MANAGEMENT_PROTOCOL.md` may describe them. | The plan should not leave the protocol undocumented. | A read of `docs/MANAGEMENT_PROTOCOL.md` (not read for this plan). |
| 7 | **Should the adapter also be applied to the MTP draft layer?** | The draft layer is a separate model with its own `self_attn.*_proj` weights (the MTP inventory). A LoRA on the main model's layers does not apply to it. | A decision by the maintainer; this plan assumes it does not apply, and says so in the adapter's documentation. |

---

## 8. A file:line index for the implementer

| Concern | Start here |
|---|---|
| The request keys and their parser | `src/program/generate.cpp:11999-12042` |
| The control vector (the pattern to copy) | `src/program/generate.cpp:1837-1918`, `include/strata/kernels/cvec.hpp`, `src/kernels/cuda/cvec.cu` |
| The per-request side-weight switch | `src/program/generate.cpp:11978`, `:12194`, `:12597` |
| The cache's key | `include/strata/core/conversation_cache.hpp:251-266` and the `cvec` field at `:166` |
| The session file's identity | `include/strata/core/conversation_file.hpp:47-83`, `src/core/conversation_file.cpp:691`, `:770-776`, `:896` |
| The model's tensor names and shapes | `include/strata/core/family_layout.hpp:154-218` |
| The model's geometry | `include/strata/core/geometry.hpp:23-68`, `:75-77` |
| The verify path (the risky one) | `src/core/verify.cpp:2663`, `:1325-1400`, `:1460-1530` |
| The prefill path | `src/prefill/prefill.cpp:3189-3213`, `:3214-...`, `:3450`, `:3525-3534` |
| The single-token path | `src/core/layer.cpp:251`, `:1130`, `:1633`; `src/core/session.cpp:804` |
| The LOGITS command | `src/program/generate.cpp:11933-11960`, `:14768-14805` |
| The server's `LOGITS` line | `serve/server.py:2144-2296`, the line at `:2168` |
| The decision path | `serve/server.py:4483-4504`, `:4510-...`, `:4620-4669` |
| The FIFO | `serve/server.py:3539`, `:3688-3716`, `:3719-3776` |
| The wire-format test to extend | `serve/test_logits_first_token.py:651-696` |
| The VRAM measurement | `tools/decision_policy/vram_headroom.py`; the engine's own reading at `src/program/generate.cpp:11680-11696` |
