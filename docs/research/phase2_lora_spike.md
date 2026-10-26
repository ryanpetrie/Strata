# Phase 2.2, step 0: the feasibility spike, measured

**Status:** measured 2026-10-21. It is step 0 of
[phase2_engine_lora_plan.md](phase2_engine_lora_plan.md) (§4 "Step 0", §6.1 M1 and M2), and it
decides whether roadmap item 2.2's exit criterion can be tested as written.

**What was measured and what was not.** Everything below is a measurement on this machine: the
engine ran, the requests were sent, the bytes were compared. The script is
`tools/decision_policy/lora_spike.py`; the raw results are in
`<opencode temp>\lora_spike\` - the temporary run directory,
`%LOCALAPPDATA%\Temp\opencode` on this machine - as `s01.json`, `s03-default.json`,
`s03-nocpushare.json`, `s02.json` and the three engine logs. The engine was **not** changed: no
file under `src/`, `include/`, `metal/` or `sycl/` was touched, and no code was written for the
adapter. One reading from the code is marked as unconfirmed static analysis (§6).

---

## 0. The answer in one page

| Question | Answer |
|---|---|
| Is the engine's output reproducible request to request? | **The generated token ids: yes, byte for byte. The answer position's logits row: no.** |
| Which of the plan's §4 Step 0 outcomes did the measurement produce? | **The third one** ("different bytes even for `LOGITS`") - for the logits row. The first one ("identical bytes across repeats") for the chat output. The verdict is split, and the split matters. |
| What does that mean for item 2.2's exit criterion? | The roadmap's criterion as written ("chat output is byte-identical with and without the adapter loaded") **is testable**, and the measurement supports it. The plan's proposed restatement ("the answer position's logits row is byte-identical") **is not testable** and must not be used. The plan's assertion A4 (the base path is unperturbed by the adapter's presence) must become a **tolerance**. |
| Is the engine deterministic at all? | The logits row converges once the expert cache stops moving: in one probe the last two identical requests gave **byte-identical** rows. The chat output was byte-identical across **three separate engine instances**. |
| Does the adapter's arithmetic round-trip (S0.2)? | **Yes.** The plan's on-disk format reproduces a numpy reference to float32 rounding (max abs diff 3.05e-4 on a vector of L2 norm 24344). But the plan's converter must transpose **both** A and B, not only B - see §5.4. |

---

## 1. What was run

| | |
|---|---|
| Machine | Windows, NVIDIA GeForce RTX 4090, 24564 MiB of VRAM |
| Engine | `C:\tools\strata\engine\strata.exe`, engine 0.1.45.3 (`engine/BUILD.json`, src `b778976b477b875f`) |
| Model | Qwen3.8-Flash-Next UD-IQ4_XS, the pack at `C:\tools\Strata-data\packs\unsloth-ud-iq4_xs` |
| Launch config | `C:\tools\strata\strata-unsloth-ud-iq4_xs.json` |
| The engine's command line | the config's `exe` and `args`, plus `--serve` - the same line the server's own log records for the live server (`strata-unsloth-ud-iq4_xs.log:139373`) |
| The engine's environment | the server's own `child_env()` (`serve/server.py:3258`) plus the config's `env` (`STRATA_HC_Q8=1`) |
| The tokenizer | the config's pack tokenizer, through the repo's own `tools/strata_tokenizer.py`, with the model's own `chat_template.jinja` |
| The server | **stopped for the measurement** (one engine instance may hold the card). The GPU had 23782 MiB free before the standalone engine started. The server was started again afterward with the same command line and the same config. |

The prompts are fixed text, so the token ids are the same in every run:

| Prompt | Tokens | What it is |
|---|---:|---|
| A | 283 | one user turn (the chat template's rendering), the request under test |
| B | 274 | one unrelated user turn (a soup recipe), the request that goes between two identical ones |
| C | 331 | A + the assistant's reply + a second user turn: the chat-shaped, growing conversation |

The requests, in the order they were sent (`ckpt=0` is the engine's one-shot mode, §5.1):

```
GEN 16 ckpt=0 <A>          gen_a1
GEN 16 ckpt=0 <B>          gen_b1      the unrelated request in between
GEN 16 ckpt=0 <A>          gen_a2
GEN 16 ckpt=0 <A>          gen_a3      nothing between it and the last one
GEN 16 ckpt=0 <B>          gen_b2      the unrelated request in between
GEN 16 ckpt=0 <A>          gen_a4
LOGITS 1024 16 ckpt=0 <A>  logits_a1
LOGITS 1024 16 ckpt=0 <A>  logits_a2
GEN 16 ckpt=0 <C>          gen_chat    the growing conversation
GEN 16 ckpt=0 <A>          gen_a5      the same request again, after the conversation
LOGITS 1024 16 ckpt=0 <A>  logits_a3
```

Every request's `DONE` line reports `reused=0` and "prompt tokens read" equal to the whole prompt
(283, 274, 331). So **each request read its prompt from token 0** and the runs are comparable, which
is what `ckpt=0` is for.

---

## 2. S0.1: the engine's run-to-run determinism

### 2.1 The generated tokens: byte-identical, everywhere

Every comparison of the `T` lines is byte-equal:

| Comparison | `T` bytes equal |
|---|---|
| A1 vs A2 (an unrelated request between) | **yes** |
| A2 vs A4 (an unrelated request between) | **yes** |
| A1 vs A4 (two unrelated requests between) | **yes** |
| A2 vs A3 (nothing between) | **yes** |
| A1 vs A5 (a conversation between) | **yes** |
| B1 vs B2 | **yes** |
| the `LOGITS` requests' `T` lines (three of them) | **yes** |
| the `LOGITS` requests' `T` lines vs the `GEN` ones | **yes** |

The generated ids were the same every time:

```
1596 1144 4087 1156 579 1622 13 14235 59987 3150 303 2250 22157 11 33791 8166
```

The two extra probes the plan asks for are therefore **negative**, which is the good case:

* **the order of the requests changes nothing** - an unrelated request between two identical
  requests does not change the answer;
* **a conversation between two identical requests changes nothing** - the same request after a
  chat-shaped request gives the same bytes.

And the ids are the same **across engine instances**, not only within one: the `S0.1` run, the
`S0.3` run and the `S0.3` run with a different engine setting gave the same 16 ids. The engine's
output therefore does not depend on the process or on the state the expert cache was in when the
request arrived - which is what the roadmap's A1 needs: the only difference between run A (no
adapter) and run B (the adapter) has to be the adapter.

### 2.2 The answer position's logits row: not byte-identical

Two identical `LOGITS` requests, nothing in between, and the rows differ:

| | `logits_a1` | `logits_a2` |
|---|---:|---:|
| the `L`/`Z` bytes | - | **different** (all 1024 rows differ) |
| the ids in the top 1024 | 1024 | 1024, of which **71 are not in the first one's** (953 are in both) |
| the top-1 id | 1596 | 1596 (the same) |
| the top-1 probability | 0.999775 | 0.999582 |
| the normalizer `Z` | 27.159889 | 26.380281 (a move of -0.779608) |
| the max abs move of a logit (over the ids in both) | - | 0.887 |
| the mean abs move of a logit | - | 0.213 |
| the total variation over the union of the two top-1024 rows | - | 1.9e-4 (each side leaves ~2e-6 below its cut) |

The same picture after the conversation (`logits_a1` vs `logits_a3`): the top-1 id is the same, the
max move is 1.008, the mean 0.220, `Z` moves -0.262, the total variation is 5.3e-5.

So the row **moves a little**, and in the `S0.1` run it was never the same bytes twice (three
pairs, three different rows), while the token it argues for does not change.

### 2.3 Why it moves: the data path, read from the engine's own counters

A follow-up probe (`S0.3` in the script) sent the same `LOGITS` request eight times, back to back,
and read the engine's own `DONE` counters:

| request | cache hits / lookups | expert blobs from the files | from the RAM copy | offloaded | the row vs the first |
|---|---:|---:|---:|---:|---|
| 1 | 5357 / 6997 | 57 | 1392 | 395 | - |
| 2 | 6089 / 7160 | 45 | 1144 | 291 | different |
| 3 | 6640 / 7378 | 41 | 897 | 201 | different |
| 4 | 6909 / 7497 | 38 | 847 | 151 | different |
| 5 | 7114 / 7593 | 29 | 645 | 112 | different |
| 6 | 7302 / 7700 | 26 | 551 | 84 | different |
| 7 | 7392 / 7746 | 19 | 505 | 75 | different |
| 8 | 7491 / 7790 | 21 | 538 | 68 | different |

The expert cache warms up while the requests run: the hit rate climbs from 76.6 % to 96.2 %, and
the blobs that come from the files and from the RAM copy fall by more than half. The engine's own
log says why that matters: *"the GPU computes the experts in the cache; it rounds differently from
the CPU, so a reply can differ slightly from a run without the cache"*. The same request is
therefore not always computed from the same experts in the same grouping, and the sum is rounded
in a different order.

**The CPU share of the prompt path is not the cause.** The engine turns it on by default for prompt
chunks below 1024 tokens (*"the idle CPU takes some of the experts the GPU would stream; answers
can differ slightly"*), so the probe was repeated with the engine's own `STRATA_PREFILL_CPU_SHARE=0`:

| probe | the rows byte-identical to the first | the `T` lines byte-identical |
|---|---:|---:|
| the default settings | 0 of 7 | 7 of 7 |
| `STRATA_PREFILL_CPU_SHARE=0` | 0 of 7 | 7 of 7 |

**A deterministic mode exists in practice.** With the CPU share off, the last two requests of the
probe (requests 7 and 8) gave **byte-identical** rows - the row had converged once the cache
stopped moving. With the default settings the cache was still moving at the last request (the file
blobs went from 19 to 21), so the rows did not converge. The convergence is therefore a property of
the cache's state, not a guarantee: the adaptive tier can move again, so this cannot be the exit
criterion. It is a useful diagnostic and a way to build a *tolerant* comparison.

### 2.4 The latency and the VRAM

| request | wall clock | the prompt read | the decode | the free VRAM at the start (nvidia-smi) |
|---|---:|---:|---:|---:|
| gen_a1 | 2.853 s | 2533.6 ms | 316.7 ms | 1641 MiB |
| gen_b1 | 2.113 s | 1797.3 ms | 312.3 ms | 1549 MiB |
| gen_a2 | 2.126 s | 1883.0 ms | 240.4 ms | 1549 MiB |
| gen_a3 | 2.249 s | 2066.9 ms | 179.7 ms | 1549 MiB |
| gen_b2 | 2.033 s | 1805.1 ms | 225.9 ms | 1549 MiB |
| gen_a4 | 2.682 s | 2474.4 ms | 205.6 ms | 1549 MiB |
| logits_a1 | 2.122 s | 1961.0 ms | 157.9 ms | 1549 MiB |
| logits_a2 | 2.207 s | 2038.7 ms | 165.3 ms | 1549 MiB |
| gen_chat | 2.126 s | 1854.6 ms | 268.4 ms | 1549 MiB |
| gen_a5 | 1.979 s | 1838.9 ms | 138.1 ms | 1535 MiB |
| logits_a3 | 1.908 s | 1743.8 ms | 161.0 ms | 1535 MiB |

The engine's own VRAM figures (from its log, read with the repo's
`tools/decision_policy/vram_headroom.py`):

| | |
|---|---:|
| free with everything loaded | 1153 MiB |
| at the 1-token window's capture | 1082 MiB |
| at the 4-token window's capture | 1068 MiB |
| at the 3-token window's capture | 1061 MiB |
| the minimum during the run | **1061 MiB** |
| samples below the engine's 1024 MiB warning threshold | **0** (the engine never warned) |

The live server's own engine logged 1176 MiB free at load (and 1206 MiB when it was started again
after the measurement). Mine logged 1153 MiB: the vision encoder was not running during the
measurement (the server was stopped), so the expert cache was bigger - 6445 experts and 14.53 GiB
against the live run's 5806 and 13.08 GiB - and the free VRAM at the end was within 23 MiB of the
live run's. **The engine's per-request free VRAM is only logged when it drops below the threshold**
(`src/program/generate.cpp:11680-11696`), so the per-request figures above are from nvidia-smi,
read just before each request line was written.

---

## 3. S0.2: the adapter's arithmetic (no GPU)

The plan's format (`docs/research/phase2_engine_lora_plan.md` §2.1) is a text index plus one
binary blob. The index's rows are the tensor's shape **in the engine's order (dim 0
contiguous)**:

```
# name rank alpha ne0 ne1 offset bytes
blk.3.attn_q.weight.lora_a  16  16.0     16  2560     0   81920
blk.3.attn_q.weight.lora_b  16  16.0  12288    16  81920  393216
```

so `lora_a` is `ne0=rank`, `ne1=in` and `lora_b` is `ne0=out`, `ne1=rank`. The engine's rule for
such a tensor: element `(i, j)` sits at `i + ne0*j`, so `ne0` is the contiguous dimension.

The measurement uses the plan's own example geometry (`blk.3.attn_q.weight` is `{in 2560, out
12288}`, rank 16, alpha 16, so `s = alpha/rank = 1`), random tensors from a fixed seed, and the
plan's formula `y += s * B @ (A @ x)`.

| | |
|---|---|
| the reference (numpy, float32) | `s * (B_peft @ (A_peft @ x))`, with `A_peft` `[16, 2560]` and `B_peft` `[12288, 16]` |
| as the plan's format does it | the file holds `A^T` and `B^T` row-major; read back with the engine's rule, the matrices are **exactly PEFT's** (`array_equal` true for both) |
| the delta against the reference | max abs diff **3.05e-4**, on a vector whose L2 norm is 24344 - the float32 rounding of the reference itself (against a float64 reference) is 2.53e-4 |
| if A is **not** transposed | max abs diff **1339**, cosine with the reference **-0.103**, norm 27175 |
| if B is **not** transposed | max abs diff **1255**, cosine **0.015**, norm 24515 |

**The verdict.** The plan's format and the plan's formula agree, and the arithmetic round-trips: the
orientation is right and the reference is reproduced to float32 rounding. The two ways to get it
wrong produce vectors of the **same magnitude** as the right answer (27175 and 24515 against
24344) and are uncorrelated with it - plausible numbers, silently wrong. That is the failure the
plan warns about, and it is real.

**The finding the plan does not have.** PEFT's safetensors keep the **last** dimension contiguous
(A's `in`, B's `rank`); the engine's format keeps **`ne0`** contiguous. Under the plan's own index
rows, **both** A and B need their bytes transposed on the way in - A's math shape maps directly,
but its bytes do not. The plan's §2.1 says only *"every LoRA `B` must be transposed on the way in"*
and that *A "maps directly"*. If the converter writes A's PEFT bytes under the index's
`ne0=rank`, the loader reads a scrambled matrix and the adapter is wrong by the numbers in the
table above, with no error anywhere. Step 1's round-trip test has to catch this (the plan already
asks for a host reference; it now has to check **both** tensors).

**The second finding: the plan mixes two storage conventions.** The index example says
`lora_b` is `ne0=out`, `ne1=rank` (out contiguous) and the §2.4 formula `B @ (A @ x)` needs the
mathematical matrix `B = [out, rank]`. The §2.1 orientation sentence says *"the engine wants
`[r, out]`"*, which is the other convention (rank contiguous, PEFT's own bytes) and does not fit
the §2.4 formula. Both conventions give the same arithmetic - the second one needs the expression
`(A @ x) @ B` instead of `B @ (A @ x)`, and it reproduces the reference to the same 3.05e-4 - but
only one of them can be in the file. Step 1 has to pick one and say which. The index example and the
formula, taken together, pick the first.

(The engine's `ne0`-contiguous rule is not a reading of the plan: it is the pack's own convention -
`include/strata/core/family_layout.hpp:12` ("shapes are in GGUF order (dim 0 contiguous)") and the
pack's `index.txt` rows, e.g. `blk.0.attn_qkv.weight ... 2560 10240`, where 2560 is `in` and
10240 is `out`.)

---

## 4. Which of the plan's three outcomes this is

The plan's §4 Step 0 table:

| The plan's outcome | What the measurement says |
|---|---|
| **1. Identical bytes across repeats** -> the exit criterion as the roadmap words it is testable | **True for the chat output.** The generated token ids are byte-identical across the 8 requests with the same prompt in the `S0.1` run (and 16 more in the two probes of §2.3), with unrelated requests and a conversation in between, and across 3 separate engine instances. |
| **2. Identical for `LOGITS`, different for `GEN`** -> restate the criterion to "the answer position's logits row is byte-identical" | **False, and the other way round.** The `GEN` output is byte-identical; the `LOGITS` row is not. |
| **3. Different bytes even for `LOGITS`** -> the criterion must become a tolerance, and the "byte-identical" framing has to be renegotiated with the maintainer before any code is written | **True for the logits row.** Two identical `LOGITS` requests never gave the same bytes. |

**So: the third outcome, with a split verdict.** The measurement produces the third outcome for the
part the plan's restatement is about (the logits row) and the first outcome for the part the
roadmap's own words are about (the chat output). The consequence is not "renegotiate everything":
it is that **the roadmap's exit criterion survives, and the plan's proposed restatement does not.**

---

## 5. What it means for item 2.2's exit criterion

### 5.1 The criterion, line by line

The plan's §5.1 lists five assertions. What the measurement says about each:

| # | The assertion | Verdict |
|---|---|---|
| A1 | the chat turn's token ids in run B are byte-identical to run A's | **testable as written.** The ids were byte-identical across repeats and across engine instances, with and without an unrelated request in between. This is the strongest form of the criterion and the measurement supports it. |
| A2 | the chat turn's text is byte-identical | **testable as written** (the same ids give the same text). |
| A3 | the decision's distribution with the adapter differs from the without-adapter one | **testable, but as a statistical difference with a stated noise floor**, not a bit difference. The plan already says so; the noise floor is now measured (§5.2). |
| A4 | the decision's distribution without the adapter in run B is identical to run A's, on the engine's raw `L`/`Z` rows | **not testable as written.** The `L`/`Z` rows are not reproducible byte for byte even with no adapter in the picture (§2.2). It must become a **tolerance**. |
| A5 | the free VRAM at the start of every request stays above 1024 MiB | **testable as written.** Measured: the minimum during the run was 1061 MiB, 0 samples below the threshold (§2.4). |

The plan's §5.4 (the engine-level unit test: with the flag off the delta kernel leaves the output
buffer **bit-identical**) is **unaffected**. That one is a property of the kernel, not of the
request path: with the flag off the kernel writes nothing at all, so no arithmetic touches the
buffer. It stays a byte-for-byte test.

### 5.2 The noise floor, measured, for the tolerance

| what | measured where | the number |
|---|---|---|
| the top-1 probability's run-to-run movement (this prompt, p1 = 0.9998) | the two probes of §2.3 (16 requests) | 2.4e-4 (0.999631 to 0.999873) with the default settings, 2.8e-4 (0.999598 to 0.999875) with the CPU share off |
| the total variation between two identical rows | §2.2, §2.3 | 1.9e-4 (the pair in §2.2), at most 2.5e-4 over all the pairs measured |
| the max move of a single logit (a tail token) | §2.2, §2.3 | 0.89 and 1.01 in §2.2, up to 2.9 in the probes |
| the confidence's run-to-run movement on the 235-item decision set | the companion doc §10.8 (already measured, not repeated here) | a median of 0.004, at most 0.11 |

The decision path's own number is the companion doc's: the confidence moves by a median of 0.004
and at most 0.11 between two runs of the same 235 items, and three picks flipped, all near-ties.
So A3's test should be written against **that** spread: the adapter's effect on the decision path
has to be larger than the run-to-run wobble of the same question, and A4's test has to be a
tolerance at the same scale. The plan's §5.1 already quotes the 0.004 and 0.11; this measurement
adds the engine-level row numbers for a single question.

### 5.3 The restatement, in the words it should take

The plan's §4 Step 0 offers one restatement. **It must not be used.** The row is not byte-
reproducible, so "the answer position's logits row is byte-identical" is a property no
implementation can have. The criterion that the measurement supports is:

> **A1/A2** - the chat turn's token ids and text are byte-identical with and without the adapter
> loaded, while the decision request uses the adapter.
> **A3** - the decision's distribution with the adapter differs from the without-adapter one by
> more than the run-to-run spread of the same question (a median confidence movement of 0.004 and
> at most 0.11, the companion doc §10.8).
> **A4** - the decision's distribution **without** the adapter, in the run that has the adapter
> loaded, agrees with the base run's to within that same spread (not byte for byte: the engine's
> own `L`/`Z` rows are not reproducible, §2.2).
> **A5** - the free VRAM stays above 1024 MiB.

This is the "third outcome" the plan describes: the "byte-identical" framing **of the logits row**
has to be renegotiated with the maintainer before any code is written. The framing **of the chat
output** does not.

### 5.4 What the plan should change, before step 1

1. **§4 Step 0's table, row 2** must be struck: the restatement it proposes is not available.
2. **§5.1's A4** must become a tolerance, with the noise floor written down (§5.2).
3. **§2.1's orientation paragraph** must say that **both** A and B are transposed on the way in
   (§3), and it must pick one of the two storage conventions for `lora_b` instead of mixing them
   (§3).
4. **§2.1's index rows** and **§2.4's formula** stay as they are: they agree with each other.

### 5.5 What is still testable, plainly

* The chat output's byte-identity (A1, A2) - the strongest thing the item claims, and it holds.
* The kernel's flag-off bit-identity (the plan's §5.4) - a kernel property, unaffected.
* The VRAM floor (A5).
* The adapter's effect as a **statistical** difference against a measured noise floor (A3).
* The base path's unperturbed-ness as a **tolerance** (A4).
* A **deterministic mode** for the raw rows, as a diagnostic: run the same request until the
  expert cache stops moving and compare the last two (observed: byte-identical once, §2.3). This
  is useful for debugging a failing parity test, and it is not a criterion.

---

## 6. The limits of this measurement

* **One prompt.** 283 tokens, and a very peaked answer (the top-1 token's probability is 0.9998).
  The bytes moved because the row moved, and the row moved by ~0.2 in logit space; on a flat
  distribution (the decision questions, where the top two are 0.26 against 0.26) the same movement
  would change the pick. The companion doc §10.8 is the better number for the decision path, and
  this measurement does not replace it.
* **One model, one card, one configuration** - the live config, unchanged.
* **16 generated tokens.** A longer generation has more chances to flip a token.
* **The engine's own request path, not the server's.** The server's FIFO and its conversation cache
  add their own effects (the companion doc §10.8's wobble); the server was stopped here, so this
  measurement is the engine's floor, not the server's.
* **The vision encoder was not running** during the measurement (the server was stopped), so the
  expert cache was bigger than the live run's (§2.4). The free VRAM at the end was within 23 MiB
  of the live run's.
* **The wobble's cause is read from the engine's counters and its own log lines** (the hit rate,
  the file and RAM blob counts, *"it rounds differently from the CPU"*). That it is the *grouping*
  of the streamed experts in the grouped GEMM that changes the summation order is **unconfirmed
  static analysis** - the counters show that the data path changes and the row changes with it,
  and no engine change was made to confirm the link.
* **S0.2 is arithmetic on random tensors**, not on a trained adapter. It says the orientation and
  the file format are right; it says nothing about an adapter's quality.

---

## 7. How to repeat it

```sh
# the host-only part (no GPU, no model)
C:\tools\strata\.venv\Scripts\python.exe tools\decision_policy\lora_spike.py --only s0.2

# the engine part (the server must be stopped: one engine instance may hold the card)
C:\tools\strata\.venv\Scripts\python.exe tools\decision_policy\lora_spike.py \
    --config C:\tools\strata\strata-unsloth-ud-iq4_xs.json --only s0.1 \
    --work <opencode temp>\lora_spike --out s01.json

# the follow-up probe: the same LOGITS request eight times, back to back
C:\tools\strata\.venv\Scripts\python.exe tools\decision_policy\lora_spike.py \
    --config C:\tools\strata\strata-unsloth-ud-iq4_xs.json --only s0.3 --repeats 8 --tag default
C:\tools\strata\.venv\Scripts\python.exe tools\decision_policy\lora_spike.py \
    --config C:\tools\strata\strata-unsloth-ud-iq4_xs.json --only s0.3 --repeats 8 --tag nocpushare \
    --env STRATA_PREFILL_CPU_SHARE=0
```

The script starts the engine itself (the launch config's `exe` and `args` plus `--serve`, with the
server's own `child_env()`), sends the request lines, and prints the byte comparisons. It changes
nothing in the engine and writes nothing next to the model store.
