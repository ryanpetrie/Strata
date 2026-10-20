# Shared-backbone System 1 decision policy in Strata

**Status:** Phase 0 and Phase 1 complete. Phases 2-6 not started.
**Date:** 2026-10-19
**Task:** explore whether Strata can serve a fast, specialized agent decision policy from the already-loaded
Qwen3.8-Flash-Next backbone, without a second model or a second copy of the weights.
**Result so far:** the prompt route is measured; the capture point is identified; the go/no-go question is answered
for the measured case. See [Go/no-go](#gonogo-recommendation).

This document separates what was **measured on this machine** from what was read out of the source. A source
reading is marked *static analysis*; it is not a runtime measurement.

---

## 1. Scope and constraints

The hard constraints from the handoff still hold:

- No second large model and no second copy of the base weights.
- No change to default behaviour, default model settings, or the user's model files.
- The server stays on `127.0.0.1` (the running instance is bound to `0.0.0.0`, see below); API-key behaviour is
  preserved. No key, private content, or secret is written to any artifact here.
- Experimental engine features are opt-in; ordinary inference is unchanged.
- The machine cannot run two Strata instances: the running one already holds ~24.1 GiB of the RTX 4090's 24.6 GiB.
  Nothing here starts a second instance.

---

## 2. Phase 0 - the checkout and the running system

### 2.1 Two clones, and the running one is not this workspace

| | This workspace | The running server |
| --- | --- | --- |
| Path | `C:\tools\strata-dev` | `C:\tools\Strata` |
| Branch | `main` | `main` |
| Commit | `e88ee39d` | `477ac13c` |
| Tag | `v0.2.0-beta-12-ge88ee39d` | `v0.1.45` |
| `serve/server.py` | 8078 lines, hash `F3EC10DB…` | 7159 lines, hash `1D50A9AD…` |
| Engine reported by `/props` | - | `build_info: "Strata 0.1.45"` |

The workspace CMake project version is `0.1.45.3` (`CMakeLists.txt:3`), which matches `AGENTS.md`. The running
server reports `0.1.45`. **The server that is up is not built from this workspace.** This matters: any engine
change made here has to be built and run before it can be measured, and that cannot happen while the current
server holds the GPU.

The working tree is clean apart from the handoff document and the new `tools/decision_policy/` and
`docs/research/` files:

```
$ git status --short --branch
## main...origin/main
?? strata_shared_backbone_decision_policy_handoff.md
```

### 2.2 The model

* Identifier: `qwen3.8-flash-next-unsloth-ud-iq4_xs`
* Architecture: `qwen4exp` (Qwen3.8-Flash-Next), 48 layers, 24 attention heads, KV heads 2
* **`n_embd` = 2560** (read from the GGUF header, `qwen4exp.embedding_length`)
* Context: 262,144 tokens
* Quantization: Unsloth UD-IQ4_XS
* Weights: three GGUF shards plus a 1.49 GB `dense.bin` pack

The capture vector this task cares about is therefore 2560 floats (10,240 bytes). A linear head over 5 actions is
`2560 × 5 + 5 = 12,805` parameters.

### 2.3 The machine

* GPU: NVIDIA GeForce RTX 4090, 24,564 MiB total, **~24,127 MiB used** while serving
* RAM: 63.7 GiB total, ~54.3 GiB used
* Driver 617.14

There is no headroom for a second model process. This is why the baseline is API-only and why no engine build was
run.

### 2.4 How a request reaches the model

The engine does **not** speak HTTP. It speaks a **line protocol over stdin/stdout**; the Python server is the HTTP
front-end.

* Engine binary: `strata` (`src/program/generate.cpp`, `main()` at `generate.cpp:2214`; built by `CMakeLists.txt:724`).
* Server side: `serve/server.py`. `Engine.generate` (`server.py:2124`) writes one `GEN <max_new><keys> <id,…>` line
  per request (`server.py:2137-2141`) and reads `T <id>` / `PP` / `DONE` / `ERR` from stdout (`server.py:2186-2217`).
* The engine does prefill and decode internally; the server does not call it once per token.
* Endpoints: `POST /v1/chat/completions`, `/v1/messages`, `/v1/responses`; `GET /v1/models`, `/models`, `/props`,
  `/slots`, `/health`, `/v1/status`, `/v1/telemetry`, `/mcp`. **There is no embeddings, logits, or hidden-state
  endpoint.**

### 2.5 Where the capture point is

This is the core engineering question, and it has a clean answer. *Static analysis, verified against source:*

* `BlockBuffers::mixed` is a `n_embd` float buffer (`include/strata/core/layer.hpp:457`).
* `lm_head_mix` (`src/core/layer.cpp:1377`) fills `mixed` from the final residual `R` through the
  `output_hc_norm/down/up` weights - this is the model's final normed hidden state.
* `lm_head` (`src/core/layer.cpp:1398`) then quantizes `mixed` to Q8_K and runs the `output.weight` GEMV to get the
  vocabulary logits. So **`mixed` is exactly the vector immediately before the vocabulary projection.**
* The plain `generate` mode already writes it: `--dump-mixed PATH` (`src/program/generate.cpp:16312-16330`) does a
  `cudaMemcpy` of `ss.block.mixed` to host and prints its dimension and RMS.
* Siblings that already exist: `--dump-final-r` (`R`, `hc × n_embd`), `--dump-residual`, `--dump-logits`
  (raw logits), `--dump-layers`, `--dump-halves`, `--dump-routing`.
* The serve path (`if (o.serve)` at `generate.cpp:8044`) returns before the dump code, so **`--dump-mixed` is not
  reachable from the running server**. Returning the vector over the serve protocol needs a new command in the
  `GEN` dispatch (`generate.cpp:11933`) and a new reader branch in `server.py`.

So: the representation exists, is named, and is one `cudaMemcpy` away. The missing piece is a serve-side command,
not the representation itself.

### 2.6 Proposed minimal intervention point

If this goes further, the smallest change is a new opt-in serve command (for example `HIDDEN <max_new> <ids>`) that
runs the existing prefill, copies `ss.block.mixed` for the last prompt position to host once, and returns it. It
would touch `src/program/generate.cpp` (dispatch + copy), `serve/server.py` (reader + endpoint), and nothing in the
layer or backend code. It would be opt-in and would not alter the `GEN` path.

---

## 3. Phase 1 - the API-only decision baseline

### 3.1 Method

* Harness: `tools/decision_policy/baseline.py` (stdlib only; no new dependencies installed).
* Example set: `tools/decision_policy/examples.jsonl`, 60 examples over 5 actions
  (`CONTINUE`, `CALL_TOOL`, `ANSWER`, `ASK_USER`, `STOP`).
* Prompt: a short system line, the state as compact JSON, the legal actions, then `Next action:`.
* Settings: `reasoning_effort: "none"`, `temperature: 0`, `max_tokens: 8`, streaming on for the timed runs.
* The server's `response_format` is documented as `prompt_and_validate` (`/v1/status`), not a true grammar, so the
  reply is validated as a strict string: whitespace and one trailing `.` are tolerated, anything else is a failure.
* 2 warm-up requests per example (not timed), then 3 timed requests. 180 timed requests total.

**Label provenance.** The targets come from a rule written out in `tools/decision_policy/make_examples.py`
(`decide()`), which the generator asserts against every stated target. These are **synthetic, rule-based labels,
not reviewed real trajectories**. The accuracy below measures agreement with that rule. It is not a learned-policy
quality claim.

### 3.2 Results (measured)

| Metric | Value |
| --- | ---: |
| Timed requests | 180 |
| Errors | 0 |
| Invalid output rate | 0.0 |
| Accuracy | 0.75 |
| Balanced accuracy | 0.7883 |
| End-to-end latency, median | **158.30 ms** |
| End-to-end latency, p95 | 182.34 ms |
| Time to first token, median | 132.91 ms |
| Time to first token, p95 | 146.91 ms |
| Prompt tokens, median | 125 |
| Completion tokens, median | 3 |
| **First-request (cold) median** | **1605.88 ms** |
| **First-request (cold) p95** | **1661.95 ms** |
| Peak VRAM during the run | 24,132 - 24,133 MiB |
| Peak RAM during the run | 54.3 - 54.6 GiB |

Three runs were taken. The numbers are stable but not identical: accuracy 0.75 / 0.7667 / 0.7667, latency median
158.30 / 163.12 / 162.95 ms. The spread is small and does not change any conclusion.

Per-action precision / recall (latest run):

| Action | Precision | Recall | Support |
| --- | ---: | ---: | ---: |
| CONTINUE | 0.55 | 0.9167 | 36 |
| CALL_TOOL | 0.9167 | 0.55 | 60 |
| ANSWER | 0.6667 | 1.0 | 30 |
| ASK_USER | 1.0 | 0.875 | 24 |
| STOP | 1.0 | 0.6 | 30 |

Confusion (rows = target, columns = predicted):

| target \ predicted | CONTINUE | CALL_TOOL | ANSWER | ASK_USER | STOP |
| --- | ---: | ---: | ---: | ---: | ---: |
| CONTINUE | 33 | 0 | 3 | 0 | 0 |
| CALL_TOOL | 27 | 33 | 0 | 0 | 0 |
| ANSWER | 0 | 0 | 30 | 0 | 0 |
| ASK_USER | 0 | 3 | 0 | 21 | 0 |
| STOP | 0 | 0 | 12 | 0 | 18 |

The costly errors are `STOP → ANSWER` (12 of 30: the model answers after the budget is spent) and
`CALL_TOOL → CONTINUE` (27 of 60: the model keeps going when a tool should be called). The model is biased toward
`CONTINUE`/`ANSWER` and away from `CALL_TOOL`.

### 3.3 The decisive finding: cold versus warm

The single most important number is the gap between a new prompt and a repeated one.

* A **new** prompt costs about **1.6 s** (median 1605.88 ms, p95 1661.95 ms).
* A **repeat** of a prompt the engine has already seen costs about **158 ms**.

The engine's own log separates them:

```
prompt 125 tokens =   0 reused + 125 read in 1521 ms (82.2 tok/s), 2 generated in 30 ms
prompt 125 tokens = 118 reused +   7 read in   99 ms (71.0 tok/s), 2 generated in 29 ms
```

So the prefill reads about 70-84 tokens/s whether or not the prefix is reused; the difference is *how many* tokens
it has to read. The decode of 2-3 tokens is 28-36 ms in both cases.

Why a shared prefix does not help: the engine checkpoints the system-prompt root only when the system prompt is
**2,048 tokens or more** (`docs/DETAILS.md:1096-1106`). This decision prompt's system line is a few dozen tokens,
so no root checkpoint is saved, and a new state is read from token 0. The 4-slot conversation cache
(`--conversation-cache-slots 4`) only helps an *exact* repeat.

**This is the number that decides the task.** A decision head still has to run the backbone forward pass over the
decision state. It saves the decode and the vocabulary projection, not the prefill. And the prefill is the slow
part for a new decision state.

### 3.4 What the head would and would not save

* The head would avoid: the 248,320-way `output.weight` GEMV, the sampler, and the token decode.
* The head would still pay: the full prefill (the expert reads), which is ~1.6 s for a new state.
* Measured decode + head projection is ~28-36 ms.

| Case | Prompt route | With a head | Head saves |
| --- | ---: | ---: | ---: |
| New decision state | ~1.6 s | ~1.6 s | ~2 % |
| Reused / cached state | ~158 ms | ~125 ms | ~20 % |

---

## 4. Phases 2-6 - not started

* **Phase 2 (data):** the blocker is real. There is no labelled decision dataset from the user's agent workflow in
  this workspace, and no way to obtain one without running the user's agent and recording trajectories. The 60
  examples here are synthetic and rule-based and must not be read as real labels. Before any learned-policy claim,
  the data-collection path has to exist.
* **Phase 3 (capture):** not run. The capture point is identified (§2.5) and `--dump-mixed` exists in plain mode,
  but the serve path needs a new command, a build, and a run. That needs the GPU, which the running server holds.
* **Phase 4 (head):** not run. It would need PyTorch/numpy, which are not installed; installing them needs the
  user's approval.
* **Phase 5 (integration):** not run.
* **Phase 6 (comparison):** the three-way comparison cannot be done until 3 and 4 exist. The prompt route's
  numbers are in §3.2 and are the reference.

---

## 5. Go/no-go recommendation

**Do not build the policy head yet.** The measured reason is that the prefill dominates and the head cannot remove
it.

* For a new decision state, the head saves about 2 % of the end-to-end time. That is not worth an engine change,
  a build across backends, a new serve command, and the concurrency and portability work that comes with it.
* For a reused state, the head would save about 20 %. But a reused state is already fast, and the orchestrator can
  reach that regime today by sending a fixed preamble long enough to be checkpointed (2,048 tokens or more) with
  the variable state at the end. That is a server-side prompt-construction change, not an engine change.

The prompt-based fast path is the recommendation for now. The one thing that would change this is a measurement
showing that the decision prompt can be made prefix-cacheable at the orchestrator level; if that is demonstrated,
the head's ~20 % becomes worth revisiting, and Phase 3's capture command is the next step.

Two caveats on the quality side, both about the labels: the 0.75-0.77 accuracy is against a synthetic rule, not real
trajectories, and the costly `STOP → ANSWER` pattern is exactly the kind of error a real orchestrator must not
make. Neither can be acted on until Phase 2 produces real labels.

---

## 6. Path to a `systemone`-style decision API

The deliverable is the reference contract served by Strata's own resources: the loaded
model, the prompt cache, the existing server path. No second model, no heads, no training,
no engine changes. One greedy decode per request yields the argmax, so the distributions
are honestly degenerate and every response says `"calibrated": false`.

### 6.0 Final design (IMPLEMENTED in the workspace, 2026-10-19)

`POST /v1/systemone` with the reference shape: `state` (string/object/array, required,
non-empty, 128k chars) + `questions` (1-10 named questions, each `type`, literal
`instructions`, and per-type `criteria`: noul true/false descriptions, choice 1-255
options, score 2-10 ordered levels as a list *or* an object of short ids to
descriptions). One prompt renders state plus all questions; the model answers
`name: value` per line; strict parsing builds the reference response (`noul`,
`choice` + full distribution + confidence, `score` + legend + distribution; `usage`
with cached-tokens detail; `timings` when present). Score levels written as
`"id: description"` are sent as `{id: description}` with the answer expected as the
short id: the model was measured answering the short id unprompted against
full-string criteria, so exact reproduction of ~100-character strings is the wrong
demand - the object form matches its instinct while the description stays visible.

Contract points, all measured or tested:
- New `serve/systemone.py` (pure: validate, render, parse, build) + `Service.systemone`
  wired like every other route (admission, `begin_request`, dispatch, `dialects`).
  Opt-in flag, default off: `"systemone": {"enabled": true}`.
- Malformed model output is a 502 (`StructuredOutputError`), never a fabricated answer -
  the same rule as `serve/structured.py`. Truncation (`finish != stop`) is a 502 too.
- Retries/failures reuse the existing paths: 400 validation, 401 key, 409/429 admission,
  503 engine, `Retry-After` where the server already sends it.
- Non-streaming, no tools, no MCP, no images, no tool_choice in the request; the server
  forces temperature 0 and thinking off whatever the web Chat settings say.
- Live check against the running model: a 3-question prompt (choice + noul + score)
  was obeyed exactly and parsed into the reference shape on the first try.
- Tests: `serve/test_systemone.py`, 14 tests, all passing with MockEngine (no GPU);
  `serve/test_server.py` still passes whole (341 tests), so no existing route changed.

### 6.1 What the goal actually needs

A decision API that returns a **structured action with a confidence**, not text; is fast for a **new** decision
state; uses the **one loaded backbone**; and leaves normal chat untouched. The measured bottleneck is the prefill,
not the decode, so the plan is organised around making the prefill cheap.

### 6.2 Stage A - orchestrator-side, no engine change (MEASURED 2026-10-19)

Stage A was run against the live server. Harness: `tools/decision_policy/stage_a.py`. Two arms, 60 examples each,
2 passes per example (first pass = new state, second pass = repeat):

| Arm | New-state median | New-state p95 | Repeat median | Accuracy | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verbose preamble + verbose states | 1418 ms | 2263 ms | 1446 ms | 1.0 | 4744 |
| **Compact preamble + compact states** | **532 ms** | **653 ms** | **477 ms** | **1.0** | **3767** |
| Phase 1 short prompt, for contrast | 1606 ms | 1662 ms | 160 ms | 0.75 | 125 |

The gate question was: does a fixed preamble make NEW decisions fast? The measured answer is **yes, with the
compact encoding: 1606 ms down to 532 ms, a 3x speedup, with synthetic accuracy rising from 0.75 to 1.0.** No
engine change was needed. The verbose arm was re-run to confirm (1439 ms then 1417 ms: stable, not a cold-cache
artifact). The full reports are `tools/decision_policy/stage_a_report.json` and `.md`.

What the engine log shows for the compact arm:

```
prompt 3766 tokens = 3714 reused + 52 read in 371 ms, 2 generated in 33 ms, 2 checkpoints
```

The 3,714-token preamble is checkpointed and reused; only the ~52-token tail is read. The verbose arm reuses
4,669 and reads ~75. The cost model that fits every measurement is: **restore the checkpoint, then pay per NEW
tail token.** The worked examples (now 45, disjoint from the test set) teach the rule well enough that agreement
with the rule is perfect on this synthetic set.

Caveats, stated plainly: the 1.0 is agreement with the synthetic rule, not real-trajectory accuracy. The compact
encoding is cryptic by design; real agent states will be messier than these examples. And 532 ms is still slower
than an exact repeat (160 ms) - the floor is the checkpoint restore plus the tail read, and no prompt
construction can remove it. But for new states, which is what an agent mostly produces, 532 ms at temperature 0
with strict validation is a genuine System-1 API today.

### 6.3 What was built instead of heads (IMPLEMENTED in the workspace, 2026-10-19)

No heads, no training, no engine changes - per the user's direction the head track
(Stages B/C as originally sketched) is dropped. What exists is the full contract on
the existing model:

- `serve/systemone.py` (new, pure): limits, request validation, one-prompt rendering
  for all batched questions, strict per-type answer parsing, reference-shaped response
  building with `"calibrated": false`.
- `serve/server.py` (wired, +~70 lines): `systemone_enabled` flag (default off),
  config `"systemone": {"enabled": true}`, admission + `begin_request` + dispatch,
  `Service.systemone` (validate → prepare → `openai_chunks` → `openai_collect` →
  parse → respond), `/v1/systemone` in the `dialects` list.
- `serve/test_systemone.py` (new): 14 tests, all passing with MockEngine; the whole
  existing `serve/test_server.py` suite still passes (341 tests), so no route changed.
- Live check: a 3-question prompt (choice + noul + score) against the running model
  was obeyed exactly and parsed into the reference shape on the first try.
- Live failure that proved the contract: the first render put the name in a colon
  line (`lit (noul): ...`) and the model echoed it (`lit (noul): yes`) - a strict
  parse rejected it as a 502 instead of misreading it. The render was fixed the
  same day (colon-free `[name | type]` headers; the reply's `name: answer` is now
  the only colon shape in the prompt) and re-verified live via the chat path.
- Second live failure, same class: with a minimal one-question prompt the model echoed
  the *instructions* (`Is the light on?: yes`). Fixed with per-question
  `start your answer line with exactly "name:"` plus an enumerated `Valid line starts:`
  closer. Six live cases then passed via the chat path with zero invalid output,
  including both former 502s, a confusable 3-option choice, criteria-bound noul,
  backtick field references, and near-identical score levels.
- Deployed verification (workspace instance on port 8095, 2026-10-19): the full
  22-check suite against the real endpoint passes 22/22 - single and batched answers
  in all three types, all eight 400 validations, both 401s, and a chat regression.
  A 5-question mixed batch answers correctly in 2.8 s cold (0 cached) and 0.7 s on
  exact repeat (492/499 cached).

Deliberate restrictions, all documented in the module: question/option/level names
hold no newlines or colons (the `name: answer` line protocol); greedy temperature 0
always; malformed model output and truncation are 502s, never fabricated answers;
unknown model names are still served (consistent with chat); non-streaming only.

### 6.4 Deploy (DEPLOYED 2026-10-19)

Superseded by events: the directories were renamed (`strata-dev` to `strata`, old
install to `strata-old`) and the workspace tree itself now serves on port 8080 with
`"systemone": {"enabled": true}` - no porting step was needed. The engine reports
0.1.45.3. Verification against the deployed endpoint: the full 22-check suite passes
22/22 (single and batched answers in all three types, all eight 400 validations,
both 401s, chat regression), and a 5-question mixed batch answers correctly
(3.2 s cold, 0 cached). Rollback is the old tree at `strata-old` with its config.

### 6.5 Real data (the only remaining research)

The synthetic 1.0 is rule-agreement, not quality. The next step, whenever the user
wants it, is logging real decisions as `(state, questions, answers, outcome,
trajectory_id)` JSONL and freezing a trajectory-split eval set - the same pipeline
validates the endpoint as-is, no architecture change needed for it.

### 6.6 What each step needs from the user

| Step | Status | Needs |
| --- | --- | --- |
| Prompt research (Phase 1 + Stage A) | Measured. Compact preamble: 532 ms new-state, 1.0 synthetic accuracy. | Nothing further. |
| `/v1/systemone` implementation | Done in the workspace, tested (14 new + 341 existing green). | Approval to port + one server restart (§6.4). |
| Real-data validation | Not started. | Access to real agent trajectories, or a decision to build the collection path. |

---

## 7. Reproducing

```sh
# 1. build the frozen example set (asserts the rule matches every stated target)
python tools/decision_policy/make_examples.py

# 2. run the baseline against the already-running server
python tools/decision_policy/baseline.py --warmup 2 --runs 3 --out-dir tools/decision_policy

# 3. run the fixed-preamble experiment (verbose + compact arms)
python tools/decision_policy/stage_a.py --runs 2 --with-compact-arm --out-dir tools/decision_policy

# 4. the endpoint's own tests (venv python: system python lacks jinja2)
C:\tools\Strata\.venv\Scripts\python.exe -m unittest serve.test_systemone -v
```

The harness reads the address, model id, and API key from the launch config
(`C:\tools\Strata\strata-unsloth-ud-iq4_xs.json`) or from `--host/--port/--model/--api-key`/`STRATA_API_KEY`. It
starts no server and loads no model. Outputs: `baseline_report.json` and `baseline_report.md`.

Nothing here is committed by default: `baseline_report.json`, `examples.jsonl`, and any future artifacts are
generated data and should stay out of version control unless the user asks.

---

## 8. Files

| Path | What it is |
| --- | --- |
| `docs/research/shared_backbone_decision_policy.md` | this document |
| `tools/decision_policy/make_examples.py` | the schema, the label rule, and the frozen example generator |
| `tools/decision_policy/examples.jsonl` | 60 synthetic, rule-labelled examples (generated) |
| `tools/decision_policy/baseline.py` | the API-only baseline harness |
| `tools/decision_policy/baseline_report.json` | the measured results (generated) |
| `tools/decision_policy/baseline_report.md` | the same, as a table (generated) |
| `serve/systemone.py` | the endpoint's pure module: validate, render, parse, respond |
| `serve/test_systemone.py` | 14 tests, MockEngine, no GPU |
| `serve/server.py` | wired: flag + config, admission, dispatch, `Service.systemone`, dialects |
| `tools/decision_policy/opendecisions.py` | OpenDecisions harness: serial + batch arms, resume-safe, official predictions output |

## 9. OpenDecisions validation (2026-10-19, validation only - the set says do not train)

jgeuter/OpenDecisions v1.0.1 (CC-BY-SA-4.0): 2,352 JevBench-format decisions
(state + typed question + labels + expected; choice 1178, noul 586, score 588).
Its task format maps 1:1 onto the endpoint, including the type names. Data quirk:
one state holds a raw U+2028, so the harness splits lines on `\n` only.

10% sample (235 items, seed 42), one attempt each, misses omitted (counted wrong
per the scorer's rules). Official `score.py` on the sample:

| Metric | Value |
| --- | ---: |
| Capability | 65.6 (Intelligence 62.0, Calibration 69.2) |
| Accuracy on predicted (223/235) | 83.4% - choice 83.2%, noul 87.0%, score 62.7% |
| Format-failure rate | 12/235 = 5.1% (all 502s, no fabricated answers) |
| Trap picked | 14/135 = 10.4% |
| Latency median / p95 | 3.2 s / 5.4 s per item cold |

Context: above the no-training baseline on their board (SemIf 58.9), below trained
4B systems (67-77); Calibration is structurally capped by design
(`"calibrated": false`). Panel-labeled real requests: 92.6%; negation/perturb
pairs and option-order: 100/100/91.7% (order-robust); weak on jaggedness 58.3%
and composite computation 72.7% (arithmetic, as the reference docs warn).

Failure modes, all measured: score levels as bare/long strings the model
paraphrases instead of reproducing (fixed for `"id: description"` levels via the
object form); one truncation at the 32-token cap; and noul answers given as
probabilities (`0.7`, `0.8875`, `0.92`) on calibration items - the model
transcribing stated rates, which the strict parser rejected. The last finding
motivated accepting decimals in [0, 1] as noul answers (reported as given);
unit-tested, awaiting deployment.

Batched arm, mixed questions (47×5, seed 43): 175/235 predicted.
Official score on the sample: Capability 54.0 (Intelligence 35.3). Per-question
latency median 1.5 s vs 3.2 s serial (2.1x amortization; batch totals 7.5 s).
Serial-vs-batch agreement on co-predicted items: 87.6%. Batch accuracy ON PREDICTED
items is 86.3% vs 83.4% serial - batching barely changes the model's judgment
(only 6 serial-right items flipped to a wrong pick). The penalty is ~entirely
format failure: 12/47 batches (25.5%) fully failed with one bad line each,
losing 46 serial-correct items as collateral, vs 5.1% serial. Zero partial
batches: the endpoint is atomic per request, by construction. Conclusion: batch
for speed, fall back to serial per item on a 502 - the hybrid keeps the
amortization and recovers the collateral.

Batched arm, same question × 5 evidences (47 batches, 47 distinct repeated
question signatures, seed 44): 230/235 predicted, Capability **66.5**
(Intelligence 60.6), correct 187/235 = **79.6%** vs 79.1% serial. Format failures
nearly vanish: 5/235 (2.1%, all one composite_gen batch with bare-sentence
levels) vs 25.5% mixed. Per-question latency median 1.1 s (**2.9x vs serial**).
No inertia signature: uniform model answers in 3/46 batches vs uniform gold in
4/46. Caveat: this arm ran with decimal-noul deployed (serial/mixed ran strict),
so its calibration-item TVD (0.272 on 60 items vs 0.546/0.400) is not comparable;
accuracy and failure-rate comparisons stand. Conclusion: same-question batching
preserves serial accuracy, formats cleaner than serial itself, and is the right
shape for an agent's fixed-vocabulary loop.
