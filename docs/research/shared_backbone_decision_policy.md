# Shared-backbone System 1 decision policy in Strata

**Status:** the endpoint is built and measured (§10, §11); the engine's stale-context
failure is under investigation and the evidence is corrected (§12).
**Date:** 2026-10-21 (the sections before §6 are the 2026-10-19 record, kept as it was
then)
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

**2026-10-20: resolved.** The directories were renamed and the workspace *is* the
running instance now (`C:\tools\strata`, port 8080, the config with the API key);
the table below is the Phase-0 snapshot of 2026-10-19, kept as it was read then.
The practical consequence of the snapshot - "any engine change made here has to be
built and run before it can be measured, and that cannot happen while the current
server holds the GPU" - still holds, and §10's measurements were taken by restarting
the server with the new code.

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

**2026-10-20: the head track was dropped by the user, and the work took another
road.** The phases below are the original plan for a *learned* policy: capture the
backbone's hidden state, train a small classifier on it, and serve the decision
without generating text. That plan is parked, not abandoned - §5 explains why the
head was not worth it for speed, and the same capture point (§2.5, §2.6) is what a
*quality* play would use, if the user wants one. What shipped instead is the
endpoint itself (§6) and, from 2026-10-20, the confidence read from the model's own
next-token distribution (§10). The phases as written:

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
no engine changes.

**§10 and §11 are the current state of this work.** This section and the ones before it are the
record of how it got there, written on 2026-10-19 and kept as it was then: the endpoint
that answered with a greedy decode and a degenerate distribution, the design decisions
that led to it, and the measurements behind them. Where they disagree with §10 and
§11, §10 and §11 are right. The four things that changed after this section was
written:

- The confidence is no longer degenerate. §10.1: the endpoint reads the model's own
  next-token distribution instead of sampling it, so a response carries a real
  probability and says `"calibrated": true`.
- The first-token path is no longer the whole story. §10.6: the options that share a
  first token are separated by a trie walk, and the fast path now carries 233 of the
  235 sample items.
- The end-to-end numbers moved: Capability 65.6 → 79.5 on the 235-item sample
  (§10.8).
- Several questions are no longer answered by one shared prompt. §11: every question
  is answered on its own, so every answer carries the model's own probability, and a
  question that fails does not cost the others their answers. Measured on the
  235-item sample: Capability 79.0 with every item predicted, against the shared
  prompt's 49.5 with 165 of 235 (§11.5).
- The engine's stale-context failure, which the 2026-10-20 handoff recorded as the
  cause of every long run's failures, is **not** what the evidence supports. §12:
  the log line that was read as a proof is the trie walk's own request, the failed
  items pass on a healthy engine, and the failure does not reproduce.

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
- Tests: `serve/test_systemone.py`, all passing with MockEngine (no GPU);
  `serve/test_server.py` still passes whole, so no existing route changed.
  (The counts in this section are the 2026-10-19 ones; the suite has grown since -
  `serve/` is 945 tests today.)

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
  building. (On 2026-10-19 every response said `"calibrated": false`; §10.1 replaced
  that with the model's own probability.)
- `serve/server.py` (wired, +~70 lines): `systemone_enabled` flag (default off),
  config `"systemone": {"enabled": true}`, admission + `begin_request` + dispatch,
  `Service.systemone` (validate → prepare → `openai_chunks` → `openai_collect` →
  parse → respond), `/v1/systemone` in the `dialects` list.
- `serve/test_systemone.py` (new): all passing with MockEngine; the whole
  existing `serve/test_server.py` suite still passes, so no route changed.
  (The counts here are the 2026-10-19 ones.)
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

### 6.5 Real data

The synthetic 1.0 is rule-agreement, not quality. The next step, whenever the user
wants it, is logging real decisions as `(state, questions, answers, outcome,
trajectory_id)` JSONL and freezing a trajectory-split eval set - the same pipeline
validates the endpoint as-is, no architecture change needed for it.

**2026-10-20: the user put this last.** A published dataset (the OpenDecisions set,
§9) is the reference for the endpoint's quality, and it is what the measurements
use; what the trajectories would add is the right answer in the *user's own*
workflow, which needs the user's notion of a good decision. Two other ways to use
the user's own data exist and cost less: the examples can live in the cached prompt
preamble (§6.2), and a policy head over the backbone's hidden state is the
architecture's training path (§2.5, §2.6).

### 6.6 What each step needs from the user

| Step | Status | Needs |
| --- | --- | --- |
| Prompt research (Phase 1 + Stage A) | Measured. Compact preamble: 532 ms new-state, 1.0 synthetic accuracy. | Nothing further. |
| `/v1/systemone` implementation | Deployed and serving (§6.4); the confidence came from the model's own distribution afterwards (§10.1). | Nothing further. |
| Real-data validation | Not started, and last in the user's priority order. | Access to real agent trajectories, or a decision to build the collection path. |

---

## 7. Reproducing

The early phases (the synthetic example set, the API-only baseline, the fixed-preamble
experiment):

```sh
# 1. build the frozen example set (asserts the rule matches every stated target)
python tools/decision_policy/make_examples.py

# 2. run the baseline against the already-running server
python tools/decision_policy/baseline.py --warmup 2 --runs 3 --out-dir tools/decision_policy

# 3. run the fixed-preamble experiment (verbose + compact arms)
python tools/decision_policy/stage_a.py --runs 2 --with-compact-arm --out-dir tools/decision_policy
```

The endpoint and its measurements (the server must be up, with `"systemone":
{"enabled": true, "diagnostics": true}` in the launch config):

```sh
# 4. the probe: the fast path fires or it does not, per case
python tools/decision_policy/logits_check.py

# 5. the evaluation: the OpenDecisions sample through the endpoint
python tools/decision_policy/opendecisions.py --items <D>\opendecisions.jsonl \
  --out <D>\sample10.jsonl --ids-file <D>\sample10.txt --batch-k 1

# 6. the score (the official scorer) and the two readings against the outcome
python <D>\score.py <D>\sample10.jsonl --data <D>\sample10_items.jsonl
python tools/decision_policy/readings_compare.py --items <D>\sample10_items.jsonl \
  --predictions <D>\sample10.jsonl --runlog <D>\sample10.runlog.jsonl
python tools/decision_policy/readings_split.py --items <D>\sample10_items.jsonl \
  --runlog <D>\sample10.runlog.jsonl

# 7. the options' shape and the walk's cost
python tools/decision_policy/trie_costs.py --items <D>\sample10_items.jsonl
python tools/decision_policy/trie_examples.py --items <D>\sample10_items.jsonl

# 8. the tests (venv python: system python lacks jinja2)
C:\tools\strata\.venv\Scripts\python.exe -m unittest serve.test_systemone serve.test_logits_first_token -v
```

`<D>` = `C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions` - the data, the
predictions and the run logs live outside the repo on purpose (the dataset's rules
say do-not-train; keep it out of git).

The harness reads the address, model id, and API key from the launch config
(`C:\tools\strata\strata-unsloth-ud-iq4_xs.json`) or from `--host/--port/--model/--api-key`/`STRATA_API_KEY`. It
starts no server and loads no model. Outputs: `baseline_report.json` and `baseline_report.md`.

Nothing here is committed by default: `baseline_report.json`, `examples.jsonl`, and any future artifacts are
generated data and should stay out of version control unless the user asks.

---

## 8. Files

| Path | What it is |
| --- | --- |
| `docs/research/shared_backbone_decision_policy.md` | this document |
| `docs/research/strata_shared_backbone_decision_policy_handoff.md` | the original brief (historical) |
| `docs/research/systemone_handoff.md` | the 2026-10-19 session handoff (historical) |
| `tools/decision_policy/make_examples.py` | the schema, the label rule, and the frozen example generator |
| `tools/decision_policy/examples.jsonl` | 60 synthetic, rule-labelled examples (generated) |
| `tools/decision_policy/baseline.py` | the API-only baseline harness |
| `tools/decision_policy/baseline_report.json` | the measured results (generated) |
| `tools/decision_policy/baseline_report.md` | the same, as a table (generated) |
| `serve/systemone.py` | the endpoint's pure module: validate, render, parse, respond |
| `serve/test_systemone.py` | the endpoint's tests, MockEngine, no GPU |
| `serve/server.py` | wired: flag + config, admission, dispatch, `Service.systemone`, dialects |
| `tools/decision_policy/opendecisions.py` | OpenDecisions harness: serial + batch arms, resume-safe, official predictions output |
| `serve/logits_first_token.py` | the first-token lookup and the trie walk: both readings, the resolution, the end factor, the checks |
| `serve/test_logits_first_token.py` | the logits path, the wire format and the fallback, MockEngine, no GPU |
| `tools/decision_policy/logits_prob.py` | the probability module: log-space softmax, the chain rule, the response shape |
| `tools/decision_policy/logits_check.py` | the probe: eight cases, each with the verdict it must produce |
| `tools/decision_policy/readings_compare.py` | the closed-vs-open comparison, with the official scorer's own correctness rule |
| `tools/decision_policy/readings_split.py` | the same comparison, split by question type |
| `tools/decision_policy/vram_headroom.py` | the minimum free VRAM during a run, from the engine's log |
| `tools/decision_policy/first_token.py` | the tokenization analysis: how many option names are multi-token, and where they collide |
| `tools/decision_policy/trie_costs.py` | the trie walk's cost per item, from the shipped code and the server's tokenizer |
| `tools/decision_policy/trie_examples.py` | the colliding questions, as the tokens see them |
| `tools/decision_policy/README.md` | the tools, what each one is for |

## 9. OpenDecisions validation (2026-10-19, validation only - the set says do not train)

**2026-10-20: this section is the baseline the §10 work is measured against.** The
numbers below are the decode-only path's; §10.8 has the same sample with the
first-token path and the trie walk (65.6 → 73.4 → 79.5). The format-failure rate
and the "Calibration is structurally capped" note below are historical: the endpoint
now reports the model's own probability, so the calibration is no longer capped by
design.

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
4B systems (67-77); Calibration was structurally capped by design
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

---

## 10. The first-token logits path (2026-10-20, milestone 1)

### 10.1 What was missing

The endpoint's confidence was the open problem. A greedy decode yields the argmax,
so a pick was reported as 1.0 and every other option as 0.0, and every response
carried `"calibrated": false`. The user asked for the confidence to come from the
model's internal state, the way a trained decision model produces it.

The model already computes that state on every request:

```
prompt -> tokens -> embeddings -> 48 layers -> hidden state (n_embd 2560)
        -> lm_head (the vocabulary projection) -> logits (~248,320 floats)
        -> softmax -> the model's own next-token distribution
```

The endpoint threw the distribution away and sampled one token from it. Reading it
instead of sampling it is the whole feature: no second model, no heads, no training.

### 10.2 The answer position, and why the assistant turn is prefilled

The answer format is `name: answer`. A prompt that simply ends with `name: ` makes
the model start its turn by writing the **name**, so the next-token logits describe
the name rather than the answer. Measured on the probe: the question's name was the
top token at the answer position in every case, with the answer third or fourth.

The fix is to prefill the assistant's turn with the answer line's prefix, so the
model continues *into* the answer. The server already had that mechanism
(`prepare(..., force=...)`, used for forced tool calls), so no new engine machinery
was needed for it.

The prefix must end at the **colon**, not at a space. In training `"urgent: no"`
tokenizes as `['urgent', ':', ' no']` - the space lives inside the merged token. A
prompt ending in a lone space token after the colon is off-distribution, and the
model reacts by wanting out: measured, `<|im_end|>` became the top token (18.78) with
`' no'` second (13.82). With the prefix ending at the colon, the answer became the
top token:

```
noul   : 874:20.82:' no'  | 9542:16.46:' yes' | 2233:15.98:' No' | 248046:15.55:'<|im_end|>'
choice : 10597:18.25:' technical' | 32419:17.92:' billing' | 248046:13.19:'<|im_end|>'
```

### 10.3 The engine command

`LOGITS <k> <max_new> <id,id,...>` on the engine's stdin protocol. It is an ordinary
GEN request that also prints the answer position's row:

| Line | What it is |
| --- | --- |
| `Z <logz>` | the full-vocabulary log-sum-exp: `P(token) = exp(logit - logz)` |
| `L <token_id> <logit>` | the top-k at the answer position, descending |
| `T <token_id>` | the tokens the request generated (the normal decode) |
| `DONE ...` | the normal DONE line |

`logz` is what makes the "open" reading exact: it normalizes over the whole
vocabulary, not just the top-k.

The command was at first a prefill-only request that broke out of the serve window
loop early. That break skipped, on every such request, `adapt_thr.join()` (the
adaptive refill thread was never joined), the `if (!adapt_ok)` check, the policy's
`observe(...)`, `++rounds`, `fs->completed` and `pcie_auto_step(...)`. The adaptive
tier is what swaps experts between the VRAM and the RAM tier
(`src.replace_resident(...)`) - VRAM copies, on a background thread - so leaving it
unjoined while the next request starts, and feeding the policy statistics that omit
those windows, faults the GPU silently. Measured, the crash landed one or two
requests *after* a fast-path request, not on it:

| path | requests before a crash |
| --- | --- |
| plain decode (log, 2026-10-19 17:11) | 235, none |
| with the fast path | 39, 22, 18 |

The break is gone. `LOGITS` is now an ordinary GEN request that also prints the
answer position's row: every counter, the policy observation and the adaptive join
run exactly as on any other request. The cost is that the fast path also pays for
the decode - the prefill dominates (~2 s against a ~30 ms decode), and the prize was
always the probabilities, not the speed.

A bonus of the redesign: the fallback needs no second engine request. The tokens
that same request generated *are* the answer, so every decision is exactly one
engine request, including the ~30% of questions whose options collide on a first
token.

### 10.4 Two readings of the same probability

| | what it is | sums to |
| --- | --- | --- |
| **closed** | softmax over the option tokens only - the forced-choice reading | the options sum to 1.0 |
| **open** | `P = exp(logit - logz)` - the model's own probability | the options sum to < 1.0 |

They differ by exactly the mass outside the options. The **answer** is identical
under both (a renormalization does not reorder options) - only the confidence
differs. So the choice between them is a calibration question, answered by
measurement rather than argument (§10.6).

The diagnostics also carry `stop` (the mass on the end-of-turn tokens) and
`outside` (everything that is not an option). Measured on the probe, after the
prefix fix, the two readings nearly coincide - so the large `<|im_end|>` mass was
noise from the off-distribution space, not uncertainty:

| case | closed | open | stop | outside |
| --- | ---: | ---: | ---: | ---: |
| noul | 0.987 | 0.962 | 0.006 | 0.026 |
| choice | 0.505 | 0.494 | 0.004 | 0.022 |
| score | 0.950 | 0.844 | 0.089 | 0.112 |

### 10.5 Two bugs, both found by the probe

**The wire format.** The engine reads `LOGITS <k> <max_new> <ids>`; the server wrote
`LOGITS <k> <ids>`. The engine answered `ERR expected: LOGITS <k 1..1024> <max_new
1..4096> <id,id,...>`, the request never ran, and every decision fell back to the
decode path: no `L`/`Z` rows, no probabilities, and the replies the fallback
produced were truncated at the front (`"ood: happy"` for `"mood: happy"`). The mock
engine in the tests answers from a script and never sees the line, so nothing caught
it. The format is now pinned by a test (`serve/test_logits_first_token.py`,
`EngineWireFormat`).

**The end-of-turn token.** The fallback read the answer from the tokens the request
generated and decoded the end-of-turn token with them, so the collision case
answered `"rule_2<|im_end|>"` - not one of the options, so a 502. The end token now
ends the answer and is cut before the text is decoded, as the chat path does.

The probe's cases after the fixes (2026-10-20, the deployed engine 0.1.45.3; the
last two rows are the trie walk's, measured the same day):

| case | expected | measured |
| --- | --- | --- |
| noul | fast path | fast path, closed 0.985, open 0.966 |
| choice | fast path | fast path, closed 0.604, open 0.593 |
| score | fast path | fast path, closed 0.964, open 0.933 |
| collide | fast path (the trie walk separates them) | fast path, closed 0.602, open 0.579, answered `rule_2` |
| prefix | falls back (the options are not the same length) | falls back, answered `rule_1` |
| multi | falls back (one question per request) | falls back |
| chat | unchanged | unchanged |
| prefix (the trie walk) | fast path (the shorter option ends where the longer one continues) | fast path, closed 0.921, open 0.885, answered `rule_1` |
| partial (the trie walk) | fast path (only two of the three options share a first token) | fast path, closed 0.704, open 0.677, answered `No change` |

### 10.6 The trie walk for the options that share a first token

The first-token lookup cannot separate two options that start with the same token
("rule_1" and "rule_2" both start with " rule"; so do "Tier 0".."Tier 3"). The
**trie walk** can: the options' token sequences form a trie, and the row at a node of
the trie tells the options that pass through it apart.

The walk needs the row at a node. The engine's LOGITS command reports the row at the
answer position - the last position of the prompt it was given - so asking again with
the node's tokens appended gives the row after them. **No engine change:** the prompt
is in the conversation cache, so the extra request reads the shared tokens, not the
prompt.

**THE RESOLUTION.** An option is told apart at the shallowest node where it is alone:
either its next token is one no other option through that node has, or its text
**ends** there (its last token, and another option continues through that node). The
second case needs the probability of the turn **ending** there, because a model that
stops after the shorter option has answered with it. Without that factor the shorter
option ("frontier") would carry the whole weight of the shared prefix and always beat
the longer one ("frontier_reasoning").

The weight of an option is then the product of a factor per node (the token that comes
next there) times the outcome it claims at its resolution node: the token it continues
with, or the end of the turn.

**The cost, measured** (`tools/decision_policy/trie_costs.py`, the shipped code and
the server's own tokenizer, over the 235 items):

| extra engine requests | items |
|---:|---:|
| 1 | 50 |
| 2 | 12 |
| 3 | 5 |
| 4 | 3 |
| 5 | 1 |
| 7 | 1 |

113 extra requests over the 72 colliding items of the sample - 113 requests that each
read a handful of cached tokens, not the prompt.

### 10.7 What the chain rule could not reach

The chain rule that came before it walked a single shared prefix, so it needed
**every** option to share one prefix and to be the same number of tokens long. A
census of the 235 items against the shipped tokenizer shows how much that left on the
table:

| the 235 items | count |
|---|---:|
| every option has its own first token (the first-token path answers them) | 163 |
| every option shares the first token | 37 |
| **only some of them share it** (a partial collision) | 35 |

Of the 37 full collisions the chain rule answered 15 - the ones whose options were
all the same length. It also read one row too few for an option longer than the
shared prefix plus one, so `jb16-multilingual-faa2edf82c` (five-token options, one
shared token) fell back - and its fallback wrote a bare `4`, which the parser
rejected, so the request answered **502**.

The same shape, on the `score` questions whose levels are long strings with a numeric
head ("0: none of the five features"): the harness refuses to shorten a level whose
head is a digit, so the raw strings stay as the option names, the model answers `0`,
and the parse - which demands the exact level string - fails. Seven of the 235 items
answered 502 that way, all of them `score` questions. The trie walk answers them from
the logits, so the model's text is never read.

### 10.8 The OpenDecisions run

The 10% sample of jgeuter/OpenDecisions v1.0.1 (235 items, seed 42, the same ids
as the 65.6 baseline), one question per request, the official `score.py` on
`sample10_items.jsonl`:

| run | Capability | Intelligence | Calibration | predicted | calibration items TVD |
|---|---:|---:|---:|---:|---:|
| the old decode-only path (the baseline) | 65.6 | 62.0 | 69.2 | 223/235 | 0.546 |
| the first-token path, the degenerate 1.0/0.0 probabilities | 74.9 | 72.3 | 77.6 | 227/235 | 0.284 |
| the first-token path, the real probabilities | 72.1 | 70.8 | 73.4 | 226/235 | 0.261 |
| the first-token path + the chain rule | 73.4 | 70.8 | 76.1 | 228/235 | 0.292 |
| **the first-token path + the trie walk** | **79.5** | **77.5** | **81.4** | **235/235** | **0.215** |
| the trie walk, run again (the noise floor) | 78.9 | 77.3 | 80.6 | 235/235 | 0.222 |
| + the answer position checked (§10.9) | 79.6 | 77.7 | 81.6 | 235/235 | 0.227 |
| + an option outside the window scored (§10.10) | 80.4 | 77.4 | 83.4 | 235/235 | 0.241 |
| + the gate softened, the noul's confidence (§10.11, §10.12) | 79.1 | 76.3 | 81.9 | 235/235 | 0.240 |

Per type, the trie walk's run: choice 89.7% accurate, noul 91.3%, score 79.7% (the
chain rule's run: 89.7%, 91.3%, 67.8%). The fast path carried **233 of the 235
items** (99%); the two that fell back did so because an option's token was outside
the engine's top-k, and the seven that answered 502 before (the model's text did
not parse) are now answered from the probabilities. The last two rows close both:
every item of the sample is on the fast path, and nothing answers 502.

**The two rows that differ only in the probabilities are the honest part.** The
second row sent the scorer a degenerate distribution (1.0 on the pick, 0.0
elsewhere), so its calibration component was just 1 - accuracy. The third row sent
the endpoint's real distribution, and the Capability **dropped 2.8 points**: the
model's own confidence is over-confident relative to its own accuracy. That is a
finding, not a bug in the reading - and it is the reason the chain rule (which
gives the colliding questions a real confidence too) is worth 1.3 Capability
points even though it changes no answer the first-token path got right.

**The trie walk is worth 6.1 Capability points**, and the reason is structural
rather than a matter of luck: 51 items moved off the decode path (they carry the
model's own probability now), the 7 that answered 502 are answered at all, and the
score questions - the ones whose options collide hardest - went from 67.8% to
79.7% accurate.

**The two readings, per type** (`tools/decision_policy/readings_split.py`, the
fast-path items of the 235):

| type | closed AUC | open AUC | closed ECE | open ECE |
|---|---:|---:|---:|---:|
| choice (105) | 0.8066 | **0.8375** | 0.0910 | **0.0597** |
| noul (69) | 0.9508 | 0.9508 | **0.0917** | 0.1008 |
| score (59) | **0.8298** | 0.8191 | 0.1069 | **0.0982** |

Over every fast-path item: the open reading AUC 0.8266 against the closed 0.8085,
and the open reading's expected calibration error 0.0426 against the closed 0.0623 -
so the contract (the closed reading for the distribution, the open one for the
confidence) still points the right way, now over 233 items instead of 177.

The open reading (the model's own probability) ranks a correct answer above a
wrong one better; the closed reading is better calibrated for a yes/no, where its
value is the probability of "yes" among the two answers, so the value and the
answer always agree. Hence the contract: `probabilities` is the closed reading
(the shape's distribution, which has to add up to 1), `confidence` is the open one
(the model's own probability of the pick), and a yes/no keeps the closed reading.

**The VRAM** during the run: the minimum free was **1082 MiB** (1206 at load)
against the engine's 1024 MiB warning threshold, and each parked conversation costs
~240 MiB with these long prompts (the largest snapshot: 488 MiB for the two slots
the config has). Four conversation-cache slots would put the card about 484 MiB
below that threshold - the failure mode that kills a verify window - so the slots
stay at 2.

**The noise floor.** The same request does not always give the same numbers: the
engine's KV cache puts a conversation in a slot whose physical layout differs from
one run to the next, and the attention's rounding differs with it. Measured by
running the whole 235-item set twice with the same code:

| | the first run | the second run |
|---|---:|---:|
| Capability | 79.5 | 78.9 |
| Intelligence | 77.5 | 77.3 |
| Calibration | 81.4 | 80.6 |
| predicted | 235/235 | 235/235 |
| the fast path | 233 items | 233 items |

The *method* is the same on every item - the fast path differed on none of the 235 -
and three picks flipped, all of them near-ties (the closed reading's top two options
0.26 against 0.26, 0.57 against 0.47, 0.32 against 0.34). The confidence moved by a
median of 0.004 and at most 0.11. So the run-to-run spread is about 0.6 Capability
points, and the 6.1 points the trie walk is worth are ten times that.

A repeat of the same request that hit the conversation cache gave exactly the same
numbers, so the wobble is the cache's layout, not the endpoint.

### 10.9 The answer position is checked too

The trie walk's check ("the model's top token at this node must be one the options
claim, or the model was going to say something else and the reading is about a choice
it is not making") ran at the nodes where the options are **told apart**. When every
option shares a first token, that node is deeper than the answer position, so the
answer position was not checked: the reading was then computed for a context the
model is not in. The model wanted to write something else at the answer position
(a digit, a different word), and the walk went on to ask what it would write *after*
the shared prefix - a sensible-looking answer to a question the model never got to.

The root is now one of the checked nodes. The options claim their first tokens
there, so the rule is the one the first-token path already states: the model's top
token at the answer position must be one of the options' first tokens.

Measured on the 235 items, the official scorer:

| run | Capability | Intelligence | Calibration | predicted | fast path |
|---|---:|---:|---:|---:|---:|
| the trie walk | 79.5 | 77.5 | 81.4 | 235/235 | 233 items |
| the trie walk, run again (the noise floor) | 78.9 | 77.3 | 80.6 | 235/235 | 233 items |
| **the trie walk + the check** | **79.6** | **77.7** | **81.6** | **235/235** | **233 items** |

The same 233 items on the fast path, the same two fallbacks, and the score is inside
the run-to-run spread. The seven items that used to answer 502 - the score questions
whose levels are long strings with a numeric head - still take the fast path with the
same picks: their model's top token at the answer position is the shared one, so the
check passes. The risk was that the check would drop them to the decode path (where
the model's bare "0" does not parse) - it does not.

### 10.10 An option outside the engine's window

The engine sends the top k logits, not the whole 248,320-token vocabulary, so an
option's token can be below the window. The endpoint used to give up on the whole
question in that case and answer it from the decode path, with the degenerate 1.0/0.0
distribution - two of the 235 items (`'creative_writing'` and `'escalate_human'`).

That was unnecessary. The window holds the **highest** logits, so an option outside it
is below every option inside it: it cannot be the pick, and its probability is below
the window's tail. It is therefore scored as **zero** and the question is answered -
the answer is provably the one the full vocabulary would give, and only the reported
probability for that option changes, from "unknown" to "approximately zero, bounded by
the window's tail".

Measured on the 235 items: the two items move onto the fast path, and nothing else
changes.

### 10.11 The gate is a flag, not a refusal

The first-token path used to refuse to use its own readings when the model's most
likely token at the answer position was not one of the options: the model was going
to write something else (a digit, a phrase, a rate), so the option distribution was
"a choice the model is not making", and the endpoint read the model's text instead.

That is the right instinct - the model did not answer with an option - but the
refusal threw away the one thing the endpoint knows: **the model was not going to
answer with an option, and how unlikely it is that it would.** The weight the model
put on everything else is already missing from the open reading, so the confidence
carries the doubt by itself. The gate is now a flag (`model_picked` in the
diagnostics) and the answer is a fallback **order**:

1. the model's top token at the answer position is one of the options -> the readings
   are the answer, as before;
2. else the model's own text is the answer when that text is a valid answer - a
   yes/no may be given as a rate, and the rate is what the reference API's `noul`
   wants;
3. else the readings are the answer after all: the forced choice, with the
   confidence carrying the doubt.

Step 3 is what removes a **502**: before, a model that wrote something that did not
parse (and was not going to write an option) produced no answer at all.

**Measured on the 235 items: the gate never fires.** Every one of the 235 items has
the model's top token at the answer position among the options' first tokens, so the
run is unchanged (the score is inside the run-to-run spread) and the change is a
safety net rather than a common path. The same is true of five cases tried by hand
against the live model - a yes/no question with "true"/"false" options, a question
with "TBD"/"N/A" options, numeric options, a question asking for a rate, and a
question asking for a sentence: the prompt's "answer with exactly one of these
option names" holds the model to the options, and it wrote one every time.

### 10.12 The noul carries a confidence

A `noul` answer is a float - the probability of yes - and it had no confidence
field, unlike the other two types. The contract for those is "the distribution is
the closed reading, the confidence is the open one", and a yes/no could not carry
the second half of it: the mass the model put on writing something other than yes
or no was invisible.

It does now, with the same meaning as the other types: **the model's own
probability of the side it picked**. The value stays the closed reading (the
probability of yes among the two answers, which measured better: the expected
calibration error 0.093 against 0.116), so a confident "no" reports a small value
and a large confidence - `{"noul": 0.03, "confidence": 0.95}` means "the answer is
no, and I am fairly sure of it". A consumer that does not know the field can
ignore it; the shape's `noul` value is unchanged, so the score is unchanged.

### 10.13 What is not done

- **The real-trajectory data** (the 60-example synthetic set is rule-labelled, so it
  measures agreement with a rule, not quality). **Last in the user's priority order
  (2026-10-20)**: a published dataset is the reference for the endpoint's quality,
  and the OpenDecisions set is it. What the trajectories would add is the right
  answer in the *user's own* workflow, which needs the user's notion of a good
  decision. Two cheaper ways to use the user's own data exist: the examples can live
  in the cached prompt preamble (§6.2), and a policy head over the backbone's hidden
  state is the architecture's training path (§2.5, §2.6).
- **A contract question to state plainly.** The reference API's `noul` type returns a
  *float* (the probability of yes) and the scorer thresholds it at 0.5. The prompt
  now says "answer yes or no", which makes the model write a boolean while the
  endpoint reports the model's internal probability. That is the right shape for the
  API, but the *answer* and the *probability* then come from different places - the
  argmax against the distribution. The noul now carries a confidence as well
  (§10.12), so the doubt is visible even though the value is the closed reading.
- **The chain rule for the options of different lengths** - **done** (§10.6): the trie
  walk tells the options apart wherever they separate and uses the end-of-turn
  probability where an option's text ends, so "frontier" and "frontier_reasoning" are
  comparable.
- **The trie walk not checking the answer position** - **done** (§10.9): the check
  now runs at the answer position as well, so the reading is never computed for a
  context the model is not in. Measured: no change to the score (inside the noise
  floor) and the same 233 fast-path items.
- **An option outside the engine's window** - **done** (§10.10): such an option is
  below every option inside the window, so it cannot be the pick; it is scored as
  zero instead of dropping the question to the decode path.
- **The gate refusing to use the readings** - **done** (§10.11): the gate is a flag
  now, and the answer is a fallback order - the model's own text when it is a valid
  answer, the readings with a small confidence when it is not. Measured: the gate
  never fires on the 235 items, so this is a safety net, and it removes a class of
  502s.
- **The exact probability of an option that continues past the separating token.** The
  walk stops where the options are told apart, so a longer option's weight is the
  probability of the prefix that identifies it, with the rest of its text assumed
  certain - the same approximation the first-token path makes. The exact weight would
  need the row at the end of that option's own text (the full trie: up to 94 rows for
  one item of the sample, against the 1-7 the walk needs).
- **The chat path's telemetry for the decision path.** The decision path takes the
  engine's turn and the engine's DONE line, but it does not go through
  `Service.run`, so the monitor's live request list does not show a decision
  request. It shows as idle while a decision runs.
- **The engine's stale context** (§12). A long run once answered another question's
  options, and the 2026-10-20 handoff recorded it as an engine cache bug that
  invalidated every long measurement. The evidence does not support that: the log
  line that was read as a proof is the trie walk's own request, and the failed items
  answer on a healthy engine. **The fault was in the server's client, not in the
  engine's cache, and it is fixed (§13).**
- **The real-trajectory data** - still last in the user's priority order, and now
  less needed: the jev-bench soft labels (§14) answer the calibration question
  against a human gold standard, and the published sets are the reference.
- **The conversation cache's slots.** Measured: the minimum free VRAM during the
  235-item run was 1082 MiB (1206 at load), against the engine's 1024 MiB warning
  threshold - and each parked conversation costs ~240 MiB of VRAM with these long
  prompts (488 MiB for the two slots the config has). Four slots would put the card
  about 484 MiB *below* that threshold, which is the failure mode that kills a
  verify window, so the slots stay at 2 until the margin is measured on a longer
  run.

---

## 11. Several questions, one at a time (2026-10-20, milestone 2)

### 11.1 What was wrong with the shared prompt

A request with more than one question was answered by **one prompt that held all of
them**: the model wrote a `name: answer` line per question and the server parsed
them. Two things were wrong with it.

**The answers carried no probability.** The shared prompt's answers came from a
greedy decode, so they were the degenerate 1.0/0.0 and the response said
`"calibrated": false` - while a request with exactly one question, which reads the
model's own distribution at the answer position (§10), said `"calibrated": true` and
carried a real probability. The same endpoint, two contracts, depending on the number
of questions.

**The request was atomic.** One unparseable line lost every question in it: §9's
batched arm measured 12 of 47 mixed batches (25.5%) failing that way, against 5.1%
for the serial arm, and the failures were correlated - one bad line, 46 serial-correct
items lost as collateral. Zero partial batches, "by construction".

**Re-measured on a fresh engine (2026-10-21), the shared prompt lost 70 of the 235
questions (30%)** - the same 5 questions per request, one attempt each, a freshly
restarted engine, Capability 49.5 against the merged path's 79.0 (§11.5). So the
atomicity is the shared prompt's own, not a stale engine: the handoff's suspicion
that these failures were the engine bug (§12) is wrong, and the 36.8 that the
2026-10-20 handoff recorded for this arm was measured on a bad engine and is too
low.

### 11.2 What it is now

**A request with several questions is answered by running the single-question path
once per question, and the answers are merged into one response.** The single-question
path is not touched: it is the same code, the same answers and the same probabilities,
so every question in the merged response carries the model's own probability and the
response says `"calibrated": true`.

* The engine's turn is taken **once per question**: the fast path holds the FIFO while
  the engine answers and gives it back before the next question. Holding it across the
  questions would serialise the requests wrongly - one request's questions would block
  every other request for the whole batch.
* The merged response says `"calibrated": true` when **every** answer in it came from
  the fast path. A question that fell back to the decode path (a degenerate 1.0/0.0
  answer) makes the whole response `"calibrated": false`, because one of the
  probabilities in it is not the model's own.
* The `usage` is the sum of the questions' (each question's prompt is its own), and
  the `timings` are merged: the counts and the milliseconds add up, the rates are
  computed from the totals.
* The response keeps its shape: the same `id`, `object`, `created`, `model`,
  `answers` and `usage` fields, the same per-answer shapes, the same status codes.

### 11.3 The failure rule

The response shape has no per-question error field, and the endpoint's rule is that a
bad answer is an error and never a fabricated distribution. So:

* a question the model could not answer is **left out** of `answers`. The other
  questions keep their answers and the request is a **200**;
* the reason goes to the server's log, and into the diagnostics when they are on;
* when **no** question could be answered there is nothing to return, and the request
  is a **502** - the same answer a one-question request gives in the same situation.

The alternative - failing the whole request when one question fails - is what the
shared prompt did, and it is what costs the 46 items of §9's batched arm.

### 11.4 The flag

The shared prompt is still reachable: `"systemone": {"enabled": true, "multi":
"shared"}` in the launch config. The default is `"separate"` (the new path). The
endpoint itself is opt-in (`"enabled": true`), so this is a change to an opt-in
endpoint's own behaviour, and the old behaviour is one config key away.

### 11.5 What it costs, and what it measures at

**The cost.** The shared prompt amortized the prefill over the questions: one read
of the state, one decode. The new path puts the state in every question's prompt, so
the prefill is repeated per question.

**The measurement (2026-10-21, the 235-item sample, a freshly restarted engine).**
The sample sent 5 questions per request (47 requests), one attempt each, the official
scorer on `sample10_items.jsonl`. The serial row is the same sample through the
one-question-per-request path (2026-10-20, a fresh engine, the same code). The shared
row is the same batched arm on the shared prompt (2026-10-21, a fresh engine).

| run | Capability | Intelligence | Calibration | predicted | fast path | per question |
|---|---:|---:|---:|---:|---:|---:|
| the serial path (1 question per request) | 79.1 | 76.3 | 81.9 | 235/235 | 235 items | **3.66 s** |
| **the merged path (5 questions per request)** | **79.0** | **74.4** | **83.7** | **235/235** | **235 items** | **6.92 s** |
| the shared path (5 questions per request) | 49.5 | 27.8 | 71.2 | 165/235 | - | 1.61 s |

**The serial row, re-measured after the concurrency fix (§13), on 2026-10-21:
79.8 (78.0, 81.7), 235/235, 235 fast-path items, 3.12 s a question.**  The
per-type accuracies are the same as the run above (choice 89.7%, noul 91.3%,
score 79.7%), so the fix changed no answer; the 0.7 points are the run-to-run
spread (§10.8), and the whole measured range for this sample is 78.9-80.4.

**The merged path's quality is the serial path's, to within the noise floor.** The
two rows differ by 0.1 Capability points, against a run-to-run spread of about 0.6
(§10.8), and the merged run predicted **every** item with **every** question on the
fast path: no fallbacks, no 502s, no collateral loss. That is the point of the
change - the merged path *is* the single-question path, run once per question, so it
cannot be worse than the single-question path.

**The shared path, re-measured on a fresh engine, is 49.5 and not the 36.8 the
2026-10-20 handoff recorded.** That number was taken on an engine that was already
bad (§12), so it was too low; §9's older batched arm (54.0, 175/235) was closer to
the truth. The shared path's cost is confirmed, and its per-question latency matches
§9 (1.61 s against 1.5 s, the request total 8.0 s against 7.5 s). **The shared
prompt's failures are its own, not a stale engine**: on a fresh engine it still lost
70 of the 235 questions (30%), because one unparseable line loses every question in
the request. That is exactly what the merged path removes.

**The merged path's cost is real and it is the prefill.** Per question it costs
**6.92 s** against the serial path's **3.66 s** (median, amortized over the request);
a 5-question request takes 34.6 s. The reason is the shape of this dataset's batched
arm: it puts **five different states in one request** (as numbered sections), so the
merged path re-reads all five sections once per question - 25 state-reads where the
serial path does 5. **That is the merged path's worst case, not its normal one.** An
agent's real request is *one* state and *many* questions, which costs one
state-read per question, not five. The number is quoted as "the merged path, when the
request carries as many states as questions", because quoting the amortized
per-question latency as if it were the general case would overstate it.

**The open design question (A1 versus A2) is now measurable.** A2 is "one prompt, one
extra request per question to read the row at that question's answer position". It
would keep the shared prompt's single read of the state and still give every question
its own probability, so on a request with one state and N questions it should cost
about N tail-reads instead of N state-reads. That is the number that decides A1
versus A2, and it is not measured yet: A2 is not implemented. What this section
establishes is the end A1 has to beat - the quality of the serial path (79.1) and the
speed of the shared path (1.61 s a question, at a third of the quality).

---

## 12. The engine's stale context: what was measured (2026-10-21)

A long OpenDecisions run once failed **43% of its items** with a 502, and the failing
answers were fragments of *other* items' option names (`creative_writing` for an
`ondrel_l/ondrel_m/ondrel_s` question, `exchange_rate_feed` for a
`nimble/steady/summit` question). The model appeared to be continuing a different
conversation. A 2026-10-20 handoff recorded this as an **engine cache bug** - the
engine restoring a conversation that was not the request's - and every long
measurement was treated as suspect because of it.

**That diagnosis was wrong on both counts: what the evidence showed, and where the
fault is.** What follows is what was measured, on this machine, on 2026-10-21.

#### 12.1 The log line that was read as a proof is the trie walk

The handoff's evidence was the engine's own line:

```
strata serve: prompt 330 tokens = 320 reused + 10 read in 104 ms
```

Classifying **all 4,342** prompt lines in the engine log
(`strata-unsloth-ud-iq4_xs.log`) shows what that line is:

| the shape | what it is | requests |
|---|---|---:|
| `N tokens = 0 reused + N read` | a base request, nothing reused | 3,404 |
| `N tokens = M reused + (N-M) read`, **1 generated** | **the trie walk's own request** | 775 |
| `N tokens = M reused + (N-M) read`, 2+ generated | a chat turn (a growing conversation) | 80 |

The second row is the one that was misread. A trie-walk request is the server's LOGITS
call with `max_new=1`, appending a node's tokens to a prompt that is already in the
cache: it reuses **its own** prompt's checkpoint and re-reads the last ~10 tokens
(the checkpoint sits at `N-9`). The `1 generated` is the tell. So the line is benign,
and it is the shape the whole trie-walk arm produces, not a symptom.

#### 12.2 The failure is environmental, not the items and not the order

The invalid run's own log is the proof. `full_trie.run.txt`:

```
25/2352 requests ok=11 fail=14 elapsed=43s
```

**It failed from its first item.** So the engine was already bad when the run
started; the run did not poison itself.

And the items are not the problem. Of the 145 failed items, **130 are in the first
300 of the set** - and a clean re-run of *the same 300 items, in the same order, with
the same code* answered **300/300** (HTTP 200 against the original run's 502):

| | the invalid run | the clean re-run |
|---|---:|---:|
| the first 300 items, answered | 170/300 | **300/300** |
| failures | 130 | **0** |

#### 12.3 It does not reproduce on a healthy engine

| the experiment | the result |
|---|---|
| the same 30 items, 4 passes in a row (120 requests) | 30/30 every pass |
| 300 distinct items, file order, warm engine | 300/300 |
| the same 30 items with a concurrent chat load on the same engine | 30/30 answered, 34 chat requests, 0 errors |

So the trigger is neither the item, nor the order, nor the request count, nor a
plain concurrency race. **The failing engine was in a state this work could not
reproduce**, and a restart cleared it.

#### 12.4 What is left, and it is not nothing

The engine log carries **42** occurrences of

```
strata serve: WARNING: only 115 MiB of VRAM free at the start of this request
    (below 1024 MiB): when the card is this full the driver can page GPU memory to
    system RAM, decode falls to a few tok/s and a window may never finish (#2065).
```

with a **minimum of 115 MiB** free, against the engine's 1,024 MiB threshold. The
engine's own text says the driver may page GPU memory to system RAM in that state.
That is a mechanism that could corrupt a KV cell or a recurrent state, and the
`qwen4exp` model has a recurrent state as well as a KV cache. **This is a
hypothesis, not a finding**: no run here was shown to fail because of it, and the
warnings do not line up with any specific failing run.

The other candidate is the **conversation cache** (`--conversation-cache-mib 8192`,
`--conversation-cache-slots 2`): it churns hard (a parked snapshot is ~240 MiB, the
eviction counter climbs through a long run) and its `reused_kv_bytes` is sometimes 0
and sometimes not, so what a restore actually reuses is not obvious from the log.

#### 12.5 The mitigation, and what it costs

The mitigation is the one the handoff describes: **a fresh engine per chunk**. It
works, and it is cheap to state: a restart reloads the model in about 60 s, so a run
of N items is split into chunks of N and the engine is restarted between them.

What the failed run means for the numbers in §10.8: **none of them.** Every one of
those runs was measured on a freshly restarted engine, and the clean re-run in §12.2
reproduced the sample's behaviour exactly (300/300, no failures). The numbers that
are suspect are the ones taken on an engine that had been running for a long time -
and the one that demonstrably was, the 327-item run, is invalid and is not used
anywhere in this document.

#### 12.6 What is still open

**2026-10-21, later the same day: the fault was found, and it is in the server's
client, not in the engine's cache.**  §13 has it: the decision path did not drain
the engine's line queue when a request was abandoned, so the next request read the
abandoned one's tokens and `DONE` as its own answer - another question's option
name, and a 502.  The evidence below is what ruled the engine's cache out; the
fault that explains the symptom is §13's, and the measurements here are still
right: the failed items answer on a healthy engine.

1. **The trigger.** The engine state that produced the wrong answers has not been
   reproduced, so there is no fix to write and no fix to measure.  The two live
   candidates are the VRAM exhaustion above and the conversation cache's eviction
   path; both need a reproduction before anything is changed.
2. **The user's own agent** shares this engine, so it shares the risk. The
   measurements here say the risk is small on a healthy engine, but they do not
   measure it.
3. **A guard.** The engine already warns below 1,024 MiB. Whether it should *refuse*
   a request rather than proceed (the warning says the driver may page) is a
   question for the engine's owner.

Nothing in this section changes the engine. The finding is about what the evidence
does and does not support.

---

## 13. The concurrency fault: the server's attribution of the engine's answer (2026-10-21)

§12 corrected the handoff's diagnosis: the log line that was read as a proof is the
trie walk's own request, and the failed items answer on a healthy engine.  But the
fault was still unexplained, and the USER's experience was the clue that the
handoff had lost: **the errors came when the decision engine was used by several
users at once.**  Nothing measured so far had exercised that.

### 13.1 The fault is in this branch's code, and it is in three parts

The engine's answer reaches the server as lines on one pipe: the `L`/`Z` rows (the
distribution at the answer position), the `T` tokens (the model's text) and a `DONE`
line.  The server reads them from **one shared queue**.  So a request's answer is
only its own if the reader that took it took *its* lines.

**(a) The decision path did not drain the queue.**  The chat path's
`Engine.generate` writes `STOP` and then **drains to `DONE`**, and its own comment
says why: *"the consumer stopped early: stop the engine, drain to DONE"*, because
the engine's lines for this request must not reach the next request, which reads
the same queue.  `Engine.logits` - the decision path's engine call, added by this
branch - wrote `STOP` and returned, **leaving the lines behind**.  A request the
server gave up on (the client hung up, the queue wait expired) therefore handed the
next request its tokens and its `DONE`: **another question's text, another
question's distribution.**  That is the reported symptom exactly - the failing
answers were other items' option names, and the strict parse turned them into
502s.

**(b) The decision path read a stopped request's tokens as an answer.**  The
engine's `DONE` line says how the request ended (`stop`, `length`, `cancel`,
`nonfinite`).  A request the engine stopped has a *partial* answer, or none.  The
chat path checks the finish reason; this path did not, and `Engine.logits` did not
even clear the previous request's, so the reason it read could be a stale one.

**(c) The trie walk took the engine's turn again.**  The walk appends a node's
tokens to the prompt it was given, so it needs that prompt still in the engine's
cache - and another request's turn takes the cache away.  The base request and the
walk are one question's work, so they now share one turn.

**(d) The decision path had no engine-silence watchdog.**  The chat path's
`generate` has one (#481): if the engine says nothing for the configured
allowance, the engine is ended.  `Engine.logits` looped on `queue.Empty` forever.
An engine that had lost step therefore left a decision request waiting for a line
that never came, **while holding the request FIFO** - so every other request
waited behind it, and the server looked stuck.

### 13.2 The measurement: the server hung, and then it did not

The reproduction is a load that looks like several users: four decision workers and
four chat workers with growing conversations, on one engine.  The queue backs up, so
the clients give up mid-request - which is the path that (a), (b) and (d) live on.

| | before the fix | after the fix |
|---|---:|---:|
| requests served before it stopped | **41** | **780 decisions + 83 chat turns** |
| decision failures | - | **0** |
| what happened next | **a deadlock**: four requests in flight, the engine idle, no progress - the server never finished | the run completed |
| the client's give-up timeout | 300 s | **60 s** (a stricter test: more requests were abandoned) |

The deadlock is the signature of (a) and (d) together: the engine had nothing left
to do, and the requests were waiting for a line that had already been consumed by
a request that had already returned.  The server's own monitor showed
`"in_flight": 4` with the engine idle for minutes.

The fix is the chat path's own behaviour, applied to the decision path: stop the
engine and **drain to `DONE`**; end the engine when the drain cannot finish; keep
the engine's `DONE` line's own prompt-token count as the identity of the answer
(so a request can tell that the lines it read were another request's, and refuse
them); do not read a stopped request's tokens as an answer; and take the engine's
turn once per question, base request and walk together.

### 13.3 The engine's own resume machinery is sound

A read of the engine's serve loop says the resume path cannot be the fault: every
way the engine continues an earlier conversation - the live session, a checkpoint,
a batch slot's checkpoint, a parked conversation - is gated by an **exact
token-prefix comparison** (`starts_with`, and the conversation cache's
`conversation_prefix`), and the checkpoint's validity invariant ("a checkpoint is
only valid while the positional cells below it still hold ITS tokens") is enforced
by an explicit prune.  The running state - the recurrences and the histories, which
cannot be rewound - is restored from a checkpoint whose ids are an exact prefix of
the new prompt, so it is correct for that prompt.

**The engine can only ever read the token ids the server wrote on the command
line.**  So "the engine restored a wrong context" requires either that the server
sent the wrong ids, or that the server read the wrong lines back.  This section is
about the second.

### 13.4 What is still open

* **The engine's batch slots** - **ruled out.**  The chat path's `generate` has a
  branch for the engine's `--batch` slots and the decision path's `logits` does
  not, so if the engine ran with batch slots the decision path's turn would not be
  exclusive *at the engine*.  The engine's log has no `batch_slots` and no batch
  lines at all, and the server reports `"concurrency": {"serving": 1}`, so the
  turn is exclusive on both sides.
* **The server's prompt cache** (`PromptEncoder`, #567) is the other way the wrong
  ids could be sent.  It is exact by construction and unit-tested, and a fresh
  *server* (not just a fresh engine) would be needed to rule it out; the clean
  re-run in §12.2 restarted the server, so it is ruled out for that run.
* **The engine's VRAM warning** (42 occurrences, 115 MiB free at the lowest) is
  still unexplained, and it is the engine's own candidate: a window that never
  finishes is exactly the condition under which the drain times out.  The fix
  makes that case survivable (the engine is ended, the next request starts fresh)
  rather than silent.
* **The multi-question path's own numbers** (§11.5) were measured before this fix.
  The serial path was re-measured after it: **79.8 (78.0, 81.7), 235/235
  predicted, 235 on the fast path**, with the same per-type accuracies as before
  (choice 89.7%, noul 91.3%, score 79.7%) - so the fix changed no answer, and the
  0.7 points against the 79.1 row are the run-to-run spread.

---

## 14. The calibration question, answered against the humans (2026-10-21)

§10.13 listed the calibration question as the one thing the OpenDecisions set
cannot answer: that set has a right answer per item, so it can say whether the
model's confidence ranks a correct answer above a wrong one, but not whether the
confidence is calibrated to the *uncertainty* a human would feel.  The jev-bench
sample has what OpenDecisions does not - **280 of its 683 records carry the human
label distribution** (`soft_label`), so the endpoint's answer can be scored
against how uncertain the humans were.

### 14.1 The run

Praveenrajus/jev-bench v0.1.1, a 3% sample (seed 20261020) of 683 records over 22
configs: choice 294, noul 222, score 167.  One item per request, one attempt each,
the official scorer on the scorer-shaped items file.  **683 of 683 answered, 0
failures** (the run shared the engine with the user's own agent for part of it).

| | Capability | Intelligence | Calibration | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|
| the jev-bench 3% sample | **55.2** | 46.7 | 63.7 | 70.7% | 74.3% | 47.9% |

The caveat the sample's own metadata carries: **jev-bench has no difficulty tier**,
so every item is filed under `"hard"` and the scorer's tier weights collapse to
one.  That is not a difficulty claim, and any write-up has to say so.

### 14.2 The distance to the human distribution

The total variation distance to the humans' distribution, over the 280 records that
have one, with two trivial baselines measured the same way:

| | mean TVD | |
|---|---:|---|
| **the endpoint's own distribution** (the closed reading) | **0.4432** | |
| the argmax with no uncertainty | 0.5045 | the endpoint is closer on **199 of 280** |
| a uniform guess | 0.4943 | the endpoint is closer on **146 of 280** |

**The endpoint's real distribution is closer to the humans' uncertainty than either
trivial baseline.**  That is the positive half of the answer: the model's own
uncertainty carries real information about human uncertainty, and a distribution
that says "I am not sure" is worth more here than a confident guess.

Per config the spread is wide, and it tracks how ambiguous the task is:
chaosnli 0.312, civil_comments 0.384, measuring_hate_speech 0.492, **go_emotions
0.722** - the six-way emotion labelling, where the humans themselves were spread
out.  Per type: noul 0.384, choice 0.470, score 0.492.

### 14.3 The other half, stated plainly

**The endpoint's pick agrees with the humans' consensus on only 123 of the 280
records (43.9%).**  The model disagrees with the humans more often than not, and
the disagreement is not random - it is largest where the task is most ambiguous.

The **AUC** of the model's own probability against "the humans' consensus was
right" is **0.6427** - informative (0.5 would be chance), but the confidence
barely separates the two cases: the mean confidence is **0.868** when the endpoint
agrees with the humans and **0.776** when it does not.  So the model is
**over-confident relative to human uncertainty**: it is usually right about where
the disagreement lies, but it is not nearly uncertain enough when it is wrong.

This is the same finding the OpenDecisions sample gave in a different form (the
real probabilities cost 2.8 Capability points against the degenerate ones, because
the model is over-confident relative to its own accuracy).  Against a human gold
standard it is sharper: the endpoint's distribution is genuinely better than a
confident guess, and genuinely worse than the humans' own uncertainty.

### 14.4 What this does and does not settle

* It answers the question §10.13 left open, against a human gold standard, with
  the numbers above.
* It does **not** say the endpoint should be more confident: the finding is that
  the model is over-confident, so a consumer that wants calibrated uncertainty
  should read the `confidence` as an upper bound, not a probability.
* The licence is per-source and mixed (several "research use only"), and the
  dataset says validation only.
* The rewrites suite asks the other half of the coherence question: not "is the
  answer right" but "does the distribution stay coherent when the question is
  reworded, negated or combined".  It is measured next.
