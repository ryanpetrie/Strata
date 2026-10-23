# Shared-backbone System 1 decision policy in Strata

**Status:** the endpoint is built and measured (§10, §11); the engine's stale-context
failure is under investigation and the evidence is corrected (§12); the decision is a
request in the Monitor like any other (§15), and the web app can ask one (§16).
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
C:\tools\strata\.venv\Scripts\python.exe -m unittest serve.test_systemone serve.test_logits_first_token serve.test_decision_monitor -v
```

`<D>` = `<opencode temp>\opendecisions` - the data, the predictions and the run logs
live outside the repo on purpose (the dataset's rules say do-not-train; keep it out
of git).  `<opencode temp>` is the machine's temporary run directory (for example
`%LOCALAPPDATA%\Temp\opencode`).

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
| `serve/test_decision_monitor.py` | the decision in the server's telemetry: the live state, the request list, the totals, the timings |
| `serve/web/decide.js` | the Decide tab: the page's own code (the client of the endpoint) |
| `serve/test_decision_page.py` | the Decide tab: the page's structure, the server's flags, the request it builds |
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

**2026-10-21: the WHOLE set has now been run, cleanly, for the first time - see**  The number is **79.8**, the same as the 235-item stratified sample's, and
all 2,352 items were answered on the fast path with no failures.  The rows below
stay as they are: they are the decode-only baseline the §10 work is measured
against, and the first-token and trie-walk rows are in §10.8.

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
  `Service.run`, so the monitor's live request list did not show a decision
  request. It showed as idle while a decision runs. **Done (§15):** the decision
  path is now a first-class request in the server's telemetry - the Monitor says
  "Deciding" while it runs, and the request list carries the decision and what it
  answered.
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

---

## 15. The decision in the Monitor (2026-10-21)

§10.13 listed an open item: the decision path takes the engine's turn and the
engine's DONE line, but it does not go through `Service.run`, so the monitor's
live request list did not show a decision request, and the model showed as idle
while a decision ran.  It is done.

### 15.1 What it was, measured

The running instance (port 8080, `qwen3.8-flash-next-unsloth-ud-iq4_xs`), one
decision request in flight, polled every 0.4 s:

| | before | after |
| --- | --- | --- |
| `/metrics` `live.state` | `idle` for the whole decision | `reading`, with `live.decision: true` and `live.phase: "deciding"` |
| the prompt's progress | not reported | 887 / 897 tokens, 99 % |
| `/metrics` `live.queued` | 0 | 0 - the decision took the turn at once |
| the request list (`/metrics` `requests`) | 12 rows before and 12 after | one new row, `kind: "decision"` |
| `/metrics` `totals` | `requests` 39 before and 39 after | `requests` +1, and `decisions` +1 |

The browser's Monitor tab follows: the header pill says **Deciding · 99 %**,
the state card says **Deciding** with the prompt's progress, and the request
table has a **Decision** row.

A decision that has to **wait** for the engine's turn is counted as queued
(`status["queued"]`, the count the chat path keeps too).  That one is covered by
a test, not by the runs above: in every live run the decision took the turn at
once.

### 15.2 What a decision row carries

One row per request, whatever path answered it (the fast path does its own
engine turn; the decode path goes through `Service.run`, which hands its
figures to the decision instead of writing a chat row of its own):

```
kind          "decision"          the row is a decision, not a chat turn
questions     3                  how many questions it answered
answers       {"verdict": "good", "healthy": "yes (1.00)"}   what it picked
path          "fast"             the model's own probability, or "decode"
calibrated    true               every answer came from the model's distribution
prompt_tokens / reused / prompt_ms / decode_ms / decode_tok_s
hit_rate / pcie_share / drafts_offered / drafts_accepted
              the engine's own figures for the request, as a chat row has them
finish        "stop"             or "error" when the request could not be answered
```

The answer itself now carries the `timings` the chat path reports
(`prompt_n`, `cache_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`,
`predicted_ms`, `predicted_per_second`), so a client can see what the decision
cost without a second request.  The several-questions path merges them the way
it merges the usage.

### 15.3 The two limits, said plainly

* **The batch slots.**  With `--batch` the chat path does not hold the engine's
  turn and keeps a status per request, so a decision leaves the shared status
  alone: the Monitor would not say "Deciding" in that mode.  The decision path
  itself is unchanged.
* **A decision is counted as a request.**  Its tokens and its time are real
  engine work, so the totals count it (`totals.decisions` counts them apart).
  A server with the endpoint on therefore reports more requests than one with
  it off - which is what happened.

---

## 16. The Decide tab (2026-10-21)

The endpoint is a JSON API: to use it a user has to write the request by hand
and read the answer as JSON.  The web app now has a **Decide** tab that does
both - it is a client of `POST /v1/systemone`, over the same origin and the
same API key, and it changes nothing about the endpoint.

### 16.1 Two things the page had to get right

**The order of the score levels is the order the model reads them in.**  The
page writes the request body by hand instead of with `JSON.stringify`, because
`JSON.stringify` puts the integer-like keys first (`"2"` before `"10"`), so a map
keyed by the user's own names can come out in another order than it went in.
The server keeps the order it receives (`validate_questions` walks the map in
its own order, and `_check_levels` returns `list(criteria)`), so a score
question whose levels are named `0`, `1`, `2` would arrive shuffled.

**A `noul` answers with a probability, not with a side.**  The response carries
the probability of yes, and the documented reading is that the value and the
answer always agree (`docs/DETAILS.md`).  The page reads the side from the value
at 0.5 - the same threshold the reference scorer uses - and says so: "Yes, the
probability of yes: 99.94 %".  It is the reading, not a new answer.

### 16.2 What was checked

* The example request the page offers passes the endpoint's own validator, and
  the limits the page checks are the endpoint's constants (a test reads both
  out of the files and compares them).
* Live, against the running instance: the three question types answered, the
  numbers shown with their meanings, the page's own check refusing a name with a
  colon before anything is sent, and a 400 shown with the server's own message.
  The 502 path (the model wrote something that did not parse) and the page's
  note for a server without `"diagnostics": true` were not exercised live; both
  are the same code path as the 400.

---

## 16. The whole OpenDecisions set, run cleanly (2026-10-21)

§9 and §10.8 measured the **235-item stratified sample**.  This is the whole set:
**all 2,352 items**, one attempt each, the official scorer on the full
`opendecisions.jsonl`.

**2,352 of 2,352 answered. 0 failures. Every item on the fast path.**  One engine,
no restarts, 1 h 57 m of engine time (median 3.93 s an item, p95 6.42 s, mean
4.03 s).  The engine was restarted once during the run, by the operator, for
unrelated reasons; the harness resumed from where it was and nothing was re-asked
(it skips the ids already in the output file).

| | Capability | Intelligence | Calibration | predicted | fast path |
|---|---:|---:|---:|---:|---:|
| **the whole set (2,352 items)** | **79.8** | **71.1** | **88.6** | 2352/2352 | 2352 items |
| the 235-item stratified sample (§10.8) | 79.8 | 76.3 | 81.9 | 235/235 | 235 items |

| type | n | accuracy | competence | calibration |
|---|---:|---:|---:|---:|
| choice | 1,178 | 86.0% | 81.6 | 87.4 |
| noul | 586 | 87.7% | 73.0 | 85.5 |
| score | 588 | 76.4% | 58.6 | 92.8 |

**The two populations agree exactly.**  A random 10% and the whole set, measured
separately, give the same Capability to the decimal (79.8).  That is the strongest
consistency check available here, and it is the first clean full-set measurement
this work has.

**The distance to the known true probabilities** on the 129 items that carry them:
mean TVD **0.241**, against 0.224 on the sample.  The model tracks known
probabilities consistently across the whole set, not only on the sample.

**The shape of the result.**  Two things stand out, and they are opposites:

* the **`score`** type is where the accuracy is thinnest (76.4%, competence 58.6)
  and where the **calibration is best** (92.8) - when the model is wrong about an
  ordered level, it tends to know it is unsure.  This is the type the trie walk
  exists for (§10.7: the score levels are long strings that collide on their first
  token), and the walk is what keeps them on the fast path rather than dropping
  them to a decode.
* the **`noul`** type is the most accurate (87.7%) and the least well calibrated
  (85.5) - it answers "no" with more confidence than it has earned.

### 16.1 The run's shape, and what the older numbers meant

The run went in blocks of 400, and the score fell as the harder families arrived:
93.2 (the easy `llm_router` family alone) → 85.2 → 83.2 → 79.8 → 79.3 → **79.8**.
Every block had **zero** failures.  The dip is the population changing, not the
endpoint degrading.

**What this replaces.**  The 2026-10-20 handoff's full-set run (327 items) failed
43% of its items and was declared invalid, and every long measurement was treated
as suspect because of it.  The cause was the server's misattribution of the engine's
answer, now fixed (§13).  This run is the replacement: the whole set, no failures,
no restarts, and the same number as the sample.

### 16.2 What it does not say

* It does not say the endpoint is good at these questions.  Capability 79.8 on a
  benchmark whose own no-training baseline is 58.9 and whose trained 4B systems
  sit at 67-77 is a middle result; §9 has the context.
* It does not measure the **real-trajectory** case (§10.13), which is still last in
  the user's priority order, nor the policy-head route (§2.5, §2.6).
* It says nothing about the **multi-question** path's speed, which is measured in
  §11.5, and nothing about the **coherence** of the distributions, which is §17's
  subject (the rewrites suite) and the jev-bench human-uncertainty result (§14).

---

## 17. The coherence of the distributions (2026-10-21)

§14 asks whether the endpoint's confidence is calibrated to human uncertainty.  The
other half of the question is **coherence**: asked the same thing two ways, does the
model give the same answer?  JevBench's rewrite suite asks it directly.  A case is
one state and a batch of typed questions, and each of its 41 relations rewrites the
QUESTIONS - never the state - and checks one law of probability over the two answer
distributions.

**THE METRIC IS NOT IN THE DATASET** - it is in the published package
(`pip install jevbench`, Apache-2.0), so the harness re-implements the laws and
names the official file and class beside each one, and it runs from a checkout with
nothing installed.  The laws, the tolerance, the binary score, the graded score and
the coverage rule were checked against the published source
(`jevbench/score.py`, `jevbench/relations/{mea,rep,log}.py`).

**THE RUN.**  12 cases, 41 relations, 492 tests, 643 requests, one question per
request, one attempt each.  **634 answered, 4 failed** (0.6%).  488 tests judged,
4 unjudged.  **All 41 relations scored at 100% coverage** - the failures did not
silently remove a relation, which was the thing to check.

| | score | graded | 95% CI |
|---|---:|---:|---|
| **overall** | **0.658** | **0.923** | [0.599, 0.717] / [0.908, 0.935] |
| REWORD (14 relations) | 0.771 | 0.918 | |
| COMBINATION (22 relations) | 0.788 | 0.957 | |
| **NEGATION (5 relations)** | **0.417** | 0.893 | |

### 17.1 What it says: coherent under rewording, incoherent under negation

**Perfect (1.000)**: `instruction_paraphrase`, `cross_lingual`, `clone_symmetry`,
`noul_as_choice`, and the whole LOG.bounds family - `frechet_and_upper`,
`inclusion_exclusion`, `entailment_strength`, `quantifier_monotonicity`,
`count_upper`, `count_lower`, `nary_and_upper`, `nary_and_lower`, `nary_or_upper`.
So the endpoint's probabilities survive a paraphrase, a translation, a renaming, an
option permutation, and the logical-composition rewrites: the same distribution
comes back.

**Broken**: `choice_complement` **0.083**, `score_cumulative` **0.000**,
`compound_negation` 0.167, `score_complement` 0.333, `de_morgan` 0.333.

The mechanism, from the harness's own arithmetic.  The official `ChoiceComplement`
is *"P(answer is not o), asked as a Noul, equals 1 - P(o) from the Choice"*; the
model does not carry its distribution across the reformulation.  Measured, on
`customer_support-037`:

| | |
|---|---|
| the base choice, P(high) | **0.001** |
| the "is this NOT high?" probe | **0.097** |
| the law wants | 0.999 |

The two answers contradict each other.  `score_cumulative` is the same shape: the
"is it level k or higher?" probe disagrees with the upper tail of the model's own
score distribution (gaps of 0.13 to 0.51 against the 0.05 tolerance).

**This is a finding about the model, not about the endpoint.**  The endpoint
faithfully reports what the model says each time; it cannot make a model consistent.
It is the same shape as §14's finding: asked one way or another, the model's
confidence is less well anchored than its accuracy.

Also weak: the **binding** group, where a question's key conflicts with its own
instruction or description (`key_instruction_conflict` 0.500,
`name_description_conflict` 0.500, `description_swap` 0.333).  That is the model
following the key rather than the prose, which is worth knowing for a consumer that
keys its questions by name.

### 17.2 The four failures, and what they say about the contract

All four are the same thing, and it is a **contract** question rather than a model
question.  The options are named `option_1` ... `option_4` with a description
attached, and the model answered with the **description**:

```
/v1/systemone: question "plan_tier" takes one of ['option_1', 'option_2', 'option_3',
'option_4'], got 'The paid pro plan'
```

The strict parse correctly refuses it - a made-up answer is worse than a missing
one - so it is a 502 rather than a wrong answer.  But this is the **same shape the
endpoint already solved for score levels**: §6.0's `{id: description}` form exists
because "the model was measured answering the short id unprompted against
full-string criteria, so exact reproduction of ~100-character strings is the wrong
demand".  A choice option that is a generic name with a description is that same
situation, and the endpoint had no answer for it.

**Done (2026-10-21, later the same day).**  A text that matches **exactly one**
option's description is read as that option; a name is still checked first, an
ambiguous description (two options carrying it) stays an error rather than a coin
flip, and an option with no description is never matched.  The answer the endpoint
returns is still one of the option names, so the contract does not change.

Measured:

| | before | after |
|---|---:|---:|
| the four requests | 0 of 4 answered | **4 of 4 answered** |
| the rewrites suite | 488 tests judged, 4 unjudged | **492 judged, 0 unjudged** |
| the overall coherence score | 0.658 [0.599, 0.717] | **0.661** [0.603, 0.719] |
| the graded score | 0.923 | 0.923 |
| the 235-item OpenDecisions sample | 79.8 | 79.2 (inside the noise floor) |
| the suite | - | 1,014 tests, 0 failures |

---

## 18. The per-type temperature (2026-10-21)

### 18.1 What was missing

The endpoint reported the model's own distribution and nothing adjusted it.  The
roadmap's item 1.1 asked for one temperature per question type, fitted on the
235-item OpenDecisions sample and applied at serve time, so that the reported
distribution is calibrated.

**The premise turned out to be wrong, and the measurement says so.**  The plan was
that the model is over-confident, so a temperature above 1 (the dataset's own README
quotes letter-readout temperatures of 5.96, 7.2444 and 7.2444, fitted on other data).
On this sample the opposite is true: the model's errors sit in the **middle** of the
confidence range, where it is UNDER-confident, so the temperature that helps is
**below 1**.  The three objectives that were fitted disagree, and the disagreement is
the finding:

| objective | what it minimizes | choice | noul | score |
|---|---|---:|---:|---:|
| nll | the negative log-likelihood of the gold label | 1.16 | 1.57 | 0.94 |
| **brier** | the multiclass Brier score | **0.62** | **0.67** | **0.77** |
| ece | the scorer's own ECE for the type | 0.58 | 0.67 | 0.75 |

The NLL is the textbook objective and it is dominated by the items the model gets
wrong with confidence (a wrong item at p = 0.99 costs -log(0.01) = 4.6), while the
ECE is a per-bin average that treats every bin alike.  The Brier score is between the
two: smooth and proper, like the NLL, but quadratic, so a confidently wrong item
costs a bounded amount.  **The Brier score is the one that is served**, and all three
are cross-validated in `tools/decision_policy/temperature_report.md`.

### 18.2 What was done

**The serving side.**  The launch config carries one temperature per question type:

```json
"systemone": {"enabled": true, "diagnostics": true,
              "temperature": {"choice": 0.62, "noul": 0.67, "score": 0.77}}
```

The key is **opt-in**: without it every type is 1.0, and 1.0 means "no temperature" -
the reported numbers are then exactly the ones §10 reported (the default path is
byte-identical).  A temperature that is not a number, is not above 0, or names a
question type the endpoint does not have is a startup error, not a silent default.

The temperature is applied to the **option-level** effective logits, which is the
closed reading (the options renormalized over themselves):

```
closed_T(i) = normalize(closed_1(i) ** (1/T))
```

That is exact, and it is why the fit could be made offline on a stored run log and
the server reproduces it: the same expression, the same order of operations.  For the
**trie walk** the two possible forms are NOT the same thing: an option's weight is a
product of per-node token probabilities, so "divide each node's logit by T" divides
by a different normalizer per option (the options' paths have different numbers of
nodes).  The option-level form is the coherent one, and the only one that can be
reproduced offline.  For a single-row question the two forms are identical.

**The open reading, plainly.**  The open reading is the model's OWN probability of each
option, `P = exp(logit - logz)`, with the full-vocabulary normalizer `logz`.  At
temperature T it would be `exp(logit/T) / Z(T)`, and **Z(T) is not available**: the
server has the engine's top-k window and the T = 1 normalizer, not the whole 248,320
token vocabulary.  So the mass the model put on the options (measured at T = 1) is
kept and the closed reading at T is what it is scaled by:

```
open_T(i) = closed_T(i) * M,    M = the mass on the options at T = 1
```

**That is NOT the model's own probability at temperature T.**  It keeps the three
properties the endpoint promises - the confidence is in [0, 1], it never exceeds the
reported distribution's maximum, and a larger T lowers it - and it is the number the
diagnostics and the response report.  The mass outside the options (`stop`,
`outside`) is left at its T = 1 value: the temperature reshapes the distribution OVER
the options.

**The PICK cannot change**: the map is monotone, so the argmax is the same under every
temperature, and a yes/no never crosses the 0.5 threshold (the sign of
`closed_T(yes) - 0.5` is the sign of `closed_1(yes) - 0.5`).  One thing does move: the
**expected level** of a `score` question - the probability-weighted mean the scorer
reads for its Intelligence - and it moved on **all 59** score items of the sample, by
up to **0.18 levels**.  That is where the Intelligence gain comes from, not only the
calibration.  Measured, not assumed.

The **decode path** (the fallback, when the model wrote something that is not an
option) reports the degenerate 1.0/0.0 distribution, and a temperature is a
mathematical no-op on it (1^(1/T) = 1, 0^(1/T) = 0), so it is not scaled.

### 18.3 The fit

`tools/decision_policy/temperature.py` reads the baseline run log and the baseline
predictions, rebuilds each item's closed distribution (cross-checked against the run
log's own readings: 235 items, 0 mismatches), fits one scalar per type, validates it
and writes the constants with their provenance.

| | |
|---|---|
| the model | `qwen3.8-flash-next-unsloth-ud-iq4_xs` |
| the arm | the serial first-token path, 235/235 predicted, every item on the fast path |
| the objective | the mean multiclass Brier score of the distribution against the gold label |
| the grid | [0.5, 10.0], step 0.01 (the optimum is below 1, so the grid has to reach there) |
| the split rule | 5-fold cross-validation, stratified by type: every type's items sorted by id, shuffled with `random.Random(42)` and dealt round-robin into the five folds; the temperature of a fold is fitted on the other four folds and applied to the fold's own items.  A 70/30 split (the same seed, the same stratification) is the second check. |
| the seed | 42 |
| the constants | choice 0.62, noul 0.67, score 0.77 |

The constants and their provenance are in
`tools/decision_policy/temperature_calibration.json`; the fit's own report, with every
table, is in `tools/decision_policy/temperature_report.md`.

### 18.4 The validation: the fit never sees the item it is scored on

The table below holds the **raw top-level ECEs** (the scorer's own `ece` function, on
the same pairs it uses: the top-label ECE on the **hard tier only** for choice - 53 of
the 107 choice items - the ECE of P(yes) for noul, the top-level ECE for score).
Out-of-fold over the five folds, and a 70/30 split as the second check:

| type | the raw top-level ECE at T = 1 | the raw top-level ECE, the fitted constants (in-sample) | the raw top-level ECE, out-of-fold (5 folds) | the raw top-level ECE, 70/30 held out |
|---|---:|---:|---:|---:|
| choice | 0.1313 | 0.1128 | **0.1224** | **0.0374** (from 0.1696) |
| noul | 0.0756 | 0.0656 | 0.0786 | **0.1721** (from 0.1791) |
| score | 0.1082 | 0.0682 | **0.0649** | **0.1031** (from 0.1783) |

(the third column is the fit on every item of the type, so it is in-sample; the fourth
is the honest one; the fifth is the 70/30 held-out slice, which is thin - 32 choice
items of which about 15 are hard, 21 noul, 18 score - so its ECEs are much larger than
the full sample's for that reason alone.)

**These are not the scorer's calibration component, and for the score type they are
only half of it.**  The scorer's per-type calibration is `100 (1 - ECE / 0.5)` for the
choice and the noul questions, and for the **score** questions the **mean** of
`100 (1 - top-level ECE / 0.5)` and `100 (1 - mean nRPS)` (`score.py:162`).  So for
choice and noul the raw ECE above, scaled, IS the component; for score it is the ECE
half, and the other half is the normalized ranked probability score.

**The scorer's own per-type calibration component** (0-100, higher is better), computed
with the scorer's `calibration` function on the same distributions:

| type | at T = 1 | out-of-fold (5 folds) | 70/30 held out |
|---|---:|---:|---:|
| choice | 73.7 | **75.5** | **92.5** (from 66.1) |
| noul | 84.9 | 84.3 | **65.6** (from 64.2) |
| score | 86.3 | **90.8** | **86.9** (from 79.2) |

(the out-of-fold column is every item scaled by the temperature of the fold it was
**not** in; the 70/30 column is the held-out slice, fitted on its own train part.
The component is the number the exit criterion is read from, and it agrees with the
raw ECE on the sign for all three types: the choice and the score improve, the noul
is a wash out-of-fold and improves on the 70/30 slice.)

The official scorer, on the out-of-fold predictions (every item scaled by the
temperature of the fold it was **not** in):

| run | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 79.8 | 78.0 | 81.7 |
| the out-of-fold fit (nll) | 79.1 | 78.3 | 79.8 |
| **the out-of-fold fit (brier)** | **81.5** | **79.4** | **83.5** |
| the out-of-fold fit (ece) | 81.2 | 79.2 | 83.3 |
| the in-sample fit (what the server gets) | 82.1 | 79.2 | 84.9 |

Per type, out-of-fold: the calibration 73.7 -> **75.5** (choice), 84.9 -> 84.3 (noul),
86.3 -> **90.8** (score); the competence 86.7 -> 86.7 (choice), 80.5 -> 80.5 (noul),
**66.9 -> 71.1** (score - the expected level moves towards the pick when the
distribution is sharpened, and the scorer reads the expected level).

**The NLL arm is the interesting one: it makes things worse** (Capability 79.1 against
the baseline's 79.8) and its ECE is worse for the choice and the noul.  A reader who
fits a temperature on this sample with the textbook objective and ships it will lose
0.7 Capability points.

How sure is the fit?  The items of the type are resampled with replacement, TWICE per
resample: the temperature is fitted on one sample and the ECE is measured on the
other, so the ECE that is compared is not the one the fit produced.  (Measuring both
on the same resample is in-sample and optimistic; the numbers below are the honest
out-of-bag ones.)  The share is the resamples where the temperature lowers the ECE.

| type | resamples | the temperature lowers the ECE | the ECE difference (5%, 95%) |
|---|---:|---:|---|
| choice | 1,000 | 64% | -0.0327, +0.0625 |
| noul | 1,000 | 67% | -0.0180, +0.0320 |
| score | 1,000 | 68% | -0.0447, +0.0769 |

**A share is not an effect size.**  It says how often the temperature wins on a
resample of 235 items; it does not say by how much, and it carries no interval for the
size.  For that, the paired item bootstrap in the next section - and the intervals
there are wide.

### 18.5 The re-run of the 235-item sample (IN-SAMPLE)

ENGINE FRESHLY RESTARTED on 2026-10-21: the server tree was stopped, the engine
process exited (the card went back to 23,219 MiB free), and the server was started
again, so the model was loaded from scratch before the run (§12.5).  The fitted
constants in the launch config, the same 235 ids, one question per request, the
official `score.py` on `sample10_items.jsonl`.

**235 of 235 answered, 0 failures, every item on the fast path.**  The commands:

```
python tools/decision_policy/opendecisions.py --items <opendecisions.jsonl> \
  --out <fix_serial_temp.jsonl> --ids-file <sample10.txt> --batch-k 1
python <score.py> <fix_serial_temp.jsonl> --data <sample10_items.jsonl>
```

| run | Capability | Intelligence | Calibration | choice | noul | score |
|---|---:|---:|---:|---|---|---|
| the baseline arm, re-scored (T = 1) | 79.8 | 78.0 | 81.7 | 89.7% / 73.7 | 91.3% / 84.9 | 79.7% / 86.3 |
| **the same run with the temperature undone (T = 1)** | 78.9 | 75.9 | 81.8 | 88.8% / 78.9 | 88.4% / 79.2 | 78.0% / 87.4 |
| **the re-run with the temperature (T = 0.62 / 0.67 / 0.77)** | **80.1** | **77.0** | **83.1** | 88.8% / **80.8** | 88.4% / **81.8** | 78.0% / 86.7 |

(per type: accuracy and the scorer's calibration component.)

**The middle row is the one that measures the temperature.**  It is the SAME run with
the temperature undone - the same engine, the same picks, the same logits, the
temperature's own arithmetic removed from the distributions (the closed reading put
back through `normalize(closed_T ** T)`) - so the difference between the middle row
and the bottom row is the temperature and nothing else:

| | Capability | Intelligence | Calibration | choice cal. | noul cal. | score cal. | score comp. |
|---|---:|---:|---:|---:|---:|---:|---:|
| the temperature's own effect on this run | **+1.2** | **+1.1** | **+1.3** | +1.9 | +2.6 | -0.7 | +3.2 |

**And how big is that?**  A **paired item bootstrap**: the 235 items are resampled with
replacement and BOTH arms are scored on the same resample, so the difference is
paired - an item's own run-to-run wobble cancels and what is left is the
temperature's effect.  The interval is the 2.5% and 97.5% percentiles of the
differences (1,000 resamples, seed 42, `tools/decision_policy/temperature.py`):

| | Capability | Intelligence | Calibration |
|---|---|---|---|
| the temperature's own effect on this run | +1.2 | +1.1 | +1.3 |
| **the paired bootstrap's 95% interval** | +1.36 [-1.10, +3.79] | +1.07 [+0.44, +1.72] | +1.64 [-3.16, +6.46] |
| the resamples where it is better | 87% | 99.8% | 75% |
| the out-of-fold fit against the baseline arm | +1.66 [-1.01, +4.41] | +1.46 [+0.72, +2.46] | +1.87 [-3.25, +7.06] |
| ... the resamples where it is better | 89% | 100% | 77% |

**The direction is right; the size is not established on 235 items.**  The
**Intelligence** gain is the part that holds up - its interval excludes zero - and
the **Calibration** gain is the least certain of the three.  A reader who wants a
number they can rely on should quote the interval, not the point estimate.

The re-run against the baseline arm is **+0.3** Capability, which is inside the
run-to-run spread of this sample (§10.8: about 0.6 points, and the arms measured on
this sample read 78.9, 79.1, 79.2, 79.5, 79.6, 79.8, 80.4).

**The picks: 5 of 235 moved against the baseline arm** (230 the same, none lost).
The temperature cannot move a pick - the map is monotone, so the argmax is the same,
and a yes/no never crosses the 0.5 threshold - so those 5 are the engine's own
run-to-run wobble (§10.8: the KV cache's layout changes the logits a little).  In
every one of them the **same pair** of options is on top in both runs, in the other
order, which is the shape of a small change in the logits rather than of a
re-ranking.  The margin column is the **smaller of the two runs'** gaps between their
top two options:

| item | type | the baseline's top two | the new run's top two | margin |
|---|---|---|---|---:|
| `jb16-jaggedness-526f673178` | choice | thursday 0.249 / tuesday 0.234 | tuesday 0.334 / thursday 0.237 | 0.015 |
| `jb16-jaggedness-5f9ee60589` | noul | yes 0.507 / no 0.470 | no 0.522 / yes 0.452 | 0.037 |
| `jb16-calibration_gen-18680f3547` | noul | yes 0.548 / no 0.429 | no 0.500 / yes 0.473 | 0.027 |
| `jb16-composite_gen-ecf4aba46e` | score | 2: 0.448 / 3: 0.332 | 3: 0.495 / 2: 0.361 | 0.116 |
| `jb16-multilingual-0746bb0c27` | choice | chunked_pipeline_mid 0.518 / frontier_direct 0.383 | frontier_direct 0.569 / chunked_pipeline_mid 0.399 | 0.136 |

(the noul's numbers are the model's own probability of yes, from the run log's open
reading.)

**The cost.**  The latency median was **3.77 s** a question and the p95 **5.82 s**,
against the baseline arm's log's 3.12 s and 5.68 s.  The temperature cannot be the
reason: it is four `pow` calls per question, measured at **1.2 us** on this machine,
and it runs after the engine's turn.  The two runs also differ in their engine
state, and both show the same growth over the run (the conversation cache's parked
conversations): **in run order**, the first 40 items took 3.23 s a question and the
last 40 took 3.90 s (the baseline arm: 2.68 s and 3.21 s).  The 40 fastest and the
40 slowest items are a different pair of figures - 2.42 s and 5.60 s, the baseline
arm's 2.06 s and 5.35 s - and they say the same thing.  A clean A/B on one engine
was not run, so the 0.65 s of median difference is not attributed to anything in
particular; it is not the temperature's arithmetic.

**The VRAM**: the minimum free was **1064 MiB** (1206 at load) against the engine's
1024 MiB warning threshold, no sample below it, the largest conversation snapshot
488 MiB - the same shape as §10.8's 1082 MiB.  These are the engine log's own
readings, and the run's window of that log (lines 113900-116522 of
`strata-unsloth-ud-iq4_xs.log`) is kept beside the run's predictions as
`fix_serial_temp.vram.log`; `python tools/decision_policy/vram_headroom.py --log
fix_serial_temp.vram.log --since-line 1` reads them back.

The mean TVD to the known true probabilities, on the 13 calibration items: **0.250**
(the baseline arm's 0.224, the temperature-undone run's 0.229).  The sharpening
makes the distribution less smooth, and on 13 items that costs a little distance to
the true probabilities - the price of a sharper answer, and the number to watch if
the endpoint is used for uncertainty rather than for a pick.

### 18.6 The limits

- **The re-run is IN-SAMPLE.**  The constants were fitted on these same 235 items, so
  its Capability is not an unbiased estimate of anything.  The honest number is the
  out-of-fold one: **81.5** against the baseline's 79.8.
- **The dataset says not to fit on it.**  Its README: "fit temperatures on your own
  data, never on OpenDecisions".  The roadmap asks for this fit, so the numbers are
  labelled as they are.  The independent check is the **jev-bench** run with the
  human soft labels (roadmap item 1.2): there the question is not "is the model right"
  but "is the confidence calibrated to the uncertainty a human would feel" (§14).
- **The open reading's meaning changed.**  The reported confidence is no longer the
  model's own probability: it is the closed reading at T scaled by the mass the model
  put on the options, measured at T = 1.  A consumer that read the confidence as "the
  model's own probability of this answer" must read it as "the forced-choice
  probability, scaled".  §18.2 has the arithmetic.
- **The exit criterion is NOT met for all three types, on the 5-fold out-of-fold
  measure.**  On the scorer's own per-type calibration component - the number the
  criterion is about; for the choice and the noul questions that ECE scaled, for the
  score questions the mean of the top-level ECE and (1 - nRPS) - the held-out value is
  better for the choice (73.7 -> 75.5) and the score (86.3 -> 90.8) and a wash for the
  noul (84.9 -> 84.3), which is what the raw top-level ECE says as well (0.1313 ->
  0.1224, 0.1082 -> 0.0649, and 0.0756 -> 0.0786 for the noul, inside the noise).  The
  noul's ECE curve is jagged and shallow around T = 1 (0.0756 at T = 1, 0.0656 at its
  in-sample minimum) and its minimum does not survive the cross-validation.  **The
  other held-out measure does show the noul improving** - the 70/30 split gives
  0.1791 -> 0.1721, and its component 64.2 -> 65.6 - so the criterion's outcome
  depends on which held-out measure is meant; the 5-fold out-of-fold is the one quoted
  here, and the 70/30 slice is thin (21 noul items).  The other half of the criterion,
  "no regression in Intelligence", **is** met: 78.0 -> 79.4 out-of-fold.
- **The size of the effect is not established.**  The paired item bootstrap puts an
  interval on it and the interval is wide (the Capability's [-1.10, +3.79] on the
  paired control, [-1.01, +4.41] out-of-fold).  The direction is right; the
  magnitude is not known, and the Intelligence gain is the part that holds up.
- **The choice ECE is a thin slice**: 53 of the 107 choice items, and the five folds
  hold 6 to 15 of them.  The fold table in the fit's report says how thin.
- **One temperature per type is one number for a whole family of questions.**  A
  temperature per question would fit better and would not be honest.
- **The constants belong to this model.**  They were fitted on one model's readings
  of one dataset; another model (or another quantization of it) needs its own fit.

---

## 19. The temperature, checked against the humans (2026-10-21)

§18 fitted one temperature per question type on the 235-item OpenDecisions sample
and shipped them in the launch config.  §18.5's re-run of those 235 items was
**IN-SAMPLE** - the constants were fitted on exactly those items - so it could not
answer the question the constants exist to answer: does the temperature still help
when the answer is not the one it was fitted against?

**This is the first out-of-sample evidence for the constants.**  The check is the
jev-bench sample with the **human label distribution** (§14): there the endpoint's
distribution is scored against how uncertain the *humans* were, not against one
right answer.  The two are different targets, and this section is where they are
compared.

**The short answer: the temperature does not transfer.  It makes the distance to
the human distribution WORSE, on all three types and on all four configs, and it
costs 3.1 Capability points on the same run.  The exit criterion is not met.**

### 19.1 The run

**ENGINE FRESHLY RESTARTED** on 2026-10-21 (§12.5): the server tree was stopped
(`taskkill /T /F` on the `strata.py serve` process, the engine `strata.exe` died with
it), the card went back to **23,450 MiB free**, and the server was started again at
19:24:33; the model was loaded from scratch and `/v1/status` answered at 19:26:46
with the temperature block `{choice: 0.62, noul: 0.67, score: 0.77}`.  The run
started at 19:27:21 and finished at 20:40:52 - **73.5 minutes** - and the engine was
not shared with anything else.

The same 683 items, the same harness, one question per request, one attempt each:

```
python tools/decision_policy/jevbench.py \
  --items <jevbench_sample3_items.jsonl> --out <jevbench_sample3_temp_predictions.jsonl>
```

**683 of 683 answered, 0 failures, 0 502s, every item on the fast path** (683 of
683).  The latency: median **4.30 s**, p95 **14.15 s**, mean 6.46 s; the first 40
items took a median 3.18 s and the last 40 a median 4.33 s.  The p95 is the engine
streaming experts off the file, not the endpoint: the server's own log reports
618,844 expert blob reads from the GGUF and an expert cache hit rate of 80-88% on
this machine, whose RAM was 58.1 of 63.7 GiB used (the server's own status, before
the run) while 36 GiB of experts were resident.  The temperature is four `pow`
calls a question, and it runs after the engine's turn.

### 19.2 The paired comparison

Comparing the new number against the stored 0.4432 would compare two different
runs, so the comparison here is **paired**: the T = 1 arm and the tempered arm come
out of the SAME run, on the SAME items, on the same engine, in the same session.

The temperature's transform is invertible, so the T = 1 arm is inside the tempered
run: the server scales the closed reading by `closed_T(i) = normalize(closed_1(i) **
(1/T))`, and the inverse is `closed_1(i) = normalize(closed_T(i) ** T)`.
`tools/decision_policy/jevbench_paired.py` does the inversion, per item, from the
run log, and writes the recovered arm as a predictions file the official scorer
takes.

**The inversion is validated, not assumed** (three independent checks, all on the
run's own data):

| the check | the result |
|---|---|
| the tempered arm's TVD, recomputed, against the TVD the harness stored per item | **0 mismatches** over 280 items (the tolerance is 1e-6; the harness rounds to 6 decimals) |
| the forward transform applied to the recovered arm, against the arm the run reported | max absolute difference **2.2e-16** |
| the noul's recovered P(yes), against the run log's own open reading (`closed[yes] = open[yes] / mass`) | max absolute difference **1.1e-16** |

**The recovered arm is a faithful stand-in for a T = 1 run.**  It scores mean TVD
**0.443944** against the stored T = 1 run's **0.443164** - a difference of 0.0008 -
and Capability 55.1 against 55.2.  The two runs differ by the engine's own
run-to-run wobble (§10.8): the recovered arm's distribution differs from the stored
run's by a mean 0.031 (choice), 0.019 (noul) and 0.027 (score) in the largest
coordinate, and **11 of the 683 picks moved** (the temperature cannot move a pick -
§18.2 - so those 11 are the engine's wobble, close to the 2.1% §18.5 saw on the
OpenDecisions sample, 5 of 235).

### 19.3 The distance to the human distribution

Over the 280 records that carry the human distribution, the same items, both arms
from the same run:

| | mean TVD | |
|---|---:|---|
| **the T = 1 arm** (the temperature's arithmetic removed) | **0.4439** | the reference for this run |
| **the fitted arm** (T = 0.62 / 0.67 / 0.77) | **0.4678** | **+0.0239, worse** |
| the stored T = 1 run (§14.1), for reference | 0.4432 | the exit criterion's number |

**The exit criterion is NOT met.**  The criterion is a TVD lower than 0.4432; the
fitted arm is 0.4678, and it is worse than its own run's T = 1 arm by 0.0239.

Per type, paired on the same items:

| type | n | T = 1 | fitted | change | better | worse | same |
|---|---:|---:|---:|---:|---:|---:|---:|
| choice | 130 | 0.4699 | 0.5124 | **+0.0425** | 13 | **117** | 0 |
| noul | 100 | 0.3844 | 0.3897 | **+0.0053** | 51 | 49 | 0 |
| score | 50 | 0.4955 | 0.5082 | **+0.0127** | 17 | 33 | 0 |

Per config, the same way (each config here is one question type, so the two tables
are the same data cut two ways):

| config | type | n | T = 1 | fitted | change |
|---|---|---:|---:|---:|---:|
| chaosnli | choice | 80 | 0.3104 | 0.3654 | **+0.0550** |
| civil_comments | noul | 100 | 0.3844 | 0.3897 | **+0.0053** |
| go_emotions | choice | 50 | 0.7253 | 0.7477 | **+0.0224** |
| measuring_hate_speech | score | 50 | 0.4955 | 0.5082 | **+0.0127** |

**Every type and every config is worse.**  The three types do not differ in
direction, only in size: the choice's temperature (0.62) does the most damage, the
noul's (0.67) the least.

### 19.4 The official scorer, paired

The recovered T = 1 arm was written as a predictions file and scored by the
**official** `score.py` in the same session, on the same items, against the
tempered arm's own predictions file:

| arm | Capability | Intelligence | Calibration | choice cal. | noul cal. | score cal. |
|---|---:|---:|---:|---:|---:|---:|
| **the T = 1 arm** (recovered, same run) | **55.1** | 46.6 | 63.6 | 69.3 | 57.8 | 63.6 |
| **the fitted arm** (same run) | **52.0** | 46.6 | 57.5 | 58.7 | 54.9 | 58.8 |
| the stored T = 1 run (§14.1), for reference | 55.2 | 46.7 | 63.7 | 68.3 | 60.4 | 62.3 |

**The temperature's own effect on this run: Capability -3.1, Intelligence -0.05,
Calibration -6.1.**  The Intelligence barely moves because the pick cannot move
(§18.2) and the accuracy per type is identical in both arms (71.1% / 74.3% /
48.5%); the score's competence dips a little (26.0 -> 25.8) because the scorer
reads the expected level and the temperature moves that.  The Calibration is where
the whole loss is, and it is lost on all three types.

The scorer's own ECEs, on the scorer's own pairs (the top-label ECE on the hard
tier for choice, the ECE of P(yes) for noul, the top-level ECE for score):

| type | ECE at T = 1 | ECE, fitted | change |
|---|---:|---:|---:|
| choice | 0.1533 | 0.2067 | **+0.0534** |
| noul | 0.2108 | 0.2253 | **+0.0145** |
| score | 0.2700 | 0.3132 | **+0.0432** |

So the temperature does not only fail against the humans' *distribution*: it also
fails against the humans' *hard* answer, on this dataset.  The ECE of P(yes) on the
jev-bench noul items is 0.2108 at T = 1 and 0.2253 at T = 0.67 - the noul's
temperature makes the binary calibration worse too.

### 19.5 Where the temperature hurts, and why

The split that explains it is the humans' own uncertainty.  Each type's items are
sorted by the entropy of the human distribution and cut in half:

| type | half | n | T = 1 | fitted | change |
|---|---|---:|---:|---:|---:|
| choice | the humans least uncertain | 65 | 0.4447 | 0.4704 | +0.0257 |
| choice | the humans most uncertain | 65 | 0.4952 | 0.5545 | **+0.0593** |
| noul | the humans least uncertain | 50 | 0.3181 | 0.3168 | -0.0013 |
| noul | the humans most uncertain | 50 | 0.4507 | 0.4626 | +0.0119 |
| score | the humans least uncertain | 25 | 0.4319 | 0.4305 | -0.0013 |
| score | the humans most uncertain | 25 | 0.5591 | 0.5859 | **+0.0268** |

**Sharpening helps where the humans are certain and hurts where they are not**, and
the effect is twice as large on the uncertain half.  Two different mechanisms, both
measurable:

* **The shape.**  The mean mass the model puts on its own top label, against the
  humans': the choice **0.7862 at T = 1 (0.8613 fitted) against the humans' 0.6215**
  - the model is already MORE peaked than the humans, so any sharpening moves the
  distribution away from theirs.  The score is the other way round (the model
  0.7977, the humans 0.8233), which is why the score's certain half is a wash.
* **The bias.**  On the noul the model over-predicts "yes" on **95 of the 100**
  items: the mean signed difference against the humans is **+0.357** on a mean
  absolute difference of 0.384, so 93% of the noul's TVD is one-directional bias,
  not uncertainty.  The model says P(yes) = 0.446 on average where the humans say
  0.089.  A temperature cannot fix a bias: it makes the wrong side more extreme.
  (The noul's shape is roughly right - the model 0.8840 at T = 1 against the humans'
  0.9184 - so here the bias is the whole story.)

§14.3 found the model over-confident relative to the humans (its mean confidence
0.868 when it agrees with the humans' consensus, 0.776 when it does not).  On the
noul the story is the other way round: the model is not over-confident, it is
**over-optimistic about the wrong answer**, and a sharper distribution is a worse
fit for that.

### 19.6 The noul

The noul was the open question, because 0.67 was fitted to sharpen P(yes) and the
jev-bench noul items are the ones where the humans are least certain.

**It does not pay.**  The noul's items are all `civil_comments`, and the humans
there are mostly certain: **74 of the 100 have a 0/1 human label** and 26 are
genuinely uncertain (0 < P(yes) < 1).  Split that way:

| the noul's items | n | T = 1 | fitted | change |
|---|---:|---:|---:|---:|
| the humans are certain (P(yes) = 0) | 74 | 0.3478 | 0.3431 | **-0.0047** (better) |
| the humans are uncertain (0 < P(yes) < 1) | 26 | 0.4885 | 0.5223 | **+0.0338** (worse) |
| all of them | 100 | 0.3844 | 0.3897 | **+0.0053** (worse) |

**The hypothesis is confirmed: the sharpening helps on the certain items and hurts
on the uncertain ones, and the uncertain ones cost more.**  The noul's overall TVD
change is a wash in size (51 items better, 49 worse) but it is the wrong way, and
the noul's ECE gets worse (0.2108 -> 0.2253) and the noul's calibration component
drops **57.8 -> 54.9**.

**The recommendation: revert the noul to T = 1.**  Nothing supports 0.67 for the
noul.  On OpenDecisions its held-out ECE was already a wash (§18.4: 0.0756 ->
0.0786 out-of-fold) and its 70/30 slice was thin; on jev-bench it is worse on both
the TVD and the ECE.  A temperature that cannot be shown to help on either dataset
should not be shipped.  The noul's constant in the config should go back to 1.0.

### 19.7 What this settles, and what it does not

* **The exit criterion is NOT met.**  The fitted arm's TVD is 0.4678 against the
  0.4432 the criterion names, and it is worse than its own run's T = 1 arm
  (0.4439).  The temperature loses on the human gold standard, on the human hard
  answer (the ECEs), and on the official scorer's Capability.
* **The two targets disagree, and that is the finding, not a contradiction.**  The
  constants were fitted against a HARD gold label (the Brier score of the
  distribution against one right answer, §18.1) and they help there: the out-of-fold
  Capability on the OpenDecisions sample is 81.5 against the baseline's 79.8
  (§18.4).  jev-bench's soft labels ask a different question - is the model's
  uncertainty as wide as the humans' - and against that question the same constants
  lose.  A scalar temperature can trade one for the other; it cannot win both.
* **The noul should go back to T = 1** (§19.6).  The choice and the score fail
  their own independent check too (TVD +0.0425 and +0.0127, ECE +0.0534 and
  +0.0432, Calibration -10.7 and -4.8 points), so the honest reading of this section
  is wider than the noul: **on this evidence the fitted constants should not ship
  as they are.**  That is the orchestrator's call to make, not this section's - but
  it should be made with these numbers in front of it.
* **The fit was per type and the three types' outcomes differ only in size.**  All
  three are worse; the choice's 0.62 is the most damaging and the noul's 0.67 the
  least.  The per-type design of §18 is not what failed here.
* **The run was clean.**  683 of 683, 0 failures, 0 502s, every item on the fast
  path, one attempt each and nothing retried, on a freshly restarted engine, with
  the engine to itself.  The scorer's rule for a missing prediction was never
  needed.
* **The limits.**  The licence is per-source and mixed (several "research use
  only") and the dataset says validation only.  jev-bench has no difficulty tier,
  so every item is filed under "hard" and the scorer's tier weights collapse to one
  - that is not a difficulty claim.  The 235-item OpenDecisions re-run (§18.5) and
  the fit itself remain in-sample; this section is the independent check, and it
  is the only one there is.  One temperature per type is one number for a whole
  family of questions, and the constants belong to this model and this
  quantization.  The run's p95 latency is the engine streaming experts off the file
  on a machine whose RAM is nearly full, not the endpoint.
* **What was not run.**  No T = 1 re-run of the 683 items was made on this engine
  (the stored run is the reference, and it is a different run), so the paired
  baseline is the **recovered** arm, not a second T = 1 run; the two agree to 0.0008
  in mean TVD and 0.1 in Capability, which is the evidence that the recovery is
  sound, but it is not the same thing as a second run.  No temperature other than
  the fitted three was tried on this dataset, so this section does not say what the
  best temperature for the human distribution would be - only that the fitted one
  is worse than none.  No second model, no engine change, no new dependency.

## 20. The refit, and the rule it is judged by (2026-10-21)

**This section is written BEFORE the run.**  The rule below is committed before the
first item is measured, so that the numbers in §20.2 cannot be read as a rule written
after the fact.  The results go into §20.3 and after; if a constant ships, it ships
because it cleared the rule, not because the rule was written to fit it.

### 20.1 The question

§19 measured the §18 constants against the human labels and they lost on every type
and every config: the TVD to the human distribution went 0.4439 -> 0.4678, Capability
55.1 -> 52.0, Calibration 63.6 -> 57.5.  The endpoint therefore ships at **T = 1**,
and the constants stay in `tools/decision_policy/temperature_calibration.json` for the
record.  The knob is not the problem; the constants were.

So the question is no longer "does a temperature help?"  It is:

> **Is there a per-type constant that helps on TWO populations at once - or is the
> endpoint's calibration population-specific?**

The second population is the jev-bench **validation** split, fitted on a pool that is
licence-clean and out of sample with respect to OpenDecisions: 500 items of every
question type, drawn per config from the configs whose licence the record clears
(`docs/research/decision_policy_training_data.md`), with the counts and the seed
(20261021) in `tools/decision_policy/soft_label_pool.json`.  750 of the 1,500 items
carry a **human distribution** (250 of every type: `go_emotions`,
`civil_comments`, `measuring_hate_speech`), which is what makes the pool able to
answer the question: the 235 OpenDecisions items carry a human distribution on 13 of
them, the jev-bench test sample on 280 of 683.

### 20.2 The rule, written down before the run

**A constant ships only if it clears ALL THREE of the following, each measured
PAIRED - out of one run per population, so the two arms share the engine, the picks,
the logits and the state, and only the arithmetic differs.**

| | the test | the baseline it must beat |
|---|---|---|
| **(a)** | the out-of-fold calibration on the pool it was fitted on improves | the pool's own **T = 1** arm (the pool run at T = 1 IS that arm) |
| **(b)** | the jev-bench **test** TVD to the human distribution does not get worse by more than **0.005** | the **T = 1** arm, paired |
| **(c)** | the OpenDecisions 235's Capability does not fall below | the **T = 1** arm, paired |

**0.005** is a fifth of what §18's constants cost on the human labels (0.0239), and it
is a margin against a run-to-run spread of about 0.0008 in mean TVD (§19).  It is
fixed now, before the numbers, so that "not worse" cannot become "better" later.

**The arms.**

* The **soft** objective - the cross-entropy against the **human distribution**, not
  against a one-hot label - is the **primary** candidate.  It is the only objective
  that fits the thing the jev-bench yardstick measures, and the §18 constants were
  fitted against one-hot labels, which is exactly what §19 showed to be
  population-specific.  The **Brier** arm (the §18 objective) is the **control**: if
  the Brier arm wins on the 235 and loses on the human labels, that is the §19
  result reproduced on a new population, and it is not a surprise.
* The **single scalar** arm (one T over the pooled records, the same objective, the
  same folds, the same no-leak rule) is the **tie-breaker**: if one number does as
  well as three, the three are not worth their complexity.
* The **noul is judged separately** and is **expected to fail**.  §19 measured its
  error to be a **bias**: the model over-predicts "yes" on 95 of the 100 items, the
  mean signed difference is +0.357 on a mean absolute 0.384, and 93% of its TVD is
  that offset (mean P(yes) 0.446 against the humans' 0.089).  A temperature divides
  the logits; it cannot move a distribution's centre.  A noul that does not clearly
  improve is left at **T = 1** and handed to item 1.4's affine fit, and **its failure
  does not block the choice and the score constants**.

**"No constant ships" is an acceptable answer, and the likeliest one.**  It would
mean the endpoint's calibration is **population-specific**, that the knob stays at
1.0, and that the next step is an **affine** calibration (item 1.4) or a
per-population calibration, not a single scalar.

### 20.3 The plan, and the state it is run from

| | |
|---|---|
| the pool | 1,500 items, 500 a type, seed 20261021, `tools/decision_policy/soft_label_pool.json` |
| the fit | `tools/decision_policy/temperature.py` on the pool's run; the Brier, the soft and the single-scalar arms all come out of it |
| the 235 | the 2026-10-21 run with the §18 constants, re-scaled **offline** to T = 1 and to each candidate (`tools/decision_policy/retemperature.py`, the inverse of "apply T" is "apply 1/T") |
| the test sample | the 683-item jev-bench sample, re-run **live** with the candidate constants, and its T = 1 arm recovered from the same run (`tools/decision_policy/jevbench_paired.py`) |
| the server | the main checkout `C:\tools\strata`, the live launch config, a **fresh engine before each measurement run** (§12.5) |
| the tools | the worktree `C:\tools\strata-wt-p1b` |

### 20.4 The pool run, and the fit on it

**THE POOL RUN (a), 2026-10-22.**  A fresh engine (the server was restarted before
it, per §12.5), the live launch config with **no temperature key** (so T = 1 for
every type), the direct readout, one question per request, one attempt each:

```
python tools/decision_policy/jevbench.py --items <p1b_pool.jsonl> \
  --out <p1b_pool_T1.jsonl> --log <p1b_pool_T1.runlog.jsonl> --config <the live config>
```

**1,500 of 1,500 answered, 0 failures, every item on the fast path**, in **6,306 s**
(1 h 45 m).  The latency: median **3,511 ms**, p95 **10,269 ms**, mean 4,204 ms.  The
slow items are the wide ones - the 151-option `clinc150` and the 77-option
`banking77` questions - and the run's own pace (8-10 s an item at the start, 3 s
later) is the conversation cache filling up, the same shape §18.5 measured.

**THE FIT**, on that run (`tools/decision_policy/temperature.py`, the same grid, the
same seed 42, the same five folds, the same no-leak rule).  **The constants are all
ABOVE 1** - the pool's population wants a FLATTER distribution, and §18's wanted a
sharper one:

| arm | what it minimizes | choice | noul | score |
|---|---|---:|---:|---:|
| nll | the negative log-likelihood of the gold label | 2.31 | 2.67 | 3.36 |
| **brier** (the control) | the multiclass Brier score | **2.27** | **2.93** | **3.29** |
| ece | the raw top-level ECE | 2.42 | 1.47 | 3.02 |
| **soft** (the primary) | the cross-entropy against the HUMAN distribution | **3.65** | **3.75** | **3.73** |
| **single scalar** | one T over the pooled records | **2.47** | 2.47 | 2.47 |

**THE POOL'S OWN T = 1 ARM** (the run above, scored by the official scorer): Capability
**49.72** (Intelligence 41.98, Calibration 57.47); the per-type calibration
**choice 44.8, noul 62.0, score 65.6**; the per-type accuracy 55.4% / 73.2% / 47.0%.

**THE OUT-OF-FOLD FIT ON THE POOL** (each item scaled by the temperature of the fold
it was **not** in):

| arm | Capability | Intelligence | Calibration | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|
| the T = 1 arm | 49.72 | 41.98 | 57.47 | 44.8 | 62.0 | 65.6 |
| the brier arm | **58.99** | 39.47 | **78.50** | 86.3 | 63.1 | 86.0 |
| the soft arm | 52.66 | 39.05 | 66.27 | 51.9 | 61.7 | 85.2 |
| the single scalar | 59.65 | 40.42 | 78.87 | 88.7 | 63.8 | 84.1 |

(the three right-hand columns are the scorer's own per-type calibration components,
0-100, higher is better.)

**How sure is the fit?**  The out-of-bag bootstrap (1,000 resamples, the temperature
fitted on one and the ECE measured on another): the temperature lowers the ECE in
**100%** of the resamples for the choice and the score and in **60%** for the noul - a
wash, as §18 found.  The paired item bootstrap, the out-of-fold fit against the T = 1
arm: Capability **+8.84 [+7.14, +10.56]** (100% > 0), Calibration **+20.20 [+17.08,
+23.31]** (100% > 0), and Intelligence **-2.51 [-3.46, -1.52]** (0% > 0) - the
flattening moves the score questions' expected level, and the scorer reads it.

These intervals are the committed `tools/decision_policy/temperature_pool_report.md`
(the fit with the bootstrap on).  The run-directory `p1b_fit.json` and `p1b_fit.md`
were written with the bootstrap **off**: `paired_bootstrap_out_of_fold` is `{}` and
the out-of-bag table has no rows.  The fit is otherwise the same - the constants and
the out-of-fold ECEs agree - so the committed report is the copy with the numbers
this section quotes.

### 20.5 The test sample, with the soft constants live

**THE RUN (c), 2026-10-22.**  A fresh engine, the launch config with the **soft**
constants (choice 3.65, noul 3.75, score 3.73) - the startup line says so - the
direct readout, the 683-item jev-bench test sample, one question per request:

**683 of 683 answered, 0 failures, every item on the fast path**, in **2,674 s**
(45 m).

**THE ARMS, ALL OUT OF THAT ONE RUN** (`tools/decision_policy/temperature_arms.py`),
so they share the engine, the picks, the logits and the state, and only the
arithmetic differs.  The inversion is checked, not assumed: the recovered T = 1 arm
put back through the served temperature reproduces the arm the run reported to
**1.1e-16**, and the noul's recovered P(yes) matches the run log's own open reading
to **3.3e-16**.

| arm | mean TVD to the humans | Capability | Intelligence | Calibration |
|---|---:|---:|---:|---:|
| **T = 1 (recovered)** | **0.4439** | 55.1 | 47.2 | 63.0 |
| soft (3.65 / 3.75 / 3.73) | 0.4111 | 56.1 | 46.0 | 66.1 |
| brier (2.27 / 2.93 / 3.29) | 0.4045 | 63.1 | 46.5 | 79.7 |
| single (2.47) | 0.4034 | 62.5 | 47.1 | 78.0 |

(The recovered T = 1 arm's 0.4439 and 55.1 are §19's stored T = 1 run, reproduced
from a different engine: the recovery is sound, and the 0.0008 mean-TVD difference
§19 measured is between the two runs, not between the two methods.)

Per type and per config, the mean TVD to the humans:

| arm | choice (130) | noul (100) | score (50) | chaosnli (80) | civil_comments (100) | go_emotions (50) | measuring_hate_speech (50) |
|---|---:|---:|---:|---:|---:|---:|---:|
| T = 1 | 0.4702 | 0.3836 | 0.4963 | 0.3116 | 0.3836 | 0.7239 | 0.4963 |
| soft | 0.4139 | 0.3811 | 0.4639 | 0.2021 | 0.3811 | **0.7527** | 0.4639 |
| brier | 0.4025 | 0.3777 | 0.4632 | 0.2103 | 0.3777 | 0.7100 | 0.4632 |
| single | 0.4012 | 0.3762 | 0.4637 | 0.2054 | 0.3762 | 0.7143 | 0.4637 |

### 20.6 The 235, the same arms

**THE ARMS OUT OF ONE RUN** - the 2026-10-21 run with §18's constants
(`fix_serial_temp.jsonl`), re-scaled offline to T = 1 and to each candidate, so they
share the engine, the picks, the logits and the state:

| arm | Capability | Intelligence | Calibration | choice cal. | noul cal. | score cal. | score comp. |
|---|---:|---:|---:|---:|---:|---:|---:|
| **T = 1 (the paired control)** | **78.9** | 75.9 | 81.8 | 78.9 | 79.2 | 87.4 | 65.7 |
| §18's constants (0.62 / 0.67 / 0.77) | 80.1 | 77.0 | 83.1 | 80.8 | 81.8 | 86.7 | 68.9 |
| soft (3.65 / 3.75 / 3.73) | 59.6 | 65.5 | 53.8 | 43.3 | 54.5 | 63.6 | 34.4 |
| brier (2.27 / 2.93 / 3.29) | 66.1 | 66.7 | 65.4 | 64.7 | 67.5 | 64.1 | 38.0 |
| single (2.47) | 69.0 | 69.3 | 68.6 | 62.9 | 70.8 | 72.1 | 46.0 |

(The picks do not move: the map is monotone, so every arm answers the same 235 items
the same way.  What moves is the reported distribution, and with it the calibration
and the score questions' expected level.)

### 20.7 The rule, applied

| | the test | the candidates | the verdict |
|---|---|---|---|
| **(a)** | the pool's out-of-fold calibration improves against the pool's own T = 1 arm | brier 44.8/62.0/65.6 -> 86.3/63.1/86.0; single -> 88.7/63.8/84.1; soft -> 51.9/61.7/85.2 | **PASS** for the brier arm and the single scalar (every type); **PASS** for the soft arm on the choice and the score, and the noul is a wash (-0.3) |
| **(b)** | the jev-bench test TVD does not get worse by more than 0.005 | soft 0.4439 -> 0.4111 (-0.0328); brier -> 0.4045 (-0.0394); single -> 0.4034 (-0.0405) | **PASS** for all three, by 6 to 8 times the margin |
| **(c)** | the 235's Capability does not fall below its T = 1 arm (78.9) | soft 59.6 (-19.3); brier 66.1 (-12.8); single 69.0 (-9.9) | **FAIL** for all three |

**NO CONSTANT SHIPS.**  Every candidate passes (a) and (b) and fails (c), and (c) is
not close: the flattening that the human labels want costs the OpenDecisions
population **9.9 to 19.3 Capability points**.

### 20.8 What that means

**The endpoint's calibration is POPULATION-SPECIFIC, and the two populations want
opposite scalings.**  §18 fitted the OpenDecisions sample and got a temperature
**below** 1 - the model is under-confident on Jev-style questions.  This fit, on the
jev-bench content judgments, got a temperature **above** 1 - the model is more peaked
than the people are (§19 measured the same: a top-label mass of 0.7862 against the
humans' 0.6215).  One scalar cannot be right for both, and the rule says a constant
that helps one population and hurts the other is not a constant we ship.  **The
endpoint stays at T = 1.**

**The noul is confirmed to be a BIAS, not a scale, and this is the second
population to say so.**  The soft arm - the only objective fitted against the human
distribution - moves the noul's TVD by **0.0025** (0.3836 -> 0.3811) while moving the
choice's by 0.056 and the score's by 0.032.  A temperature cannot move a
distribution's centre, and the noul's centre is wrong: the model over-predicts "yes"
on 95 of the 100 items (§19).  The noul goes to item 1.4's **affine** fit, and the
rule's clause that the noul's failure does not block the choice and the score
constants is what is left to say - except that the choice and the score constants
fail (c) too, so nothing is left to ship.

**What the run does say that is worth keeping.**  The pool's own T = 1 arm is poorly
calibrated (the choice's calibration component 44.8) and the fit fixes that on the
pool's own population (86.3) with a large, well-bracketed effect (+8.84 Capability,
+20.20 Calibration, both 100% of the resamples).  A temperature is a real tool on the
population it is fitted on; it is the *transfer* that fails.  The next step is
therefore not a better constant but a **population-aware** calibration: an affine map
(item 1.4) for the noul's bias, and a decision about what the endpoint should report
when the question's population is unknown.

**THE LIMITS.**

* **The two populations are not the same kind of question.**  OpenDecisions is
  Jev-style typed decisions; the jev-bench pool is content judgments (toxicity,
  emotion, hate speech, intent classification, similarity).  The result is a
  statement about *those two* populations, not about every question the endpoint
  will ever be asked.
* **The 235's arms come from a run on 2026-10-21**, re-scaled offline.  The
  re-scaling is exact (the transform is invertible) and the T = 1 arm it produces is
  §18.5's "the same run with the temperature undone" (78.9 / 75.9 / 81.8) - so the
  arms are paired, but they are one engine's readings, not a fresh run.
* **The pool's fit is a fit on 1,500 items of which 750 carry a human
  distribution**; the soft arm's 250 items per type are what it optimizes, and the
  other 250 per type (the shapes without a human distribution) are in the objective's
  denominator for the other arms only.
* **The test sample's arms come from ONE run with the soft constants**; the other
  arms are re-scalings of it, so they share that run's engine wobble.
* **What was not run.**  No live run of the 235 with the candidate constants (the
  offline re-scaling is exact and the test sample's live run is the same arithmetic
  on another population); no run of the pool with the candidate constants (the fit
  is the offline arithmetic, and the pool's run at T = 1 is its T = 1 arm); no affine
  calibration (that is item 1.4); no second model, no engine change, no new
  dependency.  The letter readout was not tried: it loses on this population too
  (Capability 48.3 against the direct arm's 55.1), and the direct readout is what
  every number above is measured on.

### 20.9 The artifacts, and where they are

The pool run's files are the base arm for roadmap item 1.5's prompt experiment (the
pool at T = 1 with today's template), and they are **kept, not overwritten** (the
temporary run directory; §8 defines `<opencode temp>`):

```
<opencode temp>\decision_policy_training\run\p1b_pool_T1.jsonl
<opencode temp>\decision_policy_training\run\p1b_pool_T1.runlog.jsonl
```

The predictions file has one row per item, `{"id": ..., "probs": {...}}` - the shape
the probe's `--base` argument reads - 1,500 rows, every item on the fast path.

The rest of this run's files, in the same directory: `p1b_fit.json` and `p1b_fit.md`
(the fit with the bootstrap **off** - its bootstrap tables are empty; the intervals
quoted in §20.4 are in the committed `temperature_pool_report.md`, which was generated
with the bootstrap on), `p1b_test_arms.md` (the arms of the test run),
`p1b_jevbench_test_soft.jsonl` and its run log (the live run with the soft
constants), `p1b_jevbench_test_soft_T1.jsonl` (the recovered T = 1 arm), and the
235's arms `p1b_235_T1_from_current.jsonl`, `p1b_235_soft.jsonl`,
`p1b_235_brier.jsonl`, `p1b_235_single.jsonl`.  The fit's own artifacts are also
committed, as `tools/decision_policy/temperature_pool_calibration.json` and
`tools/decision_policy/temperature_pool_report.md` - the §18 pair
(`temperature_calibration.json`, `temperature_report.md`) is left as the record of
the constants that are not shipped.

---

## 21. The letter readout (2026-10-21)

Roadmap item 1.3, the letter-readout benchmark: most decision readouts in the
literature show the options as lettered choices and read the probability of the
letter.  Strata shows the options by their own names and reads the probability of
the option's own tokens, and when two options share their first token ("rule_1" and
"rule_2" both start with " rule") it walks the options' trie to tell them apart
(§10.6).  The question this section answers: **is the letter readout better?**

**The answer: on the official scorer it is, by 3.4 Capability points, and it is
faster and simpler.  But the win is not in the picks** - the letter readout picked
204 items right and the direct readout 206 - **it is in the score questions'
distance to the gold level and in the calibration.**  The details are below, and
§21.8 says what the result does and does not settle.

### 21.1 What was compared

Two arms, one session, on the same engine state, on the same 235 ids, one question
per request, and **no temperature in either arm**.

The temperature is out because it is not being shipped: §19 measured the fitted
constants against the human labels on jev-bench and they were worse on every type
and every config, so the `"systemone": {"temperature": ...}` key was taken out of
the launch config.  Measuring at T = 1 compares the letter readout against the
readout that actually ships, and keeps the readout the only variable.  The launch
config's `systemone` block now reads `{"enabled": true, "diagnostics": true}`.

The arms differ in exactly one key of the launch config:

| arm | the config's `systemone` block |
|---|---|
| the direct readout (today's) | `{"enabled": true, "diagnostics": true}` |
| the letter readout | `{"enabled": true, "diagnostics": true, "readout": "letters"}` |

The letter readout is `serve/systemone_letters.py`: the options are shown as
`A) option_name ("description")`, the model answers with the letter, and the letter
is mapped back to the option by its place.  The client never sees the letter - the
response is the shape the endpoint already has, so the official scorer reads both
arms unchanged.  The design, the tokenization and the letter set are in
`docs/research/letter_readout.md`.

### 21.2 The run

**ENGINE FRESHLY RESTARTED before each arm** (§12.5): the server tree was stopped
(the engine process exited with it), the server was started again, and the model was
loaded from scratch before the run.

| | the direct arm | the letter arm |
|---|---|---|
| the server started | 20:57:09 | 21:19:10 |
| the model loaded | ~37 s later | ~42 s later |
| the run | 1084 s, ended 21:17:33 | 892 s, ended 21:35:06 |
| the minimum free VRAM | 1113 MiB | 1101 MiB (the engine's threshold is 1024) |

Both arms used the same build (the letter readout on top of this branch), the same
model, the same ids and the same harness.  The commands:

```
python tools/decision_policy/opendecisions.py --items <opendecisions.jsonl> \
  --out <arm>.jsonl --ids-file <sample10.txt> --batch-k 1
python <score.py> <arm>.jsonl --data <sample10_items.jsonl>
```

**Both arms: 235 of 235 answered, 0 failures, no 502, every item on the fast path.**
The harness omits a failed request from the predictions file and the scorer counts
a missing answer as wrong, so the 235 is the whole sample.

### 21.3 The Capability

| run | Capability | Intelligence | Calibration | choice | noul | score |
|---|---:|---:|---:|---|---|---|
| the stored baseline, re-scored | 79.8 | 78.0 | 81.7 | 89.7% / 73.7 | 91.3% / 84.9 | 79.7% / 86.3 |
| **the direct arm, measured today** | 79.3 | 77.4 | 81.1 | 89.7% / 75.0 | 91.3% / 83.0 | 79.7% / 85.3 |
| **the letter arm, measured today** | **82.7** | **82.4** | **83.0** | 90.7% / 78.2 | 92.8% / 85.8 | 72.9% / 84.9 |

(per type: accuracy and the scorer's calibration component.)

**The letter readout is 3.4 Capability points above the direct readout measured on
the same engine, the same day, the same minute.**  It is also 2.9 above the stored
baseline, and the direct arm's own number today (79.3) is 0.5 below the stored one
(79.8) - the run-to-run spread of this sample is about 0.6 points (§10.8: the arms
measured on it read 78.9, 79.1, 79.2, 79.5, 79.6, 79.8, 80.4), so the gap is well
outside the spread.

### 21.4 Where the win comes from - and where it does not

**The letter readout did not pick more right answers.**  It picked **204** of the 235
right, the direct readout **206**.  221 picks are the same and 14 moved; on those 14
the direct readout was right 8 times and the letter readout 6.  The +5 Intelligence
comes from somewhere else, and it is worth being exact about where:

* **The score questions.**  The scorer's score competence is not "was the argmax
  right" - it is the **distance from the expected level to the gold level**
  (`score.py`'s NMAE against a mid-scale guess).  The letter readout's mean
  normalized distance is **0.0908**, the direct readout's **0.1013**, so the
  competence is 74.1 against 65.1 - while the letter readout picks the right level
  on **43 of 59** score items and the direct readout on **47 of 59**.  The letter
  readout's distribution over the levels sits closer to the gold level even where
  its argmax is the wrong level.
* **The yes/no questions.**  64 of 69 on the right side against 63 of 69, and the
  tier weights turn that into 86.7 against 80.5: the letter readout has one wrong
  hard-tier yes/no, the direct readout three.
* **The choice questions.**  A wash: 86.5 against 86.7.
* **The calibration.**  83.0 against 81.1, mostly the choice questions' ECE
  (78.2 against 75.0).

So the honest reading: **the letter readout's distribution over the options is
better shaped, and the scorer's metrics reward that** - but it is not the model
deciding more items correctly.  A consumer that only wants the argmax gets the
direct readout's 206, not the letter readout's 204.

**Two numbers go the other way.**  The mean TVD to the known true probabilities on
the 13 calibration items is **0.256** for the letter arm against **0.216** for the
direct arm: where the state fixes the answer distribution exactly, the letter
readout's distribution is further from it.  And the score questions' raw accuracy
is worse (72.9% against 79.7%).

**No position bias.**  The letter readout picked the first letter on **25.3%** of
the 166 choice and score items; an unbiased readout would pick it 24.5% of the time.

### 21.5 The latency

| | the direct arm | the letter arm | the stored baseline |
|---|---:|---:|---:|
| median per question | 3951 ms | **3606 ms** | 3121 ms |
| p95 per question | 7288 ms | **5427 ms** | 5682 ms |

**Against the arm measured in the same session, the letter readout is faster**: the
median is 345 ms lower, the p95 1861 ms lower, it was faster on **168 of the 235
items**, and the paired median difference is **-229 ms**.

**Against the stored baseline the medians do not agree**: 3121 ms (stored) against
3606 ms (today's letter arm) - the letter readout is 485 ms slower there, and 255 ms
faster on the p95.  The stored run is a different day and a different engine state,
and today's direct arm is 830 ms slower than the stored one on the median, so the
machine today is slower than when the baseline was measured.  The same-session
comparison is the one to read; the stored baseline says the two are a wash.

**The direct arm's run had a slow patch**: 16 consecutive items (118 to 130) took
7 to 42 seconds each, against 1.8 to 4 seconds elsewhere in the run.  It is
environmental - the picks in that window are ordinary - and it inflates the direct
arm's median and p95.  Without those 16 items the direct arm's median is 3826 ms
and the letter arm's 3595 ms on the same 219 items: the letter readout is still
faster, by 231 ms.

### 21.6 The simplicity, and the engine requests

The letter readout needs **one engine request per question**.  That is structural,
not a measurement: the letters are one token each and no two of them are the same
token, so the options cannot collide and the trie walk is never needed.  Computed
over the 235 sample items with the server's own tokenizer
(`tools/decision_policy/letter_costs.py`): **235 items, 0 with a first-token
collision, 0 extra engine requests.**  The direct path over the same items
(`tools/decision_policy/trie_costs.py`): **72 items collide, 113 extra engine
requests** (one to seven per colliding item).

The latency shows the walk's price: on the 72 colliding items the direct arm's
median was 4316 ms and the letter arm's 3774 ms (the paired difference -551 ms);
on the 163 items that need no walk, 3762 ms against 3572 ms (-165 ms).

The letter readout is also simpler in the parts that are not measured: the prompt
does not depend on how the options happen to tokenize (the direct path has to walk
a trie over them), and the answer is one token instead of an option name of ~3.1
tokens on average.

### 21.7 The exit criterion

The roadmap's rule: **ship the letter readout only if it wins on at least two of
{Capability, latency, simplicity} without losing the other.**

| | verdict |
|---|---|
| Capability | **wins**: 82.7 against 79.3, the same session (- and 79.8 stored) |
| latency | **wins** against the same-session arm (median 3606 against 3951 ms, p95 5427 against 7288 ms); a wash against the stored baseline |
| simplicity | **wins**: one engine request per question, 0 extra over the 235 items against the direct path's 113 |

**Two of three, and the third does not lose: the exit criterion is met.**  The
letter readout is shipped as the opt-in readout it is - the config's `"systemone":
{"readout": "letters"}` - and the default path stays as it is.

### 21.8 The limits

* **The win is in the distribution, not in the picks** (§21.4).  The letter readout
  picked 204 items right against the direct readout's 206.  A reader who wants the
  Capability number should read it as "the letter readout's distribution over the
  options is better shaped", not as "the model decides more items correctly".
* **One run per arm.**  The direct arm's own spread on this sample is 0.5 points
  (79.8 stored, 79.3 today) and the letter arm's advantage is 3.4, but a second run
  of each arm would say how much of the 3.4 is the run.  The same day's two runs
  are paired item by item (14 moved picks), which cancels the item mix but not the
  engine's run-to-run wobble on the items that moved.
* **The 13 calibration items go the other way**: the mean TVD to the known true
  probabilities is 0.256 for the letter arm against 0.216 for the direct arm.  The
  endpoint is also used for uncertainty, not only for a pick.
* **The score questions' raw accuracy is worse** (72.9% against 79.7%): the letter
  readout's argmax is the wrong level more often, even though its expected level
  sits closer to the gold.
* **This is the OpenDecisions sample, which says not to fit on it and is
  validation-only.**  Nothing was fitted here - the two readouts were measured as
  they are - but the sample is still a validation set, and the jev-bench check
  (§19) is the one that decides a change of the shipped readout.
* **The machine.**  Today's runs are slower than the stored baseline's (the direct
  arm's median 3951 ms against the stored 3121 ms), so the latency numbers are a
  property of this session, not of the endpoint.  The comparison that matters is
  the same-session one.
* **What was not run.**  No second run of either arm; no jev-bench run with the
  letter readout (that is the independent check, and it is the obvious next step);
  no run with a temperature (the temperature is not being shipped); no GPU build
  (no engine file changed); no test with more than 26 options against a live model
  (the sample's widest question has 7 options, so that path was not exercised
  live); no second model.

---

## 22. The letter readout on jev-bench, against the humans (2026-10-21)

§21 measured the letter readout on the 235-item OpenDecisions sample: it won the
official Capability by 3.4 points.  This section is the **independent check**, on the
population that carries the **human label distributions** - jev-bench's 3% sample,
683 items, 280 of them with a human distribution over the options.  It is the check
that decides whether the letter readout could ever be the default, because the
question there is not "is the model right" but "is the model's uncertainty as wide as
the humans'".

**The answer: the letter readout loses, clearly.**  Capability **48.3** against the
direct readout's **55.1** (or 55.2 for the stored live run), and the mean distance to
the human distribution goes from **0.4439** to **0.4688**.  It loses on every type,
and its calibration drops by 11.5 points.

### 22.1 The run

**ENGINE FRESHLY RESTARTED** (§12.5): the server tree was stopped, the engine
process exited with it, and the server was started again from the worktree with the
letter-arm config (`"systemone": {"readout": "letters"}`, no temperature), so the
model was loaded from scratch before the run.  The server started at 21:56:43, the
model was loaded ~42 s later, the run took 2396 s and ended at 22:37:30.

```
python tools/decision_policy/jevbench.py --items <jevbench_sample3_items.jsonl> \
  --out <jevbench_sample3_letter_predictions.jsonl> \
  --log <jevbench_sample3_letter_predictions.runlog.jsonl> --config <the letter config>
python <score.py> <jevbench_sample3_letter_predictions.jsonl> --data <jevbench_sample3_items.jsonl>
```

**683 of 683 answered, 0 failures, no 502.**  **539 items took the letter fast path;
144 fell back**, every one of them with the reason "more options than letters" (§22.4).
The minimum free VRAM was **1099 MiB** (1207 at load), above the engine's 1024 MiB
warning threshold, no sample below it.

### 22.2 The official scorer, against item 1.2's direct arm

| arm | Capability | Intelligence | Calibration | choice | noul | score |
|---|---:|---:|---:|---|---|---|
| the direct arm (the stored T = 1 run) | **55.2** | **46.7** | **63.7** | 70.7% / 68.3 | 74.3% / 60.4 | 47.9% / 62.3 |
| the direct arm (item 1.2's paired arm) | 55.1 | 46.6 | 63.6 | 71.1% / 69.3 | 74.3% / 57.8 | 48.5% / 63.6 |
| **the letter arm** | **48.3** | **44.4** | **52.2** | 69.7% / 44.9 | 72.5% / 56.5 | 47.3% / 55.2 |

(per type: accuracy and the scorer's calibration component; the competence numbers are
in §22.5.)  The letter arm is **6.9 Capability points below** the direct arm, and the
gap is almost all calibration (-11.5), not accuracy: the choice questions' accuracy is
70.7% against 69.7%, the yes/no 74.3% against 72.5%, the score 47.9% against 47.3%.

### 22.3 The distance to the human distribution

The mean TVD over the 280 items that carry a human distribution, from the run logs'
own stored per-item values:

| | the direct arm | the letter arm |
|---|---:|---:|
| **all 280 items** | **0.4432** | **0.4688** |
| the 130 choice items | 0.4697 | 0.5268 |
| the 100 yes/no items | 0.3842 | **0.3559** |
| the 50 score items | 0.4922 | 0.5437 |

The letter readout is further from the humans on the choice questions and on the score
questions, and **closer on the yes/no questions** - the one type where the humans are
nearly certain (their entropy is 0.216 bits) and a sharper answer is the right one.

### 22.4 The 144 items the letter readout could not serve

Twenty-six letters, and jev-bench's multi-intent classification configs have more
options than that: go_emotions (28 options, 50 items), ledgar (100, 23), massive (60,
23), banking77 (77, 24), clinc150 (151, 24).  Those 144 questions have no letter for
every option, so the letter readout declines them - and the endpoint then answers them
from the **decode path**, which reports a degenerate 1.0/0.0 distribution and no
probability at all.

| the 144 items | Capability | Intelligence | Calibration | mean TVD |
|---|---:|---:|---:|---:|
| the direct arm | **58.4** | **60.3** | **56.5** | **0.7219** |
| the letter arm | 38.1 | 58.2 | 18.1 | 0.8327 |

The calibration collapses (56.5 → 18.1) because a confident wrong answer is what a
degenerate distribution is.  **And the letter readout loses even on the 539 items it
did serve**: Capability 57.8 → 53.8, Calibration 67.0 → 60.8, the mean TVD 0.3826 →
0.3897.  On those items the choice questions' accuracy is the same (80.0% against
80.0%) and the competence is the same (71.1 against 71.1); the loss is the
calibration again (78.1 → 70.7).

So the fallback is not the whole story: **the readout itself is worse on the items it
can serve**, and the fallback makes it worse still.

### 22.5 The mechanism: is the letter readout hedging?

The hypothesis from §21's breakdown was that the letter readout's distribution over
the ordinal levels is **flatter**, which pulls the expected level toward the middle
and flatters the scorer's nMAE term.  **Measured, it is not flatter - but the
apparent sharpness over the whole sample is mostly the fallback.**  The entropy of
the reported distribution, the mass on its mode, and (for the score questions) the
distance of the expected level from the gold:

| | the direct arm | the letter arm |
|---|---:|---:|
| the choice questions' entropy, all 294 items | 0.531 bits | 0.188 bits |
| ... the 150 items the letter path served | 0.401 bits | 0.368 bits |
| ... the 80 served items with a human distribution | 0.541 bits | 0.504 bits |
| the choice questions' mass on the mode, all 294 | 0.866 | 0.950 |
| ... the 150 served | 0.902 | 0.903 |
| ... the 80 served with a human distribution | 0.860 | 0.862 |
| the yes/no questions' entropy | 0.312 bits | 0.251 bits |
| the score questions' entropy | 0.816 bits | 0.705 bits |
| the score questions' mass on the mode | 0.756 | 0.791 |
| the score questions' argmax accuracy | 47.9% | 47.3% |
| the score questions' nMAE (the expected level's distance to the gold) | 0.2384 | 0.2450 |
| the score questions' SIGNED distance (the expected level - the gold) | +0.1348 | +0.2738 |

**The 0.531 -> 0.188 drop over all 294 choice items is not the letter readout.**  144
of those items have more options than there are letters, so the letter arm answers
them from the decode path with a degenerate 1.0/0.0 distribution; that is the drop.
Over the **150 it served** the entropy is **0.401 -> 0.368 bits** and the mass on the
mode **0.902 -> 0.903**; over the **80 served items that carry a human distribution**
it is **0.541 -> 0.504** bits and **0.860 -> 0.862**.  The letter readout is only
marginally sharper where it works, not twice as sharp.

On the items with a human distribution the shape is not the whole story: the
**humans are uncertain** (the choice questions' human distributions carry 1.144 bits
of entropy), the direct readout answers with 0.541 bits on the 80 it served, and the
letter readout with **0.504** - not the 0.310 the all-130 figure showed, which
included the 50 `go_emotions` items the letter arm cannot serve.  That is what the
calibration component measures: over the 130 choice items with a human distribution
it is 44.9 against 68.3, and even over the 539 items the letter arm served it is
70.7 against 78.1.

And on the score questions the OpenDecisions effect does not transfer: on that sample
the letter readout's sharper distribution happened to centre better (the nMAE 0.0908
against 0.1013, with the mode right less often: 72.9% against 79.7%); on jev-bench the
same sharpening moves the expected level **further** from the gold (the signed
distance +0.135 → +0.274, the nMAE 0.2384 → 0.2450), so the score type loses on both
terms.

### 22.6 The cost, and what it buys

| | the direct arm | the letter arm |
|---|---:|---:|
| engine requests for the 683 items | 683 + **2171** (the trie walk: 163 colliding items, 1 to 30 extra each) | **683** (one per item; 539 with a real distribution, 144 with a degenerate one) |
| latency median / p95 | 3137 / 9419 ms | 3293 / 6619 ms |
| the paired median on the 539 served items | | **+209 ms** (the letter arm is slower) |
| the paired median on the 144 fallback items | | **-1058 ms** (the letter arm is faster) |

The letter readout makes **fewer engine requests** (683 against 2854: the direct path's
trie walk costs 2171 extra rows on the 163 items whose options share their first
token, up to 30 for 24 clinc150 items - a static count from the shipped code with the
server's own tokenizer, `tools/decision_policy/trie_costs.py`) and it has a better
p95, but it is slower on the median of the items it serves.  The 144 fallbacks cost
no second request: the model's own text came with the same turn.

So the trade is a real one: **the letter readout buys speed and simplicity with the
quality of the distribution**, and on the population with the humans' labels it buys
too much.

### 22.7 What this settles

* **The OpenDecisions win does not generalize.**  §21: +3.4 Capability.  Here:
  **-6.9**.  Two populations, opposite signs, and the one with the human labels is
  the one that decides the default question.
* **The hedging hypothesis is refuted.**  The letter readout's distributions are
  sharper, not flatter, in every type on both populations; the OpenDecisions score
  gain was a sharpening that happened to centre better on that sample, and it does
  not transfer.
* **The letter readout cannot serve a question with more than 26 options**, and on
  jev-bench that is 21% of the items; the fallback (the decode path) reports a
  degenerate distribution and costs 20 Capability points on those items.  That part
  is a fix, not a verdict: those questions should be read the direct way (the
  options' own tokens, with the trie walk), not answered blind.
* **The limits.**  One run of the letter arm, against a direct arm measured in
  another session (item 1.2's), so the run-to-run wobble is not paired out of the
  latency comparison; the jev-bench sample is validation-only and carries a per-source
  licence, and its score questions are six-level similarity ratings, not the ordinal
  routing questions of OpenDecisions, so the two populations' score types are not the
  same thing.  The 280 items with a human distribution are the ones the distance
  numbers rest on.
* **What was not run.**  No second run of the letter arm; no direct arm re-run in
  this session (the stored one is the reference); no run with a letter set larger than
  26 letters; no run with the >26-options fallback changed to read those questions
  the direct way; no GPU build (no engine file changed); no second model.

### 22.8 The recommendation

**Do not ship the letter readout as the default.**  The roadmap's exit criterion for
item 1.3 is met on the OpenDecisions sample (§21: it wins Capability and simplicity
without losing the latency), but this section is the independent check, and there the
letter readout **loses Capability by 6.9 points**, loses the distance to the humans
(0.4688 against 0.4432) and loses 11.5 points of calibration.  "Ship only if it wins
on at least two of {Capability, latency, simplicity} without losing the other" does
not survive a second population: the win was population-specific.

What the letter readout is worth keeping for: it is **faster and simpler** where it
works (one engine request per question, no trie walk, 683 requests against 2854 on
jev-bench), and on the yes/no questions it is **closer to the humans** (the TVD 0.3559
against 0.3842).  So the flag stays, the default stays the direct readout, and the
next steps are: read the questions with more than 26 options the direct way instead
of falling back to the decode path, and then measure the letter readout again on both
populations - the OpenDecisions sample (where it won) and jev-bench (where it lost) -
with that fix in, on one engine state, so the two can be compared paired.

## 23. The prompt probe and the noul's arms (2026-10-22)

Roadmap item 1.5: the server's per-type template, varied one axis at a time, and
measured against the licence-clean pool.  The design, the variant list, the
measurement design and the pre-registered rule are in
[prompt_variants.md](prompt_variants.md).  This section is the record: what was
run, what came out, and what it means.

**Every number here comes out of a run on this machine on 2026-10-22**, and the
files it came from are named.  The base arm is item 1.1b's pool run: the pool at
T = 1, the direct readout, today's template, 1,500 items, in
`...\decision_policy_training\run\p1b_pool_T1.jsonl` (and its run log beside it).
The design set is the pool's **500 noul items** in two halves: **half A**, the 250
`civil_comments` items with a human distribution (the humans' mean P(yes) 0.1096,
a 9.2% gold "yes" rate), and **half B**, the 250 `paws` and `strategyqa` items
with a hard label at a 45.2% gold "yes" rate.  Half A is where the bias is
measured; half B is where a base-rate shift shows up as a loss.

### 23.1 The probe (P1): the noul's template, mirrored

The design's §6 made the probe the first thing to run, because it can kill the
order axis before three hours of arms are spent.  P1 is the noul's template with
the two sides' order reversed in the criteria line **and** in the answer line;
the answer tokens do not change, so the two arms are two presentations of the
same question.

The arm is a launch-config value, so it is a server start: a copy of the live
config with `"systemone": {"prompt": {"noul": "swap"}}`, and the server started
from the worktree with it.  The server printed
`[strata] decisions: the questions are rendered in another prompt variant (noul swap)`
and `/v1/status` carried `"prompt": {"noul": "swap", "choice": "base", "score": "base"}`.

```
python tools/decision_policy/prompt_probe.py \
  --items ...\decision_policy_training\items\p1b_pool.jsonl \
  --out ...\decision_policy_training\run\p1b_pool_swap.jsonl \
  --base ...\decision_policy_training\run\p1b_pool_T1.jsonl \
  --report ...\decision_policy_training\run\p1b_pool_swap_report.md \
  --half A --config ...\strata-unsloth-ud-iq4_xs-swap.json
```

250 items, **250 answered, 0 failures**, 1,136 s; the fast path on **250 of 250**;
the median latency 2,394 ms; the prompt tokens median 257 (the base arm's, on the
same items: the median latency 3,248 ms, the prompt tokens median 257 - the swap
costs no tokens).

### 23.2 The probe's three readouts, and the base arm beside them

| | the base arm | the probe (swap) |
|---|---:|---:|
| the mean P(yes) | 0.4142 | 0.3930 |
| the bias (P(yes) - the humans') | +0.3047 | +0.2834 |
| the mean TVD to the human distribution | 0.3313 | 0.3125 |
| the AUC against the humans' hard label | 0.8435 | 0.8426 |
| the picks that moved | - | **15 of 250 (6.0%)** |
| the **swap gap** `mean(P_yes(base) - P_yes(swap))` | - | **+0.0212** |

The base arm's numbers are the pool run's, on the same 250 items.

**The flips, item by item.**  The 15 items whose pick moved were all items where
the humans say "no"; on them the base arm's accuracy was 0.133 and the swap
arm's 0.867 - **13 got better and 2 got worse**, and the accuracy against the
humans' majority went from 0.620 to 0.664 (+11 items of 250).  The flipped items
were not coin flips: their median distance from the 0.5 threshold was 0.100, and
only 1 of the 250 items sat within 0.02 of the threshold.

**The effect's size against the engine's own noise.**  The swap's per-item change
in P(yes) is a mean |change| of 0.0433 (SD 0.0743); the engine's own run-to-run
wobble, measured on the jev-bench noul slice, is a mean |change| of 0.0193 (SD
0.0337).  So the order is **2.2 times the engine's own noise** - a real effect,
not a re-run of the same template.

**And it does not move the ranking at all.**  The AUC went 0.8435 -> 0.8426: a
change of **-0.0010**, where the AUC's own noise on 250 items is about 0.047.  A
lever that moves the level and not the ranking is, in the design's own words, a
**base-rate** effect.

### 23.3 The base rate's own check, and the reading

The design's §6 priced the probe at one arm on half A.  The measurement came back
between the design's two branches, and half B is where the design's own rule (c)
lives, so **the probe's arm was run on half B as well** (250 items, 249 answered
and 1 already done, 0 failures, 679 s).  The arm list changed; the rule did not.

| half B (the 45.2% "yes" population) | the base arm | the probe (swap) |
|---|---:|---:|
| the mean P(yes) | 0.4872 | 0.4961 |
| the accuracy against the hard label | 84.4% | **82.0%** |
| the ECE of P(yes) | 0.0743 | **0.1045** |
| the picks that moved | - | 10 of 250 |

Of the 10 flips on half B, **2 got better and 8 got worse**.

**The pre-registered rules, applied to the probe's arm:**

| rule | the numbers | verdict |
|---|---|---|
| (a) calibration, on half A | the TVD 0.3313 -> 0.3125 (+0.0188), the bias toward zero +0.0212; the margin 0.007 | **PASS** |
| (b) ranking, on half A | the AUC 0.8435 -> 0.8426 (-0.0010); the margin -0.05 | **PASS** |
| (c) the base rate, on half B | the accuracy 84.4% -> 82.0% (**6 items** worse), the ECE 0.0743 -> 0.1045 (**+0.0301**); the margins 3 items and 0.02 | **FAIL** |

**The reading: the position is a real component, and the order arms stay.**  The
design's §6 set one conjunction for killing the order axis: the swap gap "below
**0.05** in absolute value **and** the flip rate is near the engine's own 1%
wobble".  The gap is 0.0212, below 0.05, but the flip rate is **6.0%**, six times
the floor, so **the conjunction did not fire**.  The design's §3.4 names the
branch the measurement is in:

> the flip rate is well above 1% and `mean(d)` is clearly positive -> the
> **position** of the two sides is a real, measurable component of the bias.  N1
> and N2 are worth their runs, and `mean(d)` prices them.

The gap is +0.0212 with a per-item SD of 0.0743, so its SE on 250 items is
0.0047: the gap is **4.5 SE from zero**.  The level moved, the ranking did not
(the AUC -0.0010, against an AUC noise of about 0.047 at this n), and the effect
is 2.2 times the engine's own per-item noise.  That is a real order effect.  The
probe's half B **fails rule (c)**, and that is recorded; but (c) is a **shipping**
guard, the design says in words that the probe "is **not** a shipping candidate",
and the probe is N1 and N2 **together**.  Its half-B failure therefore cannot be
charged to either arm: N1 rewrites the criteria line, half B has no criteria line,
and N1 is inert there by construction.  The probe's half-B loss belongs to N2 and
to the mirror as a whole.  Killing N1 and N2 on the combined arm's (c) would be a
rule tightened after the numbers were seen, which §6 forbids.  **N1 and N2 are
run.**

**A correction, made before N1 and N2 were run.**  An earlier version of this
section, committed as `909d77d4`, read the same numbers as "the order axis is
DEAD", because the probe's arm fails (c) on half B.  That reading applied a
shipping guard to a diagnostic arm, and it treated N1 + N2 together as if it were
each of them.  The pre-registered trigger did not fire, §3.4's second branch
applies, and this section is corrected on 2026-10-22 **before the two arms were
run**.  The probe's own numbers in 23.1-23.3 stand; only their reading changes.

### 23.4 The arm list (written down before the arms ran)

**Run: all five.**  N1 `criteria-order`, N2 `answer-order`, N3 `rule-form`, N4
`criteria-first` and N5 `decision-rule`.  The probe kept the order axis (23.3),
so N1 and N2 are run; N3, N4 and N5 were always in the list.  Each is one axis,
500 noul items (half A and half B) on a freshly restarted engine.  The probe
changed **no rule**; it changed the reading of the arm list, and that reading is
written above before any of the five was scored.  (**N2 was later cut off by the
engine hand-back; it is incomplete and not compared - 23.5.**)

**The rule is unchanged.**  The pre-registered rules (a) to (f) of the design's
§6 apply to each arm exactly as they were written:

- (a) half A's mean TVD falls by more than **0.007** and its mean signed
  difference moves toward zero by more than **0.007**;
- (b) half A's AUC does not fall by more than **0.05** (a non-inferiority check:
  the AUC's MDE at n = 250 is about 0.13, so an improvement is not measurable and
  is not asked for);
- (c) half B's accuracy does not fall by more than **3 items** of 250 and half
  B's ECE does not rise by more than **0.02**;
- (d) the 235's noul accuracy does not fall by more than **2 items** of 69 and
  the 235's Capability does not fall by more than **0.6 points**;
- (e) the jev-bench test sample's noul: the mean TVD to the human distribution
  does not rise by more than **0.005** and the ECE of P(yes) does not rise by
  more than **0.02**;
- (f) the fast-path rate does not fall and the median latency does not rise by
  more than **10%**.

The base arm's numbers for the yardsticks are the stored ones: the 235's
`p1b_235_T1_from_current.jsonl` (Capability **78.9**, Intelligence 75.9,
Calibration 81.8, the noul's accuracy 61 of 69) and the jev-bench test sample's
`p1b_jevbench_test_soft_T1.jsonl` (the noul's mean TVD **0.3836** on the 100
`civil_comments` items, the ECE 0.3941, the AUC 0.6413).

**One change to a variant's text, made before any arm was scored** (2026-10-22):
N5's added rule line said *"answer yes only when the state satisfies the \"yes\"
criterion"*, and a noul without criteria has no such criterion - the words had no
referent on half of the design set.  The two options were to make N5 inert
without criteria (so half B could not catch N5's base-rate shift and rule (c)
would lean on the 235 for it) or to reword the line.  **The line was reworded**
to *"answer yes only when the state supports the yes answer; otherwise answer no"*:
the rule now names the ANSWER, not the criterion, so it is well formed on both
halves, and the axis is unchanged.  N5 therefore stays live on half B and rule
(c) applies to it unchanged.

### 23.5 The arms

**Four of the five arms are complete; `answer-order` is not.** Each completed arm
ran 500 noul items (half A and half B) on a freshly restarted engine, answered
**500 of 500, 0 failures, 500 of 500 on the fast path**, and was compared item by
item against the same base arm (`p1b_pool_T1.jsonl`). The command is the probe's,
with `--half both`:

```
python tools/decision_policy/prompt_probe.py \
  --items ...\decision_policy_training\items\p1b_pool.jsonl \
  --out ...\run\p1b_pool_<variant>.jsonl --arm <variant> --half both \
  --base ...\run\p1b_pool_T1.jsonl \
  --config ...\strata-unsloth-ud-iq4_xs-<variant>.json
```

and the numbers are `prompt_arm.py`'s on the same two files.

**The design set.** The base arm's half A: mean P(yes) 0.4142, the bias +0.3047,
the TVD 0.3313, the AUC 0.8435, the accuracy against the humans' majority 155 of
250; half B: accuracy 84.4% (211 of 250), ECE 0.0743.

| variant | half A P(yes) | half A bias | half A TVD | half A AUC | half A acc | half B acc | half B ECE | (a) | (b) | (c) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| base | 0.4142 | +0.3047 | 0.3313 | 0.8435 | 155/250 | 84.4% | 0.0743 | - | - | - |
| N1 `criteria-order` | 0.3748 | +0.2652 | 0.2967 | 0.8412 | 168/250 | 83.6% | 0.0800 | PASS | PASS | PASS |
| N3 `rule-form` | 0.3925 | +0.2829 | 0.3140 | 0.8328 | 164/250 | 82.4% | 0.0943 | PASS | PASS | **FAIL** |
| N4 `criteria-first` | 0.3363 | +0.2268 | 0.2626 | 0.8422 | 177/250 | 83.6% | 0.0847 | PASS | PASS | PASS |
| N5 `decision-rule` | 0.3870 | +0.2775 | 0.3086 | 0.8395 | 161/250 | 82.8% | 0.0922 | PASS | PASS | **FAIL** |

**The yardsticks.** The jev-bench noul's base: mean TVD 0.3836, ECE 0.3941. The
235's base is two runs, because rule (d)'s validity clause asked for a repeat and
the repeat did not settle it (last part of this section).

| variant | 235 noul | 235 Capability | jev TVD | jev ECE | (d) | (e) |
|---|---:|---:|---:|---:|---|---|
| base run 1 (stored) | 61/69 (88.4%) | 78.9 | 0.3836 | 0.3941 | - | - |
| base run 2 (repeat) | 63/69 (91.3%) | 80.7 | - | - | - | - |
| N1 `criteria-order` | 61/69 (88.4%) | 78.0 | 0.3509 | 0.3518 | **FAIL** | PASS |
| N3 `rule-form` | 60/69 (87.0%) | 78.6 | 0.3669 | 0.3783 | PASS/FAIL | PASS |
| N4 `criteria-first` | 63/69 (91.3%) | 80.7 | 0.3212 | 0.3192 | **PASS** | PASS |
| N5 `decision-rule` | 62/69 (89.9%) | 78.5 | 0.3510 | 0.3451 | PASS/FAIL | PASS |

**Rule (f), the cost.** Every completed arm's fast-path rate is **100%** (500 of
500, and 100 of 100 on the jev noul), and every arm's median latency is below the
base arm's, so (f) passes for all four. The latencies are **not comparable across
arms**: the engine was shared with a local chat client through most of the
session (the client's prompts are in the engine log), so the queue time is not the
arm's. The fast-path rates and the prompt-token counts are the comparable part:
100% everywhere, and the token medians base 251, N1 251, N3 254, N4 251, N5 267.

**The verdicts.**

- **N1 `criteria-order` does not ship: guard (d) fails.** (a) and (b) pass by a
  wide mark (TVD +0.0346, the bias toward zero +0.0394, the AUC -0.0023), (c)
  passes (2 items, ECE +0.0057; N1 is inert on half B, so that is noise), and half
  A's accuracy rises 155 -> 168 of 250. On the 235 the noul's accuracy is 61 of 69
  - level with base run 1 and 2 items below the repeat, exactly at the item margin
  - but Capability is **78.0**, **-0.9** against run 1 and **-2.7** against the
  repeat, past the 0.6 margin either way. **FAIL whichever base is used.** The
  fall is not all in the type N1 touches - the choice's calibration is 75.4 (run
  1: 78.9) and N1 changes no choice prompt - but the noul is 2 items lower too.
  **Recorded as a population-specific effect**: the order lever helps the
  9.2%-yes population and does not clear the 55.1%-yes one.
- **N2 `answer-order` is INCOMPLETE and is not part of the comparison.** It was
  stopped by the orchestrator's engine hand-back. It answered **13 of 500 items**
  in its first attempt (the shared engine; kept as
  `p1b_pool_answer-order.jsonl.partial-shared`), and the restarted attempt wrote
  **9 rows - the same first 9 ids, re-measured** - before the server was stopped
  (kept as `p1b_pool_answer-order.jsonl`). **At most 13 distinct items of 500 were
  answered**, and no number from the arm is used. There is no design-set, 235 or
  jev-bench result for N2. **N2's question - whether the answer line's order
  carries the effect N1 did not - is therefore unanswered.**
- **N3 `rule-form` does not ship: guard (c) fails.** (a) passes (TVD +0.0173, bias
  +0.0217) and (b) passes (-0.0107), but half B loses **5 items** (84.4% ->
  82.4%; the margin is 3) and its ECE rises **+0.01996** (inside 0.02 by 0.00004).
  On the 235 it loses 1 noul item and 0.3 Capability against run 1, 3 items and
  2.1 against the repeat - (d) turns on the base - but (c) decides it. The
  jev-bench noul improves (TVD -0.0166, ECE -0.0158). It is the design's picture
  of a base-rate shift: the level falls on half A and the accuracy falls on half
  B, and the yardsticks do not see it.
- **N4 `criteria-first` clears the pre-registered rule; it is not applied.** It is
  the only arm that passes (a) to (f). The branch's default is `base` and the live
  config carries no `prompt` key; enabling it is one config key
  (`"systemone": {"prompt": {"noul": "criteria-first"}}`). The
  half A bias falls 0.3047 -> 0.2268 (toward zero by 0.0779, eleven times the
  0.007 margin), the TVD 0.3313 -> 0.2626, the AUC is flat (-0.0013) and the
  accuracy against the humans' majority rises 155 -> 177 of 250; the jev-bench
  noul's TVD falls 0.0623 and its ECE 0.0749. On the 235 it **does not fall**: its
  noul is 63 of 69, **equal to the repeat** and 2 above run 1, and its Capability
  is **80.7, equal to the repeat** and 1.8 above run 1. Rule (d) is a
  non-inferiority guard, so that is a **PASS**; the *size* of the 235 "gain"
  against run 1 is not established, because 1.8 is exactly the base's own spread
  (below). It is a **weighing** effect and not a base-rate shift. N4 is **inert on
  half B by construction** (half B has no criteria line to move), so half B's
  2-item loss and +0.0104 ECE are noise and (c) is a noise check for this arm; the
  base-rate evidence is the **235**, the 55.1%-yes population, where the variant
  **does not lose**. The rule's warning - that a variant which lowers the yes rate
  to match half A will cost the 235 - did not happen here: this one lowers half
  A's mean P(yes) from 0.4142 to 0.3363 and the 235 stays level. The bias is cut
  by about a quarter, not removed.
- **N5 `decision-rule` does not ship: guard (c) fails.** (a) and (b) pass (TVD
  +0.0227, bias +0.0272, AUC -0.0040); half B loses **4 items** and its ECE rises
  +0.0179 (the ECE is inside 0.02, the accuracy is not). On the 235 it gains 1
  noul item and loses 0.4 Capability against run 1; against the repeat it loses 1
  item and 2.2 Capability - (d) turns on the base, but (c) decides it. The
  jev-bench noul improves (TVD -0.0326, ECE -0.0490). The added rule line pushes
  the base rate down on **both** halves - half A's mean P(yes) 0.4142 -> 0.3870
  and half B's 0.4872 -> 0.4536 - and half B is where that shows.

**The N5 line, and why it stayed live on half B.** The decision (in 23.4) was to
**reword**, not to make N5 inert without criteria: the line is *"answer yes only
when the state supports the yes answer; otherwise answer no"*, it is well formed
on both halves, and (c) applies to N5 unchanged. It names the answer, not the
criterion, so the axis is still "the template states when to answer yes". No
neighbour's text was changed and no neighbour had to be re-interpreted; the
comparison stays one-axis. The change was made before any arm was scored.

**The base 235 run, and a finding against the rule's 0.6 margin.** Rule (d) says
the base arm's own 235 run must land within **0.6** of the stored 79.8 or it is
invalid and repeated. The stored base run, `p1b_235_T1_from_current.jsonl`, scores
**78.9** - 0.9 outside - so it was repeated on a freshly restarted engine
(`p1b_235_base-rerun.jsonl`). **The repeat scores 80.7** - 0.9 outside on the
other side. Two fresh-engine base runs of the same 235, same code, same template,
span **1.8 Capability points** (78.9 and 80.7) and **2 noul items** (61 and 63 of
69), with two noul picks moving. The pre-registered window is 0.6, three times
narrower than the base's own spread as realized here, and **neither base run is
inside it**. The clause cannot be satisfied on this tree; the honest reading is
that **the 235's overall Capability is not a 0.6-point instrument at this n**, not
that either run is broken. Every (d) reading above is given against **both** base
runs. The two verdicts that decide the item are stable across the choice: **N1
fails (d) against both** (-0.9 and -2.7) and **N4 passes against both** (does not
fall: +1.8 and 0.0). N3's and N5's (d) turn on the base, and both already fail
(c). The noul's accuracy margin (2 items of 69) is likewise at the base's own
2-item wobble. Nothing in §6 is changed: this is a fact about the yardstick, for
the roadmap owner.

**The conditions, and the interruption.** The arm configs in the worktree contain
**no `--conversation-cache-min-tokens` argument** (checked after the stop), so the
arms were started without the flag the orchestrator added mid-experiment. If that
change reached any of the runs by another path, it reached their **latency** and
not their picks: every latency figure here is labelled as a shared-engine figure,
none is used for a verdict, and (f) rests on the **100% fast-path rate** in every
arm. The engine was shared with a local chat client through much of the session,
so no latency is presented as an isolated measurement. The `answer-order` arm was
cut off by the engine hand-back and is incomplete, as stated above; the live
config is the orchestrator's and is not edited by this session.

### 23.6 What was not run

- The **choice's and the score's arms** (C1-C4, S1-S4) are implemented and
  tested, but not run: the design's §6 says they need their own pre-registration,
  and the noul's arms come first.
- **`answer-order` (N2)**: incomplete, 13 distinct items of 500, and **not part of
  the comparison** (23.5). Its question is unanswered. It was cut off by the
  engine hand-back, not by a failure of the arm.
- **The base arm's pool run** was not re-run: item 1.1b's 500-item run is the base
  arm. **The 235's base was re-run**, because rule (d)'s validity clause required
  it; both runs are in 23.5, and together they show the 0.6-point margin is
  smaller than the base's own spread.
- **Combinations** (N1 + N4, N4 + N5, and the like) were not run: one axis per
  arm, and a combination is a new pre-registration.
- **No second seed, no second engine state per arm**, beyond the base 235's
  repeat.
- **No temperature.**  Every arm ran at T = 1, the direct readout, the same model
  and the same quantization as the base arm.
- **The engine was shared with a local chat client** through most of the arms
  (the client's prompts are in the engine log). No latency figure is presented as
  an isolated measurement; the fast-path rate is 100% everywhere, and the client
  does not change the model's logits beyond the measured run-to-run wobble.
---

## 24. The affine calibration, and the rule it is judged by (2026-10-22)

Roadmap item 1.4.  The per-type temperature is one scalar, and a scalar moves a
distribution's **scale**.  On the noul questions the measured error is the **centre**:
the model over-predicts "yes" on 95 of the 100 human-labelled items, the mean signed
difference is **+0.357** on a mean absolute **0.384** (93% of the TVD), and its mean
P(yes) is **0.446** against the humans' **0.089** (§19.6).  The soft refit of §20 moved the
noul's TVD by **0.0025** while moving the choice's by 0.056 and the score's by 0.032
(§20.8).  A temperature cannot move a distribution's centre.  This item fits **two
parameters per type**.

**This section is written BEFORE the fit is run.**  The rule below is the rule of §20.2,
committed on 2026-10-21 and unchanged: it is not re-written here.  What is new is how it
is read per question type, and that reading is fixed here, before the numbers.  The tool
was smoke-tested on a **coarse grid** (steps of 0.5 and 2 folds) before this section was
committed, so that a broken tool could not masquerade as a measurement; the rule and the
expected outcome were not changed by it, and the numbers below are the full-grid fit
(steps of 0.01 and 5 folds).  Said plainly rather than hidden.

### 24.1 The map, and why the three types do not share one form

One map on the **closed** distribution - the same object the temperature scales, the
distribution over the item's own labels - with a scale `T` and a shift `b` along a
per-label coordinate `s_l` in the type's natural unit:

```
p'_l = normalize( p_l ** (1/T) * exp(b * s_l) )
```

| type | parameters | the coordinate `s_l` | what `b` means |
|---|---|---|---|
| noul | T, b | 0 for no, 1 for yes | `logit(P'(yes)) = logit(P(yes))/T + b`: the shift, in log-odds - exactly "a scale and a shift" |
| score | T, b | the level's index 0..K-1 | a tilt along the ordinal axis, in log-odds per level: a location parameter |
| choice | T | none | **not identified**: a shift that is the same for every option is cancelled by the normalizer |

**Why the score gets a tilt and not a shift.**  Its measured error is a **location**
error: on the jev-bench direct arm the expected level sits **+0.1348** above the gold on
the level scale (§22.5), and the humans' own expected level is higher still (the bias
against the humans is +0.291 on the pool and +0.339 on the test - the fit below).  A
location error needs a location parameter, and for an ordinal distribution the smooth,
invertible one is a tilt along the level index.  A literal translation `p'_k = p_{k-d}`
is not smooth and is undefined at the ends; a shift that is the same for every level is a
no-op under the normalizer.

**Why the choice gets no second parameter.**  The choice's measured error is a **shape**
error, not a location one: the model's top-label mass is 0.7862 against the humans'
0.6215 (§19.5), and a scale is what fixes a shape.  There is also no order among the
options to attach a shift to, and a constant shift over the options is cancelled by the
normalizer.  Two parameters per type **where two are meaningful**, not two everywhere, is
the honest answer: a fitted "shift" for the choice would be a number with no referent.

**The map is invertible**, `undo(q) = apply(q, 1/T, -b*T)`, so a stored run carries every
arm inside it - which is what makes the offline paired comparison exact (the
`retemperature.py` trick generalised).  The tool checks the round trip on every item of
every population and records the worst difference.  With `b = 0` the map **is** the
server's `apply_temperature`, called, not copied.  Unlike a temperature it can **move the
pick**: the noul can cross 0.5 and the score's argmax can move.  That is the point - the
noul's answer is what is wrong - and every population's accuracy is reported beside its
TVD because of it.  The map is **not in the server**; this item measures it.

### 24.2 The objective

The **cross-entropy against the human distribution** (`target_probs`) is the primary: it
is a strictly proper scoring rule (its minimum is the human distribution itself), it is
what the jev-bench yardstick measures, and it is the **same `soft` objective** as item
1.1b's primary arm, so the two fits are like for like.  The **mean TVD** to the human
distribution is the **control**: TVD is the metric the rule reads, but it is a distance
and not a proper scoring rule, so it is not the shipped objective - the fit below shows
what it does to the noul (it drives it to a near-degenerate "no").

### 24.3 The fit and the populations

**The pool** is the licence-clean fitting set of item 1.1b:
`tools/decision_policy/soft_label_pool.json`, 1,500 items, 500 a type, of which **250 a
type carry a human distribution** (`go_emotions`, `civil_comments`,
`measuring_hate_speech`).  The fit reads the stored T = 1 pool run
(`...\decision_policy_training\run\p1b_pool_T1.jsonl` and its run log) and nothing else.
**OpenDecisions is never fitted on.**

**The validation** is the temperature fit's own: 5-fold cross-validation, stratified by
type, seed **42** (`temperature.stratified_folds`), the map of a fold fitted on the other
four folds' items of the type that carry a human distribution, and applied to the fold's
own items.  The tool asserts that no item is in both the fold and its fit.  The grid is
T in [0.5, 10] and b in [-8, 8], coarse then fine; the objective is convex in (1/T, b),
so the grid's best point is the global minimum up to the grid.

**The three populations, each out of one stored run**, so the arms are paired - the same
engine, the same picks, the same logits, the same state, only the arithmetic differs:

| population | items | the T = 1 arm | the affine arm |
|---|---:|---|---|
| the pool (the fit's own) | 1,500 (750 with humans) | the pool run, `p1b_pool_T1.jsonl` | **out of fold** |
| OpenDecisions | 235 (13 with humans) | `p1b_235_T1_from_current.jsonl` | the full pool fit |
| the jev-bench test sample | 683 (280 with humans) | `p1b_jevbench_test_soft_T1.jsonl` | the full pool fit |

Reported per type and per population: the mean **TVD** to the human distribution, the
**mean signed difference** in the type's own unit (the noul's P(yes) gap; the score's
expected-level gap in levels; for the choice the **top-label mass gap**, because a
per-label signed difference sums to zero by construction on a simplex), the **AUC**, and
the official scorer's numbers (`...\opendecisions\score.py`, run on the re-scaled
predictions), with the paired item bootstrap for the intervals.

### 24.4 The rule, and how it is read per type

The rule is §20.2's, **unchanged**: a constant ships only if **(a)** the pool's
out-of-fold calibration improves against the pool's own T = 1 arm, **(b)** the jev-bench
test TVD does not get worse than T = 1 by more than **0.005**, and **(c)** the
OpenDecisions 235's Capability does not fall below its T = 1 arm - all paired.  The noul
is judged separately and its failure does not block the other types.

Two readings of that rule are fixed here:

* **(a) and (b) are read per type** - a map is fitted per type, so the type is the unit.
  **(c) is the population-level guard**: the Capability of the arm made of the types that
  pass (a) and (b).  The scorer's Intelligence and Calibration are means over the three
  types of each type's own component, and a type's map changes only that type's
  component, so every subset's Capability is **exact** from the per-type arms - the tool
  asserts the subset table reproduces the all-three arm to 1e-9.  If the candidate set
  fails (c), the **noul is removed first** (it is judged separately) and (c) is re-checked
  for the others; a set that still fails (c) ships nothing.
* **The noul's separate judgement** is its own (a), its own (b), and its own 235 row (its
  accuracy and competence on that population) - the table reports all three.

### 24.5 The expected outcome, before the numbers

**The noul's shift should help against the humans and should cost the 235.**  The fit
sees 250 `civil_comments` items whose humans say yes **9.2%** of the time (§23), while the
235's noul items are a majority-"yes" population (the model's accuracy there is 61 of 69
in the stored run) and are **the best-calibrated of the three types on that sample**
(noul 79.2 against the choice 78.9 and the score 87.4).  A shift fitted to a 9%-yes
population should lower the 235's yes rate until its noul answers flip, and (c) should
fail.  The choice's scale-only map is the temperature again, and the soft T on this pool
was 3.65 with (a) and (b) passing and (c) failing by 19.3 (§20.7) - the same is likely.
The score's tilt is the one with a chance to behave differently, because it targets the
location error; but it is fitted against the **humans'** expected level, which is not the
gold level, so it may over-correct the decision while fixing the distance to the people.

**"No constant ships" is therefore a live and likely outcome, and a population-specific
result is still a result.**  What the item can establish either way: whether a shift on
the noul's logit closes the human gap out of sample, whether discrimination moves (a
monotone map cannot move the noul's AUC - only its level and its picks), and whether the
choice's and the score's maps are scale problems or something else.

The fit's artifacts are `tools/decision_policy/affine_calibration.json` and
`tools/decision_policy/affine_report.md`; the tool is `tools/decision_policy/affine.py`.
The results follow in 24.6.

### 24.6 The fit

The full-pool fit, on the 250 items a type that carry a human distribution
(`tools/decision_policy/affine.py`; the command is at the end of this section):

| objective | type | T | b | the objective at (T, b) | at T = 1, b = 0 |
|---|---|---:|---:|---:|---:|
| **soft** (primary) | choice | **3.65** | n/a | 7.7164 | 9.9116 |
| **soft** | noul | **2.63** | **-2.29** | 0.2821 | 0.9068 |
| **soft** | score | **3.73** | **-0.54** | 0.8502 | 1.3746 |
| tvd (control) | choice | 1.90 | n/a | 0.6502 | 0.6776 |
| tvd | noul | 0.82 | -5.50 | 0.0864 | 0.3313 |
| tvd | score | 0.70 | -2.80 | 0.2930 | 0.4106 |

**The noul's shift is the item, and it is large and stable.**  `b = -2.29` in
log-odds, at `T = 2.63`.  In the population's own numbers: the pool's mean P(yes) was
**0.4142** to the humans' **0.1096**; the out-of-fold mapped mean is **0.1080**.  Across
the five folds `T` is 2.35 to 3.20 and `b` is **-2.22 to -2.41** - the sign and the size do
not turn on the split.  The cross-entropy falls from 0.9068 to 0.2821 nats, which is what
moving the centre to the people buys.

**The choice's fit has no shift and finds the temperature again: 3.65.**  That is the
soft temperature of §20.4 to two decimals, and it must be: with no coordinate to shift
along, the affine map *is* the temperature.  The choice's row is therefore not a new
measurement - it is item 1.1b's result reproduced by a different tool, and it behaves
the same way (below).

**The score's tilt is -0.54 in log-odds per level at `T = 3.73`**, and 24.10 shows why
that parameter is the wrong shape for the type.

**The TVD control drives the noul to a near-degenerate "no"** (`T = 0.82`, `b = -5.50`):
TVD is not a proper scoring rule, and minimizing the distance rewards a distribution that
says "no" to nearly everything on a 9%-yes population.  It wins its own metric on the test
sample (0.1171 against the soft arm's 0.1234), and on the 235 it fails (c) too (42.82 all
three, 24.11) - it is a control, not a candidate.  The cross-entropy is the right primary
for a decision, and this is the measurement that says so.

**The map is invertible to floating point.**  The worst `|p - undo(apply(p))|` over every
item of every population is **5.6e-16** (the per-population maxima are in
`affine_calibration.json`), so the offline paired arms are exact arithmetic on the stored
logits.

```
python tools/decision_policy/affine.py \
  --items ...\decision_policy_training\items\p1b_pool.jsonl \
  --runlog ...\run\p1b_pool_T1.runlog.jsonl \
  --predictions ...\run\p1b_pool_T1.jsonl
```

(1,500 items, 0 cross-check problems, 1.8 min; the arms are written to
`...\run\p1a_pool_affine.jsonl`, `p1a_od235_affine.jsonl`, `p1a_jevtest_affine.jsonl`.)

### 24.7 The pool, out of fold

Each item is mapped by the map of the fold it was **not** in.  The baseline is the pool
run's own T = 1 arm.

| arm | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| T = 1 | 49.72 | 41.98 | 57.47 |
| the affine fit, out of fold | **55.05** | 40.47 | **69.63** |
| paired bootstrap | **+5.35 [+0.39, +10.36]** | -1.56 [-6.14, +2.74] | **+12.26 [+5.38, +19.28]** |

| type | TVD T = 1 | TVD affine | bias T = 1 | bias affine | AUC T = 1 | AUC affine | calibration T = 1 | calibration affine | competence T = 1 | competence affine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| choice | 0.6776 | 0.7049 | +0.1780 | -0.2959 | 0.9195 | 0.9195 | 44.78 | 51.89 | 54.32 | 54.32 |
| noul | 0.3313 | **0.1088** | +0.3047 | **-0.0015** | 0.8716 | 0.8684 | 61.99 | **73.72** | 46.40 | 52.40 |
| score | 0.4106 | 0.3637 | +0.2907 | +0.0010 | 0.6895 | 0.6860 | 65.64 | **83.28** | 25.21 | 14.69 |

The noul's mean P(yes) is **0.4142 at T = 1**, **0.1080** with the map, against the
humans' 0.1096; its accuracy rises 73.2% to 76.2% (366 to 381 of 500).  The choice's TVD
gets **worse** (0.6776 -> 0.7049) while its calibration component improves - the CE fit is
not the TVD's fit, and the two measures disagree on a 28-option emotion question.  The
score's competence falls (25.21 -> 14.69) while its calibration rises: 24.10 is why.

The paired item bootstrap of the TVD difference (affine - T = 1): noul **-0.2225
[-0.2588, -0.1863]**, score -0.0469 [-0.0741, -0.0178], choice **+0.0272 [+0.0031,
+0.0537]**, all -0.0807 [-0.1013, -0.0615].

**(a) passes for every type** - the out-of-fold calibration improves for all three - and
that is the first half of what the fit was asked to do.

### 24.8 The OpenDecisions 235

The full pool fit, applied to the 235's stored T = 1 arm.  The types' components are
separately measurable, so the Capability of **any subset** of the three maps is exact
(the tool asserts the table reproduces the all-three arm to 1e-9).

| arm | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| T = 1 | **78.87** | 75.90 | 81.84 |
| the affine fit, all three | **41.53** | 35.34 | 47.73 |
| paired bootstrap | **-35.80 [-43.26, -27.57]** | -39.67 [-47.71, -29.42] | -31.93 [-42.18, -20.77] |

| types mapped | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| none (T = 1) | 78.87 | 75.90 | 81.84 |
| choice | 72.93 | 75.90 | 69.96 |
| score | 68.50 | 58.44 | 78.56 |
| noul | 57.84 | 52.80 | 62.88 |
| choice + score | 62.56 | 58.44 | 66.68 |
| choice + noul | 51.90 | 52.80 | 51.01 |
| noul + score | 47.47 | 35.34 | 59.60 |
| all three | 41.53 | 35.34 | 47.73 |

**Every single type loses on the 235**, and the noul loses the most.  The per-type rows:

| type | accuracy T = 1 | accuracy affine | TVD T = 1 | TVD affine | bias T = 1 | bias affine | calibration T = 1 | calibration affine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| choice | 88.79% (95/107) | 88.79% | 0.2868 (3 items) | 0.0667 | +0.2868 | +0.0559 | 78.95 | 43.32 |
| noul | **88.41% (61/69)** | **56.52% (39/69)** | 0.1589 (6 items) | 0.6486 | **-0.1114** | -0.6486 | 79.16 | 22.29 |
| score | 77.97% (46/59) | 49.15% (29/59) | 0.2902 (4 items) | 0.3327 | +0.1486 | -0.3544 | 87.41 | 77.57 |

**The noul's sign is the finding.**  On the 235's noul items the model's mean P(yes) is
**0.6776** against the humans' **0.7890** - it is already **under**-eager there, the
opposite of the jev-bench content judgments, and the sign of its bias is **-0.1114**.  The
shift fitted on a 9.2%-yes population moves **26 of 69** answers and gets **22 fewer right
(61 -> 39)**, so its bias goes to -0.6486 and its competence from 76.32 to 7.01.  A
shift that is right for one population is wrong, in sign, for the other.

The choice's accuracy **cannot** move (the map is monotone, so the pick is fixed), and yet
its Capability falls: its calibration component goes 78.95 -> 43.32, because the flattening
that helped its own pool's ECE reports a different confidence on this one.  That is
§20.7's failure reproduced exactly.

The score moves **28 of its 59** answers and gets **17 fewer right (46 -> 29)**, and its
within-item AUC falls 0.8983 -> 0.7189: the tilt is not monotone inside the item, so it
can cost the ordering of the levels, which no scale can.

### 24.9 The jev-bench test sample

| arm | Capability | Intelligence | Calibration | mean TVD |
|---|---:|---:|---:|---:|
| T = 1 | 55.08 | 47.18 | 62.98 | 0.4439 |
| the affine fit, all three | 55.28 | 42.93 | 67.63 | **0.3074** |
| paired bootstrap | +0.01 [-6.99, +6.78] | -4.48 [-10.94, +2.03] | +4.50 [-4.75, +13.37] | |

| type | TVD T = 1 | TVD affine | bias T = 1 | bias affine | AUC T = 1 | AUC affine | accuracy T = 1 | accuracy affine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| choice | 0.4702 | 0.4139 | +0.1691 | -0.1771 | 0.9123 | 0.9123 | 70.75% | 70.75% |
| noul | 0.3836 | **0.1234** | +0.3560 | **+0.0330** | 0.8577 | **0.8577** | 75.23% (167/222) | 74.77% (166/222) |
| score | 0.4963 | 0.3986 | +0.3394 | -0.0333 | 0.7446 | 0.7213 | 47.31% (79/167) | 49.10% (82/167) |

**The noul's human gap closes out of sample, and the ranking does not move.**  The mean
P(yes) goes **0.4451 -> 0.1222** against the humans' **0.0892**; the TVD falls **0.3836 ->
0.1234** (the paired item bootstrap: **-0.2602 [-0.3214, -0.1993]**, 67% of items better),
the bias **+0.3560 -> +0.0330**, and the AUC is **0.8577 either way** - a monotone map
cannot move a binary ranking, and this one did not.  **91 of the 222 answers move and only
one net answer is lost** (167 -> 166).  The gain is the **centre, not new information**:
the AUC is unchanged and **33% of the items are further from the humans** - the shift
moves the whole distribution, and the mean gap closes because the mean was where the error
was.  This is the item's exit criterion's first half, met out of sample: the shift
generalizes to a population it was not fitted on, and it costs no discrimination.

The choice is the temperature again (3.65, the soft arm's), so its TVD falls
0.4702 -> 0.4139 and its calibration falls 68.11 -> 51.25; the Capability of the
choice-only arm is 52.27 against 55.08.  The score moves **79 of 167** answers and gains
**3 net** (79 -> 82), and its expected level still gets worse (24.10).

### 24.10 The score's location is per config, not per type

The score type spans K = 3, 4, 5 and 6 levels on these populations, and the pool carries
a human distribution for **one** score config only, `measuring_hate_speech` (K = 3).  The
per-config measurements (the report's full table; `level - gold` is the mean signed
distance of the expected level to the gold, §22.5's measure):

| population | config | K | n | level - gold, T = 1 | level - gold, affine | accuracy T = 1 | accuracy affine |
|---|---|---:|---:|---:|---:|---:|---:|
| pool | helpsteer2_helpfulness | 5 | 125 | -0.3906 | -1.3214 | 0.34 | 0.19 |
| pool | helpsteer2_verbosity | 5 | 125 | +0.5087 | -0.4587 | 0.47 | 0.48 |
| pool | measuring_hate_speech | 3 | 250 | +0.3387 | +0.0490 | 0.54 | 0.71 |
| test | helpsteer2_helpfulness | 5 | 24 | -0.1923 | -1.1032 | 0.38 | 0.25 |
| test | helpsteer2_verbosity | 5 | 24 | +0.7106 | -0.2916 | 0.46 | 0.46 |
| test | measuring_hate_speech | 3 | 50 | +0.4511 | +0.0784 | 0.40 | 0.74 |
| test | sst5 | 5 | 23 | -0.1567 | -1.1365 | 0.57 | 0.39 |
| test | stsb | 6 | 23 | -0.5602 | -1.3799 | 0.43 | 0.13 |
| test | yelp5 | 5 | 23 | +0.2127 | -0.4099 | 0.70 | 0.70 |

**Two things, and the second is the larger.**  (1) The raw level index is **not
commensurable** across K: a tilt of 0.54 in log-odds per level moves the expected level by **0.29
levels on the 3-level config** and **0.93 to 0.97 on the 5-level ones** (the deltas in the
table above) - the fit was made on a K = 3 config and applied to K = 5 and 6, and the
response scales with K.  The scorer's own nMAE normalizes by K - 1, so the natural unit is the
**normalized position k/(K-1)**, not the index.  (2) **The configs disagree in sign at
T = 1**: `helpsteer2_helpfulness` is 0.39 levels *below* the gold and
`helpsteer2_verbosity` 0.51 *above* it on the pool; the same opposition is on the test
(-0.19 and +0.71) and in the 235's larger cells.  So the aggregate **+0.14** (§22.5's
+0.1348 on its own run; +0.1401 on this one) is not one bias: it is opposite errors
cancelling.  Even a correctly normalized tilt cannot serve two configs that want
opposite directions.  **A per-type location is the wrong unit for the score**; a
per-config location is the right one, and that is a different item.

This is a correction to 24.1's choice of coordinate, recorded rather than quietly made:
the pre-registered map used the level's index, the measurement shows the index is the
wrong unit and the type is the wrong granularity, and the corrected map (a per-config
location on the normalized position) is **not run here**.

### 24.11 The rule, applied

| type | (a) the pool's OOF calibration | (b) the jev test TVD | its own 235 row | verdict |
|---|---|---|---|---|
| choice | 44.78 -> 51.89: **PASS** | 0.4702 -> 0.4139 (-0.0563): **PASS** | competence 85.72 -> 85.72, calibration 78.95 -> 43.32 | does not ship |
| noul | 61.99 -> 73.72: **PASS** | 0.3836 -> 0.1234 (-0.2602): **PASS** | competence 76.32 -> 7.01, 61 -> 39 of 69 | does not ship |
| score | 65.64 -> 83.28: **PASS** | 0.4963 -> 0.3986 (-0.0976): **PASS** | competence 65.66 -> 13.28, 46 -> 29 of 59 | does not ship |

**(c) fails for every candidate.**  The set that passes (a) and (b) is all three types;
its 235 Capability is **41.53 against 78.87**.  With the noul removed (it is judged
separately) the choice and the score are **62.56, still 16.3 below**; each type alone is
72.93 (choice), 57.84 (noul) and 68.50 (score).  There is no subset of this map that keeps
the 235 where it was.  The TVD control fails (c) as well and is nowhere near the bar:
42.82 all three (its noul's own 235 competence is 13.55 against the soft arm's 7.01, both
far below the 76.32 the type had).

**NO CONSTANT SHIPS.**  The endpoint stays at T = 1, and the map stays out of the server.

A correction to 24.5's parenthetical, before it is read as a contradiction: "best
calibrated of the three types" is §1.1's **raw ECE** on the 235 (noul 0.0756 against the
choice 0.1313 and the score 0.1082), not the stored arm's **calibration components**
(79.2 / 78.9 / 87.4, where the score is highest - its component also carries
(1 - nRPS), so it is not an ECE).  By the raw ECE the noul is the best; the stored T = 1
arm's own ECEs are 0.1042 (noul) and 0.1053 (choice).  The argument does not change.

### 24.12 What it means

**The noul's bias is real, it is movable, and it is population-specific.**  A shift on
the logit closes the human gap on the pool's out-of-fold arm (0.3313 -> 0.1088) and on a
population it was not fitted on (the jev-bench test: 0.3836 -> 0.1234), leaves the AUC
unchanged, and costs one net answer of 222.  **That is what a temperature could not do**, and
it settles the mechanism question §19.6 left open: the noul's error is in the reported
level, and a level is what a shift moves.

**And the same shift is wrong on the 235, in sign.**  There the model is already
*under*-eager (mean P(yes) 0.6776 against 0.7890, bias -0.1114); the shift drags it to
0.1404, moves 26 of 69 answers and gets 22 fewer right.  The two populations want opposite
shifts, so **no per-type constant can serve both** - the same verdict as §20.8's for the
temperature, now with the right lever.  The product answer is unchanged and sharper: **a
per-workload calibration**, not one constant, and the form of that calibration for the
noul is a **shift**, not a scale.

**The choice has no second parameter, and the measurement is the check.**  Its
scale-only map is the temperature to two decimals, passes (a) and (b), and fails (c) by
5.9 Capability points - the 1.1b result reproduced.  A "two-parameter" fit for the choice
would be a fitted number with no referent; the honest report says so.

**The score's failure is the item's most useful design finding.**  Its expected level is
off, but the error is **per config and both signs** (+0.51 and -0.39 on the two 5-level
configs of the pool) and the type's level count varies from 3 to 6, so a per-type
location is the wrong granularity *and* the raw level is the wrong unit.  A per-config
location on the normalized position k/(K-1) is the map the measurement points to; it is
not run, and it is a new pre-registration.

**Where this sits beside 1.5.**  The `criteria-first` prompt (item 1.5) cuts the noul's
bias on the same pool half A from **0.3047 to 0.2268** - about a quarter - with no 235
cost, and it passes its guards.  The affine shift cuts it to **~0.00** and costs the 235
21 Capability points.  So the prompt is still **the cheaper and the shippable lever**; the
affine fit is the proof that the level can be moved, and the evidence that a constant
fitted on one population is not the way to move it.  A combination (the prompt's
`criteria-first` plus a *small* shift, or a per-workload shift chosen from the workload's
own data) is a **new experiment**, not a result of this one.

**The exit criterion, read honestly.**  Its first half - the noul's TVD to the human
distribution falls out of sample - is **met**: 0.3836 -> 0.1234 on the test sample, with
no AUC loss.  Its second half - "nothing else regresses on either population" - is **not
met**: the choice and the score regress on the 235, and the noul itself regresses there.
The clause "if the shift does not generalize, the bias is in the prompt or in the task,
not in the readout" resolves the other way: **the shift does generalize to the humans, and
it is the population that changes the answer**, which is a stronger statement than either
branch of the criterion.

### 24.13 The limits, and what was not run

* **No live run of the map.**  The 235's and the test sample's arms are offline
  re-scalings of one stored run each.  The arithmetic is exact (the inversion is checked
  to 5.6e-16) and the arms share the engine, the picks, the logits and the state; a live
  run would add the engine's own wobble (about 0.0193 mean |change| in P(yes) on the jev
  noul slice, and 1.8 Capability points on the 235, §23.5) and nothing else.  The picks a
  live run would report are the argmax of the same logits, so only that wobble would
  separate them from the arms here.
* **The map is not in the server.**  Shipping one would need a change to
  `serve/logits_first_token.py` and a new launch-config key with a byte-identity test;
  nothing was changed there, and the rule says nothing ships.  (That is a static reading
  of the server's module: `apply_temperature` is the scale only, and there is no affine
  key.)
* **The 235's per-type TVD rows are thin**: 3 choice, 6 noul and 4 score items carry a
  human distribution there.  The accuracies (107 / 69 / 59 items) and the calibration
  components are the numbers to read on that population, not the TVDs.
* **The fit sees one noul population and one score config with humans.**  The noul's 250
  human items are all `civil_comments` (9.2% yes); the 250 `paws`/`strategyqa` items have
  no human distribution, so the fit cannot see the 45.2% base rate.  The score's 250
  human items are all `measuring_hate_speech` (K = 3), which is 24.10's cause.
* **No normalized or per-config score map was run**, no per-config noul map, no
  combination with the 1.5 prompt, no second pool and no second seed.  Each is a new
  pre-registration.
* **The same caveats as §20:** the two populations are different kinds of question
  (Jev-style typed decisions against content judgments), the constants belong to this
  model and this quantization, and the pool is licence-clean but validation-only.
* **No engine, no server, no GPU, no new dependency.**  Everything above comes from the
  stored runs in
  `...\decision_policy_training\run\` and `...\opendecisions\`; the fit and the report are
  `tools/decision_policy/affine_calibration.json` and
  `tools/decision_policy/affine_report.md`.
## 25. The conversation cache: a one-shot decision no longer takes a slot (2026-10-22)

The decision endpoint sends short prompts, one per question, and each one used to leave a
parked conversation behind: 25,738 parks against 649 restores in the user's 2-slot log, 98.3%
of them under 2,000 tokens, one eviction for every two parks. A parked conversation costs a
whole slot whatever its length, so two decisions filled the cache and the third evicted the
chat they were interleaved with - whose next turn then re-read its whole history (the log's
4,847 tokens in 7.6 s). The diagnosis and the options are in
[conversation_cache_oneshot.md](conversation_cache_oneshot.md); this section is the
measurement of the option that was built.

**Sections 20-23 are on `feat/phase1-integration`; this branch's copy of this document ends
at 19. The number is the roadmap's, so the merge is an append.**

### 25.1 The key, and the rule that makes it safe

The engine now takes `park=0` on a `GEN` or `LOGITS` line: the conversation that request
read is not parked by the request that replaces it, while its session stays continuable
(`ckpt` stays 1), so a trie walk's next step still reuses it. `ckpt=0` would also stop the
continuation, which is why it was rejected: 96% of the log's 14,393 walk steps reuse the
session. The server sends the key on the decision path only; for this measurement it was
behind `STRATA_DECISION_PARK=1`, so the default path was byte for byte what it was, and it
became the default at step 6 (§25.6). The as-built line map is in the plan's
§10; the rule that a conversation **taken** from the cache is always parked again (it must
be put back) is `conversation_park_allowed()` and is tested without a GPU.

### 25.2 The measurement

The cache-built engine (`build-cache-cuda\strata.exe`), a copy of the live config whose
`exe` is that binary, arm A with the switch unset and arm B with it set, three runs each
alternating A B A B A B, a fresh server and engine per run. Each run: four chat turns that
reach **7,258 tokens**, **20 decisions** with fresh state text, then chat turn 5. Three
departures from the plan's recipe, each forced and each recorded:

* **`--resident-budget-gib 20`** instead of the live 36. With 36 GiB and another session
  holding 9.6 GB, the host had 261 MiB available and the engine's own admission refused every
  park (`need 671 MiB plus 2560 MiB floor`). Arm A could not park, which is not today's
  behaviour. The 20 GiB copy leaves about 7 GB and the floor is met: the chat's snapshot is
  704 MiB at 7,258 tokens, a decision's 263 MiB at 1,822. All other arguments are the live
  ones.
* **The decisions are 1,815-1,881 tokens**, not under 500. The shipping config keeps
  `--conversation-cache-min-tokens 1024`, and a 300-token decision is already skipped by it.
  Over the minimum, the shipping config parks them, which is the state under test.
* **Latency is not reported.** The card was shared with about 10 GB of other work. The
  readouts are the cache's own and do not depend on the clock.

| run | arm | chat 5: prompt = reused + read | reuse | evictions | parked | `skip parking (park=0)` |
|---|---|---|---:|---:|---:|---:|
| A1 | before | 7,467 = **0** + 7,467 | 0.0% | **19** | 2 | 0 |
| A2 | before | 7,477 = **0** + 7,477 | 0.0% | **19** | 2 | 0 |
| A3 | before | 7,458 = **0** + 7,458 | 0.0% | **19** | 2 | 0 |
| B1 | after | 7,453 = **7,423** + 30 | **99.6%** | **0** | 1 | 25 |
| B2 | after | 7,459 = **7,429** + 30 | **99.6%** | **0** | 1 | 25 |
| B3 | after | 7,461 = **7,430** + 31 | **99.6%** | **0** | 1 | 25 |

Every run answered all 25 requests. In the before arm the 20 decisions read their whole
prompt (`0 reused`) and park; the third pushes the chat out and **19 evictions** follow; the
fifth chat turn reads all 7,467 tokens. In the after arm the decisions read the same prompt
and pay the same read, but take no slot: **zero evictions**, the chat stays the only entry,
and its fifth turn resumes 99.6% of itself. The remaining parks in B (1, 2 and 3 across the
runs) are the chat path parking and restoring its own conversation between turns, which is
today's behaviour.

The walk is untouched in both arms: its steps read `1,877 = 1,867 reused + 10 read` and
`1,879 = 1,878 reused + 1 read` (99.5% / 99.9%) in every run. The 25 skips in B are the 20
decision switches plus 5 walk steps that try to park their base request's conversation and
are suppressed too - the key working on the walk, not a fault.

### 25.3 The 235, and the answers

| run | items | fast path | Capability | Intelligence | Calibration | answers moved vs `fix_serial.jsonl` |
|---|---:|---|---:|---:|---:|---:|
| stored `fix_serial.jsonl` | 235 | 235/235 | 79.8 | 78.0 | 81.7 | - |
| before (A) | 235 | 235/235 | **79.8** | 78.1 | 81.4 | 2 |
| after (B) | 235 | 235/235 | **78.3** | 76.4 | 80.3 | 6 |

Both arms answered every item on the fast path and both are inside the sample's 1.8-point
run-to-run spread. The moves are the engine's own wobble (the two stored base runs differ in
2 noul items); the key changes no arithmetic.

### 25.4 The verdict

The pre-registered rule passes on every run: after, `M5 >= 0.9 x N5` (99.6%) and
`evictions = 0`; before, `M5 = 0` and `evictions >= 18` (19). The walk's reuse is intact in
both. It is not the "not worth it" case either: the chat's entry is the one the decisions
evicted, and it is the one the after arm keeps.

### 25.5 The limits, and what was not run

* One model, one card, one engine build, 3 runs per arm, 2 cache slots and the shipping
  1,024-token minimum. Another slot count or no minimum changes which requests park, not
  what `park=0` does.
* The 300-token decisions of the plan's §6.1 cannot show a difference under the shipping
  minimum; they were not run, and the minimum's own effect (the plan's stopgap) is not
  re-measured here.
* The 20 GiB resident copy is the one departure from the live config; it changes what the
  engine keeps in RAM, not the cache's rules.
* **Not run:** the latency comparison (the card is shared); three runs per arm on the 235
  (one each); a default-on arm (the plan's step 6; it was run afterwards, §25.6); the
  walk's answer identity on a stored run; any AMD or Mac engine. The 235's run-to-run
  spread is quoted from §23, not re-measured.

### 25.6 Step 6: on by default, and the verification (2026-10-22)

The key is now the server's default (`STRATA_DECISION_PARK=0` restores the old call), the
live config's `--conversation-cache-min-tokens` is back to 0, and `engine\strata.exe` is
rebuilt from the merged source for sm_89. One run of the same recipe on the **shipping**
config (36 GiB resident, no minimum) reproduced the effect: the chat's fifth turn
`7,463 = 7,404 + 59` (**99.2%** reused), **0 evictions**, 25 `park=0` skips, the walk at
`1,878 = 1,868 + 10` and `1,879 = 1,878 + 1`. The 235: **81.3** (79.8 baseline), 235/235 on
the fast path, 5 answers moved. One run, not three; latency not claimed. The numbers, the
backups and the limits are the companion doc's §12.



## 26. The score's location, per config (1.4b) (2026-10-22)

**26.1 is written BEFORE the run**, like 20.2 and 24.2: the rule below is committed before the first effect is
read.  The counts in the data table were taken first (which items exist, in which config, with which labels), and
they are not effects.

### 26.1 The question, the data, and the rule

1.4 closed with a finding and an open item.  The finding: the score's error at T = 1 is not one location.  On the
jev-bench sample `helpsteer2_helpfulness` sits 0.39 levels BELOW the gold and `helpsteer2_verbosity` 0.51
ABOVE it, so the type-wide +0.14 is two opposite errors cancelling; and a tilt fitted on the raw level index is
not the same size on a 3-level question as on a 5-level one.  The open item (1.4b) asks whether a location on the
NORMALISED position `u_k = k/(K-1)`, fitted PER CONFIG, transfers where the type-wide constant did not.

**What the data on this machine can answer, counted before any effect was looked at.**

| population | score items | with a human distribution | with a stored T = 1 run |
|---|---:|---:|---|
| the licence-clean pool | 500 | **250** (`measuring_hate_speech`, K = 3) | 1,500 of 1,500 |
| the pool's `helpsteer2_helpfulness` / `_verbosity` | 125 + 125 (K = 5) | **0** - gold level only | yes |
| the jev-bench sample, `measuring_hate_speech` | 50 | **50** | 683 of 683 |
| the jev-bench sample, `helpsteer2_*` | 24 + 24 (K = 5) | **0** - gold only | yes |
| the OpenDecisions 235 | 69 | 13 (its calibration items) | yes |

Three consequences, and they shape the experiment rather than excuse it:

* **The unit does not matter for a fit inside ONE config.**  With K fixed, `b * k/(K-1)` and `b' * k` describe
  the same one-parameter family (`b' = b/(K-1)`) and give identical distributions.  Normalising matters when one
  b is SHARED across configs of different K - which is what 1.4 did, and what made its tilt incommensurable - and
  it matters for comparing fitted b's across configs.  So "per config on the normalised position" is, for a
  single config, "one location fitted for that config", and this section says so instead of letting the
  normalisation look like the thing being tested.
* **Only one config can be fitted against PEOPLE**: `measuring_hate_speech` (K = 3), 250 pool items, transferred
  to the 50 jev-bench sample items of the same config.  The two `helpsteer2` configs have no human distribution
  anywhere on this machine, so their arm can only be fitted against the GOLD LEVEL - a location toward the gold,
  not toward the annotators' spread.  That is a weaker target, and the rule below says so.
* **The OpenDecisions 235 cannot be the yardstick for a per-config constant.**  Its 69 score items are
  OpenDecisions' own families at K = 3, 4 and 5; none of them is a jev config, so a b fitted for
  `measuring_hate_speech` would arrive there as a foreign constant - the transfer 1.1b and 1.4 already measured
  failing.  The 235 is therefore reported for information (its score competence at T = 1, unchanged) and is
  **not** a pass condition here.

**The map.**  For a score question with K levels, `u_k = k/(K-1)`:

    p'_k = normalize( p_k * exp(b_c * (u_k - 0.5)) ),   one b_c per config

`T = 1` in every arm (a location, not a scale: 1.4's scale arm is measured and does not ship).  The `- 0.5` is
cosmetic - a constant in the exponent cancels in the normaliser - and it makes `b_c = 0` the identity and `b_c`
comparable across configs.

**Arm H (against people).**  Config `measuring_hate_speech`.  Fit b by minimising the mean cross-entropy against
the human distribution (the `soft` objective of 1.1b and 1.4), 5-fold leak-free CV
(`temperature.stratified_folds`, seed 42), one b per held-out fold fitted on the other four.

**Arm G (against the gold).**  Configs `helpsteer2_helpfulness` and `helpsteer2_verbosity`.  Fit b by
minimising the mean cross-entropy against the ONE-HOT gold level, same folds and seed.  This is the only target
the data carries for these configs, and a pass here says the level error is movable - not that the reported
distribution is better calibrated against people.

**The rule, fixed now.**

| | the test | the baseline it must beat |
|---|---|---|
| **(a)** | Arm H: the pool's out-of-fold TVD to the human distribution improves | the pool's T = 1 arm (one run, paired) |
| **(b)** | Arm H: the jev-bench sample's TVD for the same config does not get worse by more than **0.005** | T = 1, paired |
| **(c)** | Arm G, each config: the pool's out-of-fold mean absolute LEVEL error (expected level minus gold) falls | T = 1, paired |
| **(d)** | Arm G, each config: the jev-bench sample's mean absolute level error does not get worse by more than **0.10 levels** | T = 1, paired |

0.005 is 1.1b's own margin (a fifth of what 1.1's constants cost on the human labels); 0.10 levels is a quarter
of the smaller of 1.4's two measured errors (0.39).  Both are fixed before the numbers, so "not worse" cannot
become "better" later.

**What a pass means, and what it does not.**  A pass means the score's error is a LOCATION that a per-config
constant can move, out of sample, which would make a per-workload calibration (the endpoint told which config it
is serving) worth building.  It does not ship a constant: the fitted b's would be constants for these
populations, and 1.1b and 1.4 measured what a constant does on a third.  A fail on (a)/(b) closes Arm H, and a
fail on (c)/(d) closes Arm G, with the finding that the score's location there is a task property, not a
calibration one.

**The limits, stated up front.**  One 3-level config with human labels; two 5-level configs with gold only; no
engine is started (every arm is the stored T = 1 run re-scaled, `retemperature.py`'s principle); the 235 is
reported, not judged; one parameter per config (no pooling across K, no per-family fit, no interaction with T).

### 26.2 The results

**Run: `tools/decision_policy/score_location.py`, `--out-json score_location.json --out-report
score_location_report.md`, 2026-10-22.**  Its output files are in `tools/decision_policy/` beside the other arms.
No engine was started: every arm is the stored T = 1 run re-scaled.

| config | K | n (pool) | b_c | pool: T = 1 | pool: out of fold | the pool, paired (map - T = 1) |
|---|---:|---:|---:|---|---|---|
| `measuring_hate_speech` | 3 | 250 | **-2.203** | TVD 0.4106 | TVD **0.3299** | **-0.0807 [-0.1012, -0.0590]** |
| `helpsteer2_helpfulness` | 5 | 125 | **+4.308** | level 1.044 | level 0.982 | -0.062 [-0.141, +0.020] |
| `helpsteer2_verbosity` | 5 | 125 | **-4.249** | level 0.752 | level 0.648 | -0.104 [-0.197, -0.006] |

| config | n (jev sample) | T = 1 | the map | the sample, paired | 26.1's tests |
|---|---:|---|---|---|---|
| `measuring_hate_speech` | 50 | TVD 0.4963 | TVD **0.3961** | **-0.1001 [-0.1405, -0.0554]** | (a) and (b) **pass** |
| `helpsteer2_helpfulness` | 24 | level 0.928 | level 0.973 | +0.045 [-0.107, +0.191] | (c) and (d) pass by the letter |
| `helpsteer2_verbosity` | 24 | level 0.790 | level **0.464** | **-0.326 [-0.509, -0.155]** | (c) and (d) pass |

**Arm H passes both of its tests, and clearly.**  On `measuring_hate_speech` one location, `b = -2.2` (toward
level 0), takes the out-of-fold TVD from 0.4106 to 0.3299 (**-0.0807 [-0.1012, -0.0590]**) on the pool it was
fitted on, and from 0.4963 to 0.3961 (**-0.1001 [-0.1405, -0.0554]**) on the jev-bench sample's 50 items of the
same config.  The five folds agree on the constant (-2.261, -2.357, -1.904, -2.137, -2.342), so this is not one
fold carrying the result.  It is also the same **mechanism** 1.4 found for the noul: the error is a signed
offset - here the model reads too much hate speech on a population whose gold levels average 0.55 out of 2, so
it sits above the people - and a location, not a scale, moves it.

**The two 5-level arms pass the letter of the rule and are NOT a calibration.**  Two things say so, and the
second is a diagnostic run after these numbers were read (so it is reported as a qualifier, not a pre-registered
test):

1. **The constants are unstable or the gain is inside the noise.**  `helpsteer2_helpfulness` fits `b = 4.308`,
   but its folds fit 3.486 to 5.291, and the pooled change (-0.062 levels) has a 95% interval through zero.
   `helpsteer2_verbosity` fits -4.249 with folds -4.067 to -4.516 (stable) and a pooled change of -0.104
   [-0.197, -0.006], which is real but small.
2. **A constant that knows nothing but the population's mean level beats the fitted map on
   `helpsteer2_verbosity`.**  "Predict the pool's mean gold level for every item" scores a mean |level error|
   of **0.431** on verbosity (against T = 1's 0.752 and the map's 0.648).  On `helpsteer2_helpfulness` the
   same prior scores 1.103, which is *worse* than T = 1's 1.044 - there the map's 0.982 is the best of the
   three, but only 0.062 better than doing nothing.  So the tilt moves the model's distribution centre toward
   the population's gold mean (verbosity) or leaves it barely better (helpfulness); it does not make the
   distribution better per item.  For a 5-level question with gold-only labels that is all a one-parameter
   location *can* do, which is why 26.1 said a pass here would be the weaker claim.

**The 235, reported and not judged.**  Its **59** score items (26.1's table said 69 - that count was wrong; no
test depends on it) at T = 1: mean |expected level - gold| **0.306 levels**, argmax accuracy **0.780**.  No arm
is applied to them, because none of their configs is a jev config: a fitted b would arrive there as a foreign
constant, which is the transfer 1.1b and 1.4 already measured failing.

**The unit, as 26.1 predicted.**  Inside one config the normalised axis and the raw level index are the same
one-parameter family, so the normalisation did no work in any of these three fits.  What 1.4 got wrong was
sharing one b across K = 3, 4 and 5 with a raw index - not the index itself.  This run therefore does **not**
vindicate "per config on the normalised axis" as a new mechanism; it measures one location per config, which is
what the data could support.

**Verdict, per 26.1.**

* **Arm H (`measuring_hate_speech`): the location is a real, transferable fix for this config** - (a) and (b)
  both pass, on out-of-fold data and on a sample it was never fitted on.  It is still a constant for one
  population, and 1.1b/1.4 measured what a constant does on a third.  **No constant ships.**
* **Arm G (`helpsteer2_helpfulness`, `helpsteer2_verbosity`): this line closes for these configs.**  The rule
  passes, but a population-mean prior beats the map on verbosity, and on helpfulness the map's 0.062-level gain
  has a 95% interval through zero with folds scattered from 3.5 to 5.3.  The finding is the one the exit
  criterion named in advance: **with gold-only labels the score's location is a task property here, not a
  calibration** - the information a per-item location would need is not in the labels this machine has.
* **The one thing that would reopen it** is a human distribution for a 5-level score config.  There is none on
  this machine (26.1's table); the licence record's rules decide where one could come from.

**The limits, and what was not run.**

* One 3-level config with human labels, two 5-level configs with gold only, one seed (42) and five folds; the
  bootstrap is the programme's paired item bootstrap, 1,000 resamples.
* The jev-bench samples are 50 and 24 items: Arm G's 24-item intervals are wide by construction, and (d)'s
  0.10-level margin is generous for that size.
* No engine, no live run: every arm is the stored T = 1 reading re-scaled, so this says nothing about the
  endpoint's latency or about a live temperature.
* No pooling across K, no per-family fit, no interaction with T, and no per-item (state-dependent) calibration -
  the last is the thing a location cannot be.
