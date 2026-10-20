# Handoff: Explore a Shared-Backbone System 1 Decision Policy in Strata

**Audience:** A coding agent operating inside the user's Strata workspace  
**Status:** Research and prototype assignment; do not change default behavior  
**Primary objective:** Determine whether Strata can serve a fast, specialized agent decision policy using the already-loaded Qwen3.8-Flash-Next backbone, without loading a second large model or duplicating its weights.

## Executive summary

The user runs Qwen3.8-Flash-Next locally through Strata. VRAM and overall memory are constrained enough that a second full model is not a viable solution. Explore a policy head that reuses Strata's current model execution and weights:

1. **Establish a baseline first.** Call the already-running Strata server with a compact decision prompt, reasoning disabled, and a very short output budget. Measure decision quality and end-to-end latency. Do not start a second inference server.
2. **Define the decision task and data.** Use a small, stable action vocabulary and representative state/action examples from the user's agent workflow.
3. **Investigate Strata's actual inference path.** Find whether the final hidden representation for the last prompt position can be captured immediately before the normal vocabulary projection, without token generation. Inspect the checked-out revision and its architecture; do not assume stock llama.cpp APIs apply unchanged.
4. **Export representations from the exact Strata model/build.** Avoid training against a different full-precision Transformers checkpoint unless an experiment explicitly tests that mismatch.
5. **Train a tiny classifier in PyTorch.** Train it on exported vectors, not by loading another large model. Start with a linear layer.
6. **Integrate an opt-in decision path.** Run the existing backbone once, score actions with the tiny head, and return a structured decision without autoregressive text generation.
7. **Benchmark and make a go/no-go recommendation.** The head saves decoding, not the cost of running the backbone. Demonstrate a real end-to-end benefit before proposing further integration.

The first deliverable is a measured feasibility report and a reproducible baseline. The policy-head integration is the next stage only if the repository architecture and baseline justify it.

## 1. Goals and constraints

### Goals

- Reuse the **single existing loaded Strata model** for ordinary text generation and decision inference.
- Add at most a small action-classifier head and its configuration; do not load another large checkpoint.
- Explore a direct decision path that does not generate a natural-language explanation or a chain of reasoning tokens.
- Keep normal chat/completions behavior unchanged.
- Measure quality, latency, and memory rather than assuming the classifier will be faster.
- Leave clear setup, reproduction, and future-work notes for the user.

### Non-goals for this first exploration

- Do not fine-tune the full Qwen model.
- Do not make LoRA, a separate small LLM, or a second Strata process the main solution. They may be documented as later alternatives, not prerequisites.
- Do not redesign Strata's expert cache, SSD lookup, MTP/speculative decoding, context/KV-cache system, or scheduler merely to pursue this prototype.
- Do not change default behavior, default model settings, or the user's model files.
- Do not claim that a classifier eliminates backbone compute: it still needs the backbone's forward pass over the decision state.

### Hard constraints

- **No second large model or second copy of the base weights.**
- Do not overwrite or discard existing local changes. Inspect `git status`, branch, and revision first.
- Follow the workspace's `AGENTS.md` and contribution instructions before editing engine code.
- Experimental engine features must be opt-in; existing API behavior must remain unchanged.
- Keep the server bound to localhost unless the current user configuration explicitly requires otherwise. Preserve existing API-key behavior.
- Never write API keys, private user content, or unredacted secrets to examples, datasets, reports, or logs.
- Avoid installing dependencies or downloading model files unless strictly necessary. Ask before costly or system-wide changes.

## 2. Phase 0 — Orient to the actual checkout

Before implementing anything:

1. Read the workspace `AGENTS.md`, the contribution guide, and the relevant sections of `docs/DETAILS.md` and `docs/HOW_IT_WORKS.md`.
2. Record the current branch, commit/revision, dirty files, installed Strata version, engine version, and build/backend in use. The workspace may be a fork or older/newer than upstream; local source is authoritative for implementation decisions.
3. Identify the real boundaries between:
   - request parsing/API serving (`serve/` and the relevant server code),
   - prompt/chat-template construction and tokenization,
   - model forward execution and final projection/logits,
   - model context/KV-state handling,
   - CPU/GPU backend code, and
   - tests and build scripts.
4. Verify that the existing server is running and responding before changing code. Use the current configured address/API key; don't launch another model process.
5. Sketch the relevant inference path in a short note with file/function references. Do not begin by reading or changing the whole codebase indiscriminately.

**Phase 0 exit criteria:** a repository map, clean understanding of the working tree, a reproducible existing-server health check, and a proposed minimal intervention point. If the current checkout differs from upstream documentation, document the difference.

## 3. Phase 1 — Build the API-only decision baseline

This phase requires no engine patch and no additional model weights. Use the existing Strata server.

### Task

Add a small script or harness component (suggested name: `tools/decision_policy/baseline.py`) that submits compact decision requests to the server's existing OpenAI-compatible endpoint. Confirm the actual API/model identifier and supported request parameters for the installed build rather than assuming them. Upstream Strata documents `POST /v1/chat/completions` and `reasoning_effort: "none"`; verify locally.

The prompt should contain only the state needed for a decision and a finite list of legal actions. Start with an action set like the following, but adapt it to the user's existing agent harness if one already defines actions:

- `CONTINUE`
- `CALL_TOOL`
- `ANSWER`
- `ASK_USER`
- `STOP`

Use deterministic settings where supported, disable reasoning, and constrain output length. Prefer a supported constrained-output/grammar feature if it works reliably with the current server; otherwise validate the returned string strictly. Treat malformed or unknown actions as failures, never as permission to execute a tool.

### Baseline state example

```json
{
  "goal": "Find and summarize the relevant documentation",
  "last_action": "SEARCH",
  "new_information_available": true,
  "answer_confidence": "medium",
  "steps_remaining": 3,
  "available_tools": ["SEARCH", "READ_DOCUMENT"]
}
```

### Measure

For a small, repeatable initial evaluation set (e.g. 50–100 examples where feasible), record:

- correctness against an explicitly reviewed target action;
- per-action precision/recall and confusion matrix;
- median and p95 end-to-end decision latency;
- prompt/input token count, generated token count, and time-to-first-token if available;
- peak VRAM and system RAM while the request is running, where measurements are available;
- model quantization/build/backend, context configuration, and machine details;
- invalid-output/fallback rate.

Run warm-up requests separately from timed requests. Use the same prompts and environment for future comparisons. Keep the input compact and stable; long agent transcripts can erase much of the benefit of short outputs.

**Phase 1 exit criteria:** a runnable baseline, a versioned test set, benchmark results, and a clear statement of whether the prompt-based route is already fast enough. Do not continue to engine changes without this reference point.

## 4. Phase 2 — Define task and training data

Create a small, documented schema for examples. A suggested JSONL record is:

```json
{
  "id": "example-0001",
  "state": {
    "goal": "Find and summarize the relevant documentation",
    "last_action": "SEARCH",
    "new_information_available": true,
    "answer_confidence": "medium",
    "steps_remaining": 3,
    "available_tools": ["SEARCH", "READ_DOCUMENT"]
  },
  "target_action": "READ_DOCUMENT",
  "label_source": "reviewed_trajectory",
  "split_group": "trajectory-014"
}
```

This is an illustrative schema, not a mandate to use these exact state fields. Reuse the user's actual orchestration vocabulary when practical.

Guidelines:

- Include routine cases and edge cases: tool results that add information, tool failure/retry, insufficient evidence, ambiguous user intent, completion, exhausted budget, and cases where the safe action is to stop or ask.
- Prefer labels reviewed against actual trajectory outcomes. Heuristic or model-generated labels must be identified as such and sampled for review.
- Split by trajectory/session or another meaningful group, not randomly by near-duplicate rows, to reduce leakage between train and test.
- Freeze an evaluation set before training. Keep it out of head fitting and threshold tuning.
- Minimize/redact sensitive state and secrets. Do not persist unnecessary conversation text.
- Clearly define any tool-execution safety rules outside the classifier. The head proposes an action; the orchestrator validates authorization and arguments.

**Phase 2 exit criteria:** a schema, initial dataset, documented label provenance, and fixed train/validation/test split. If there's no credible labeled data, report that blocker and build the data-collection path before claiming learned-policy quality.

## 5. Phase 3 — Investigate hidden-state capture in Strata

This is the core engineering feasibility question.

### What to find

Trace a normal chat request through prompt templating, tokenization, prefill/forward execution, final hidden representation, vocabulary projection/logits, and token sampling. Identify the final representation for the **last input position after processing the decision prompt but before decoding any output token**. The policy classifier should consume a single vector for that position, not the entire sequence of hidden states.

Do not assume the model is implemented exactly like stock Hugging Face Transformers or current upstream llama.cpp. Strata has a custom execution path and model-specific optimizations; inspect the actual graph and backend implementation.

### Recommended instrumentation experiment

- Add a narrowly scoped debug/experimental path to capture the one vector needed for a prompt.
- Make it opt-in (for example, a build/debug flag or local-only experimental request path); do not change ordinary inference semantics.
- Capture the vector from the same Strata model format, quantization, backend, model revision, and prompt serialization that will be used at inference.
- Prefer capturing/copying the final vector once over transferring all token/sequence activations between CPU and GPU.
- Record the vector dimension, dtype, source model/quantization, engine revision, tokenizer/template version, and example ID with the dataset metadata.
- Validate the capture point by checking dimensions, finite values, repeatability under deterministic settings, and that ordinary generation produces unchanged outputs when instrumentation is off.
- Keep diagnostics local. Don't expose raw hidden states on a public/network API by default.

### Architecture questions to answer in writing

1. Can the current forward graph expose the target representation without duplicating model state or running a second model?
2. Does Strata currently retain the final pre-projection vector, or would the graph/backend need a small change to make it available?
3. How is the assistant-prefix/chat template represented, and which final input token's state is captured? Ensure the same construction is used during training and serving.
4. Can the existing request path stop after prompt processing and head evaluation, without entering the autoregressive sampling loop or invoking MTP drafting unnecessarily?
5. What synchronization or device-to-host transfers would the capture path introduce on CUDA/HIP/Metal/SYCL backends?
6. Can the implementation be restricted to one backend initially while documenting the portability implications?

**Phase 3 exit criteria:** one end-to-end experiment exports vectors from a small set of examples using the exact running model path; the capture point is documented; normal inference remains unchanged when the option is disabled. If this requires disproportionate restructuring, stop and report the smallest viable alternative before undertaking a large refactor.

## 6. Phase 4 — Train a tiny head in PyTorch

Training should consume exported vectors and action labels. Do **not** load Qwen into PyTorch for this stage.

Start with a linear classifier:

- input: final hidden vector of dimension `D`;
- output: logits for `A` allowed actions;
- trainable parameters: approximately `D × A + A`;
- objective: cross-entropy over the labeled action;
- baseline: majority-class predictor and/or simple non-learned rules, plus the API-only model baseline.

A small MLP is a follow-up only if the linear model is not adequate and validation results justify its added complexity.

Deliver a reproducible training/evaluation script and a compact artifact format (e.g. safetensors or NumPy/PyTorch state dict if already supported by project conventions). Document class ordering explicitly so action IDs cannot be misread. Save training parameters, dataset version/hash, train/validation metrics, and the checkpoint hash.

Evaluate:

- overall accuracy and balanced accuracy;
- per-action precision/recall and confusion matrix;
- false `ANSWER`/false `STOP` and other operationally costly errors;
- calibration or confidence/abstention behavior if used;
- performance across held-out trajectories and relevant edge cases.

Do not interpret the largest softmax probability as a calibrated probability. A confidence-based fallback requires validation/calibration, and the final orchestrator must still validate the proposed action.

**Phase 4 exit criteria:** the head beats simple baselines on held-out data and does not have unacceptable failure patterns. If it does not, diagnose the data/representation/task formulation before adding model complexity.

## 7. Phase 5 — Add an opt-in inference path

Only proceed if Phase 3 proved capture feasible and Phase 4 produced a useful classifier.

### Preferred architecture

```text
Incoming decision state
        |
        v
Existing Strata prompt/template + tokenizer
        |
        v
Existing loaded Qwen3.8-Flash-Next forward pass
        |
        v
Final hidden vector for last input position
        |
        v
Tiny policy head (linear classifier initially)
        |
        v
Action scores + policy metadata
        |
        v
Orchestrator validates action, permissions, and required arguments
```

Normal chat/completions continues to use the existing output projection, sampler, and generation path. The decision route should not require a second server or a second base checkpoint.

### API and safety

- Keep it opt-in; use a documented feature flag or an experimental local-only endpoint/route. Do not silently change normal chat behavior.
- Reuse the existing server's authentication and local-access protections.
- Validate request size, action schema, and allowed actions. Return a well-defined error for malformed state or an unavailable/unloaded head.
- Never let the head bypass tool permissions, safety rules, human approval requirements, or argument validation.
- Support an abstain/fallback result where confidence is inadequate; the caller may then invoke the ordinary model route or request a stronger review.
- Keep action selection separate from action execution. Inference must not itself perform external side effects.
- Keep the trained head in a separate small artifact/config, not merged into or substituted for the base model weights.

### Implementation requirements

- Avoid changing shared model state in a way that races with normal generation.
- Understand existing server serialization/concurrency before adding stateful data buffers.
- Use proper lifetime/ownership for the captured tensor/vector and avoid copying more data than needed.
- Add unit tests for action ordering, shape/dtype mismatch, absent model/head, malformed inputs, and fallback behavior.
- Add an integration test proving that the server can serve a normal completion and a decision call using the same loaded model process.
- If the internal model context serializes requests, follow its existing locking/scheduling model; do not assume concurrent calls are safe.

**Phase 5 exit criteria:** decision inference works through the experimental path; normal generation still passes its regression checks; logs and tests demonstrate only one base model process/checkpoint is used.

## 8. Phase 6 — Benchmark, compare, and make a go/no-go decision

Compare three methods on the exact same frozen set:

1. API-only decision prompt (no engine changes beyond baseline).
2. Tiny policy head using representations from the existing Strata inference path.
3. A simple rule/majority-class baseline for context.

Measure on the same machine, model quantization, engine revision, state serialization, and backend. Separate warm-up from measured runs. Report results rather than broad claims.

| Dimension | Required report |
|---|---|
| Decision quality | Overall/balanced accuracy, per-action precision/recall, costly error examples |
| Latency | Median and p95 end-to-end latency; breakdown for prefill/forward, classifier, and decoding where possible |
| Memory | Peak VRAM/RAM, incremental memory attributable to the head, and whether any second model was loaded |
| Throughput | Sequential requests first; concurrent tests only if they represent the real intended use |
| Reliability | Invalid request rate, abstention/fallback rate, repeatability, backend/build used |
| Maintainability | Lines/modules changed, new dependencies, build/test burden, backend portability |

Do not claim speedup merely because the classifier returns one label. The full backbone forward pass may dominate latency, particularly for offloaded experts or long prompts. Report both end-to-end results and the measured time spent in the head.

Set a go/no-go threshold after measuring the prompt baseline. A useful initial goal is materially lower median and p95 decision latency with no unacceptable reduction in action quality, especially for high-cost errors. The precise quality tolerance depends on the user's agent, so surface the tradeoff rather than hiding it in a single score.

If prefill dominates and the head brings little benefit, recommend keeping the simple prompt-based fast path unless a better research direction is demonstrated. Do not proceed to layer skipping, model surgery, or a separate model as an unapproved escalation.

## 9. Expected deliverables

At minimum:

1. `docs/research/shared_backbone_decision_policy.md` — findings, architecture map, assumptions verified/unverified, tradeoffs, and go/no-go recommendation.
2. A reproducible API-only baseline script and a small benchmark report.
3. A documented decision/action schema and example dataset format.
4. If feasible, a narrowly scoped hidden-state capture experiment with sample vectors/metadata stored only in an appropriate local test-data location; never commit private user data.
5. If the experiment merits continuation, PyTorch train/evaluation code and a small classifier artifact generated from synthetic or explicitly permitted test data.
6. If and only if quality and speed warrant it, an opt-in integration, regression tests, and build/benchmark instructions.
7. A final summary listing changed files, exact commands run, tests passed/failed/not run, measured results, limitations, and next recommendation.

Do not commit generated datasets, model artifacts, benchmark results, or user information by default. Follow repository conventions and .gitignore. Ask before adding large artifacts to version control.

## 10. Suggested working order and stop conditions

Proceed in this order; do not skip directly to an engine patch:

- [ ] Phase 0: inspect checkout, instructions, runtime, and inference path.
- [ ] Phase 1: implement and measure the API-only decision baseline.
- [ ] Phase 2: define labels/schema and freeze a clean evaluation split.
- [ ] Phase 3: prove last-prompt-position hidden-state capture in Strata.
- [ ] Phase 4: train/evaluate a tiny head on exported vectors.
- [ ] Phase 5: integrate an opt-in direct decision path only if justified.
- [ ] Phase 6: compare quality/latency/memory and write the recommendation.

**Stop and report before major refactoring** if any of the following is true:

- The local checkout's architecture or license materially differs from the assumptions here.
- Capturing the required state would require duplicating or separately loading the model.
- The only available representation comes from a different checkpoint/quantization and can't be validated as compatible.
- There is no defensible labeled dataset and no feasible way to obtain one.
- The head does not beat simple baselines or does not provide a meaningful measured speed improvement.
- The required patch would destabilize regular generation, introduce unsafe concurrency, or change default behavior.

A negative result is useful. The objective is not to force a classifier into Strata; it is to discover the smallest reliable path to low-latency decisions while respecting the user's one-model memory constraint.

## 11. Reference links

Use these as starting points, but verify against the local revision:

- Strata repository: <https://github.com/Niko1221/Strata>
- Strata agent instructions: <https://github.com/Niko1221/Strata/blob/main/AGENTS.md>
- Strata implementation overview: <https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md>
- Strata detailed technical notes: <https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md>
- Strata contribution rules: <https://github.com/Niko1221/Strata/blob/main/CONTRIBUTING.md>
- Strata AI/setup guide and API examples: <https://github.com/Niko1221/Strata/blob/main/docs/AI_SETUP.md>
- Qwen3.8-Flash-Next model card: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- llama.cpp / ggml upstream: <https://github.com/ggml-org/llama.cpp>
- PyTorch: <https://pytorch.org/docs/stable/index.html>

## Final instruction to the implementing agent

Start with Phase 0 and Phase 1. Preserve the user's working tree and existing Strata setup. Provide a concise status report with evidence after those phases, then continue in order only where the measured results and code inspection justify it. Do not represent a static code-reading inference as a runtime measurement, and do not claim a performance win until it has been measured on the user's actual setup.
