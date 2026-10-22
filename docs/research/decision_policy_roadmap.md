# Decision policy: implementation roadmap

**Status:** plan, written 2026-10-21. Companion to
[shared_backbone_decision_policy.md](shared_backbone_decision_policy.md) (the
measured record) and [decision_policy_literature_and_gaps.md](decision_policy_literature_and_gaps.md)
(the literature context and the ranked gap list). This document does not repeat
their content; it references them by section. It is a *roadmap*, not a spec — each
item below becomes its own scoped change when picked up.

**Organizing principle:** cheapest, highest-ROI, lowest-risk items first; training
and data work only after the cheap wins are banked and measured; speculative items
last. Every item must be measured on the 235-item OpenDecisions sample (and, where
relevant, jev-bench) before it is called done, and the result recorded in the
companion doc §10–§11 style.

---

## 0. How to read this

- **Effort** is rough (days / weeks). **Gain** is expected Capability points on the
  235-item sample, where the literature gives a number; otherwise a qualitative
  estimate.
- **Depends on** lists prerequisites (code, data, infra, or user approval).
- **Exit criterion** is what "done" looks like. Without it, an item is not done.
- Items are grouped into phases; phases are ordered. Within a phase, items can be
  done in parallel where noted.

---

## Phase 0 — Already done (for context)

These shipped and are measured; listed so the roadmap's starting line is clear.
See companion doc §§6–14.

- `/v1/systemone` endpoint (opt-in), pure module `serve/systemone.py`, tests.
- First-token logits path (`LOGITS` engine command, trie walk, open/closed
  readings, `model_picked` gate, outside-window scoring).
- Merged-per-question path (§11) — every answer carries its own probability.
- Concurrency fix (§13) — drain-to-DONE, finish-reason checks, per-question turn.
- OpenDecisions + jev-bench validation runs.

Baseline to beat: **Capability 79.0 (merged path) / 79.8 (serial, post-fix) on
the 235-item sample**, 235/235 predicted, 235 on the fast path.

---

## Phase 1 — Easy wins (days, no training, no engine change)

### 1.1 Fit per-type temperature scaling (literature doc §3.A)

**Status: PARTIAL (2026-10-21).** The temperature is fitted per type,
cross-validated and served; the constants (choice 0.62, noul 0.67, score 0.77)
are in `tools/decision_policy/temperature_calibration.json` and go into the launch
config's `"systemone": {"temperature": {...}}`. The held-out ECE is lower for the
choice (0.1313 -> 0.1224) and the score (0.1082 -> 0.0649) questions on the 5-fold
out-of-fold split, and lower for the noul on the 70/30 split (0.1791 -> 0.1721) but
not on the 5-fold out-of-fold (0.0756 -> 0.0786, a wash) — so the exit criterion is
met for two of the three types, on the out-of-fold measure. Its other half, "no
regression in Intelligence", is met: 78.0 -> 79.4 out-of-fold. Out-of-fold
Capability 81.5 against the baseline's 79.8, with a paired item bootstrap of +1.66
[-1.01, +4.41]: the direction is right, the size is not established on 235 items.
The constants were fitted on these same 235 items, so the re-run is IN-SAMPLE, and
the dataset's README says "Fit temperatures on your own data, never on
OpenDecisions" — the independent check is the jev-bench run with the human soft
labels (item 1.2). The temperatures are BELOW 1, not above: on this sample the model
is under-confident, and the textbook objective (the NLL) would cost 0.7 points. The
open reading's meaning changed: the reported confidence is no longer the model's own
probability. Recorded in the companion doc §18.

- **Effort:** 2–4 days.
- **Gain:** expected +1 to +2 Capability points (recovers part of the 2.8 lost to
  over-confidence). Improves Calibration component directly.
- **Depends on:** the 235-item sample + scorer (already in hand); a train/held-out
  split of it; the softmax site to apply `logits / T` — which is
  `serve/logits_first_token.py` (`first_token_probs`, `trie_readings`), not a
  `serve/logits_prob.py` (that file does not exist; the probability helpers are
  `tools/decision_policy/logits_prob.py`).
- **Work:** split sample; grid-search or optimize one scalar per type on train;
  validate on held-out; store constants in the launch config (per-model, like the
  `llm-logits` family's `calibration.json`); apply in the probability module.
  scipy is not installed here: stdlib + numpy. The objective that worked was the
  **Brier score**, not the NLL (the NLL's out-of-fold Capability was 79.1 against
  the Brier's 81.5).
- **Exit criterion:** held-out ECE lower than current for all three types, and a
  re-run of the 235-item sample recorded in companion doc §10 style with the new
  numbers. No regression in Intelligence.
- **Note:** the temperature here is a **calibration constant on the reported
  distribution**, not the engine's sampling temperature. The decision path never
  samples: the pick is an argmax, and this constant cannot change it.

### 1.1b Refit the temperatures on the jev-bench validation split (added 2026-10-21)

- **Effort:** 1–2 days (mostly compute: one ~1,500-item run, ~90 min).
- **Gain:** makes the shipped constants out-of-sample. The 1.1 constants were
  fitted on the 235 items they were then measured on, and the dataset says never to
  fit on OpenDecisions.
- **Depends on:** the jev-bench **validation** splits (on the Hub; the local copy
  is test-only) and `tools/decision_policy/soft_labels.py` (the converter). Pool
  and seed are fixed in the repo; only licence-verified, training-permitted
  configs are used (see `docs/research/decision_policy_training_data.md`).
- **Work:** fit on the validation pool (Brier objective; also try the soft-label
  objective, the cross-entropy against the human distribution); compare against the
  1.1 constants on **both** the OpenDecisions 235 and the jev-bench test sample,
  paired (same session, same engine state) because the run-to-run spread is ~0.6-0.9
  Capability points.
- **Exit criterion:** the out-of-sample constants are measured on the 235-item
  sample, and the shipped constants are the ones that win there **and** do not
  worsen the jev-bench TVD. If per-type does not beat a single scalar, ship a
  single scalar and say so.

### 1.2 Re-run the temperature-fitted numbers on jev-bench (soft labels)

- **Effort:** 1–2 days (mostly compute).
- **Gain:** confirms calibration improvement against a human gold standard
  (companion doc §14), not just OpenDecisions.
- **Depends on:** 1.1.
- **Exit criterion:** jev-bench TVD to human distribution lower than the current
  0.4432; reported in companion doc §14 style.
- **Note:** the noul's constant is the one the evidence contradicts (its 5-fold
  out-of-fold ECE got worse). This run decides whether the noul keeps T = 0.67 or
  goes back to T = 1.0.

### 1.3 Letter-readout benchmark (literature doc §3.F) — optional, low priority

- **Effort:** 3–5 days (prompt + parser variant; no engine change).
- **Gain:** unknown; may simplify engineering, may cost Capability. Informational.
- **Depends on:** nothing new.
- **Work:** implement an A/B/C letter readout variant of the single-question path
  behind a flag; run the 235-item sample both ways; compare Capability, fast-path
  rate, latency, and 502 rate.
- **Exit criterion:** a written comparison in the companion doc. Ship only if it
  wins on at least two of {Capability, latency, simplicity} without losing the
  other.

---

## Phase 2 — The big lever: self-distillation (weeks)

This is the item that can move Strata from "competitive with zero-shot" to "beats
Jev", per the literature (+4.8 Capability points). It is also the largest
engineering lift in this roadmap.

### 2.1 Training infrastructure prerequisite

- **Effort:** 1–2 weeks (and user approval).
- **Depends on:** user approval to install PyTorch + PEFT/TRL (or Unsloth) into
  the venv; a GPU training slot (the same 4090, but training needs it free — so
  either stop the server or use off-hours).
- **Work:** install; verify a minimal LoRA training run on a tiny dataset completes
  end-to-end on the 4090; document the command in `tools/decision_policy/README.md`.
- **Exit criterion:** one successful trivial training run, recorded, reproducible.
- **Blocker note:** nothing in Phases 2–4 can start without this.

### 2.2 Engine support for per-request LoRA application (literature doc §3.B)

**Plan written 2026-10-21: `docs/research/phase2_engine_lora_plan.md`** (the
design, the file:line map, the nine reviewable steps, the exit criterion and the
open questions). Read it before starting; the summary that matters most:
the answer position's logits come from the **verify** path, not the prefill path,
so three forward paths need the hook; the conversation cache already carries the
control vector's flag in its key, so the adapter state can join it the same way;
and a routed-expert adapter at rank 16 needs ~7.55 GB against ~58 MiB of headroom
above the 1024 MiB threshold, so **attention-only is the only MoE-compatible
option** — which is why 2.4 is an escalation and not a default. Step 0 of the plan
is a ~2-minute spike: if the engine's own output is not byte-reproducible
request-to-request, this item's exit criterion has to be restated before any code
is written.

- **Effort:** 2–4 weeks (the biggest single engineering item in this roadmap).
- **Depends on:** 2.1; familiarity with Strata's serve loop and the LOGITS/GEN
  dispatch.
- **Work:** add the ability to apply a LoRA adapter on the LOGITS path only, with
  base weights untouched on the GEN path. Design: adapter loaded alongside base
  weights (small VRAM overhead); LOGITS command applies delta; GEN command does
  not. Must interact correctly with the conversation cache and the FIFO turn
  (companion doc §13). MoE-aware adapter format (vLLM's
  `enable_mixed_moe_lora_format` is the reference).
- **Exit criterion:** a unit/integration test showing chat output is byte-identical
  with and without the adapter loaded, while a decision request uses the adapter.
  VRAM headroom re-measured (companion doc §10.8: 1082 MiB minimum today; adapter
  adds ~10–100 MB — must stay above the 1024 MiB threshold).

### 2.3 Self-distillation training run (literature doc §3.C)

- **Effort:** 1–2 weeks of compute + tuning.
- **Depends on:** 2.1, 2.2.
- **Work:** for each decision example, run Qwen with thinking on → capture the
  answer-position distribution → train an attention-only LoRA adapter (thinking
  off) to match it. No external labels. Start with attention-only (literature doc
  §5, option 1) — simplest, tooling-supported, avoids MoE router complexity.
- **Exit criterion:** adapter trained; decision path uses it; 235-item sample
  re-run recorded with new Capability. Target: beat the current 79.8; stretch:
  approach 84 (the "Releasing the Inner Jev" trained number, on a different
  backbone — treat as an upper bound, not a promise).

### 2.3b Supervised arm: train against HUMAN soft labels (added 2026-10-21)

- **Effort:** 1–2 weeks of compute + tuning (the training itself is 2.3's job; this
  is the data arm).
- **Why:** self-distillation copies the model's **own** thinking-mode distribution.
  If that mode is over-confident, the adapter inherits the over-confidence, so 2.3
  cannot close a human-agreement gap (Calibration 81.7, TVD to the human
  distribution 0.4432, 43.9% agreement with human consensus). Human labels can.
- **Depends on:** 2.1, 2.2, and the data below.
- **Data:** the jev-bench **train** splits of the three soft-label configs —
  `civil_comments` (CC0-1.0), `measuring_hate_speech` (CC-BY-4.0), `go_emotions`
  (Apache-2.0) — about 24,000 rows carrying human distributions, plus the upstream
  `google/civil_comments` (CC0-1.0, `toxicity` = the annotator share) and
  `ZefanCai/Open-Jev-v1.1` where its `target_basis` is a human annotation. The
  licences and the exclusions are in
  `docs/research/decision_policy_training_data.md`; the converter is
  `tools/decision_policy/soft_labels.py`. **Excluded:** chaosnli (non-commercial),
  mnli and yelp5 (research use), sst5 (unspecified), fever_evidence (GPL present).
  OpenDecisions stays validation-only — its card says "Please do not use for
  training".
- **Work:** train a second adapter against the human distribution (cross-entropy on
  the soft label, not on a one-hot label), then compare it with the 2.3 adapter on
  the 235-item sample **and** on held-out jev-bench test rows.
- **Exit criterion:** measured improvement over 2.3 on the sample. If the
  self-distilled adapter already matches, stop — do not ship a data-dependent
  adapter for a marginal gain.
- **Caveat, stated up front:** these are toxicity / emotion / hate-speech content
  judgments, not Jev-style decisions. They teach the readout to express human
  uncertainty on binary and small-K social judgments. That is a real but narrow
  skill, and it is the only cheap way to get calibrated probabilities without
  proprietary labels or the user's own trajectories.

### 2.4 Escalation: attention + shared expert (only if 2.3 plateaus)

- **Effort:** days on top of 2.3.
- **Depends on:** 2.3 measured and plateaued.
- **Exit criterion:** measured improvement over 2.3 on the sample. If none,
  revert to attention-only and note it.

### 2.5 Maybe: selected-expert LoRA (literature doc §5, option 3)

- **Effort:** weeks (custom training code, router analysis).
- **Depends on:** 2.3 measured, and a clear gap remaining.
- **Caveat:** the MoE²-LoRA critique — skipped experts that still fire at
  inference get no update. Only pursue if 2.4 is insufficient and the gap is
  large. Not recommended as a default.

---

## Phase 3 — Real labels / agent trajectories (weeks to months)

Only worth starting once Phase 2's self-distillation is banked and measured.
Self-distillation (2.3) already supplies training signal without external data;
this phase adds *your own workflow's* right answer.

**Amended 2026-10-21.** Human uncertainty is no longer only reachable through
this phase: 2.3b has a public, licence-clean source of it (the jev-bench train
splits). 3.1 still comes last, for a different reason than "there is no data":
it answers a different question. 2.3b asks *are the probabilities honest against
humans*; 3.1 asks *does the policy suit the people who use Strata*. The first is
cheap and unblocked; the second needs the user's own agent, and it stays last in
the user's priority order.

### 3.1 Logging path for real decisions

- **Effort:** 1–2 weeks.
- **Work:** log `(state, questions, answers, outcome, trajectory_id)` as JSONL
  from a real agent using `/v1/systemone`; freeze a trajectory-split eval set.
  Companion doc §6.5 describes this; it is "last in the user's priority order"
  and remains so.
- **Depends on:** a real agent using the endpoint (the user's own workflow).
- **Exit criterion:** a frozen eval set with at least N real decisions, documented
  split, stored outside the repo (dataset rules: do-not-train; keep out of git).

### 3.2 Train on real labels (LoRA, attention-only or +shared)

- **Effort:** 1–2 weeks (compute + tuning).
- **Depends on:** 3.1, 2.1, 2.2.
- **Work:** LoRA on the real labels; compare against the self-distilled adapter
  from 2.3 on *your* eval set, not just OpenDecisions.
- **Exit criterion:** measured improvement on the user's own eval set. If the
  self-distilled adapter already matches, stop — do not ship a data-dependent
  adapter for marginal gain.

---

## Phase 4 — Refinements (only if measured gaps remain)

### 4.1 Coherence improvements (literature doc §3.D)

- **Effort:** open-ended; weeks.
- **Work:** training with coherence losses (additivity, order-invariance) on top of
  the LoRA adapter; or architectural change to score options independently
  (Strata's merged-per-question path is a partial version).
- **Depends on:** 2.3.
- **Caveat:** additivity is unsolved across the whole literature (~7% pass). Do
  not promise a fix; measure and report honestly.

### 4.2 MoE expert selection (literature doc §5, option 3)

- See 2.5. Only if 2.4 is insufficient.

---

## Maybe / someday

These are real ideas with real value, but they are not on the critical path and
have unclear ROI. Revisit when Phases 1–3 are measured.

- **A second decision model for comparison.** Running OpenJev or SII-Decision
  locally to A/B against Strata's own output. Useful for calibration, but costs
  VRAM (the 4090 has ~1 GB free) and violates the "no second model in memory"
  spirit unless run on a separate machine. Their **training data** is the more
  useful find: SII-Decision-4B-Training-Data is the self-distillation shape in
  public, and Open-Jev-v1.1 carries typed choice/noul/score with distribution
  targets (2.3b).
- **Thinking mode at serve time ("Thinking Jev").** LLM-as-Jev shows budgeted
  test-time reasoning boosts accuracy on hard tasks. Contradicts the fast-path
  goal, but could be an opt-in "slow but accurate" mode for hard decisions. Would
  reuse the existing GEN path.
- **Multimodal decisions.** LLM-as-Jev supports image decisions zero-shot.
  Strata's vision modes (off/on/gpu/cpu) already exist; wiring them into
  `/v1/systemone` is mostly plumbing. Value depends on whether agent states
  contain images.
- **Conversation-cache slot increase (2 → 4).** Blocked today by VRAM (companion
  doc §10.8: 4 slots ≈ 484 MiB below the 1024 MiB threshold). Re-measure after
  the adapter is added; the adapter's VRAM cost may change the math.
- **Real-trajectory collection as a product feature.** Logging + opt-in telemetry
  so users can build their own labels. Only if there is demand.
- **Coherence as a published benchmark contribution.** If Strata's readout scores
  well on JevBench coherence, that is a publishable result. Not required for the
  product.

---

## Blockers and risks

1. **GPU exclusivity.** The 4090 holds ~24.1 GiB; training needs the card free.
   All Phase 2 work implies stopping the server or off-hours runs. Mitigation:
   chunked runs with restarts (companion doc §12.5).
2. **No PyTorch in the environment.** Phase 2.1 requires user approval. Until then,
   Phases 2–4 cannot start; Phase 1 is fully unblocked.
3. **Engine LoRA support (2.2)** is the biggest engineering risk. If it proves
   infeasible, the fallback is a separate, smaller process for decisions — which
   violates the "no second model in memory" constraint and should be rejected.
   Validate feasibility (a spike) before committing to 2.3.
4. **Over-confidence may persist.** Temperature fitting (1.1) helps calibration
   on held-out sets but may not fully close the gap to human uncertainty
   (companion doc §14: 43.9% agreement with human consensus). Report honestly.
5. **MoE training complexity.** Attention-only LoRA (2.3) avoids this; expert
   selection (2.5) does not. Stay on attention-only unless measured need.
6. **Data licences (new, 2026-10-21).** Training data is not a free choice, and
   one dataset's own manifest is wrong. `docs/research/decision_policy_training_data.md`
   has the record, every licence quoted from its source. The short form:
   **OpenDecisions is validation-only** ("Please do not use for training") and
   stays that way; jev-bench's **train** splits are usable config by config, but
   `chaosnli` is **non-commercial** (CC BY-NC 4.0 — jev-bench's manifest records
   it wrongly as CC BY-SA), `mnli` and `yelp5` are research-use, `sst5` is
   unspecified, and `fever_evidence` also carries GPL-3.0. CC-BY-SA sources
   (`arc_challenge`, `boolq`) are share-alike, which a released adapter would
   inherit. Every source's test split stays a yardstick.
7. **The training data is not the same shape as the task (new, 2026-10-21).** The
   public human-uncertainty sources are toxicity, emotion and hate-speech content
   judgments — binary and small-K, not Jev-style typed decisions. They teach the
   readout to express human uncertainty; they do not teach it to decide. Say that
   plainly in any write-up rather than letting the numbers imply more.

---

## Success metrics

Measured on the 235-item OpenDecisions sample (same ids, same scorer), unless
noted. Baseline: Capability 79.8, Intelligence 78.0, Calibration 81.7,
235/235 predicted, 235 on the fast path.

| Phase | Metric | Target | Result so far |
|---|---|---|---|
| 1.1 | Calibration (ECE) on held-out | Lower than current for all three types | **2 of 3** (choice and score, out-of-fold); the noul is a wash |
| 1.1 | Capability | +1 to +2 over baseline | Out-of-fold 81.5 vs 79.8 (+1.66, CI [-1.01, +4.41]); the size is not established |
| 1.1b | Capability on the 235 (out-of-sample constants) | Beat the in-sample constants | not run |
| 1.2 | jev-bench TVD to the human distribution | Lower than 0.4432 | running |
| 2.3 | Capability | Beat 79.8; stretch toward 84 | not run |
| 2.3b | Capability | Beat the self-distilled adapter (2.3) | not run |
| 2.3 | Fast-path rate | Stay at 99%+ (no regression) | 235/235 today |
| 3.2 | Capability on user's own eval set | Beat self-distilled adapter (2.3) | not run |
| All | Latency (per question, median) | No regression vs. baseline 3.12 s | 3.77 s in the 1.1 run (3.23 s -> 3.90 s in run order); the temperature's own cost is ~1 µs a question |
| All | VRAM minimum free | Stay above 1024 MiB threshold | 1064 MiB, 0 samples below |

**Rule:** no item is "done" until its numbers are recorded in the companion doc in
the §10/§11 table style, with the run date and engine state noted. Numbers taken
on a long-running engine are invalid (companion doc §12.5).