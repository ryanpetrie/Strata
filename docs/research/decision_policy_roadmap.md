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

**Status: PARTIAL (2026-10-21), and the constants are NOT shipped.** The
temperature is fitted per type, cross-validated, and the mechanism is in the code
(opt-in: `"systemone": {"temperature": {...}}`, absent = 1.0 = the old numbers
exactly). The fitted constants (choice 0.62, noul 0.67, score 0.77) are in
`tools/decision_policy/temperature_calibration.json` and are **not** put in any
launch config: item 1.2 measured them against the human labels and they were
worse on every type and every config. The held-out ECE is lower for the choice
(0.1313 -> 0.1224) and the score (0.1082 -> 0.0649) questions on the 5-fold
out-of-fold split, and lower for the noul on the 70/30 split (0.1791 -> 0.1721)
but not on the 5-fold out-of-fold (0.0756 -> 0.0786, a wash) — so the exit
criterion is met for two of the three types, on the out-of-fold measure, on the
one dataset. Its other half, "no regression in Intelligence", is met: 78.0 ->
79.4 out-of-fold. Out-of-fold Capability 81.5 against the baseline's 79.8, with
a paired item bootstrap of +1.66 [-1.01, +4.41]: the direction is right, the size
is not established on 235 items. The constants were fitted on these same 235
items, so that re-run is IN-SAMPLE, and the dataset's README says "Fit
temperatures on your own data, never on OpenDecisions". The temperatures are
BELOW 1, not above: on this sample the model is under-confident, and the textbook
objective (the NLL) would cost 0.7 points. **If** a temperature is configured, the
open reading's meaning changes — the reported confidence is then the closed reading
scaled by the option mass, not the model's own probability — which is one more
reason the constants are not shipped. Recorded in the companion doc §§18 and 19.

**What the failure means.** The endpoint's calibration is **population-specific**:
sharpening helps on OpenDecisions-style questions and hurts on jev-bench's
content judgments, where the model is already more peaked than the humans are.
The knob stays (it is the mechanism, and it is opt-in); the constants do not
ship. Item 1.1b asks whether a constant fitted on a third population helps on
both, and item 1.4 asks for the tool that fits a **bias** rather than a scale.

- **Effort:** 2–4 days.
- **Gain:** expected +1 to +2 Capability points, on the premise that the model was
  over-confident. **The premise was wrong on this sample** (it is under-confident
  there) and the gain did not transfer; see the status above.
- **Depends on:** the 235-item sample + scorer (already in hand); a train/held-out
  split of it; the softmax site to apply `logits / T` — which is
  `serve/logits_first_token.py` (`first_token_probs`, `trie_readings`), not a
  `serve/logits_prob.py` (that file does not exist; the probability helpers are
  `tools/decision_policy/logits_prob.py`).
- **Work:** split sample; grid-search or optimize one scalar per type on train;
  validate on held-out; store the constants beside the fit (per-model), and put
  them in a launch config **only** once a rule like 1.1b's says they transfer;
  apply in the probability module.
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

**Status: DONE (2026-10-22). No constant ships, and that is the finding.** The
rule was written into the companion doc **before** the numbers were read (§20.2).
The pool run answered 1,500 of 1,500 with 0 failures, every item on the fast path,
on a fresh engine. The fitted constants are all **above** 1 — the pool's
population wants a **flatter** distribution, the opposite of the OpenDecisions
sample:

| arm | choice | noul | score |
|---|---:|---:|---:|
| soft (cross-entropy against the human distribution) | 3.65 | 3.75 | 3.73 |
| brier | 2.27 | 2.93 | 3.29 |
| single scalar | 2.47 | 2.47 | 2.47 |

On the pool's **own** population the fit works, and large: out-of-fold Capability
49.72 -> **58.99** (brier), Calibration 57.47 -> **78.50**, paired bootstrap
**+8.84 [+7.14, +10.56]** and **+20.20 [+17.08, +23.31]** (Intelligence -2.51
[-3.46, -1.52], because the flattening moves the score type's expected level).
On the **jev-bench test** sample every candidate passes: the TVD falls
0.4439 -> 0.4034-0.4111 and Capability **rises** 55.1 -> 62.5-63.1.

On the **OpenDecisions 235** the same constants **fail hard**: 78.9 -> 59.6 (soft),
66.1 (brier), 69.0 (single). So the rule's (a) and (b) pass for every candidate and
**(c) fails by 9.9 to 19.3 Capability points**. **No constant ships; the endpoint
stays at T = 1.**

**The mechanism is the result:** the two populations want **opposite scalings** —
OpenDecisions wants sharpening (T below 1; the model is under-confident there),
the jev-bench content judgments want flattening (T above 1; the model is more
peaked than the people are). One scalar cannot serve both, and a constant that
helps one population and hurts the other is not one we ship. **A temperature is a
real tool on the population it is fitted on; it is the transfer that fails.** The
product answer is not a better constant but a **per-workload** calibration and an
endpoint that says so.

**The noul is a bias, confirmed on a second population.** The soft arm — the only
objective fitted against the human distribution — moves the noul's TVD by
**0.0025** while moving the choice's by 0.056 and the score's by 0.032. A
temperature cannot move a distribution's centre, and the noul's centre is wrong.
It goes to item 1.4.

- **Effort:** 1–2 days, mostly compute — but it is **four** runs, not one: the
  1,500-item pool (~87 min), the 235 with the current constants (~14 min), the
  235 live with the new ones (~14 min), the 683-item test sample (~40 min),
  plus the fit (5–25 min) and four restarts. About **2 h 50 min**.
- **Gain:** makes the shipped constants out-of-sample. The 1.1 constants were
  fitted on the 235 items they were then measured on, and the dataset says never to
  fit on OpenDecisions. After 1.2 showed the constants do not transfer to another
  population, this item's question is sharper: **is there a constant that helps on
  two populations at once, or is the endpoint's calibration population-specific?**
- **Depends on:** the jev-bench **validation** splits — **already on this
  machine**, fetched by the data branch's `download_soft_labels.py` into the shared
  data root (the copy under `…\opendecisions\jevbench_data` is still test-only) —
  and `tools/decision_policy/soft_labels.py` (the converter).
- **The pool is fixed in the repo:** `tools/decision_policy/soft_label_pool.json`,
  seed **20261021**, 1,500 items, 500 a type: choice go_emotions 250 /
  banking77 80 / ledgar 80 / massive 50 / clinc150 40; noul civil_comments 250 /
  paws 170 / strategyqa_closed 50 / strategyqa_grounded 30; score
  measuring_hate_speech 250 / helpsteer2_helpfulness 125 /
  helpsteer2_verbosity 125. 750 of them carry a human distribution. The pool keeps
  **only licences that oblige nothing beyond attribution** — boolq (CC BY-SA 3.0)
  was swapped out for 80 more paws items **before any compute**, so the pool holds
  no share-alike source at all.
- **Work:** fit on the validation pool. The **soft-label objective is the primary
  candidate** — the cross-entropy against the human distribution — because the
  Brier objective fits against a one-hot gold label, and 1.2 measured exactly
  those constants losing against the humans; the Brier arm is the control. The
  pool run at T = 1 **is** the pool's own T = 1 arm, so no extra run is needed.
  Compare against **T = 1** (what ships now), not against the in-sample
  constants: they are already measured to lose on the human labels. The arms are
  **paired without a second engine**: `tools/decision_policy/retemperature.py`
  re-scales one run, so the arms share the engine, the picks, the logits and the
  state exactly. Fit the **single scalar** arm too (one T over the pooled
  records, same objective, same folds, same no-leak rule): on the 235 it ties
  the per-type arm on Capability (81.5 against 81.5).
- **The decision rule, pre-registered before the run.** A constant ships only if
  it clears all three: **(a)** the out-of-fold calibration on the pool it was
  fitted on improves against that pool's own T = 1 arm; **(b)** the jev-bench
  test TVD does not get worse than the T = 1 arm by more than **0.005** (a fifth
  of what 1.1's constants cost there); **(c)** the OpenDecisions 235's out-of-fold
  Capability does not fall below its T = 1 arm. All three measured paired.
- **The noul is expected to fail this test and its failure does not block the
  choice and score constants.** 1.2 measured the noul's error to be a **bias** —
  93% of its TVD is a signed offset, the mean P(yes) is 0.446 against the
  humans' 0.089 — and no scalar can move a distribution's centre. A noul that
  does not clearly improve stays at T = 1 and goes to item 1.4.
- **Exit criterion:** the out-of-sample constants are measured on the 235-item
  sample **and** on the jev-bench test sample, and the shipped constants are the
  ones that clear the three tests above. A constant that helps one population and
  hurts the other is not a constant we ship. If per-type does not beat a single
  scalar, ship a single scalar and say so. **"No constant ships" is an acceptable
  and likely answer**: it would mean the endpoint's calibration is
  population-specific, and the next step is 1.4's affine fit and a
  per-population calibration, not one number.

### 1.2 Re-run the temperature-fitted numbers on jev-bench (soft labels)

**Status: DONE (2026-10-21). The exit criterion is NOT met.** The 683-item
sample, one question per request, a freshly restarted engine, 683/683 predicted,
0 failures, 683/683 on the fast path. The paired comparison (the same run, the
temperature undone offline, so the arms share the engine, the picks and the
logits) over the 280 items that carry the human distribution:

| | T = 1 | the fitted constants | change |
|---|---:|---:|---:|
| TVD to the human distribution | **0.4439** | 0.4678 | **+0.0239 (worse)** |
| Capability | **55.1** | 52.0 | **-3.1** |
| Intelligence | 46.6 | 46.6 | -0.05 |
| Calibration | **63.6** | 57.5 | **-6.1** |

**Every type and every config got worse** (choice +0.0425, noul +0.0053, score
+0.0127; chaosnli +0.0550, civil_comments +0.0053, go_emotions +0.0224,
measuring_hate_speech +0.0127). Recorded in the companion doc §19.

**The mechanism, measured rather than guessed.** The choice's model is already
**more peaked than the humans** (top-label mass 0.7862 against 0.6215), so
sharpening moves away from them. The noul's error is a **bias**: the model
over-predicts "yes" on 95 of the 100 items, the mean signed difference is +0.357
on a mean absolute 0.384 — 93% of the TVD — and its mean P(yes) is 0.446 against
the humans' 0.089. **A temperature cannot fix a bias**, and that is what the
noul's failure is. The noul's constants were already the weakest (a wash on
OpenDecisions) and are now clearly negative here: revert them to 1.0.

Two facts worth keeping, and the first one corrects an earlier draft of this file.
**A constant guess is worse than the model, not better.** On the 280 jev-bench
items that carry a human distribution, a uniform guess sits at mean TVD **0.4943**
against the model's **0.4432** (companion doc §14.2), and on the 100 noul items a
constant P(yes) = 0.5 sits at **0.4184** against the model's **0.3844**. So the
distribution carries real information; it is badly **biased**, not useless. (The
0.4220 quoted in an earlier draft was the upstream `civil_comments` 50,000-row
sample, a different population — not this one.) The second fact: a
temperature-fitted constant is **population-specific** — the same three numbers
help on OpenDecisions-style questions and hurt here.

- **Effort:** 1–2 days (mostly compute).
- **Gain:** confirms calibration improvement against a human gold standard
  (companion doc §14), not just OpenDecisions.
- **Depends on:** 1.1.
- **Exit criterion:** jev-bench TVD to human distribution lower than the current
  0.4432; reported in companion doc §14 style. **NOT met** — see above.

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

**Status: DONE (2026-10-21/22). The criterion was met on OpenDecisions and the
change still does not ship.** The A/B on the 235, both arms in one session on a
fresh engine each, no temperature: the direct readout **79.3 (77.4, 81.1)**, the
letter readout **82.7 (82.4, 83.0)**, and **0 extra engine requests against the
direct path's 113** (72 of the 235 items collide on their first token, and the
trie walk pays for them). That is Capability and simplicity won, latency not lost
(median 3.61 s against 3.95 s in-session). The per-type breakdown already warned
that it was not uniform: the letter readout is better on choice (90.7% against
89.7%) and noul (92.8% against 91.3%), and **worse on score** (72.9% against
79.7% accuracy) while its score *competence* rises, because the scorer reads the
expected level's distance.

**Then the human labels killed it.** On the jev-bench test sample (683 items, the
direct arm already measured there): the letter arm is **48.3 (44.4, 52.2)**
against the direct arm's **55.1 (46.6, 63.6)**, and its TVD to the humans is
**0.4688 against 0.4439**. The mechanism, stated carefully because the first draft
of this file overstated it: over **all 294** choice items the letter readout looks
far sharper (entropy 0.188 bits against the direct readout's 0.531, top-option mass
0.950 against 0.866), but **144 of those 294 are items the letter path cannot serve**
(more than 26 options) and answers degenerately. Over the **150 it served**, the
entropy is **0.401 → 0.368 bits**, and over the **80 served items that carry a human
distribution, 0.541 → 0.504**. So the large drop is the fallback, not the readout:
where it works, the letter path is only marginally sharper (calibration 70.7 against
78.1 on the served items, 44.9 against 68.3 over all of them), and it still loses to
the people. The verdict is unchanged; the mechanism is milder than the first draft
said. The OpenDecisions win and the jev-bench loss are the same effect as the
temperature's: **decisiveness raises the OpenDecisions metric and loses against
human uncertainty.**

So the letter readout stays an **opt-in flag** (`"systemone": {"readout":
"letters"}`), the default stays the direct readout, and the item is recorded in the
companion doc as a **population-specific** result. What would change the verdict:
letters for a question with more than 26 options currently falls back to the
decode path (144 of the 683 items, reporting a degenerate 1.0/0.0) — reading those
the direct way, then one paired run of both arms on both populations, is the
measurement that would settle it.

### 1.4 Fit an AFFINE calibration per type, not a temperature (added 2026-10-21)

**Status: DONE (2026-10-22). No constant ships, and the noul's bias is proved
movable and population-specific.** The map is one invertible exponential-family
step on the closed distribution, `p'_l = normalize(p_l**(1/T) · exp(b·s_l))`, with
`s_l` the type's own axis: for the noul, 0/1 for no/yes (so `b` is a shift in
log-odds — a scale *and* a shift); for the score, the level's index (a tilt along
the ordinal axis); for the **choice, there is no second parameter** — a shift that
is the same for every option is cancelled by the normaliser, so its map is the
scale-only temperature again. Fitted on the licence-clean pool against the **human
distribution** (the same `soft` objective as 1.1b), leak-free 5-fold CV, seed 42:
noul **T 2.63, b −2.29** (pool mean P(yes) 0.4142 → 0.1080, against the humans'
0.1096); score T 3.73, b −0.54; choice T 3.65 (which is 1.1b's soft temperature
exactly, as it must be).

**What it found:**
- **The noul's bias is real and the shift closes it out of sample** — the jev-bench
  test's noul TVD falls **0.3836 → 0.1234** (bias +0.356 → +0.033), the **AUC is
  unchanged (0.8577)**, and only one net answer of 222 moves. This is the thing a
  temperature could not do (1.1b moved the same TVD by 0.0025).
- **And the 235 rejects the same shift in sign**: there the model is already
  *under*-eager (bias −0.111), so the shift moves 26 of 69 noul answers and takes
  Capability **78.9 → 41.5**. Every type loses alone (choice 72.9, noul 57.8, score
  68.5). So the rule's (c) fails by a wide margin and **no constant ships** — the
  bias is real, movable, and **population-specific**; one per-type number cannot
  serve both. The form of a per-workload noul calibration is a **shift, not a
  scale**.
- **The score's "location error" is per config, not per type** — the brief's premise
  was wrong: at T = 1 `helpsteer2_helpfulness` sits 0.39 levels *below* the gold and
  `helpsteer2_verbosity` 0.51 *above* it, so the aggregate +0.14 is two opposite
  errors cancelling; and the raw level index is not commensurable across K (a tilt
  fitted on a 3-level config moves a 5-level question three times as far). A
  corrected map — a location on the normalised position `k/(K−1)`, per config — is
  **not run** (see 1.4b).
- **Beside 1.5**: `criteria-first` cuts the same half-A bias 0.3047 → 0.2268 with no
  235 cost; the affine shift cuts it to ~0.00 and costs the 235 21 points. **The
  prompt is the cheaper, shippable lever; the affine fit proves the level can be
  moved and that a constant fitted on one population is not the way.**
- **Effort:** 2–4 days (mostly compute; the mechanism is a small change to the
  probability module).
- **Gain:** targets the failure the temperature could not fix. The noul's error on
  the human labels is a **bias** — the model over-predicts "yes" on 95 of the 100
  items, mean signed difference +0.357 on a mean absolute 0.384 — and 93% of that
  TVD is the bias, not the scale. A temperature divides the logits; it cannot move
  a distribution's centre. **An affine calibration can**: `p ∝ softmax(z/T + b)`,
  a scale and a shift per question type.
- **Depends on:** 1.1 (the knob and the fit tooling exist), and a fitting set with
  human distributions — the jev-bench validation pool from 1.1b.
- **What the plan asked:** extend the fit to two parameters per type, fit on the
  pool, and evaluate paired on **both** the OpenDecisions 235 and the jev-bench test
  sample (`retemperature.py` generalises to an affine map); the exit criterion was
  the noul's TVD falling out-of-sample with nothing else regressing. *The result is
  the status above: the first half passes, the second fails, and the reason is a
  sign reversal between the two populations.*

### 1.4b The score's location, per config, on the normalised level (added 2026-10-22)

- **Effort:** 1–2 days, offline for the fit and the paired evaluation (the harness
  and the arms exist); a GPU run only if the fitted map is then measured live.
- **Why:** 1.4 showed the score's error is opposite in sign between its 5-level
  configs at T = 1, and that the level index is the wrong unit across K. The map to
  try is a location on the normalised position `k/(K−1)`, fitted **per config**,
  with its own pre-registration (the 1.1b/1.5 rule).
- **Exit criterion:** the score's TVD falls out-of-sample on the pool **and** the
  235's score competence does not fall; if the per-config map does not transfer
  either, the score's location is a task property, not a calibration one, and this
  line of work closes with that finding.
- **Depends on:** 1.4's fit tooling (`tools/decision_policy/affine.py`), the pool's
  stored run (`p1b_pool_T1.jsonl`), and the 235's stored score items. **Checked
  before starting, 2026-10-22, and the data is weaker than this entry first said:**
  of the pool's three score configs only **measuring_hate_speech** (K = 3, 250
  items) carries a human distribution; the pool's `helpsteer2_helpfulness` and
  `helpsteer2_verbosity` items (125 each, K = 5) carry a **gold level only**
  (`target_probs` is null), and the jev-bench sample's distributions cover
  civil_comments, go_emotions, chaosnli and measuring_hate_speech - **no
  helpsteer2 at all**. So 1.4's "0.39 levels below / 0.51 above" was measured
  against the **gold level**, not against people: there is no human distribution
  for a 5-level score question anywhere on this machine.
- **Work:** swap the raw level index for `k/(K−1)` in the tilt, fit one location
  per config, and evaluate paired without a second engine (`retemperature.py`'s
  re-scaling), reporting both populations as 1.4 did. Two arms, and they are not
  equal: **measuring_hate_speech** is fitted against the human distribution (the
  soft objective, the 1.1b rule applies in full); **the two 5-level configs** can
  only be fitted against the gold level, so their arm is a location toward the gold
  and the rule has to say so before the numbers are read. The pre-registered rule
  also has to name what happens if the two configs' locations disagree again -
  because that is the outcome 1.4 makes likely.

### 1.5 Per-type instructions — a pre-registered ablation, not a hill-climb (added 2026-10-21)

- **Effort:** 3–5 days, mostly compute. The design is small (the server's per-type
  template is 4–5 lines); the *discipline* around it is the work.
- **Gain:** unknown, and possibly the cheapest fix we have. The noul's measured
  failure is a **bias** (1.2: it over-predicts "yes" on 95 of 100 items, 93% of
  its TVD), and a bias in a template is a prompt problem before it is a
  calibration problem. If a prompt change removes the bias, it beats 1.4's affine
  shift, which beats nothing.
- **Depends on:** nothing new. The template lives in `serve/systemone.py`
  (`_question_lines`, `render_answer_prompt`) and the item's own instructions come
  from the dataset, so the variant space is the **fixed lines and the order** the
  server wraps around them — a small space, per type.
- **The method, because "hill-climbing" is the trap.** Ten variants tried on the
  235-item sample and the best one picked is **noise-chasing**: the sample's
  run-to-run spread was **1.8 Capability points** — measured, not assumed: two
  identical base runs of the 235 scored 78.9 and 80.7 and differed in **6 answers
  of 235** — so the best of ten wins by selection alone unless the effect is
  measured paired. So: a **pre-registered, paired ablation** with a design set and
  yardsticks, the same discipline as 1.1b:
  1. **Design set:** the licence-clean pool (item 1.1b's 500 a type, 250 of them
     with a human distribution). Variants are chosen there.
  2. **Yardsticks:** the OpenDecisions 235 and the jev-bench test sample. A
     variant is only *measured* there after it has been chosen, and the margin is
     pre-declared.
  3. **One axis per variant.** Each variant changes exactly one thing (the option
     order, the criteria wording, the instruction line, the preamble), so an
     effect is attributable.
  4. **Paired runs.** Same items, same session, same engine state; the official
     scorer for the numbers.
  5. **Discrimination and calibration are separate metrics.** A prompt that shifts
     the base rate moves the TVD without making the model any better at ranking
     the answers; a prompt that improves the AUC is a genuine improvement. Report
     both, and do not let a base-rate shift masquerade as calibration.
- **First target: the noul**, because it is where the bias was measured. Later
  targets: the choice and score templates (the score template says "answer with
  exactly one of these level names, lowest first" — a plausible lever).
- **Note:** item 1.3 is already a prompt experiment in flight — the lettered
  readout changes the prompt *and* the readout together. Its result tells us
  something about the prompt's shape; it is not a clean prompt ablation.
- **Exit criterion:** a written comparison (per type, per variant, per population)
  in the companion doc; a variant ships only if it wins on the design set **and**
  clears the pre-declared margin on both yardsticks, and only if it improves
  discrimination or calibration **without** hurting the other.

**Status: designed and implemented (2026-10-21/22); the probe and the arms have run.** The design and the pre-registered rule are in
`docs/research/prompt_variants.md`: the rendered prompts (byte-verified against
the renderer), the variant table (**N1** criteria order, **N2** answer order,
**N3** rule form, **N4** criteria first, **N5** an explicit decision rule, plus
the probe **P1** `swap`), and the mechanism (`"systemone": {"prompt": {type:
name}}`, absent = byte-identical, 14 variants implemented with a byte-identity
test and one test per variant).

**What the design measured, and it changes the item's shape.** The per-item SD of
the change in P(answer) between two runs of the same code is **0.0337** on the
noul, so at 250 items the **minimum detectable effect on the mean TVD is 0.006**
(0.007 Bonferroni-corrected for the best of five). The **AUC's** MDE at 250 items
is **0.165** — so on the noul, **discrimination can only be a non-inferiority
guard, never an improvement criterion**. And the obvious metric is gameable: a
**constant** P(yes) = 0 scores a mean TVD of **0.0892** against the model's
0.3842 on the jev-bench noul slice and is worthless. The TVD cannot be the
criterion alone.

**The design set is two populations, and that is what makes the base-rate check
free.** The pool's 500 noul items are 250 `civil_comments` (**9.2%** gold "yes",
with the human distribution) and 250 `paws`/`strategyqa` (**45.2%** gold "yes").
One run, one engine: a variant that merely shifts the prior gains on one half and
loses on the other.

**The probe (2026-10-22): the order axis is real, and small.** Swapping the two
sides of the noul template and the answer line (`no means … ; yes means …`,
`answer no or yes`) moves P(yes) by **+0.0212** (SE 0.0047) on the 250
`civil_comments` items, flips **15 of 250** picks (**6.0%**, against the engine's
~1% wobble), and leaves the **AUC unchanged (−0.0010)**. Half B is 6 items worse
with the ECE 0.0301 higher. So the template's order matters — the template names
"yes" first in three places — but it is a prior shift, not a discrimination
change, and it fails the pre-registered guards. **All five arms were attempted; N1,
N3 and N5 failed a guard, N2 stopped after 13 distinct items of 500 and is not in the
comparison, and one variant cleared every test:**

**N4 `criteria-first` — the criteria line before the instruction header — is the one
shipping candidate.** Per the pre-registered rule: (a) the design set's TVD to the
human distribution falls **0.0687** (bias +0.0779 toward zero), ~10× the MDE; (b) the
AUC is **−0.0013**, inside the non-inferiority margin; (c) half B loses 2 items of 250
with the ECE **+0.0104** (4 worse, 2 better - not significant); (d) the 235 is **80.7**
against a base re-run's 80.7, with the noul 63 of 69; (e) the jev-bench noul slice's
TVD falls **0.0623**; (f) the fast-path rate and the token count are unchanged (the
latency was **not isolated** - the arms shared an engine - so it is not claimed).
Three further checks, all offline from the stored runs:

| the check | base | N4 |
|---|---|---|
| pooled over every human-labelled noul item (350, two populations): ECE | **0.3362** on the soft T = 1 arm (0.3349 on the stored direct run) | **0.2726** (`-0.0677 [-0.0875, -0.0464]`) |
| the same, AUC (ranking) | 0.7921 | 0.7928 (`+0.0016 [−0.0337, +0.0364]`) |
| the 235's own noul items (69, decision-style): ECE | 0.0878-0.1042 | **0.0737**; bias −0.064 → **−0.036** |
| broad or outliers? (250 items) | — | **178 improved, 72 worsened** |

**And it costs nothing on OpenDecisions**: the Capability is **80.7**, the same as a
base re-run's, and **4 of the 235 picks move** (2 noul, 2 score), net zero on the
noul. That is what a calibration fix should look like - the answers stay, the
confidence gets honest.

**And it reproduces on a fresh engine state** (2026-10-22, the 100-item check that was
this item's one gap): a deterministic 100-item subset of the pool's 250
human-distribution noul items, base and N4 on a fresh engine each arm: the mean TVD
against the humans falls **0.3036 -> 0.2416**, a paired change of **-0.0620
[-0.0827, -0.0423]** against the full design set's **-0.0687 [-0.0859, -0.0522]** -
same sign, 90% of the size, overlapping interval, the same mechanism (the level moves,
the ranking does not: AUC +0.0010). Eight answers move, **seven toward the gold**; the
bias +0.292 -> +0.222; the gold answers 69 -> 75 of 100. Recorded in
`docs/research/n4_repro_100.md`.

**Not applied**: the branch's default is `base`, and enabling N4 is one config
key (`"systemone": {"prompt": {"noul": "criteria-first"}}`) the operator adds.

---

## The conversation cache — a one-shot decision must not evict a chat (added 2026-10-22)

**Priority: next after the Phase 1 handover.** This is the first item in this
roadmap that **touches the engine**, and it is not about the decision API's
quality: it is a product bug the user is hitting while using Strata for chat and
coding.

**The bug.** A decision request is a short prompt (under ~500 tokens) and is
**one-shot**, but it parks a conversation-cache snapshot like any other request.
With two cache slots (`--conversation-cache-slots 2`) and a parked snapshot around
240 MiB, a burst of decisions evicts the user's **chat** conversations, so the
next chat turn re-reads its whole prompt. The chat path is where the cache earns
its keep; the decision path is where it churns.

**The user's proposed mitigation:** never park a conversation under 2k tokens.
Whether that is the right lever is the plan's first question, because it would also
stop a *short chat* from reusing its first turn, and because the decision path's
**trie walk** depends on the request's own checkpoint (`ckpt=0` would make each
walk step re-read the whole prompt — the cost is `trie_costs.py`'s step count
times the prefill rate). The alternatives are a one-shot marker on the request, a
separate budget or priority for one-shot entries, or a smaller slot policy. The
letter readout (1.3) is relevant here: it needs **no** trie walk, so if it ever
became the default the cache dependency would shrink on its own.

**Plan:** `docs/research/conversation_cache_oneshot.md` (committed `3f8154e9`). **The
plan's verdict changes the item**, and one number decides it: **the parked state is
parked and thrown away.** Over the engine log's 64 lifetimes, 26,179 parks, **649
restores (2.5%)**, and **97.5% of the parks carry `reused_kv_bytes=0`** — that is,
the conversation had read from scratch when it parked. The log has the bug in a
row: nine decisions, each `0 reused`, `evictions` 5 -> 13, one per request. And the
cost is a chat turn that re-reads 4,847 tokens in **7,601 ms** instead of
`3390 reused + 125 read` in **1,534 ms**.

**Why a decision parks at all:** parking is a switch-time decision taken by the
*arriving* request (`generate.cpp:12441`, `park_current`): the previous session is
copied to host RAM as one entry, costing **one slot and ~237 MiB whatever its
length** (the smallest snapshot seen is 226.1 MiB, for a 22-token park), and
`make_room` evicts the oldest when the slots are full. A decision request parks
because the server sends `LOGITS k max_new ids` with no keys (`server.py:2168`),
`ckpt` defaults to 1, and `live_ok = prompt_cache > 0 && req_ckpt`
(`:14934`) — **the engine has no signal that the decision endpoint will never
extend that conversation.**

**The user's 2k threshold: right as a stopgap, wrong as the rule.** 98.3% of the
parks are under 2,000 tokens (median 819), so a minimum length removes nearly all
the traffic that hurts — and the flag **already exists**
(`--conversation-cache-min-tokens`), so it is a config-only change with no code and
no risk, revertable by deleting the flag. But length stands in for "will this be
continued" and the two differ: a 1,500-token **chat** will be continued and would
re-read it every turn, while a 5,000-token **decision** would still park and still
evict.

**The fix: a `park=0` request key (option b2).** The server tells the engine not
to park a one-shot conversation; the engine honours it for a conversation read
from scratch, so the **trie walk keeps its checkpoint and session reuse** (96% of
walk steps reuse the session; only 3.7% follow a restore, so `ckpt=0` — option b1 —
is rejected: it would make every walk step re-read the full prompt, tens of seconds
for the colliding items). It costs 0 latency, 0 memory, and leaves the chat's hit
rate intact at every length. The engine change is five small edits (the flag, the
parse, the guard, the `live_ok` set, and the restore path forcing a restorable
entry back to parkable) plus one argument on the server.

**A correction to this file's own VRAM claim:** the conversation cache is **host
RAM, not VRAM** (`generate.cpp:9128`, "ordinary host vectors"; the admission line
reads `/proc/meminfo`). Two slots are ~474 MiB of an **8,192 MiB** budget, so the
slot count — not VRAM — is the binding constraint, and the "4 slots ≈ 484 MiB below
the 1,024 MiB threshold" note under "Maybe / someday" should be re-checked for that
framing.

**Status: MEASURED (2026-10-22) and the fix works.** Built (CUDA; the first build was
`CMAKE_CUDA_ARCHITECTURES=75` and its sm_75 SASS **died on the 4090** with
`dequant launch: unspecified launch failure` — rebuilt for **89**), and measured with
three runs an arm, a fresh engine each, alternating:

| run | `park` | chat 5: `N = M reused + K read` | reuse | evictions | parked | `skip parking (park=0)` |
|---|---|---|---:|---:|---:|---:|
| A1 | off | 7,467 = **0** + 7,467 | 0.0% | **19** | 2 | 0 |
| A2 | off | 7,477 = **0** + 7,477 | 0.0% | **19** | 2 | 0 |
| A3 | off | 7,458 = **0** + 7,458 | 0.0% | **19** | 2 | 0 |
| B1 | on | 7,453 = **7,423** + 30 | **99.6%** | **0** | 1 | 25 |
| B2 | on | 7,459 = **7,429** + 30 | **99.6%** | **0** | 1 | 25 |
| B3 | on | 7,461 = **7,430** + 31 | **99.6%** | **0** | 1 | 25 |

**The walk's reuse is intact in both arms** — its steps read
`1,877 = 1,867 reused + 10` and `1,879 = 1,878 reused + 1` (99.5% / 99.9%) — so the
fix does not break the trie walk. All 25 requests answered in every run. The 235,
one run per arm: **79.8** (before) and **78.3** (after), both 235/235 on the fast
path, inside the 1.8-point spread (2 and 6 answers moved against the stored arm).

**The plan's mechanism sentence was wrong, and the measurement says so.** A walk
step does **not** continue the live session: it resumes a checkpoint 9-10 tokens back
and therefore calls `park_current` on the base request's conversation too — which is
why the key had to go on the steps as well, and why the 25 skips are 20 decision
switches plus 5 walk steps. The key on the steps is still required and works; the
plan's explanation of *why* was not.

**Two forced departures from the recipe**, both recorded: (i) the config copies set
`--resident-budget-gib 20` instead of 36, because with 36 plus the user's other
session (9.6 GB) the host had **261 MiB** free and the engine refused **every** park
(`skip parking (physical RAM admission; need 671 MiB plus 2560 MiB floor)`) — arm A
could not have parked at all; (ii) the 20 decisions were 1,815-1,881 tokens, not
under 500, because the shipping `--conversation-cache-min-tokens 1024` already skips
short decisions, so the parked traffic at the shipping setting is the over-1k kind.

**Order of work:** (1) the plan (done); (2) the config stopgap, which needs one
restart to take effect and is the user's call; (3) the `park=0` key (done, CUDA
built; **HIP not built** — no ROCm on this machine); (4) the server sends it, behind
`STRATA_DECISION_PARK` (done); (5) the measurement (done, above); (6) **not done:**
make it the default, and set the minimum-length threshold back to 0. That needs the
engine rebuilt from the cache branch and the user's OK.

**Exit criterion: MET** — the chat's fifth turn reuses 99.6% of its prefix with the
key on (0% with it off), the evictions are 0 (19), the walk's reuse is intact, and
the 235 is 235/235 on the fast path inside the spread. The latency half was **not
measured** (the card was shared, so a latency number would be meaningless) and is not
claimed.

**Risks, updated:** an engine change means a build and a GPU slot; the CUDA build
must be for the card's own architecture (sm_89 here — sm_75 builds and then dies at
run time); HIP cannot be built here. The `reused_kv_bytes=0` question is answered in
the companion doc §25 (it is the K/V a restored conversation already held, not how
much of the cache a snapshot reuses).

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

**Step 0 is DONE (2026-10-22) and it produced a split verdict** (report:
`docs/research/phase2_lora_spike.md`, script `tools/decision_policy/lora_spike.py`):

- **The chat output IS byte-identical across repeats.** The `T` lines (the
  generated ids) matched byte for byte in every comparison — 24 requests over
  three separate engine instances, with unrelated requests and a chat-shaped
  conversation in between. Order does not matter; a conversation in between does
  not matter. **The exit criterion's A1/A2 survive as written** — "chat output is
  byte-identical with and without the adapter loaded" is testable.
- **The LOGITS rows are never byte-identical** — 0 of 7 in each probe. The top-1
  id is the same, its probability moves 2.4e-4 to 2.8e-4, the max |Δlogit| is
  0.89-2.9, 71 of the top-1024 ids change, and the total variation is at most
  2.5e-4. The cause is measured, not guessed: **the expert cache warms while the
  requests run** (hit rate 76.6% -> 96.2%), so the same request is not computed
  from the same experts in the same grouping. It is not the CPU share.
- **So the plan's proposed restatement ("the answer position's logits row is
  byte-identical") must be struck** — no implementation can have that property.
  A4 (the adapter's effect on the logits) becomes a **tolerance** against the
  measured noise floor: the top-1 probability moves ~2.8e-4 here, and the
  companion doc records 0.004 median / 0.11 max on the 235-item set.
- **A deterministic mode exists in practice**: once the cache stopped moving, two
  consecutive requests gave byte-identical rows. If A4 needs exactness, that is
  the route (pin the expert placement, then measure).
- **The adapter format has two orientation bugs that would have produced silently
  wrong adapters**: both A and B need their bytes transposed from PEFT's storage
  (the plan flags only B; getting A wrong gives a max diff of 1339 and a cosine of
  -0.10 - plausible-looking and wrong), and the plan mixes two storage conventions
  for `lora_b`. Fix the format document before step 1.

**Step 1 is DONE (2026-10-22): the format, the loader and the converter, host-side, no GPU.**
`tools/lora_pack.py` writes an adapter (and `--validate` reads it back and compares it with the
source exactly); `include/strata/core/lora_pack.hpp` + `src/core/lora_pack.cpp` read it against the
model's own layout table, with every refusal of the plan's 3.5; `src/core/lora_pack_test.cpp` and
`tools/test_lora_pack.py` pin the **same twelve fp16 values** from both ends, so the orientation
cannot drift between the converter and the reader. The format document's orientation paragraph and
its `lora_b` convention are corrected (plan 2.1). **What it does not do:** convert a full-model
adapter - the converter maps the QSA projections and the shared expert and refuses every other
module **by name**, because the 36 GDN layers' Hugging Face module names are still open question 1
of the plan's 7. That is the planned behaviour: a guessed name is a silently wrong adapter.

Renegotiate **only the logits-row framing** with the maintainer, not the item.

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
  `google/civil_comments` (CC0-1.0, `toxicity` = the annotator share, measured: every
  value is exactly k/n) and `ZefanCai/Open-Jev-v1.1`, whose human-annotated rows
  (82,045, WANLI, CC BY 4.0) are **one-hot** — shapes and hard labels, not
  uncertainty, so they do not serve this item's purpose. The licences and the
  exclusions are in `docs/research/decision_policy_training_data.md`; the
  converter is `tools/decision_policy/soft_labels.py`. **Excluded:** chaosnli
  (non-commercial), mnli and yelp5 (research use), sst5 (unspecified),
  fever_evidence (GPL present). OpenDecisions stays validation-only — its card
  says "Please do not use for training".
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
- **Conversation-cache slot increase (2 → 4).** **Not the fix** (2026-10-22): the
  cache is **host RAM, not VRAM** — two slots are ~474 MiB of an 8,192 MiB budget,
  so the slot count is the binding constraint and the old "4 slots ≈ 484 MiB below
  the 1,024 MiB threshold" framing was wrong. And four slots would still be flushed
  by twenty decisions (the cache item's plan, option d1), because a decision parks
  a slot it never reuses. Fix the parking, not the count.
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

**The noise floor, measured (2026-10-22):** two base runs of the 235
scored **78.9 and 80.7** and differed in **6 answers of 235**. (One of the two is
the T = 1 inverse of a run that had been served with §18's temperatures, not a fresh
run; the inversion is exact and checked, but it is derived, so read the 1.8 as an
upper bound on the wobble.) So the sample's
run-to-run spread is **around 1.8 Capability points**, not the 0.6-0.9 an earlier
draft assumed, and a single-run difference below that is not a result. The paired
comparisons — one run, re-scaled offline to another setting — are the only
decisive ones, which is why 1.1b and 1.5 are built that way.

| Phase | Metric | Target | Result so far |
|---|---|---|---|
| 1.1 | Calibration (ECE) on held-out | Lower than current for all three types | **2 of 3** (choice and score, out-of-fold); the noul is a wash — but see 1.2 |
| 1.1 | Capability | +1 to +2 over baseline | Out-of-fold 81.5 vs 79.8 (+1.66, CI [-1.01, +4.41]); the size is not established |
| 1.1 | The constants ship | — | **No.** They lose on the human labels (1.2). The knob ships; the constants do not |
| 1.1b | Capability on the 235 (out-of-sample constants) | Beat the **T = 1** arm on both populations, per the three pre-registered tests | **No constant ships.** (a) and (b) pass for every candidate; (c) fails by 9.9 to 19.3 Capability points on the 235 |
| 1.2 | jev-bench TVD to the human distribution | Lower than 0.4432 | **NOT met**: 0.4439 -> 0.4678; Capability 55.1 -> 52.0 |
| 1.3 | The letter readout ships | Wins on two of three without losing the other | **Does not ship.** Met on OpenDecisions (82.7 vs 79.3, 0 extra requests); loses on jev-bench (48.3 vs 55.1). Opt-in flag stays |
| 1.4 | The noul's TVD to the human distribution | Lower than **0.3844** (the shipping T = 1 arm; not 0.3897, which was the failed T = 0.67 arm) | **0.1234 out-of-sample** on the jev-bench test — and the same shift takes the 235 from 78.9 to 41.5 (wrong in sign there). **No constant ships** |
| 1.4b | The score's TVD / the 235's score competence | Per-config location on `k/(K−1)`; TVD down and competence not down | not run |
| 1.5 | A prompt variant ships | The pre-registered rule's (a)-(f) | **N4 `criteria-first` clears all six** (TVD -0.0687, AUC -0.0013, 2 items on half B, 235 = a base re-run, jev noul -0.0623), and **reproduces** at -0.0620 [-0.0827, -0.0423] on a fresh-engine 100-item run. Not applied; one config key away |
| Cache | A chat survives a decision burst; the walk's reuse intact | `M5 ≥ 0.9 × N5`, `evictions = 0` for the fix; `M5 = 0`, `evictions ≥ 18` without | **MET**: 99.6% reuse (0%), 0 evictions (19), the walk 99.5%/99.9% intact, the 235 235/235. Not the default yet |
| Cache | The decision path's latency | No regression beyond the margin | **not measured** (shared card); not claimed |
| 2.3 | Capability | Beat 79.8; stretch toward 84 | not run |
| 2.3b | Capability | Beat the self-distilled adapter (2.3) | not run |
| 2.3 | Fast-path rate | Stay at 99%+ (no regression) | 235/235 today |
| 3.2 | Capability on user's own eval set | Beat self-distilled adapter (2.3) | not run |
| All | Latency (per question, median) | No regression vs. baseline 3.12 s | 3.77 s in the 1.1 run (3.23 s -> 3.90 s in run order); the temperature's own cost is ~1 µs a question |
| All | VRAM minimum free | Stay above 1024 MiB threshold | 1064 MiB, 0 samples below |

**Rule:** no item is "done" until its numbers are recorded in the companion doc in
the §10/§11 table style, with the run date and engine state noted. Numbers taken
on a long-running engine are invalid (companion doc §12.5).

---

## What Phase 1 established (2026-10-22)

Five interventions were measured against two populations - the OpenDecisions
235-item sample, and the jev-bench test sample, which carries **human**
distributions. They agree on one thing, and it is the result:

| the intervention | OpenDecisions 235 | jev-bench (human labels) |
|---|---|---|
| a per-type temperature fitted on OpenDecisions (1.1) | **+1.7** Capability out-of-fold | **-3.1** Capability, TVD 0.4439 -> 0.4678 (1.2) |
| a per-type temperature fitted on the jev-bench validation pool (1.1b) | **-9.9 to -19.3** | the Brier/single arms +7.4 to +8.0 and the soft (primary) arm +1.0, TVD 0.4439 -> 0.4034 |
| the letter readout (1.3) | **+3.4** Capability | **-6.9**, TVD 0.4439 -> 0.4688 |
| the prompt's option order (1.5 probe) | not measured | a prior shift of +0.0212 P(yes), AUC unchanged |
| **the prompt's criteria-first (1.5, N4)** | **80.7, the same as a base re-run; 4 picks of 235 move, net zero on the noul** | **the pooled noul ECE falls 0.3362 -> 0.2726 [-0.0875, -0.0464], AUC unchanged** |

**For the interventions that change how decisively the endpoint answers, the
OpenDecisions number moved opposite to the human-label agreement.** The sharpening
temperature and the letter readout raised OpenDecisions and moved the distribution
further from the people; the pool's flattening constants did the reverse. Three of
those three rows, two populations, two mechanisms — not a law, but enough to say the
two numbers measure different things and must be reported together.

**N4 breaks the pattern, and that is the point of it.** It barely changes an answer
(4 of 235) and
still moves the distribution toward the people: it is the first intervention here
that improves the human-label agreement with **no OpenDecisions cost at all**. A
template that makes the model read the definition before the question fixes a
calibration error without touching the decision — which is exactly what the
temperature was asked to do and could not.

The letter readout is the clearest case of the *other* effect, with a correction the
review caught: over **all 294** jev-bench choice items the direct readout answers
with 0.531 bits against the letter's 0.188, but 144 of those are items the letter
path cannot serve and answers degenerately — over the **150 it served** it is
0.401 -> 0.368 bits, and over the **80 served items that carry a human
distribution**, 0.541 -> 0.504. So the letter readout is only marginally sharper
where it works, and the large drop is the fallback. The verdict does not change (it
loses on the humans), but the mechanism is milder than the first draft said.

So three things are true and they are all unwelcome:

1. **The endpoint is miscalibrated, and the direction of the error depends on the
   population.** OpenDecisions wants a sharper distribution than the model
   produces; the content judgments want a flatter one. **No single constant can
   serve both**, which is why 1.1b's pre-registered rule returns "no constant
   ships" and why that is a finding rather than a failure.
2. **The noul's error is a bias, not a scale.** It over-predicts "yes" on 95 of
   100 items, and the soft objective moves its TVD by 0.0025 while moving the
   choice's by 0.056. A temperature cannot move a distribution's centre; that is
   item 1.4's job (an affine scale and shift). The prompt experiment (1.5) found
   that the **order** levers (N1, N2) only shift the prior and fail the base-rate
   guard, while **N4 `criteria-first`** — recasting the definition as the thing the
   model reads first — improves the calibration on three populations without moving
   an answer.
3. **The OpenDecisions Capability is a weak proxy for honest probabilities.** It
   rewards decisiveness, and every decisive change measured here cost the
   human-label agreement. **Report both, always** — the capability number alone
   would have shipped two changes that make the probabilities worse, and would have
   missed the one prompt change that makes them better.

**What ships from Phase 1, then, is not a number but the machinery and the
honesty:** the per-type temperature knob (opt-in, default off) with a fit harness
that takes any pool, any objective, a leak-free cross-validation and paired
bootstrap intervals; the letter readout as an opt-in flag; the prompt-variant
mechanism; the offline re-scaling tool that makes paired comparisons possible
without a second engine; and this record, which says what was measured, on what,
and which of it failed.