# The Phase 2 training arms: self-distillation (2.3) and human soft labels (2.3b)

## The pre-registration, written 2026-10-23 before any adapter exists

**Status: a plan, not a result.** Nothing here has been trained. It fixes the two arms'
data, recipe, evaluation, margins and stop rules **before** the first adapter is trained,
because the program's other items (1.1-1.5, 1.4b) each learned that a rule written after
the numbers is a rule that can be read backwards. The engine hook the arms need does not
exist yet: **2.2's step 2 (the LoRA kernel) is the blocker**, and it is weeks of work. This
document is what makes those weeks have a target.

**Read with** `phase2_engine_lora_plan.md` (2.2's design and file map),
`phase2_training_env.md` (2.1's environment, installed and smoke-tested on the 4090),
`decision_policy_training_data.md` (the license record every data choice below comes from),
`decision_policy_literature_and_gaps.md` §3.C and §3.E (why these two arms), and
`shared_backbone_decision_policy.md` §27 (the metric family this rule reuses).

---

## 0. In one page

* **2.3, self-distillation.** The model's own answer distribution **with thinking on** is
  the teacher; a LoRA adapter makes the model's **thinking-off** answer position match it.
  No external labels. At serve time it is still one forward pass.
* **2.3b, the human arm.** The same recipe, but the target is **people's answer
  distribution** - the jev-bench train split's soft labels for three configs, 24,000 rows
  in all. This is the arm that can close a human-agreement gap; self-distillation copies
  whatever the model already believes.
* **The shared control.** A third adapter trained with the same recipe and steps on
  **shuffled targets**. It measures how much a LoRA shifts the readout by existing at all,
  so a 2.3/2.3b gain cannot be "any adapter helps".
* **What decides.** The license-clean pool's 750 human-labeled items are the
  **development** set (disjoint from the training split by split); the **jev-bench test
  sample** (683 items, 280 with a human distribution) and the **OpenDecisions 235** are the
  yardsticks. The rule is §6; the margins are the measured per-type run-to-run noise, the
  same arithmetic as `prompt_variants.md` §6 and companion doc §27.
* **What stops it.** 2.2's hook, the GDN layer naming (2.2's open question 1), the VRAM
  headroom, and - the one nobody has costed yet - **the teacher capture**: 2.3 needs one
  thinking-on pass per training row, and §3 sets a 100-row pilot before the size is fixed.

---

## 1. The two arms, and why both

| arm | target | data | what it can fix | what it cannot |
|---|---|---|---|---|
| **2.3** self-distillation | the model's own thinking-on distribution at the answer position | the states only (no labels) | the gap between what the model can work out and what it says in one pass | a gap between the model and **people**: it copies the model's own bias, as the literature says |
| **2.3b** human soft labels | people's distribution for the same question | the jev-bench train split's soft labels: `civil_comments` 8,000 (noul), `go_emotions` 8,000 (choice), `measuring_hate_speech` 8,000 (score) | a human-agreement gap, per type, on the configs that have human labels | the configs that have no human distribution (HelpSteer2's 5 levels, most of the 235's routing questions) - the adapter carries no target for them |

The two are not alternatives to each other; they are the two things a decision adapter can
learn. The program's own result says why 2.3 alone is not enough: the endpoint's error on
the human-labeled noul items is **93% a bias** (companion doc §19), and a bias copied from
the teacher stays. 2.3b is the only arm whose target is outside the model. **The order is
2.3 first**, because it needs no labels and it prices the teacher; 2.3b is evaluated against
it, not only against the base.

---

## 2. The data

### 2.1 The training set (measured from the manifest, not assumed)

`<data root>\decision_policy_training\<config>\train.jsonl`, jev-bench's own train split:

| config | question type | rows | with a human distribution | license (record §7) |
|---|---|---:|---:|---|
| `civil_comments` | noul | 8,000 | 8,000 | CC0 1.0, nothing owed |
| `go_emotions` | choice (28 options) | 8,000 | 8,000 | Apache-2.0, nothing owed |
| `measuring_hate_speech` | score (3 levels) | 8,000 | 8,000 | CC BY 4.0, attribution |
| **total** | | **24,000** | **24,000** | |

The counts are the manifest's (`manifest.json`, 22 files, 32,328 rows, 44,966,196 bytes);
the soft-label share is 100% for these three. **The train split is the only split a training
run may train on** (record §7); the validation split is what selection reads, and the test
split is the yardstick, never trained on.

**What is excluded, and why** (record §5 and §7): `chaosnli` (non-commercial),
`mnli`, `yelp5`, `sst5` (unusable or unspecified licenses), `fever_evidence`, `sms_spam`
and `stsb` (terms not verified), **OpenDecisions** (its card says "Please do not use for
training"), and Open-Jev-v1.1's 1,500 `customer-control-v1` rows (TypeSafe terms not read).
A released adapter is trained on the list above only, so it owes attribution and no
share-alike (record §5.2).

**The states for 2.3.** The same 24,000 rows, unordered - the arm needs the state and the
question, not the label. If the first adapter underfits, the **second** pass may add
`ZefanCai/Open-Jev-v1.1`'s 145,639 usable train rows (147,139 less the 1,500
`customer-control` rows) as **shapes only** (its generated content is CC0; its WANLI rows
keep CC BY 4.0 attribution), never its targets. That is a second
decision, written here so it is not invented later.

### 2.2 What the adapter is trained on at the answer position

The teacher's and the human target are both a **distribution over the question's answer
options** at the answer position - the same object the endpoint reports (`noul`: P(yes);
`choice`: the options; `score`: the levels). The prompt is the one the endpoint sends
(the question, its criteria, the answer-line prefix), the same rendering the direct readout
uses, so the adapter is trained on the input it will see. This also means the prompt
variants and the readout flags are **frozen at their shipping values** for the run: N4 on
the noul, letters off, temperature absent. An adapter fitted on another template is a
different experiment.

### 2.3 The split, and what the model is selected on

* **Train:** the train split, 24,000 rows (§2.1).
* **Develop (choose the checkpoint, the rank, the epoch):** the 750 human-labeled items of
  the license-clean pool's **validation** split (1,500 items; 250 a type), the same pool
  1.4b and 1.5 used, disjoint from the train split by split. It is also the set the rule's
  (a) is measured on.
* **Yardsticks (after the checkpoint is chosen):** the jev-bench **test** sample and the
  235. Same order as every other item in this program: choose on the develop set, confirm
  on the yardsticks, never the other way round.
* **The civil_comments caveat:** jev-bench's train and validation rows are the upstream
  **validation** split, and the upstream repeats comments across splits (record §5.7.1:
  71 of 8,492 comments appear in the upstream test split). The measured overlap **between
  the training rows and the jev test sample is 0 shared texts** (record §7), so the
  yardstick is clean; the develop set shares texts with the train rows only in the way the
  upstream does, which is why the develop number is a *selection* number and the yardsticks
  are the claim.

---

## 3. The teacher capture (2.3), and its pilot

Self-distillation needs one **thinking-on** answer-position distribution per training row.
That is the arm's real cost, and it has not been measured yet.

**The pilot, first.** 100 rows, stratified by config (34/33/33), on a fresh engine, with
thinking on, capturing the answer-position row exactly as the endpoint's `diagnostics`
does. It measures the per-row wall time and the thinking-token count. **The full capture's
size is then fixed by that number**, with a target of **6,000 rows (2,000 a type)** if the
pilot puts it under 20 hours, and 3,000 if it does not. The pilot is not optional: a
capture whose size was guessed is a run that either wastes a day or underfits.

**What could make the pilot fail** (any of these stops 2.3, not the whole plan):
the thinking-on pass exceeds the model's context on the long states (`civil_comments`'
long comments are the longest training rows); the answer-position row cannot be read with
thinking on without changing the endpoint's code path; or the per-row cost puts even 3,000
rows beyond a week.

**The capture is one engine run, not one per arm**, and it is stored as
`{id, state, question, target}` rows outside the repository. A capture is reusable by any
later adapter, which is why it is worth doing once and keeping.

---

## 4. The adapter and the recipe

Chosen by `phase2_engine_lora_plan.md` §3.3 and its VRAM finding, not re-opened here:

* **Attention-only LoRA**, rank **16**, on the QSA projections (the modules the step-1
  converter maps); the shared expert's 60.9 MB does not fit at rank 16. **Routed-expert
  adaptation is out** at this rank (~7.55 GB against ~58 MiB of headroom over the 1024 MiB
  threshold); it is roadmap 2.5's escalation, only if 2.3 plateaus.
* **The same recipe for all three adapters** (2.3, 2.3b, the shuffled control): the same
  optimizer, learning rate, batch size, steps, seed and rank. The **only** difference is the
  target tensor. A different recipe for the winner is a second experiment.
* **The loss** is the soft cross-entropy (KL) between the model's answer-position
  distribution and the target, over the question's answer options only - the same
  renormalization the endpoint reports. A hard-label loss is not used for either arm; the
  whole point is the distribution.
* **Steps and epochs:** fixed by the develop set, with the base control trained for the
  same number of steps. The training set is small (24,000 rows) and the adapter is small;
  the epoch count is expected to be 1-3, and the develop set's curve decides it.
* **The control** is **not** a formality. A LoRA changes the logits before it learns
  anything; the shuffled-target adapter measures that shift. **If the control moves the
  develop set's TVD by more than half of an arm's gain, the arm's gain is not attributable**
  and the arm is reported as inconclusive.

---

## 5. What is measured

Per type, on the develop set and each yardstick, the same family as §27:

* **the mean TVD to the human distribution** (the primary; lower is better),
* **the mean signed difference** for the noul (the bias; toward zero),
* **the mean expected level** for the score (levels from the humans' distribution),
* **the top-label mass**, reported (the shape),
* **the ranking**: the AUC of P(yes) against the humans' hard label (noul) and the accuracy
  against the humans' majority (choice, score),
* **the ECE** of the reported confidence,
* **the 235's** official Capability, Intelligence, Calibration and per-type accuracy,
* **the fast-path rate**, the median prompt tokens and the latency (reported, not judged).

Chat is measured too, because the adapter must not change it: the 2.2 exit criterion is
**chat output byte-identical with and without the adapter**, and the decision path using it.
An arm that moves a chat answer is withdrawn whatever its decision numbers.

---

## 6. The rule (pre-registered)

An adapter ships only if **all** of the following hold. The margins are the measured
run-to-run noise of each type (the per-item SD of the change in P(answer) between two runs:
noul **0.0337**, choice **0.0348**, score **0.0255**, `prompt_variants.md` §5.2), turned
into a minimum detectable effect at the develop set's n = 250 by the same arithmetic as §27
(2.802 x SE, and 3.08 x SE after the Bonferroni correction over the two arms):
**0.0066 for the noul, 0.0068 for the choice, 0.0050 for the score.**

> **(a) The develop set, per type.** The mean TVD to the human distribution falls by more
> than the type's margin (above). For the noul, the mean signed difference also moves toward
> zero by more than the noul's margin. For the score, the mean signed level difference
> moves toward zero by more than the score's margin **and** the arm beats the mean-level
> prior's mean absolute error (the baseline that closed 1.4b's 5-level line).
>
> **(b) The ranking, per type.** The AUC (noul) or the accuracy against the humans' majority
> (choice, score) does not fall by more than **0.05 AUC** or **3 items of 250**. A
> non-inferiority guard: an adapter that flattens the distribution can lower the TVD and
> lose the picks, which is exactly what a temperature did in 1.2.
>
> **(c) The control, per type.** The arm's TVD gain must exceed the shuffled-target
> control's gain by more than the type's margin. **This is the check that separates
> "training helped" from "a LoRA shifted the logits".**
>
> **(d) The jev-bench test sample.** Its 280 human-labeled items' mean TVD does not rise
> by more than **0.005**, and its official Capability does not fall by more than **1.8**
> points, against a base arm run in the same session.
>
> **(e) The 235.** Capability does not fall by more than the measured **1.8-point** spread
> against the base arm's own run, which must itself land within 1.8 of the stored **79.8**
> (78.0, 81.7), or the run is invalid and is repeated. The noul's accuracy on its 69 items
> does not fall by more than **2 items**.
>
> **(f) The cost and the contract.** The fast-path rate does not fall below the base arm's;
> the median prompt tokens do not rise; the chat output is byte-identical; the VRAM stays
> above the 1024 MiB threshold at serve time. Latency is reported, not judged (no isolated
> timing run is part of this protocol).
>
> **Between the arms.** 2.3b's margin is measured against **2.3**, not only against the
> base: the human arm must beat the self-distilled one on the develop set by the type's
> margin. If it does not, **the answer is that the teacher's distribution is enough on this
> model and the data-dependent adapter is not worth shipping** (2.3b's own exit criterion,
> roadmap §2.3b). If neither arm clears (a)-(c), the finding is that a LoRA at this scale
> does not move the readout, and the program stops Phase 2 rather than spending 2.5's
> VRAM on an escalation with no measured cause.

**If the develop set and a yardstick disagree, the arm does not ship**; it is recorded as
population-specific with both numbers, as N1 and the letter readout were. No margin is
re-chosen after a number is seen, and no arm is re-run on a different item order.

---

## 7. The order of work, and what blocks it

1. **2.2 step 2: the LoRA kernel** (weeks). Device tables, the flag, the delta kernel,
   CUDA built; HIP cannot be built on this machine (no ROCm) and is recorded as a limit.
   The GDN layers' Hugging Face names (2.2's open question 1) must be settled for the
   converter, or the adapter excludes those layers by name - a decision the plan already
   allows but the training code has to know.
2. **The 100-row teacher pilot** (§3), a GPU day, once the hook exists. It fixes the
   capture's size, and a capture is one run.
3. **The 2.3 adapter**, trained, evaluated on the develop set, then the yardsticks if it
   clears (a)-(c).
4. **The 2.3b adapter**, same recipe, the human targets, compared with 2.3.
5. **The control**, same recipe, shuffled targets, run beside both (it can be trained
   before them: it does not need the capture).

**Nothing in steps 2-5 starts before step 1's exit criterion**, which is 2.2's: chat output
byte-identical with the adapter loaded, the decision path using it, the VRAM measured.

---

## 8. What this document does not claim, and the limits

* **No number here comes from a training run.** The data counts are the manifest's and the
  license record's; the margins are the measured engine noise, not an adapter's.
* **The 24,000 rows are three configs of one dataset.** A noul adapter trained on
  `civil_comments` may overfit to toxicity; the develop set's 250 noul items are the same
  population, so the 235 (11 families, rule (e)) is the guard that matters. The same holds
  for the choice (`go_emotions`' 27 emotions plus neutral (28 classes)) and the score (`measuring_hate_speech`'s 3
  levels). **A 5-level human distribution does not exist on this machine** (1.4b), so 2.3b
  has no target for the score's 5-level configs; its 250-item develop slice is 3-level.
* **The adapter cannot carry a target it was not trained on.** For the 235's routing
  questions (11 families, none in the training set) the adapter is a shift with no fitted
  meaning; that is what rule (e) is for.
* **The teacher is the same model.** 2.3 inherits the teacher's bias by construction; its
  value is the reasoning it internalizes, not a correction.
* **One model, one quantization, one card.** The training environment is 2.1's
  (`C:\tools\strata-train`, torch 2.11.0+cu128), the engine is the shipping one, and the
  adapter format is 2.2's. A result here is a candidate for this model, not a general claim.
* **The license record governs.** No OpenDecisions row, no chaosnli, no `customer-control`
  row, no test split. The adapter's release owes the attribution in record §5.3.
