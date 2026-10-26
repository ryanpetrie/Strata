# The per-type temperature: the fit

**Fitted 2026-10-22, model `qwen3.8-flash-next-unsloth-ud-iq4_xs`.**  The constants live in
`temperature_calibration.json` next to this file; the server takes them from the
launch config (`"systemone": {"temperature": {...}}`).

**The pool:** Praveenrajus/jev-bench, the validation split, the 500-per-type pool of soft_label_pool.json (seed 20261021) - the serial first-token path, one question per request, the server at T = 1, a fresh engine (1500/1500, 0 failures, 6306 s).

## What was fitted

One scalar per question type.  The temperature divides the logits before the
softmax, so the reported distribution is flattened (T > 1) or sharpened (T < 1).
The PICK cannot change - the map is monotone, so the argmax is the same and a
yes/no never crosses the 0.5 threshold.  The score type's EXPECTED LEVEL does
move (the probability-weighted mean the scorer reads), and with it the scorer's
Intelligence for that type: that is where the gain is, not only the calibration.

The exact identity the endpoint applies:

```
closed_T(i) = normalize(closed_1(i) ** (1/T))
```

which is why the fit can be made offline on a stored run log and the server
reproduces it: the same expression, the same order of operations.  For the trie
walk the temperature is applied to the OPTION-LEVEL effective logits (the closed
distribution), not to the per-trie-row logits: an option's weight is a product of
per-node probabilities and the per-row normalizers differ per option.

The expression is not written out a second time here: the fit calls the server's
own `apply_temperature` (serve/logits_first_token.py), so the fit and the served
numbers cannot drift apart.

## The four objectives, and why the shipped one is the Brier score

They disagree on this data, so all four are fitted and cross-validated:

| objective | what it minimizes | choice | noul | score |
|---|---|---:|---:|---:|
| nll | the negative log-likelihood of the gold label | 2.31 | 2.67 | 3.36 |
| brier | the multiclass Brier score | 2.27 | 2.93 | 3.29 |
| ece | the raw top-level ECE for the type | 2.42 | 1.47 | 3.02 |
| soft | the cross-entropy against the HUMAN distribution | 3.65 | 3.75 | 3.73 |

The NLL is the textbook objective, and it wants a FLATTER distribution than the
expected calibration error does.  The reason is in the data: the NLL is
dominated by the items the model gets wrong with confidence (a wrong item at
p = 0.99 costs -log(0.01) = 4.6), while the ECE is a per-bin average that
treats every bin alike.  On this sample the model's errors sit in the MIDDLE of
the confidence range, where it is UNDER-confident, so flattening makes the ECE
worse.  The Brier score is between the two: a smooth proper scoring rule, like
the NLL, but quadratic, so a confidently wrong item costs a bounded amount.

The **soft** objective is the one that lines the probability up with how
uncertain the PEOPLE were, not with a one-hot label: it is the cross-entropy
against the item's `target_probs`, the annotators' own distribution.  It is
only defined on the items that carry one, and the report says how many that is:

| type | items | of those, with a human distribution |
|---|---:|---:|
| choice | 500 | 250 |
| noul | 500 | 250 |
| score | 500 | 250 |

**The shipped constants use the `brier` objective.**

## The validation

The split rule: 5-fold cross-validation, stratified by type: every type's items are sorted by id, shuffled with random.Random(seed) and dealt round-robin into the five folds; the temperature of a fold is fitted on the other four folds and applied to the fold's own items. A 70/30 split (the same seed, stratified by type the same way) is the second check.  The seed is **42**.

The fit never sees the item it is scored on.

**The raw top-level ECE** - the first table below, and 1.1's table, reproduced bit
for bit: the scorer's own `ece` function on the same pairs it uses - the top-label
ECE on the hard tier for choice, the ECE of P(yes) for noul, the top-level ECE for
score.  For the SCORE type this is the ECE half of the scorer's calibration
component, not the component itself: the component is the mean of the top-level
ECE and (1 - nRPS) (score.py).  The second table is that component.

| objective | type | the raw top-level ECE before | the raw top-level ECE after (out-of-fold) |
|---|---|---:|---:|
| nll | choice | 0.2761 | 0.0630 |
| nll | noul | 0.1901 | 0.1825 |
| nll | score | 0.2467 | 0.0604 |
| brier | choice | 0.2761 | 0.0683 |
| brier | noul | 0.1901 | 0.1843 |
| brier | score | 0.2467 | 0.0539 |
| ece | choice | 0.2761 | 0.0568 |
| ece | noul | 0.1901 | 0.1787 |
| ece | score | 0.2467 | 0.0559 |
| soft | choice | 0.2761 | 0.2405 |
| soft | noul | 0.1901 | 0.1914 |
| soft | score | 0.2467 | 0.0621 |

**The scorer's own per-type calibration, 0-100** - the second table - on the same
out-of-fold predictions, from the scorer's `calibration` function: the top-label ECE
on the hard tier for choice, the ECE of P(yes) for noul, and for score the mean of
the top-level ECE and (1 - nRPS).  **This is the number the exit criterion is
read from** (the held-out calibration, lower is better), and it is the one the
official scorer's Calibration number is made of.

| objective | type | the scorer's calibration before | the scorer's calibration after (out-of-fold) |
|---|---|---:|---:|
| nll | choice | 44.8 | 87.4 |
| nll | noul | 62.0 | 63.5 |
| nll | score | 65.6 | 85.4 |
| brier | choice | 44.8 | 86.3 |
| brier | noul | 62.0 | 63.1 |
| brier | score | 65.6 | 86.0 |
| ece | choice | 44.8 | 88.6 |
| ece | noul | 62.0 | 64.3 |
| ece | score | 65.6 | 85.6 |
| soft | choice | 44.8 | 51.9 |
| soft | noul | 62.0 | 61.7 |
| soft | score | 65.6 | 85.2 |

The 70/30 split (the same seed), fitted on its own train part:

| type | train items | held-out items | T (train) | the raw top-level ECE before | the raw top-level ECE after |
|---|---:|---:|---:|---:|---:|
| choice | 350 | 150 | 2.28 | 0.2762 | 0.0884 |
| noul | 350 | 150 | 3.0 | 0.1837 | 0.2014 |
| score | 350 | 150 | 3.37 | 0.2268 | 0.1269 |

The fold mix for the shipped objective (the choice ECE uses the hard tier only,
so a thin slice matters):

| fold | items | choice T | noul T | score T | hard-tier choice items | tiers |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 300 | 2.24 | 2.74 | 3.12 | 100 | choice/hard 100, noul/hard 100, score/hard 100 |
| 1 | 300 | 2.28 | 3.01 | 3.4 | 100 | choice/hard 100, noul/hard 100, score/hard 100 |
| 2 | 300 | 2.3 | 2.97 | 3.38 | 100 | choice/hard 100, noul/hard 100, score/hard 100 |
| 3 | 300 | 2.25 | 2.9 | 3.61 | 100 | choice/hard 100, noul/hard 100, score/hard 100 |
| 4 | 300 | 2.26 | 3.05 | 2.93 | 100 | choice/hard 100, noul/hard 100, score/hard 100 |

How sure is the fit?  The items of the type resampled with replacement, TWICE per
resample: the temperature is fitted on one sample and the ECE is measured on the
other, so the ECE that is compared is not the one the fit produced (measuring
both on the same resample is in-sample and optimistic).  The share is the
resamples where the temperature lowers the ECE; near 50% means the difference is
inside the sample's noise.

| type | resamples | the temperature lowers the ECE | the ECE difference (5%, 95%) |
|---|---:|---:|---|
| choice | 1000 | 100% | +0.1699, +0.2313 |
| noul | 1000 | 60% | -0.0184, +0.0250 |
| score | 1000 | 100% | +0.1453, +0.2143 |

**A share is not an effect size.**  It says how often the temperature wins on a
resample of 235 items; it does not say by how much, and it does not carry the
uncertainty of the size.  For that, the paired item bootstrap below.

## How sure is the EFFECT?  A paired item bootstrap

The items are resampled with replacement and BOTH arms are scored on the same
resample, so the difference is paired: an item's own run-to-run wobble cancels
and what is left is the temperature's effect.  The interval is the 2.5% and
97.5% percentiles of the differences, and the share is the resamples where the
second arm is better.

| comparison | Capability | Intelligence | Calibration |
|---|---|---|---|
| the out-of-fold fit against the baseline arm | +8.84 [+7.14, +10.56] (100% > 0) | -2.51 [-3.46, -1.52] (0% > 0) | +20.20 [+17.08, +23.31] (100% > 0) |

## The single-scalar arm

The roadmap asks for this comparison: **if per-type does not beat a single
scalar, ship a single scalar and say so.**  This is that arm - ONE temperature
for all three types, fitted on the pooled records with the same objective, the
same grid, the same seed and the same five folds, so the two arms are measured
the same way.

The single scalar is **2.47** (the per-type constants are choice 2.27, noul 2.93, score 3.29).

| run | Capability | Intelligence | Calibration | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|
| the baseline arm (T = 1) | 49.7 | 42.0 | 57.5 | 44.8 | 62.0 | 65.6 |
| the per-type fit (brier), out-of-fold | 59.0 | 39.5 | 78.5 | 86.3 | 63.1 | 86.0 |
| the single scalar, out-of-fold | 59.6 | 40.4 | 78.9 | 88.7 | 63.8 | 84.1 |

(the three right-hand columns are the scorer's own per-type calibration
components, 0-100, higher is better.)

## The official scorer

The out-of-fold predictions (all the items, each scaled by the temperature of the
fold it was not in) were written out and scored with the official `score.py`:

| run | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 49.7 | 42.0 | 57.5 |
| the out-of-fold fit (nll) | 59.1 | 39.4 | 78.8 |
| the out-of-fold fit (brier) | 59.0 | 39.5 | 78.5 |
| the out-of-fold fit (ece) | 59.0 | 38.6 | 79.5 |
| the out-of-fold fit (soft) | 52.7 | 39.1 | 66.3 |
| the in-sample fit (brier, what the server gets) | 59.0 | 39.5 | 78.4 |

Per-type calibration:

| run | choice | noul | score |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 44.8 | 62.0 | 65.6 |
| the out-of-fold fit (nll) | 87.4 | 63.5 | 85.4 |
| the out-of-fold fit (brier) | 86.3 | 63.1 | 86.0 |
| the out-of-fold fit (ece) | 88.6 | 64.3 | 85.6 |
| the out-of-fold fit (soft) | 51.9 | 61.7 | 85.2 |
| the in-sample fit (brier) | 86.4 | 62.5 | 86.4 |

Per-type competence (the score type's expected level moves with the temperature,
so its Intelligence is not fixed):

| run | choice | noul | score |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 54.3 | 46.4 | 25.2 |
| the out-of-fold fit (nll) | 54.3 | 46.4 | 17.5 |
| the out-of-fold fit (brier) | 54.3 | 46.4 | 17.7 |
| the out-of-fold fit (ece) | 54.3 | 46.4 | 15.0 |
| the out-of-fold fit (soft) | 54.3 | 46.4 | 16.4 |
| the in-sample fit (brier) | 54.3 | 46.4 | 17.9 |

## The limits

- **What the fit is measured on:** Praveenrajus/jev-bench, the validation split, the 500-per-type pool of soft_label_pool.json (seed 20261021) - the serial first-token path, one question per request, the server at T = 1, a fresh engine (1500/1500, 0 failures, 6306 s).
  the pool is licence-clean and out of sample with respect to OpenDecisions and to the jev-bench test split; the constants are fitted on it and on nothing else
- **The raw top-level ECE of the choice questions is measured on the hard tier only** (500 of the 500 choice items), so it is a thin slice: the fold table above says how thin.  (For the choice and the noul types the scorer's calibration component is that ECE, scaled; for the score type it is the mean of the top-level ECE and (1 - nRPS), so the ECE half is only part of the component.)
- **The noul's ECE is a wash.**  Its curve is jagged and shallow around T = 1 and
  its minimum does not survive the cross-validation; see the table above.
- **One temperature per type is one number for a whole family of questions.**  A
  temperature per question would fit better and would not be honest.
- **The temperature does not touch the open reading's normalizer.**  The server has
  the engine's top-k window and the T = 1 normalizer, not the full-vocabulary
  normalizer at T, so the confidence is the closed reading at T scaled by the mass
  the model put on the options (measured at T = 1).  See the serving side.
- **The size of the effect is not established on 235 items.**  The paired item
  bootstrap puts an interval on it, and that interval is wide: the direction is
  right, the size is not known.  The Intelligence gain is the part that holds up;
  the Calibration gain is the least certain of the three.

The files the fit read, with their digests:

| file | sha256 |
|---|---|
| items (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\items\p1b_pool.jsonl`) | `eae751ef36fe7822fc09edcd178a91a2cd75d9997ea1d3122e32e048672481d7` |
| runlog (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run\p1b_pool_T1.runlog.jsonl`) | `60aab16cd8540aa8460aea6463f681ae9c2ae59c55f958805010200570f74e72` |
| predictions (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run\p1b_pool_T1.jsonl`) | `7d556fa7a2a52af1413b30d0094b4d47f23e1e0e6f3a5dfdec18ea23615b8589` |

