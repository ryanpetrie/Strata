# The per-type temperature: the fit

**Fitted 2026-10-21, model `qwen3.8-flash-next-unsloth-ud-iq4_xs`.**  The constants live in
`temperature_calibration.json` next to this file; the server takes them from the
launch config (`"systemone": {"temperature": {...}}`).

**The pool:** jgeuter/OpenDecisions v1.0.1, the 235-item 10% sample (seed 42) - the serial first-token path, every item on the fast path.

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
| nll | the negative log-likelihood of the gold label | 1.16 | 1.57 | 0.94 |
| brier | the multiclass Brier score | 0.62 | 0.67 | 0.77 |
| ece | the raw top-level ECE for the type | 0.58 | 0.67 | 0.75 |
| soft | the cross-entropy against the HUMAN distribution | 4.36 | 1.2 | 2.22 |

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
| choice | 107 | 3 |
| noul | 69 | 6 |
| score | 59 | 4 |

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
| nll | choice | 0.1313 | 0.1452 |
| nll | noul | 0.0756 | 0.1128 |
| nll | score | 0.1082 | 0.0622 |
| brier | choice | 0.1313 | 0.1224 |
| brier | noul | 0.0756 | 0.0786 |
| brier | score | 0.1082 | 0.0649 |
| ece | choice | 0.1313 | 0.1301 |
| ece | noul | 0.0756 | 0.0824 |
| ece | score | 0.1082 | 0.0498 |
| soft | choice | 0.1313 | 0.3462 |
| soft | noul | 0.0756 | 0.1002 |
| soft | score | 0.1082 | 0.1912 |

**The scorer's own per-type calibration, 0-100** - the second table - on the same
out-of-fold predictions, from the scorer's `calibration` function: the top-label ECE
on the hard tier for choice, the ECE of P(yes) for noul, and for score the mean of
the top-level ECE and (1 - nRPS).  **This is the number the exit criterion is
read from** (the held-out calibration, lower is better), and it is the one the
official scorer's Calibration number is made of.

| objective | type | the scorer's calibration before | the scorer's calibration after (out-of-fold) |
|---|---|---:|---:|
| nll | choice | 73.7 | 71.0 |
| nll | noul | 84.9 | 77.4 |
| nll | score | 86.3 | 91.0 |
| brier | choice | 73.7 | 75.5 |
| brier | noul | 84.9 | 84.3 |
| brier | score | 86.3 | 90.8 |
| ece | choice | 73.7 | 74.0 |
| ece | noul | 84.9 | 83.5 |
| ece | score | 86.3 | 92.3 |
| soft | choice | 73.7 | 30.8 |
| soft | noul | 84.9 | 80.0 |
| soft | score | 86.3 | 77.1 |

The 70/30 split (the same seed), fitted on its own train part:

| type | train items | held-out items | T (train) | the raw top-level ECE before | the raw top-level ECE after |
|---|---:|---:|---:|---:|---:|
| choice | 75 | 32 | 0.5 | 0.1696 | 0.0374 |
| noul | 48 | 21 | 0.6 | 0.1791 | 0.1721 |
| score | 41 | 18 | 0.88 | 0.1783 | 0.1031 |

The fold mix for the shipped objective (the choice ECE uses the hard tier only,
so a thin slice matters):

| fold | items | choice T | noul T | score T | hard-tier choice items | tiers |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 48 | 0.68 | 0.81 | 0.95 | 15 | choice/easy 2, choice/hard 15, choice/judge 3, choice/standard 2, noul/easy 3, noul/hard 3, noul/judge 6, noul/standard 2, score/easy 2, score/hard 7, score/judge 2, score/standard 1 |
| 1 | 48 | 0.88 | 1.01 | 0.78 | 13 | choice/easy 3, choice/hard 13, choice/judge 2, choice/standard 4, noul/easy 2, noul/hard 6, noul/judge 4, noul/standard 2, score/easy 1, score/hard 8, score/judge 2, score/standard 1 |
| 2 | 47 | 0.74 | 0.5 | 0.76 | 9 | choice/easy 6, choice/hard 9, choice/judge 5, choice/standard 1, noul/easy 2, noul/hard 4, noul/judge 3, noul/standard 5, score/easy 1, score/hard 6, score/judge 3, score/standard 2 |
| 3 | 47 | 0.5 | 0.7 | 0.57 | 6 | choice/easy 7, choice/hard 6, choice/judge 5, choice/standard 3, noul/easy 4, noul/hard 3, noul/judge 4, noul/standard 3, score/easy 2, score/hard 6, score/judge 2, score/standard 2 |
| 4 | 45 | 0.5 | 0.62 | 0.7 | 10 | choice/easy 6, choice/hard 10, choice/judge 5, noul/easy 3, noul/hard 5, noul/judge 5, score/easy 2, score/hard 6, score/judge 3 |

How sure is the fit?  The items of the type resampled with replacement, TWICE per
resample: the temperature is fitted on one sample and the ECE is measured on the
other, so the ECE that is compared is not the one the fit produced (measuring
both on the same resample is in-sample and optimistic).  The share is the
resamples where the temperature lowers the ECE; near 50% means the difference is
inside the sample's noise.

| type | resamples | the temperature lowers the ECE | the ECE difference (5%, 95%) |
|---|---:|---:|---|
| choice | 1000 | 64% | -0.0327, +0.0625 |
| noul | 1000 | 67% | -0.0180, +0.0320 |
| score | 1000 | 68% | -0.0447, +0.0769 |

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
| the out-of-fold fit against the baseline arm | +1.66 [-1.01, +4.41] (89% > 0) | +1.46 [+0.72, +2.46] (100% > 0) | +1.87 [-3.25, +7.06] (77% > 0) |
| the temperature run against the same run with the temperature undone | +1.36 [-1.10, +3.79] (87% > 0) | +1.07 [+0.44, +1.72] (100% > 0) | +1.64 [-3.16, +6.46] (75% > 0) |

The paired control: the arm that ran with the temperature, and the SAME run
with the temperature undone (the closed reading put back through
`normalize(closed_T ** T)`) - the same engine, the same picks, the same
logits, the temperature's arithmetic removed.  The difference between the two
is the temperature and nothing else.

| run | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| the same run with the temperature undone | 78.9 | 75.9 | 81.8 |
| the run with the temperature | 80.1 | 77.0 | 83.1 |

## The official scorer

The out-of-fold predictions (all the items, each scaled by the temperature of the
fold it was not in) were written out and scored with the official `score.py`:

| run | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 79.8 | 78.0 | 81.7 |
| the out-of-fold fit (nll) | 79.1 | 78.3 | 79.8 |
| the out-of-fold fit (brier) | 81.5 | 79.4 | 83.5 |
| the out-of-fold fit (ece) | 81.2 | 79.2 | 83.3 |
| the out-of-fold fit (soft) | 68.0 | 73.5 | 62.6 |
| the in-sample fit (brier, what the server gets) | 82.1 | 79.2 | 84.9 |

Per-type calibration:

| run | choice | noul | score |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 73.7 | 84.9 | 86.3 |
| the out-of-fold fit (nll) | 71.0 | 77.4 | 91.0 |
| the out-of-fold fit (brier) | 75.5 | 84.3 | 90.8 |
| the out-of-fold fit (ece) | 74.0 | 83.5 | 92.3 |
| the out-of-fold fit (soft) | 30.8 | 80.0 | 77.1 |
| the in-sample fit (brier) | 77.4 | 86.9 | 90.5 |

Per-type competence (the score type's expected level moves with the temperature,
so its Intelligence is not fixed):

| run | choice | noul | score |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 86.7 | 80.5 | 66.9 |
| the out-of-fold fit (nll) | 86.7 | 80.5 | 67.8 |
| the out-of-fold fit (brier) | 86.7 | 80.5 | 71.1 |
| the out-of-fold fit (ece) | 86.7 | 80.5 | 70.4 |
| the out-of-fold fit (soft) | 86.7 | 80.5 | 53.2 |
| the in-sample fit (brier) | 86.7 | 80.5 | 70.4 |

## The limits

- **What the fit is measured on:** jgeuter/OpenDecisions v1.0.1, the 235-item 10% sample (seed 42) - the serial first-token path, every item on the fast path.
  the constants are fitted on these same items, so a re-run of the sample is IN-SAMPLE; the honest number is the out-of-fold one below
- **The 235-item re-run is in-sample.**  The constants are fitted on those items,
  so the in-sample row above is not an unbiased estimate of anything.  The honest
  number is the out-of-fold one.
- **The dataset says not to fit on it** (its README: "fit temperatures on your own
  data, never on OpenDecisions").  The roadmap asks for this fit, so the numbers
  are labelled as they are, and the independent check is the jev-bench run with
  human soft labels (roadmap item 1.2).
- **The raw top-level ECE of the choice questions is measured on the hard tier only** (53 of the 107 choice items), so it is a thin slice: the fold table above says how thin.  (For the choice and the noul types the scorer's calibration component is that ECE, scaled; for the score type it is the mean of the top-level ECE and (1 - nRPS), so the ECE half is only part of the component.)
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
| items (`C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\sample10_items.jsonl`) | `ad61ec162c4f95b4ddadc6733f2da557b4a8dbeade37f611b41282d2f89100c8` |
| runlog (`C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\fix_serial.runlog.jsonl`) | `778db6e107621865bb776a956558cc4b1d240bc2e0ef1759e8b099e31759c453` |
| predictions (`C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\fix_serial.jsonl`) | `6ca1b3126673affcb4dbf566a58ec160d87106d5dfc713e779a5aff2a3942727` |

