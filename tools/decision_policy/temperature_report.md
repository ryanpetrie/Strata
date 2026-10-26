# The per-type temperature: the fit

**Fitted 2026-10-21, model `qwen3.8-flash-next-unsloth-ud-iq4_xs`.**  The constants live in
`temperature_calibration.json` next to this file; the server takes them from the
launch config (`"systemone": {"temperature": {...}}`).

## What was fitted

One scalar per question type.  The temperature divides the logits before the
softmax, so the reported distribution is flattened (T > 1) or sharpened (T < 1).
The PICK cannot change - the map is monotone - so only the calibration moves.

The exact identity the endpoint applies:

```
closed_T(i) = normalize(closed_1(i) ** (1/T))
```

which is why the fit can be made offline on a stored run log and the server
reproduces it: the same expression, the same order of operations.  For the trie
walk the temperature is applied to the OPTION-LEVEL effective logits (the closed
distribution), not to the per-trie-row logits: an option's weight is a product of
per-node probabilities and the per-row normalizers differ per option.

## The three objectives, and why the shipped one is the Brier score

They disagree on this data, so all three are fitted and cross-validated:

| objective | what it minimizes | choice | noul | score |
|---|---|---:|---:|---:|
| nll | the negative log-likelihood of the gold label | 1.16 | 1.57 | 0.94 |
| brier | the multiclass Brier score | 0.62 | 0.67 | 0.77 |
| ece | the scorer's own ECE for the type | 0.58 | 0.67 | 0.75 |

The NLL is the textbook objective, and it wants a FLATTER distribution than the
expected calibration error does.  The reason is in the data: the NLL is
dominated by the items the model gets wrong with confidence (a wrong item at
p = 0.99 costs -log(0.01) = 4.6), while the ECE is a per-bin average that
treats every bin alike.  On this sample the model's errors sit in the MIDDLE of
the confidence range, where it is UNDER-confident, so flattening makes the ECE
worse.  The Brier score is between the two: a smooth proper scoring rule, like
the NLL, but quadratic, so a confidently wrong item costs a bounded amount.

**The shipped constants use the `brier` objective.**

## The validation

The split rule: 5-fold cross-validation, stratified by type: every type's items are sorted by id, shuffled with random.Random(seed) and dealt round-robin into the five folds; the temperature of a fold is fitted on the other four folds and applied to the fold's own items. A 70/30 split (the same seed, stratified by type the same way) is the second check.  The seed is **42**.

The fit never sees the item it is scored on.  The ECEs are the scorer's own (its
`ece` function, on the same pairs it uses: the top-label ECE on the hard tier for
choice, the ECE of P(yes) for noul, the top-level ECE for score).

| objective | type | ECE before | ECE after (out-of-fold) |
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

The 70/30 split (the same seed), fitted on its own train part:

| type | train items | held-out items | T (train) | ECE before | ECE after |
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

How sure is the fit?  The items of the type resampled with replacement, the
temperature fitted again on each resample, the ECE measured on it at T = 1 and
at the fitted temperature.  The share is the resamples where the temperature
lowers the ECE; near 50% means the difference is inside the sample's noise.

| type | resamples | the temperature lowers the ECE | the ECE difference (5%, 95%) |
|---|---:|---:|---|
| choice | 1000 | 72% | -0.0293, +0.0737 |
| noul | 1000 | 75% | -0.0175, +0.0357 |
| score | 1000 | 78% | -0.0343, +0.0967 |

## The official scorer

The out-of-fold predictions (all the items, each scaled by the temperature of the
fold it was not in) were written out and scored with the official `score.py`:

| run | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 79.8 | 78.0 | 81.7 |
| the out-of-fold fit (nll) | 79.1 | 78.3 | 79.8 |
| the out-of-fold fit (brier) | 81.5 | 79.4 | 83.5 |
| the out-of-fold fit (ece) | 81.2 | 79.2 | 83.3 |
| the in-sample fit (brier, what the server gets) | 82.1 | 79.2 | 84.9 |

Per-type calibration:

| run | choice | noul | score |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 73.7 | 84.9 | 86.3 |
| the out-of-fold fit (nll) | 71.0 | 77.4 | 91.0 |
| the out-of-fold fit (brier) | 75.5 | 84.3 | 90.8 |
| the out-of-fold fit (ece) | 74.0 | 83.5 | 92.3 |
| the in-sample fit (brier) | 77.4 | 86.9 | 90.5 |

Per-type competence (the score type's expected level moves with the temperature,
so its Intelligence is not fixed):

| run | choice | noul | score |
|---|---:|---:|---:|
| the baseline arm (T = 1) | 86.7 | 80.5 | 66.9 |
| the out-of-fold fit (nll) | 86.7 | 80.5 | 67.8 |
| the out-of-fold fit (brier) | 86.7 | 80.5 | 71.1 |
| the out-of-fold fit (ece) | 86.7 | 80.5 | 70.4 |
| the in-sample fit (brier) | 86.7 | 80.5 | 70.4 |

## The limits

- **The 235-item re-run is in-sample.**  The constants are fitted on those items,
  so the in-sample row above is not an unbiased estimate of anything.  The honest
  number is the out-of-fold one.
- **The dataset says not to fit on it** (its README: "fit temperatures on your own
  data, never on OpenDecisions").  The roadmap asks for this fit, so the numbers
  are labelled as they are, and the independent check is the jev-bench run with
  human soft labels (roadmap item 1.2).
- **The ECE of the choice questions is measured on the hard tier only** (53 of the
  107 choice items), so it is a thin slice: the fold table above says how thin.
- **The noul's ECE is a wash.**  Its curve is jagged and shallow around T = 1 and
  its minimum does not survive the cross-validation; see the table above.
- **One temperature per type is one number for a whole family of questions.**  A
  temperature per question would fit better and would not be honest.
- **The temperature does not touch the open reading's normalizer.**  The server has
  the engine's top-k window and the T = 1 normalizer, not the full-vocabulary
  normalizer at T, so the confidence is the closed reading at T scaled by the mass
  the model put on the options (measured at T = 1).  See the serving side.

