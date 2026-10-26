# N4 `criteria-first` on 100 items: the paired change reproduces

**2026-10-22.** This is the addition §23 needs ("The prompt probe and the noul's arms"):
the design set's N4 evidence is one run per arm on 250 items, and this asks whether the
**paired change** comes back on a fresh engine state with a smaller, independent draw.
Neither run is in git; both arms' artefacts are under
`<opencode temp>\decision_policy_training\run\`.

## The design set, for comparison

On the design set's half A (the pool's 250 `civil_comments` noul items with a human
distribution), §23 measured the base template at a mean TVD of **0.3313** and N4 at
**0.2626**: a paired change of **-0.0687 [-0.0859, -0.0522]**, with the bias cut from
+0.3047 to +0.2268. N4 is a reorder - the `yes means ... ; no means ...` criteria line is
emitted before the `[q | noul] ...` header - so it costs no tokens.

## This run

**The sample.** The same half A (250 items), sorted by id, then
`random.Random(42).sample(items, 100)` - deterministic, seed 42 (the repository's own seed
for folds and splits). The draw is **100 distinct items**, gold "yes" **11/100 (11.0%)**
against the design set's 9.2%, the humans' mean P(yes) **0.1227** against 0.1096. The rule
is in `cache_measure\make_subset.py`; a systematic every-5th pick landed on 3 gold "yes" of
100 and was rejected before the arms ran, for representativeness, not for a result.

**The arms.** The server from `feat/phase1-integration` (`C:\tools\strata-wt-int`), a copy
of the live config with `"systemone": {"enabled": true, "diagnostics": true}` and no
temperature and no readout, and the live `engine\strata.exe`. Base: no `prompt` key. N4:
`"prompt": {"noul": "criteria-first"}`. One fresh engine start per arm; 100 items each, no
resumption across arms. The harness is `tools/decision_policy/prompt_probe.py`; the
predictions are `p1b_cache_base100.jsonl` and `p1b_cache_n4_100.jsonl`.

**The readouts** (the harness's own, and the paired bootstrap over the 100 items, 10,000
resamples, seed 42, in `cache_measure\n4_stats.py`):

| | base | N4 | change |
|---|---:|---:|---:|
| the mean TVD to the human distribution | **0.3036** | **0.2416** | **-0.0620 [-0.0827, -0.0423]** |
| the ECE of P(yes) against the gold label | 0.3044 | 0.2349 | -0.0695 |
| the bias (mean P(yes) - the human's) | +0.2917 | +0.2223 | -0.0694 |
| the AUC against the humans' hard label | 0.9285 | 0.9295 | +0.0010 |
| the answers (P(yes) >= 0.5) against the gold | 69/100 | 75/100 | +6 |

The answers moved on **8 of 100 items, 7 of them toward the gold and 1 away**; the
harness's flip rate is 8.0% against the engine's ~1% run-to-run wobble, and its swap gap
(mean P(yes) base minus N4) is **+0.0695**.

## Does it reproduce?

**Yes, in direction, in size and in mechanism.**

* **Direction and size.** -0.0620 against -0.0687: the same sign, 90% of the magnitude, and
  the two intervals overlap. The 100-item interval is wider, as a quarter of the items must
  make it.
* **The mechanism, not just the number.** N4 moves the *level* of P(yes) down (bias -0.0694)
  and leaves the ranking where it was (AUC +0.0010). That is the design set's signature (bias
  -0.0779, AUC -0.0013), not a base-rate shift: the answers that move go toward the gold, 7
  to 1.
* **The base is a fresh engine state, not the stored one.** Its TVD is 0.3036 where the
  design set's base was 0.3313 - a different 100 items and a different engine state - so the
  two changes are not a re-reading of one run.

## The limits

* 100 of 250 items, one run per arm, one model, one card. The absolute TVDs of this sample
  are not the design set's (a different draw); only the paired change is compared.
* The ECE is against the gold label, the only hard label these items have; the design set
  reports its calibration as the bias and the TVD to the human distribution, which are the
  numbers the rule reads. The two ECE arms are the same items and the same label, so the
  change is comparable even though its level is not the design set's.
* The engine was slow and shared; no latency is reported, and none is needed for a
  template comparison (the prompt lengths are unchanged).
* **Not run:** the 250 items again (the design set stands), the other variants (N1, N2, N3,
  N5), the 235's rule (d) on this sample, a second model.
