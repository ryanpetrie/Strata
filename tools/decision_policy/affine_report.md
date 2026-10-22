# The per-type affine calibration: the fit

**Fitted 2026-10-22, model `qwen3.8-flash-next-unsloth-ud-iq4_xs`.**  Roadmap item 1.4.  The
constants live in `affine_calibration.json` next to this file; **they are not in
any launch config**, and the map is not in the server - this is the fit and the
measurement, not a shipped feature.

**The pool:** Praveenrajus/jev-bench, the validation split, the 500-per-type pool of soft_label_pool.json (seed 20261021) - the serial first-token path, one question per request, the server at T = 1, a fresh engine (6306 s, 1500/1500, 0 failures).

**The objective:** the mean cross-entropy of the mapped distribution against the item's HUMAN distribution (target_probs), over the items that carry one.  The control is the TVD objective.

## The map

One scale and, where the type has an axis to shift along, one shift:

```
p'_l = normalize( p_l ** (1/T) * exp(b * s_l) )
```

| type | parameters | the coordinate `s_l` | what `b` means |
|---|---|---|---|
| noul | T, b | 0 for no, 1 for yes | `logit(P'(yes)) = logit(P(yes))/T + b`: the shift, in log-odds |
| score | T, b | the level's index 0..K-1 | a tilt along the ordinal axis, in log-odds per level: a location parameter |
| choice | T | none | **not identified**: a shift that is the same for every option is cancelled by the normalizer |

The map is invertible: `undo(q) = apply(q, 1/T, -bT)`, and the round trip is
checked on every item of every population (the worst difference is in the JSON).
With `b = 0` it is the server's own `apply_temperature`, called, not copied.
Unlike a temperature it can move the pick; every accuracy below is reported
because of that.

## The fit

5-fold cross-validation, stratified by type, seed 42; the fold rule is
`temperature.stratified_folds`, the temperature fit's own function.  A fold's
(T, b) is fitted on the other folds' items of the type that carry a human
distribution, and applied to the fold's own items.  The full-pool fit (what the
two yardsticks are measured with) is below; the per-fold fits are in the JSON.

| objective | type | items fitted | T | b | the objective at (T, b) | at T = 1 |
|---|---|---:|---:|---:|---:|---:|
| soft | choice | 250 | 3.65 | n/a | 7.7164 | 9.9116 |
| soft | noul | 250 | 2.63 | -2.290 | 0.2821 | 0.9068 |
| soft | score | 250 | 3.73 | -0.540 | 0.8502 | 1.3746 |
| tvd | choice | 250 | 1.90 | n/a | 0.6502 | 0.6776 |
| tvd | noul | 250 | 0.82 | -5.500 | 0.0864 | 0.3313 |
| tvd | score | 250 | 0.70 | -2.800 | 0.2930 | 0.4106 |

The grid: T from 0.50 to 10.00 (coarse 0.10, then fine 0.01), b from -8.0 to 8.0
(coarse 0.1, then fine 0.01).  The objective is convex in (1/T, b), so the best
grid point is the global minimum up to the grid, not a local one.

## The populations, paired

Every arm of a population comes out of that population's ONE stored run: the
T = 1 distribution is the stored one, and the affine arm is the same distribution
put through the map.  The engine, the picks, the logits and the state are shared;
only the arithmetic differs.  `TVD` is the mean total variation distance to the
human distribution over the items that carry one; `bias` is the type's own signed
measure (the noul's P(yes) gap, the score's expected-level gap in levels, the
choice's top-label mass gap); `AUC` is the probability that the gold answer
outranks a random other (binary for the noul, multiclass for the choice and the
score).

### The pool, out of fold (the fit's own population)

| arm | Capability | Intelligence | Calibration | mean TVD |
|---|---:|---:|---:|---:|
| T = 1 | 49.72 | 41.98 | 57.47 | 0.4732 |
| affine (out of fold) | 55.05 | 40.47 | 69.63 | 0.3925 |

| type | n (with humans) | TVD T = 1 | TVD affine | bias T = 1 | bias affine | AUC T = 1 | AUC affine |
|---|---:|---:|---:|---:|---:|---:|---:|
| choice | 500 (250) | 0.6776 | 0.7049 | +0.1780 | -0.2959 | 0.9195 | 0.9195 |
| noul | 500 (250) | 0.3313 | 0.1088 | +0.3047 | -0.0015 | 0.8716 | 0.8684 |
| score | 500 (250) | 0.4106 | 0.3637 | +0.2907 | +0.0010 | 0.6895 | 0.6860 |

| type | calibration T = 1 | calibration affine | competence T = 1 | competence affine |
|---|---:|---:|---:|---:|
| choice | 44.78 | 51.89 | 54.32 | 54.32 |
| noul | 61.99 | 73.72 | 46.40 | 52.40 |
| score | 65.64 | 83.28 | 25.21 | 14.69 |

The arms of one type at a time (the subset Capability is exact, from the
per-type components):

| the types mapped | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| none | 49.72 | 41.98 | 57.47 |
| choice | 50.91 | 41.98 | 59.84 |
| noul | 52.68 | 43.98 | 61.38 |
| choice+noul | 53.86 | 43.98 | 63.75 |
| score | 50.91 | 38.47 | 63.35 |
| choice+score | 52.09 | 38.47 | 65.72 |
| noul+score | 53.86 | 40.47 | 67.26 |
| choice+noul+score | 55.05 | 40.47 | 69.63 |

The paired item bootstrap of the whole-population arm against T = 1
(the same items, resampled, scored in both arms):

| metric | mean change | 95% interval | share of resamples better |
|---|---:|---|---:|
| capability | +5.35 | [+0.39, +10.36] | 0.98 |
| intelligence | -1.56 | [-6.14, +2.74] | 0.23 |
| calibration | +12.26 | [+5.38, +19.28] | 1.00 |

The paired item bootstrap of the TVD difference (affine - T = 1):

| type | mean | 95% interval | share of items better |
|---|---:|---|---:|
| choice | +0.0272 | [+0.0031, +0.0537] | 0.63 |
| noul | -0.2225 | [-0.2588, -0.1863] | 0.62 |
| score | -0.0469 | [-0.0741, -0.0178] | 0.61 |
| all | -0.0807 | [-0.1013, -0.0615] | 0.62 |

### The OpenDecisions 235 (out of sample; never fitted on)

| arm | Capability | Intelligence | Calibration | mean TVD |
|---|---:|---:|---:|---:|
| T = 1 | 78.87 | 75.90 | 81.84 | 0.2288 |
| affine (all three) | 41.53 | 35.34 | 47.73 | 0.4171 |

| type | n (with humans) | TVD T = 1 | TVD affine | bias T = 1 | bias affine | AUC T = 1 | AUC affine |
|---|---:|---:|---:|---:|---:|---:|---:|
| choice | 107 (3) | 0.2868 | 0.0667 | +0.2868 | +0.0559 | 0.9607 | 0.9607 |
| noul | 69 (6) | 0.1589 | 0.6486 | -0.1114 | -0.6486 | 0.9499 | 0.9499 |
| score | 59 (4) | 0.2902 | 0.3327 | +0.1486 | -0.3544 | 0.8983 | 0.7189 |

| type | calibration T = 1 | calibration affine | competence T = 1 | competence affine |
|---|---:|---:|---:|---:|
| choice | 78.95 | 43.32 | 85.72 | 85.72 |
| noul | 79.16 | 22.29 | 76.32 | 7.01 |
| score | 87.41 | 77.57 | 65.66 | 13.28 |

The arms of one type at a time (the subset Capability is exact, from the
per-type components):

| the types mapped | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| none | 78.87 | 75.90 | 81.84 |
| choice | 72.93 | 75.90 | 69.96 |
| noul | 57.84 | 52.80 | 62.88 |
| choice+noul | 51.90 | 52.80 | 51.01 |
| score | 68.50 | 58.44 | 78.56 |
| choice+score | 62.56 | 58.44 | 66.68 |
| noul+score | 47.47 | 35.34 | 59.60 |
| choice+noul+score | 41.53 | 35.34 | 47.73 |

The paired item bootstrap of the whole-population arm against T = 1
(the same items, resampled, scored in both arms):

| metric | mean change | 95% interval | share of resamples better |
|---|---:|---|---:|
| capability | -35.80 | [-43.26, -27.57] | 0.00 |
| intelligence | -39.67 | [-47.71, -29.42] | 0.00 |
| calibration | -31.93 | [-42.18, -20.77] | 0.00 |

### The jev-bench test sample, 683 items (out of sample)

| arm | Capability | Intelligence | Calibration | mean TVD |
|---|---:|---:|---:|---:|
| T = 1 | 55.08 | 47.18 | 62.98 | 0.4439 |
| affine (all three) | 55.28 | 42.93 | 67.63 | 0.3074 |

| type | n (with humans) | TVD T = 1 | TVD affine | bias T = 1 | bias affine | AUC T = 1 | AUC affine |
|---|---:|---:|---:|---:|---:|---:|---:|
| choice | 294 (130) | 0.4702 | 0.4139 | +0.1691 | -0.1771 | 0.9123 | 0.9123 |
| noul | 222 (100) | 0.3836 | 0.1234 | +0.3560 | +0.0330 | 0.8577 | 0.8577 |
| score | 167 (50) | 0.4963 | 0.3986 | +0.3394 | -0.0333 | 0.7446 | 0.7213 |

| type | calibration T = 1 | calibration affine | competence T = 1 | competence affine |
|---|---:|---:|---:|---:|
| choice | 68.11 | 51.25 | 64.90 | 64.90 |
| noul | 59.68 | 68.47 | 50.45 | 49.55 |
| score | 61.16 | 83.18 | 26.19 | 14.35 |

The arms of one type at a time (the subset Capability is exact, from the
per-type components):

| the types mapped | Capability | Intelligence | Calibration |
|---|---:|---:|---:|
| none | 55.08 | 47.18 | 62.98 |
| choice | 52.27 | 47.18 | 57.36 |
| noul | 56.40 | 46.88 | 65.91 |
| choice+noul | 53.59 | 46.88 | 60.29 |
| score | 56.78 | 43.23 | 70.32 |
| choice+score | 53.97 | 43.23 | 64.70 |
| noul+score | 58.09 | 42.93 | 73.25 |
| choice+noul+score | 55.28 | 42.93 | 67.63 |

The paired item bootstrap of the whole-population arm against T = 1
(the same items, resampled, scored in both arms):

| metric | mean change | 95% interval | share of resamples better |
|---|---:|---|---:|
| capability | +0.01 | [-6.99, +6.78] | 0.51 |
| intelligence | -4.48 | [-10.94, +2.03] | 0.10 |
| calibration | +4.50 | [-4.75, +13.37] | 0.84 |

## The score's configs, and the unit of its location

The score type is the one the tilt was meant for, and the per-config table is why
the type is the wrong unit for it.  `level - gold` is the mean signed distance of
the expected level to the GOLD level (the measure of §22.5), in levels; the
humans' own expected level is not the gold's either, and a single per-type
location cannot serve configs that disagree in sign.

| population | config | K | n | level - gold, T = 1 | level - gold, affine | accuracy T = 1 | accuracy affine | TVD T = 1 | TVD affine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pool | helpsteer2_helpfulness | 5 | 125 | -0.3906 | -1.3214 | 0.34 | 0.19 | n/a | n/a |
| pool | helpsteer2_verbosity | 5 | 125 | +0.5087 | -0.4587 | 0.47 | 0.48 | n/a | n/a |
| pool | measuring_hate_speech | 3 | 250 | +0.3387 | +0.0490 | 0.54 | 0.71 | 0.4106 | 0.3637 |
| jevtest | helpsteer2_helpfulness | 5 | 24 | -0.1923 | -1.1032 | 0.38 | 0.25 | n/a | n/a |
| jevtest | helpsteer2_verbosity | 5 | 24 | +0.7106 | -0.2916 | 0.46 | 0.46 | n/a | n/a |
| jevtest | measuring_hate_speech | 3 | 50 | +0.4511 | +0.0784 | 0.40 | 0.74 | 0.4963 | 0.3986 |
| jevtest | sst5 | 5 | 23 | -0.1567 | -1.1365 | 0.57 | 0.39 | n/a | n/a |
| jevtest | stsb | 6 | 23 | -0.5602 | -1.3799 | 0.43 | 0.13 | n/a | n/a |
| jevtest | yelp5 | 5 | 23 | +0.2127 | -0.4099 | 0.70 | 0.70 | n/a | n/a |

The tilt was fitted on the only score config the pool carries a human
distribution for, `measuring_hate_speech` (K = 3), and it fixes that config on
every measure.  The other configs have K = 5 (and 6 on the test), so a tilt of
0.54 in log-odds **per level** is about 1.1 levels of expected level on a 5-level
question and 0.36 on a 3-level one: the raw level index is not commensurable
across the K the type spans.  And the magnitude is not the whole story: at T = 1
the 5-level configs disagree in **sign** (`helpsteer2_helpfulness` is 0.39 levels
*below* the gold and `helpsteer2_verbosity` 0.51 *above* it on the pool), so even a
correctly scaled tilt could not serve both.  **The score's location error is
per config, not per type** - the aggregate +0.14 of §22.5 is two opposite errors
cancelling.  The normalized position k/(K-1) would fix the unit, not the sign; a
per-config location is a different item.

## The control: the TVD objective

The same map, fitted on the mean TVD to the human distribution instead of the
cross-entropy.  It is the control, not the shipped objective: TVD is a distance
and not a proper scoring rule, and its fit drives the noul to a near-degenerate
"no" (the table below).  The primary objective's numbers are the ones above.

| type | pool OOF TVD, T = 1 | soft | tvd | jev test TVD, T = 1 | soft | tvd | the 235's competence, T = 1 | soft | tvd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| choice | 0.6776 | 0.7049 | 0.6505 | 0.4702 | 0.4139 | 0.4101 | 85.72 | 85.72 | 85.72 |
| noul | 0.3313 | 0.1088 | 0.0879 | 0.3836 | 0.1234 | 0.1171 | 76.32 | 7.01 | 13.55 |
| score | 0.4106 | 0.3637 | 0.3142 | 0.4963 | 0.3986 | 0.3244 | 65.66 | 13.28 | 6.73 |

The whole-population Capability of the all-three arm: the 235 41.53 (soft) and
42.82 (tvd); the jev test 55.28 and 60.91; the pool out of fold 55.05 and 57.02 (T = 1:
49.72 on the pool, 78.87 on the 235, 55.08 on the test).

## The rule, applied

The pre-registered rule (§20.2): (a) the pool's out-of-fold calibration improves
against its own T = 1 arm; (b) the jev-bench test TVD does not get worse by more
than 0.005; (c) the 235's Capability does not fall below its T = 1 arm.  The noul
is judged separately.

| type | (a) pool calibration | (b) jev test TVD | the 235, this type's own competence | verdict |
|---|---|---|---|---|
| choice | 44.78 -> 51.89: PASS | 0.4702 -> 0.4139 (-0.0563): PASS | 85.72 -> 85.72 | does not ship |
| noul | 61.99 -> 73.72: PASS | 0.3836 -> 0.1234 (-0.2602): PASS | 76.32 -> 7.01 | does not ship |
| score | 65.64 -> 83.28: PASS | 0.4963 -> 0.3986 (-0.0976): PASS | 65.66 -> 13.28 | does not ship |

(c), the whole-population guard: the 235's Capability is 78.87 at T = 1 and 41.53
with the maps of choice, noul, score applied: **FAIL**.  Without the noul it is 62.56: **FAIL**.

**The verdict: no constant ships.**

## The limits

- **The 235's and the test sample's arms are offline re-scalings of a stored
  run**, not a live run of the map: the arithmetic is exact and the inversion is
  checked, but the engine's own reading is the stored one.  A live run would add
  the engine's run-to-run wobble (about 0.0193 mean |change| in P(yes) on the
  jev noul slice, and 1.8 Capability points on the 235, §23.5) and nothing else.
- **The map is not in the server.**  Shipping one would be a change to
  `serve/logits_first_token.py` and a new launch-config key, with its own
  byte-identity test; this item measured the map.
- **The noul's two parameters are one number's worth of evidence**: the fit is on
  250 `civil_comments` items whose humans say yes 9.2% of the time.  The second
  population that carries humans in the pool (the 250 `paws`/`strategyqa` items)
  has no human distribution, so the fit cannot see its 45.2% base rate - that is
  what the 235 and the test sample are for.
- **The choice and the score are judged on fewer human items in the pool** (250
  each), and the score's tilt is fitted on 250 items of ONE config only
  (`measuring_hate_speech`, K = 3): the other two score configs of the pool
  carry no human distribution, which is the cause of the finding below.
- **The score's location is per config, not per type** (see the per-config table
  above): the type spans K = 3 to 6, the raw level index is not commensurable
  across K, and the configs disagree in sign at T = 1.  The tilt as fitted is one
  number for a family that does not share a location, and its failure is that,
  not the model's.
- **One map per type is one map for a whole family of questions.**  A per-question
  map would fit better and would not be honest.
- **No new dependency, no engine, no server, no GPU.**  Everything above comes
  from the stored runs.

The files the fit read, with their digests:

| file | sha256 |
|---|---|
| items (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\items\p1b_pool.jsonl`) | `eae751ef36fe7822fc09edcd178a91a2cd75d9997ea1d3122e32e048672481d7` |
| runlog (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run\p1b_pool_T1.runlog.jsonl`) | `60aab16cd8540aa8460aea6463f681ae9c2ae59c55f958805010200570f74e72` |
| predictions (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run\p1b_pool_T1.jsonl`) | `7d556fa7a2a52af1413b30d0094b4d47f23e1e0e6f3a5dfdec18ea23615b8589` |
| od235_items (`C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\sample10_items.jsonl`) | `ad61ec162c4f95b4ddadc6733f2da557b4a8dbeade37f611b41282d2f89100c8` |
| od235_predictions (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run\p1b_235_T1_from_current.jsonl`) | `ab495c5ca5e73306d8e8c0af293d6fa84d2592a5a51acc60565f38586d1c60b0` |
| test_items (`C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\jevbench_sample3_items.jsonl`) | `df190fab6ab927866d254623c461be3b611c0d4ea089162cdc90e178edcf7729` |
| test_predictions (`C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run\p1b_jevbench_test_soft_T1.jsonl`) | `a789fba3758a199fe4357a10001acdaec947810f2d7d336feff76829343b63fb` |

