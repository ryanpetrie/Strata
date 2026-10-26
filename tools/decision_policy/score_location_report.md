# 1.4b: one location per score config (the run of 2026-10-22)

The map: `p'_k = normalize(p_k * exp(b_c * (u_k - 0.5)))`, `u_k = k/(K-1)`, `T = 1`; one `b_c` per
config, fitted on the licence-clean pool's stored T = 1 run and transferred to the jev-bench
sample's items of the same config.  The rule is companion doc 26.1, committed before this ran.

| config | K | n (pool) | b_c | pool T=1 | pool out-of-fold | the pool, paired |
|---|---:|---:|---:|---|---|---|
| `measuring_hate_speech` | 3 | 250 | -2.203 | TVD 0.4106 | TVD 0.3299 | -0.0807 [-0.1012, -0.0590] |
| `helpsteer2_helpfulness` | 5 | 125 | 4.308 | level 1.044 | level 0.982 | -0.062 [-0.141, 0.020] |
| `helpsteer2_verbosity` | 5 | 125 | -4.249 | level 0.752 | level 0.648 | -0.104 [-0.197, -0.006] |

| config | n (jev sample) | T=1 | the map | the sample, paired | passes the rule? |
|---|---:|---|---|---|---|
| `measuring_hate_speech` | 50 | 0.496 | 0.396 | -0.100 [-0.141, -0.055] | **yes** |
| `helpsteer2_helpfulness` | 24 | 0.928 | 0.973 | 0.045 [-0.107, 0.191] | **yes** |
| `helpsteer2_verbosity` | 24 | 0.790 | 0.464 | -0.326 [-0.509, -0.155] | **yes** |

## The 235, for information

The 235's 59 score items at T = 1: mean |expected level - gold| 0.306 levels, argmax accuracy 0.780. No arm is applied to them: none of their configs is a jev config, so a fitted b would arrive there as a foreign constant (26.1).

## Every fold's b

| config | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| `measuring_hate_speech` | -2.261 | -2.357 | -1.904 | -2.137 | -2.342 |
| `helpsteer2_helpfulness` | 5.291 | 5.257 | 3.773 | 3.801 | 3.486 |
| `helpsteer2_verbosity` | -4.059 | -4.516 | -4.197 | -4.067 | -4.443 |
