#!/usr/bin/env python3
"""tools/decision_policy/affine.py - fit a per-type AFFINE calibration against the humans.

ROADMAP ITEM 1.4.  The per-type temperature (temperature.py, items 1.1 and 1.1b) is
one scalar per question type, and it moves a distribution's SCALE, not its centre.  On
the noul questions the measured error is a centre: the model over-predicts "yes" on 95
of the 100 human-labelled items, the mean signed difference is +0.357 against a mean
absolute 0.384 (93% of the TVD), and the model's mean P(yes) is 0.446 against the
humans' 0.089 (companion doc §19.6).  A temperature cannot fix that, and the soft arm
of the 1.1b refit moved the noul's TVD by 0.0025 while moving the choice's by 0.056
(§20.8).  This tool fits TWO parameters per type, in the type's own units.

THE MAP, PER TYPE, AND WHY THE THREE ARE NOT THE SAME
=====================================================

All three are one exponential-family map on the CLOSED distribution - the same object
the temperature scales, i.e. the distribution over the item's own labels:

    p'_l  =  normalize( p_l ** (1/T) * exp(b * s_l) )

with `T` a scale (the temperature) and `b` a shift along `s_l`, a per-label coordinate
in the type's natural unit:

  noul   s(no) = 0, s(yes) = 1.  So, over the two answers,

             logit(P'(yes)) = logit(P(yes)) / T + b

         which is exactly "a scale AND a shift" on the logit, and `b` is in
         log-odds.  This is the type the item exists for: its error is the centre.

  score  s(level) = the level's index (0..K-1).  So `b` is a TILT along the ordinal
         axis, in log-odds per level: a location parameter for an ordinal
         distribution.  The score's measured error is a location error - the expected
         level is off by +0.135 levels - and a tilt moves the expected level.  It is
         the smooth, invertible way to move an ordinal distribution's location; a
         literal translation p'_k = p_{k-d} is not smooth and is undefined at the
         ends.  (A shift that is the SAME for every label is a no-op under the
         normalization - softmax is shift-invariant - so the second parameter has to
         be attached to the label's coordinate.  That is why the choice's b below is
         not a parameter at all.)

  choice s(l) = 0 for every option.  With no order among the options there is no
         coordinate to attach a shift to, and a constant shift over all options is
         cancelled by the normalizer: **a shift is not identified for the choice**.
         The choice's measured error is a SHAPE error (the model's top-label mass is
         0.7862 against the humans' 0.6215), which is what a scale fixes.  So the
         choice's map is the scale-only temperature, and the honest report says so:
         two parameters per type where two are meaningful, not two everywhere.

The map is INVERTIBLE, exactly, and the inverse is the same map:

    p_l  =  normalize( p'_l ** T * exp(-b * T * s_l) )   =  apply( p', T' = 1/T, b' = -b*T )

so a run carries every arm inside it, which is what makes the offline paired
comparison (retemperature.py's trick) exact.  The script checks the round trip and
reports the worst absolute difference over every item of every population.  When
b = 0 the map IS the temperature, and it is applied by the server's own function
(serve/logits_first_token.py::apply_temperature, imported through temperature.scale),
so the b = 0 arm cannot drift from the served one.  The affine part is NOT in the
server: this item measures the map, and shipping it would be a separate change.

Unlike a temperature, an affine map CAN move the pick: the noul can cross 0.5 and the
score's argmax can move.  That is the point - the noul's answer is what is wrong - and
it is why every population's accuracy is reported beside its TVD.

THE OBJECTIVE
=============

Fitted against the HUMAN distribution (the item's `target_probs`), not a one-hot gold
label, because that is the thing the jev-bench yardstick measures and the thing §19
showed the one-hot-fitted constants to lose on.  Two objectives are fitted:

  soft  the cross-entropy against the human distribution (the PRIMARY, and the same
        `soft` objective as item 1.1b's primary arm).  Cross-entropy is a strictly
        proper scoring rule: its minimum is the human distribution itself, and the
        comparison with 1.1b's soft arm is then like for like.
  tvd   the mean total variation distance to the human distribution (the CONTROL).
        TVD is the metric the rule reads, but it is not a proper scoring rule - it is
        a distance, and per-coordinate it is minimized by a median rather than by the
        distribution - so it is the control and not the shipped objective.

The module docstring of temperature.py explains why the temperature fit has four
objectives; this fit has two because it is against the humans only.

THE VALIDATION
==============

The same leak-free 5-fold cross-validation as the temperature fit, and the same
function: `temperature.stratified_folds` (seed 42, every type's items sorted by id,
shuffled, dealt round-robin).  A fold's (T, b) is fitted on the other four folds'
items OF THE TYPE THAT CARRY A HUMAN DISTRIBUTION (the objective is undefined
without one), and applied to the fold's own items.  Out-of-fold predictions cover
every item; the fit never sees the item it is scored on, and the code asserts it.
The paired item bootstrap is temperature.py's own (`paired_bootstrap`) for the
scorer's numbers, and the same resampling for the TVD difference.

THE POPULATIONS, AND THE RULE
=============================

  the pool        the licence-clean jev-bench validation pool (item 1.1b's 1,500
                  items, 250 per type with a human distribution) - the fit's own
                  population, reported OUT OF FOLD;
  the 235         the OpenDecisions sample - out of sample, and never fitted on;
  the jev test    the 683-item jev-bench test sample - out of sample.

The pre-registered rule is the companion doc's §20.2, unchanged: a constant ships only
if (a) the pool's out-of-fold calibration improves against the pool's own T = 1 arm,
(b) the jev-bench test TVD does not get worse than T = 1 by more than 0.005, and (c)
the OpenDecisions 235's Capability does not fall below its T = 1 arm, all paired.  The
noul is judged separately and its failure does not block the other types (§20.2).  The
rule and the expected outcome were written down before the numbers (companion doc §24,
committed before this run).

USAGE
  python tools/decision_policy/affine.py \
    --items <p1b_pool.jsonl> --runlog <p1b_pool_T1.runlog.jsonl> \
    --predictions <p1b_pool_T1.jsonl> \
    --out-json tools/decision_policy/affine_calibration.json \
    --out-report tools/decision_policy/affine_report.md \
    --out-dir <the run directory>

Pure standard library (it imports temperature.py, which imports the server's
apply_temperature).  No engine, no server, no network.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temperature as tp  # noqa: E402  (the fold rule, the scorer, the transform, the bootstrap)

TYPES = ("choice", "noul", "score")
SEED = 42
FOLDS = 5
T_MIN, T_MAX = 0.5, 10.0
B_MIN, B_MAX = -8.0, 8.0
COARSE_T_STEP = 0.1
COARSE_B_STEP = 0.1
FINE_T_STEP = 0.01
FINE_B_STEP = 0.01
OBJECTIVES = ("soft", "tvd")
# The two parameters are only meaningful where the type has a coordinate to attach
# the shift to.  The choice does not (see the module docstring), so its map is the
# scale-only temperature.
PARAMETERS = {"choice": ("temperature",), "noul": ("temperature", "shift"),
              "score": ("temperature", "shift")}
DEFAULT_SCORER = tp.DEFAULT_SCORER
DEFAULT_CONFIG = tp.DEFAULT_CONFIG
HERE = Path(__file__).resolve().parent
RUN_DIR = Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\run")
POOL_ITEMS = Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\items\p1b_pool.jsonl")
POOL_RUNLOG = RUN_DIR / "p1b_pool_T1.runlog.jsonl"
POOL_PREDICTIONS = RUN_DIR / "p1b_pool_T1.jsonl"
OD_DIR = Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions")
OD_235_ITEMS = OD_DIR / "sample10_items.jsonl"
OD_235_PREDICTIONS = RUN_DIR / "p1b_235_T1_from_current.jsonl"
JEV_TEST_ITEMS = OD_DIR / "jevbench_sample3_items.jsonl"
JEV_TEST_PREDICTIONS = RUN_DIR / "p1b_jevbench_test_soft_T1.jsonl"


# ------------------------------------------------------------------ the map

def coordinates(qtype: str, labels: list[str]) -> dict[str, float]:
    """The natural unit of the type's shift, per label.

    noul: 0 for "no", 1 for "yes"  -> b is in log-odds.
    score: the level's index       -> b is in log-odds per level (a tilt).
    choice: 0 for every option     -> b is not identified; the map is the scale.
    """
    if qtype == "noul":
        return {lab: (1.0 if lab == "yes" else 0.0) for lab in labels}
    if qtype == "choice":
        return {lab: 0.0 for lab in labels}
    if qtype == "score":
        return {lab: float(int(lab)) for lab in labels}
    raise SystemExit("not a question type: %r" % qtype)


def apply_map(closed: dict[str, float], qtype: str, labels: list[str],
              temperature: float, shift: float = 0.0) -> dict[str, float]:
    """-> the closed distribution under the affine map.

    p'_l = normalize(p_l ** (1/T) * exp(b * s_l)).  With b = 0 this is the server's
    own temperature (called, not copied), so the b = 0 arm is the served transform
    exactly.  A probability of 0 stays 0 (the log is -inf); every item still sums to
    1 because the normalizer is recomputed over the item's own labels.
    """
    if temperature <= 0.0:
        raise SystemExit("a temperature must be above 0")
    if shift == 0.0:
        return dict(tp.scale(closed, temperature))
    coords = coordinates(qtype, labels)
    weights: dict[str, float] = {}
    for lab in labels:
        p = float(closed.get(lab, 0.0))
        if p <= 0.0:
            weights[lab] = 0.0
        else:
            weights[lab] = math.exp(math.log(p) / temperature + shift * coords[lab])
    total = sum(weights.values())
    if total <= 0.0:
        # every weight underflowed; the temperature's own guard returns the reading
        # unchanged, and the map must not invent a distribution
        return dict(closed)
    return {lab: w / total for lab, w in weights.items()}


def undo_map(closed: dict[str, float], qtype: str, labels: list[str],
             temperature: float, shift: float = 0.0) -> dict[str, float]:
    """-> the distribution the map came from (the exact inverse).

    apply(p, T, b) = normalize(p**(1/T) exp(b s)), so
    apply(q, 1/T, -b T) = normalize(q**T exp(-b T s)) = p.  Expressed as the same
    function so there is one place that knows the map.
    """
    return apply_map(closed, qtype, labels, 1.0 / temperature, -shift * temperature)


def round_trip(closed: dict[str, float], qtype: str, labels: list[str],
               temperature: float, shift: float = 0.0) -> float:
    """-> the worst |difference| between p and undo(apply(p)); 0 is exact."""
    there = apply_map(closed, qtype, labels, temperature, shift)
    back = undo_map(there, qtype, labels, temperature, shift)
    return max(abs(back[lab] - float(closed.get(lab, 0.0))) for lab in labels)


# ------------------------------------------------------------------ the data

def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:                       # split on \n only: the pool holds a raw U+2028
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def closed_of_prediction(task: dict, probs: dict) -> dict[str, float]:
    """The closed distribution over the item's labels, from a predictions row."""
    qtype = task["question"]["type"]
    labels = list(task["labels"])
    if qtype == "noul":
        p = float(probs["yes"])
        return {"yes": p, "no": 1.0 - p}
    return {lab: float(probs.get(lab, 0.0)) for lab in labels}


def records_from_predictions(items: list[dict], predictions: list[dict]) -> tuple[list[dict], list[str]]:
    """-> ({id: record}, the problems) for a population, from its predictions file.

    A record holds the item's closed distribution (what the map acts on), the gold
    label, the item's human distribution when it has one, and the scorer-shaped task.
    """
    preds = {p["id"]: p for p in predictions}
    records: dict[str, dict] = {}
    problems: list[str] = []
    for task in items:
        iid = task["id"]
        pred = preds.get(iid)
        if pred is None:
            problems.append("%s: no prediction" % iid)
            continue
        closed = closed_of_prediction(task, pred.get("probs") or {})
        if abs(sum(closed.values()) - 1.0) > 1e-6:
            problems.append("%s: the distribution sums to %r" % (iid, sum(closed.values())))
        target, problem = tp.human_target(task)
        if problem:
            problems.append("%s: %s" % (iid, problem))
        qtype = task["question"]["type"]
        gold = (("yes" if task["expected"] == "yes" else "no") if qtype == "noul"
                else (task["expected"] if qtype == "choice" else str(int(task["expected"]))))
        records[iid] = {"id": iid, "type": qtype, "tier": task.get("tier"),
                        "labels": list(task["labels"]), "closed": closed, "gold": gold,
                        "target": target, "task": task}
    return [records[i] for i in sorted(records)], problems


# ------------------------------------------------------------------ the objective

def prepared(records: list[dict], qtype: str) -> list[dict]:
    """The type's records that carry a human distribution, with the pieces the
    objective needs, precomputed once.

    Each item's labels, the log of the closed reading and the shift coordinate are
    fixed; only T and b change, so the fit's inner loop is the map and the objective.
    """
    out = []
    for rec in records:
        if rec["type"] != qtype or not rec.get("target"):
            continue
        labels = rec["labels"]
        coords = coordinates(qtype, labels)
        out.append({"labels": labels,
                    "logp": [math.log(rec["closed"].get(lab, 0.0))
                             if rec["closed"].get(lab, 0.0) > 0.0 else None for lab in labels],
                    "coords": [coords[lab] for lab in labels],
                    "target": {lab: float(rec["target"].get(lab, 0.0)) for lab in labels}})
    return out


def objective_value(items: list[dict], temperature: float, shift: float, objective: str) -> float:
    """The mean objective over the items, under the map.

    soft: -sum_l q_l log p'_l, over every label of the item (q_l may be 0, and 0 log 0
          is 0, so the zeros contribute nothing).
    tvd:  0.5 sum_l |p'_l - q_l|.
    """
    if not items:
        return float("inf")
    total = 0.0
    for it in items:
        logp, coords = it["logp"], it["coords"]
        weights = []
        for lp, s in zip(logp, coords):
            if lp is None:
                weights.append(0.0)
            else:
                weights.append(math.exp(lp / temperature + shift * s))
        z = sum(weights)
        if z <= 0.0:
            return float("inf")
        if objective == "soft":
            value = 0.0
            for w, (lab, q) in zip(weights, it["target"].items()):
                if q <= 0.0:
                    continue
                p = w / z
                value -= q * math.log(p if p > 1e-300 else 1e-300)
        else:
            value = 0.0
            for w, (lab, q) in zip(weights, it["target"].items()):
                value += abs(w / z - q)
            value *= 0.5
        total += value
    return total / len(items)


def fit_map(items: list[dict], qtype: str, objective: str,
            t_min: float = T_MIN, t_max: float = T_MAX,
            b_min: float = B_MIN, b_max: float = B_MAX,
            coarse_t: float = COARSE_T_STEP, coarse_b: float = COARSE_B_STEP,
            fine_t: float = FINE_T_STEP, fine_b: float = FINE_B_STEP) -> dict:
    """-> the (T, b) the objective likes best, on a coarse grid then a fine one.

    The objective is convex in (1/T, b): the map's log-weights are linear in them and
    the cross-entropy of a softmax is a log-sum-exp minus a linear term.  A grid is
    used anyway, because it is deterministic and its shape can be printed - the same
    reason the temperature fit sweeps its scalar.  The choice's map has no shift, so
    its grid is over T alone.
    """
    identified = "shift" in PARAMETERS[qtype]
    shifts = ([round(b_min + i * coarse_b, 6)
               for i in range(int(round((b_max - b_min) / coarse_b)) + 1)]
              if identified else [0.0])
    steps_t = int(round((t_max - t_min) / coarse_t))
    best = None
    for i in range(steps_t + 1):
        t = round(t_min + i * coarse_t, 6)
        for b in shifts:
            value = objective_value(items, t, b, objective)
            if best is None or value < best[2]:
                best = (t, b, value)
    # refine around the coarse winner, on the fine grid
    t0, b0, _ = best
    fine_ts = [round(max(t_min, min(t_max, t0 + i * fine_t)), 6)
               for i in range(-10, 11)]
    fine_bs = ([round(max(b_min, min(b_max, b0 + i * fine_b)), 6)
                for i in range(-10, 11)] if identified else [0.0])
    for t in sorted(set(fine_ts)):
        for b in sorted(set(fine_bs)):
            value = objective_value(items, t, b, objective)
            if value < best[2]:
                best = (t, b, value)
    t, b, value = best
    at_one = objective_value(items, 1.0, 0.0, objective)
    return {"temperature": t, "shift": b, "objective_value": value,
            "objective_at_t1": at_one, "items": len(items),
            "at_boundary": bool(t in (t_min, t_max) or (identified and b in (b_min, b_max))),
            "grid": {"temperature": [t_min, t_max, coarse_t, fine_t],
                     "shift": [b_min, b_max, coarse_b, fine_b] if identified else None},
            "shift_identified": identified}


# ------------------------------------------------------------------ the cross-validation

def fold_maps(records: list[dict], objective: str, folds: int = FOLDS, seed: int = SEED,
              **grid) -> list[dict]:
    """-> one entry per fold: the maps fitted on the other folds, and the held ids."""
    fold_items = tp.stratified_folds(records, folds, seed)
    table = []
    for f in range(folds):
        held = fold_items[f]
        train = [r for g in range(folds) if g != f for r in fold_items[g]]
        held_ids = {r["id"] for r in held}
        train_ids = {r["id"] for r in train}
        overlap = held_ids & train_ids
        if overlap:
            raise AssertionError("fold %d's own items are in its fit: %s"
                                 % (f, ", ".join(sorted(overlap)[:5])))
        maps = {}
        for qtype in TYPES:
            items = prepared(train, qtype)
            maps[qtype] = fit_map(items, qtype, objective, **grid)
        table.append({"fold": f, "n": len(held), "maps": maps})
    return table


def out_of_fold_maps(records: list[dict], table: list[dict]) -> dict[str, dict]:
    """-> {id: the map of the fold the item was held out of}.

    The fold assignment is rebuilt with the same function the fits used, so the two
    cannot drift apart.
    """
    per_item: dict[str, dict] = {}
    fold_of: dict[str, int] = {}
    fold_items = tp.stratified_folds(records, len(table), SEED)
    for f, held in fold_items.items():
        for rec in held:
            fold_of[rec["id"]] = f
    for rec in records:
        per_item[rec["id"]] = table[fold_of[rec["id"]]]["maps"][rec["type"]]
    return per_item


# ------------------------------------------------------------------ the metrics

def tvd(probs: dict, target: dict, labels: list[str]) -> float:
    return 0.5 * sum(abs(float(probs.get(lab, 0.0)) - float(target.get(lab, 0.0)))
                     for lab in labels)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def expected_level(probs: dict, labels: list[str]) -> float:
    return sum(float(int(lab)) * float(probs.get(lab, 0.0)) for lab in labels)


def top_mass(probs: dict) -> float:
    return max(float(v) for v in probs.values())


def auc_binary(scores: list[float], labels: list[float]) -> float | None:
    """The AUC with ties counted as half, i.e. the Mann-Whitney statistic."""
    pos = [s for s, y in zip(scores, labels) if y >= 0.5]
    neg = [s for s, y in zip(scores, labels) if y < 0.5]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def auc_rank(records: list[dict], dists: dict[str, dict]) -> float | None:
    """The probability that the gold label outranks a uniformly random other label.

    The multiclass AUC (one-vs-rest, macro over the other labels), averaged over the
    items.  Ties count as half.  For the choice the gold is the expected option; for
    the score it is the expected level.
    """
    values = []
    for rec in records:
        probs = dists[rec["id"]]
        gold = rec["gold"]
        labels = rec["labels"]
        if gold not in labels:
            continue
        p_gold = float(probs.get(gold, 0.0))
        others = [float(probs.get(lab, 0.0)) for lab in labels if lab != gold]
        if not others:
            continue
        values.append(sum(1.0 if p_gold > o else (0.5 if p_gold == o else 0.0)
                          for o in others) / len(others))
    return mean(values)


def bias(records: list[dict], dists: dict[str, dict], qtype: str) -> float | None:
    """The mean signed difference, in the type's own unit.

    noul:   P(yes) - the humans' P(yes) (the measured +0.357 of §19.6).
    score:  the expected level - the humans' expected level (the +0.135 level gap).
    choice: the top-label mass - the humans' top-label mass.  A per-label signed
            difference is NOT defined here: both distributions sum to 1, so the
            per-label differences sum to 0 by construction.  The top mass is the
            signed shape measure §19.2 used (0.7862 against 0.6215), and it is the
            one the choice's error is.
    """
    values = []
    for rec in records:
        if rec["type"] != qtype or not rec.get("target"):
            continue
        probs, target = dists[rec["id"]], rec["target"]
        if qtype == "noul":
            values.append(float(probs["yes"]) - float(target.get("yes", 0.0)))
        elif qtype == "score":
            values.append(expected_level(probs, rec["labels"])
                          - expected_level(target, rec["labels"]))
        else:
            values.append(top_mass(probs) - top_mass(target))
    return mean(values)


def type_metrics(records: list[dict], dists: dict[str, dict], qtype: str) -> dict:
    """The four numbers the brief asks for, on the type's items that carry humans,
    plus discrimination on every item of the type."""
    sub = [r for r in records if r["type"] == qtype]
    with_target = [r for r in sub if r.get("target")]
    tvds = [tvd(dists[r["id"]], r["target"], r["labels"]) for r in with_target]
    if qtype == "noul":
        auc = auc_binary([float(dists[r["id"]]["yes"]) for r in sub],
                         [1.0 if r["gold"] == "yes" else 0.0 for r in sub])
    else:
        auc = auc_rank(sub, dists)
    return {"n": len(sub), "with_human_distribution": len(with_target),
            "mean_tvd": mean(tvds), "bias": bias(records, dists, qtype), "auc": auc}


def paired_tvd_bootstrap(records: list[dict], arm_a: dict[str, dict], arm_b: dict[str, dict],
                         qtype: str | None = None, resamples: int = 1000,
                         seed: int = SEED) -> dict:
    """The paired item bootstrap of the mean-TVD difference (B - A).

    Both arms are measured on the same resample of items, so an item's own wobble
    cancels and what is left is the map's effect on the distance to the humans.
    """
    sub = [r for r in records if r.get("target") and (qtype is None or r["type"] == qtype)]
    if not sub or resamples <= 0:
        return {}
    diffs = [tvd(arm_b[r["id"]], r["target"], r["labels"])
             - tvd(arm_a[r["id"]], r["target"], r["labels"]) for r in sub]
    rng = random.Random(seed)
    n = len(sub)
    out = []
    for _ in range(resamples):
        out.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return {"n": n, "mean": sum(diffs) / n,
            "p025": out[int(0.025 * (len(out) - 1))],
            "p975": out[int(0.975 * (len(out) - 1))],
            "share_better": sum(1 for d in diffs if d < 0) / n,
            "resamples": resamples}


# ------------------------------------------------------------------ one population

def arm_distributions(records: list[dict], params: dict[str, dict],
                      per_type: bool = False) -> dict[str, dict]:
    """-> {id: distribution} with the maps of `params` applied to the items' types.

    `params` maps a type to {"temperature": T, "shift": b}; a type that is absent is
    left at T = 1, b = 0.  With per_type the result is {type: {id: distribution}} for
    the type's own arm (the type's map, every other type at T = 1).
    """
    if per_type:
        out = {}
        for qtype in TYPES:
            out[qtype] = {r["id"]: (apply_map(r["closed"], qtype, r["labels"],
                                              params[qtype]["temperature"],
                                              params[qtype].get("shift", 0.0))
                                    if r["type"] == qtype else dict(r["closed"]))
                          for r in records}
        return out
    out = {}
    for r in records:
        p = params.get(r["type"])
        out[r["id"]] = (apply_map(r["closed"], r["type"], r["labels"],
                                  p["temperature"], p.get("shift", 0.0))
                        if p else dict(r["closed"]))
    return out


def score_population(scorer_path: Path, records: list[dict],
                     dists_by_arm: dict[str, dict[str, dict]]) -> dict:
    """The official scorer's numbers for every arm of one population, and the
    per-type metrics of every arm against the first arm (the T = 1 baseline)."""
    base_name = next(iter(dists_by_arm))
    out = {"baseline": base_name, "arms": {}}
    for name, dists in dists_by_arm.items():
        scored = tp.score_with_official_scorer(scorer_path, records, dists)
        metrics = {q: type_metrics(records, dists, q) for q in TYPES}
        out["arms"][name] = {"scorer": scored, "metrics": metrics}
        if name != base_name:
            out["arms"][name]["paired_tvd"] = {
                q: paired_tvd_bootstrap(records, dists_by_arm[base_name], dists, q)
                for q in TYPES}
            out["arms"][name]["paired_tvd"]["all"] = paired_tvd_bootstrap(
                records, dists_by_arm[base_name], dists)
    return out


def round_trip_check(records: list[dict], params: dict[str, dict]) -> dict:
    """The worst |p - undo(apply(p))| over the items, per type.

    The inversion is checked, not assumed: the offline arms are only exact if the map
    is invertible to floating point.
    """
    worst = {q: 0.0 for q in TYPES}
    for rec in records:
        p = params.get(rec["type"])
        if not p:
            continue
        worst[rec["type"]] = max(worst[rec["type"]],
                                 round_trip(rec["closed"], rec["type"], rec["labels"],
                                            p["temperature"], p.get("shift", 0.0)))
    return worst


def combine_capability(arms: dict[str, dict], types: list[str],
                       baseline: str = "T = 1") -> dict:
    """-> the scorer's Capability for the arm made of `types`' maps, exactly.

    The scorer's Intelligence and Calibration are each a mean over the three types of
    that type's own component, and a type's map changes only that type's component
    (its items and tier weights do not move).  So the per-type arms ("only <type>")
    carry every subset's Capability exactly, with no extra scorer run: a type in
    `types` contributes its mapped component, every other type its `baseline` one.
    """
    per_type_cal = {}
    per_type_comp = {}
    for qtype in TYPES:
        key = ("only " + qtype) if qtype in types else baseline
        arm = arms[key]["scorer"]
        per_type_cal[qtype] = (arm.get("calibration_per_type") or {}).get(qtype)
        per_type_comp[qtype] = (arm.get("competence_per_type") or {}).get(qtype)
    def weighted(values: dict[str, float]) -> float | None:
        present = [v for v in values.values() if v is not None]
        return sum(present) / len(present) if present else None
    cal = weighted(per_type_cal)
    comp = weighted(per_type_comp)
    return {"capability": None if cal is None or comp is None else (cal + comp) / 2,
            "intelligence": comp, "calibration": cal}


# ------------------------------------------------------------------ the report

def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def signed(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=POOL_ITEMS, help="the fitting pool's items")
    ap.add_argument("--runlog", type=Path, default=POOL_RUNLOG, help="the pool run's run log")
    ap.add_argument("--predictions", type=Path, default=POOL_PREDICTIONS,
                    help="the pool run's predictions (the T = 1 arm)")
    ap.add_argument("--scorer", type=Path, default=DEFAULT_SCORER)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-json", type=Path, default=HERE / "affine_calibration.json")
    ap.add_argument("--out-report", type=Path, default=HERE / "affine_report.md")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the arms' predictions files are written (not the repo)")
    ap.add_argument("--objective", choices=OBJECTIVES, default="soft")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--t-min", type=float, default=T_MIN)
    ap.add_argument("--t-max", type=float, default=T_MAX)
    ap.add_argument("--b-min", type=float, default=B_MIN)
    ap.add_argument("--b-max", type=float, default=B_MAX)
    ap.add_argument("--coarse-t", type=float, default=COARSE_T_STEP)
    ap.add_argument("--coarse-b", type=float, default=COARSE_B_STEP)
    ap.add_argument("--fine-t", type=float, default=FINE_T_STEP)
    ap.add_argument("--fine-b", type=float, default=FINE_B_STEP)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--235-items", dest="od_items", type=Path, default=OD_235_ITEMS)
    ap.add_argument("--235-predictions", dest="od_predictions", type=Path,
                    default=OD_235_PREDICTIONS)
    ap.add_argument("--test-items", type=Path, default=JEV_TEST_ITEMS)
    ap.add_argument("--test-predictions", type=Path, default=JEV_TEST_PREDICTIONS)
    ap.add_argument("--date", default=None)
    ap.add_argument("--source-dataset", default=(
        "Praveenrajus/jev-bench, the validation split, the 500-per-type pool of "
        "soft_label_pool.json (seed 20261021)"))
    ap.add_argument("--source-arm", default=(
        "the serial first-token path, one question per request, the server at T = 1, "
        "a fresh engine (6306 s, 1500/1500, 0 failures)"))
    ap.add_argument("--source-note", default=(
        "the pool is licence-clean and out of sample with respect to OpenDecisions and "
        "to the jev-bench test split; the map is fitted on it and on nothing else"))
    args = ap.parse_args()

    when = args.date or _date.today().isoformat()
    model = args.model
    if model is None and args.config.exists():
        model = json.loads(args.config.read_text(encoding="utf-8")).get("model_name")
    for label, path in (("items", args.items), ("runlog", args.runlog),
                        ("predictions", args.predictions), ("scorer", args.scorer),
                        ("the 235's items", args.od_items),
                        ("the 235's predictions", args.od_predictions),
                        ("the test sample's items", args.test_items),
                        ("the test sample's predictions", args.test_predictions)):
        if not path.exists():
            raise SystemExit("the %s file is not there: %s" % (label, path))
    grid = {"t_min": args.t_min, "t_max": args.t_max, "b_min": args.b_min, "b_max": args.b_max,
            "coarse_t": args.coarse_t, "coarse_b": args.coarse_b,
            "fine_t": args.fine_t, "fine_b": args.fine_b}

    # ---- the pool, with the run log's cross-check
    pool_items = read_jsonl(args.items)
    pool_runlog = read_jsonl(args.runlog)
    pool_predictions = read_jsonl(args.predictions)
    pool_by_id, problems = tp.build_records(pool_items, pool_runlog, pool_predictions)
    pool = [pool_by_id[i] for i in sorted(pool_by_id)]
    print("[affine] the pool: %d items, %d cross-check problems" % (len(pool), len(problems)))
    for p in problems[:10]:
        print("  !", p)
    pool_counts = {q: sum(1 for r in pool if r["type"] == q and r.get("target")) for q in TYPES}

    # ---- the fit, every objective, full pool and out of fold
    scorer = tp.load_scorer(args.scorer)
    fits: dict[str, dict] = {}
    folds: dict[str, list] = {}
    for objective in OBJECTIVES:
        fits[objective] = {}
        for qtype in TYPES:
            fits[objective][qtype] = fit_map(prepared(pool, qtype), qtype, objective, **grid)
            print("[affine] %-5s %-6s T %-6s b %-8s objective %.6f (at T = 1: %.6f)"
                  % (objective, qtype, fits[objective][qtype]["temperature"],
                     fits[objective][qtype]["shift"],
                     fits[objective][qtype]["objective_value"],
                     fits[objective][qtype]["objective_at_t1"]))
        folds[objective] = fold_maps(pool, objective, args.folds, args.seed, **grid)

    # ---- the fitted arms: the pool out of fold, then the two yardsticks
    pool_t1 = {r["id"]: dict(r["closed"]) for r in pool}
    populations: dict[str, dict] = {}
    oof: dict[str, dict] = {}
    for objective in OBJECTIVES:
        per_item = out_of_fold_maps(pool, folds[objective])
        oof[objective] = per_item
        arms = {"T = 1": pool_t1,
                "affine (out of fold)": {
                    r["id"]: apply_map(r["closed"], r["type"], r["labels"],
                                       per_item[r["id"]]["temperature"],
                                       per_item[r["id"]].get("shift", 0.0))
                    for r in pool}}
        for qtype in TYPES:
            arms["only " + qtype] = {
                r["id"]: (apply_map(r["closed"], r["type"], r["labels"],
                                    per_item[r["id"]]["temperature"],
                                    per_item[r["id"]].get("shift", 0.0))
                          if r["type"] == qtype else dict(r["closed"]))
                for r in pool}
        populations.setdefault("pool", {})[objective] = score_population(args.scorer, pool, arms)
        populations["pool"][objective]["subsets"] = subsets_of(populations["pool"][objective]["arms"])
        check_decomposition(populations["pool"][objective]["arms"], "affine (out of fold)")
        print("[affine] %-5s the pool out of fold: Capability %.2f (T = 1: %.2f)"
              % (objective,
                 populations["pool"][objective]["arms"]["affine (out of fold)"]["scorer"]["capability"],
                 populations["pool"][objective]["arms"]["T = 1"]["scorer"]["capability"]))

    # ---- the 235 and the jev-bench test sample: the full pool fit, per type and all
    full_maps = {o: {q: fits[o][q] for q in TYPES} for o in OBJECTIVES}
    population_records: dict[str, list[dict]] = {"pool": pool}
    for name, items_path, preds_path in (
            ("od235", args.od_items, args.od_predictions),
            ("jevtest", args.test_items, args.test_predictions)):
        records, problems = records_from_predictions(read_jsonl(items_path), read_jsonl(preds_path))
        population_records[name] = records
        print("[affine] %s: %d items, %d problems" % (name, len(records), len(problems)))
        for p in problems[:10]:
            print("  !", p)
        base = {r["id"]: dict(r["closed"]) for r in records}
        for objective in OBJECTIVES:
            by_type = arm_distributions(records, full_maps[objective], per_type=True)
            arms = {"T = 1": base,
                    "affine (all three)": arm_distributions(records, full_maps[objective])}
            for qtype in TYPES:
                arms["only " + qtype] = by_type[qtype]
            populations.setdefault(name, {})[objective] = score_population(args.scorer, records, arms)
            populations[name][objective]["items"] = len(records)
            populations[name][objective]["items_by_type"] = {
                q: sum(1 for r in records if r["type"] == q) for q in TYPES}
            populations[name][objective]["with_human_distribution"] = {
                q: sum(1 for r in records if r["type"] == q and r.get("target")) for q in TYPES}
            populations[name][objective]["subsets"] = subsets_of(
                populations[name][objective]["arms"])
            check_decomposition(populations[name][objective]["arms"], "affine (all three)")
            print("[affine] %-7s %-5s Capability %.2f (T = 1: %.2f)"
                  % (name, objective,
                     populations[name][objective]["arms"]["affine (all three)"]["scorer"]["capability"],
                     populations[name][objective]["arms"]["T = 1"]["scorer"]["capability"]))
        # the round trip on this population's own items
        populations[name]["round_trip"] = round_trip_check(records, full_maps[args.objective])
    populations["pool"]["round_trip"] = round_trip_check(
        pool, {q: fits[args.objective][q] for q in TYPES})

    # ---- the paired bootstrap of the scorer's numbers (the primary objective)
    primary = args.objective
    paired = {}
    for name in ("pool", "od235", "jevtest"):
        records = population_records[name]
        t1 = {r["id"]: dict(r["closed"]) for r in records}
        arm = (pool_oof_arm(records, oof[primary]) if name == "pool"
               else arm_distributions(records, full_maps[primary]))
        paired[name] = tp.paired_bootstrap(scorer, records, t1, arm,
                                           args.bootstrap, args.seed)
        print("[affine] %-7s paired bootstrap: Capability %s [%s, %s]"
              % (name,
                 signed(paired[name].get("capability", {}).get("mean"), 2),
                 signed(paired[name].get("capability", {}).get("p025"), 2),
                 signed(paired[name].get("capability", {}).get("p975"), 2)))

    # ---- the rule
    rule = apply_rule(populations, fits, primary, pool_counts)

    # ---- the arms' predictions files, for the record
    out_dir = args.out_dir or RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in ("pool", "od235", "jevtest"):
        records = population_records[name]
        dists = (pool_oof_arm(records, oof[primary]) if name == "pool"
                 else arm_distributions(records, full_maps[primary]))
        path = out_dir / ("p1a_%s_affine.jsonl" % name)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                probs = dists[r["id"]]
                row = {"id": r["id"], "probs": ({"yes": probs["yes"]} if r["type"] == "noul"
                                                else {lab: probs[lab] for lab in r["labels"]})}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[name] = str(path)
        print("[affine] wrote %s" % path)

    # ---- the JSON
    out = {
        "version": 1,
        "model": model,
        "fitted_on": when,
        "objective": primary,
        "objective_text": {
            "soft": "the mean cross-entropy of the mapped distribution against the item's "
                    "HUMAN distribution (target_probs), over the items that carry one",
            "tvd": "the mean total variation distance between the mapped distribution and "
                   "the item's HUMAN distribution, over the items that carry one",
        }[primary],
        "map": {
            q: {"form": ("p'_l = normalize(p_l ** (1/T) * exp(b * s_l))" if q != "choice"
                         else "p'_l = normalize(p_l ** (1/T))  (no shift: not identified)"),
                "parameters": list(PARAMETERS[q]),
                "coordinate": {"choice": "none",
                               "noul": "0 for no, 1 for yes (log-odds)",
                               "score": "the level's index (log-odds per level)"}[q]}
            for q in TYPES},
        "grid": grid,
        "seed": args.seed,
        "folds": args.folds,
        "split_rule": ("5-fold cross-validation, stratified by type: every type's items are "
                       "sorted by id, shuffled with random.Random(seed) and dealt round-robin "
                       "into the five folds (temperature.stratified_folds, the temperature "
                       "fit's own function); a fold's (T, b) is fitted on the other four "
                       "folds' items of the type that carry a human distribution, and "
                       "applied to the fold's own items."),
        "items_with_human_distribution": pool_counts,
        "fits": fits,
        "folds": folds,
        "populations": populations,
        "paired_bootstrap": paired,
        "rule": rule,
        "round_trip": {
            "pool": populations["pool"].get("round_trip", {}),
            "od235": populations["od235"].get("round_trip", {}),
            "jevtest": populations["jevtest"].get("round_trip", {}),
        },
        "arms_written": written,
        "source": {
            "dataset": args.source_dataset,
            "arm": args.source_arm,
            "note": args.source_note,
            "files": {label: {"path": str(path), "sha256": tp.file_digest(path)}
                      for label, path in (
                          ("items", args.items), ("runlog", args.runlog),
                          ("predictions", args.predictions),
                          ("od235_items", args.od_items),
                          ("od235_predictions", args.od_predictions),
                          ("test_items", args.test_items),
                          ("test_predictions", args.test_predictions))},
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[affine] wrote %s" % args.out_json)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render(out), encoding="utf-8")
    print("[affine] wrote %s" % args.out_report)
    return 0


def pool_oof_arm(records: list[dict], per_item: dict[str, dict]) -> dict[str, dict]:
    """-> the pool's out-of-fold arm: every item mapped by its held-out fold's map."""
    return {r["id"]: apply_map(r["closed"], r["type"], r["labels"],
                               per_item[r["id"]]["temperature"],
                               per_item[r["id"]].get("shift", 0.0))
            for r in records}


def subsets_of(arms: dict[str, dict]) -> dict[str, dict]:
    """-> the Capability of every subset of the three types' maps, exactly."""
    out = {}
    for mask in range(8):
        types = [TYPES[i] for i in range(3) if mask & (1 << i)]
        out["+".join(types) if types else "none"] = combine_capability(arms, types)
    return out


def check_decomposition(arms: dict[str, dict], all_key: str) -> None:
    """The subset table must reproduce the arm it was measured from, exactly.

    If it does not, the decomposition (a type's map changes only its own components)
    is wrong, and every (c) reading built on it would be wrong with it.
    """
    combined = combine_capability(arms, list(TYPES))["capability"]
    measured = arms[all_key]["scorer"].get("capability")
    if combined is None or measured is None or abs(combined - measured) > 1e-9:
        raise AssertionError("the subset table does not reproduce %s: %r vs %r"
                             % (all_key, combined, measured))


def apply_rule(populations: dict, fits: dict, primary: str, pool_counts: dict) -> dict:
    """The companion doc's §20.2 rule, per type, on the paired arms.

    (a) the pool's out-of-fold calibration improves against the pool's own T = 1 arm;
    (b) the jev-bench test TVD does not get worse than T = 1 by more than 0.005;
    (c) the 235's Capability does not fall below its T = 1 arm.
    The overall (whole-population) numbers are the rule's; the per-type numbers are
    its decomposition.  The noul is judged separately (§20.2): if the set that passes
    (a) and (b) fails (c), the noul is removed first and (c) is re-checked for the
    others; a type that does not survive its own (c) does not ship.
    """
    pool = populations["pool"][primary]["arms"]
    od = populations["od235"][primary]["arms"]
    jev = populations["jevtest"][primary]["arms"]
    rows = {}
    for qtype in TYPES:
        cal_before = (pool["T = 1"]["scorer"].get("calibration_per_type") or {}).get(qtype)
        cal_after = (pool["affine (out of fold)"]["scorer"].get("calibration_per_type") or {}).get(qtype)
        tv_before = pool["T = 1"]["metrics"][qtype]["mean_tvd"]
        tv_after = pool["affine (out of fold)"]["metrics"][qtype]["mean_tvd"]
        jev_before = jev["T = 1"]["metrics"][qtype]["mean_tvd"]
        jev_after = jev["affine (all three)"]["metrics"][qtype]["mean_tvd"]
        # the type's OWN 235 arm: its competence and its accuracy
        od_own = od["only " + qtype]["scorer"]
        od_base = od["T = 1"]["scorer"]
        rows[qtype] = {
            "a_calibration_before": cal_before,
            "a_calibration_after": cal_after,
            "a_pass": (cal_before is not None and cal_after is not None and cal_after > cal_before),
            "pool_tvd_before": tv_before, "pool_tvd_after": tv_after,
            "b_jevtest_tvd_before": jev_before, "b_jevtest_tvd_after": jev_after,
            "b_delta": (None if jev_before is None or jev_after is None
                        else jev_after - jev_before),
            "b_pass": (jev_before is not None and jev_after is not None
                       and jev_after - jev_before <= 0.005),
            "od235_competence_before": (od_base.get("competence_per_type") or {}).get(qtype),
            "od235_competence_after_own_map": (od_own.get("competence_per_type") or {}).get(qtype),
            "od235_accuracy_before": (od_base.get("accuracy_per_type") or {}).get(qtype),
            "od235_accuracy_after_own_map": (od_own.get("accuracy_per_type") or {}).get(qtype),
        }
    # (c) is a whole-population guard: the Capability of the arm made of the types that
    # passed (a) and (b), from the per-type arms, exactly
    passing = [q for q in TYPES if rows[q]["a_pass"] and rows[q]["b_pass"]]
    cap_t1 = combine_capability(od, [])["capability"]
    cap_candidate = (combine_capability(od, passing)["capability"] if passing else cap_t1)
    c_pass = (cap_t1 is not None and cap_candidate is not None
              and cap_candidate >= cap_t1)
    # the noul is judged separately: if the candidate set fails (c), remove it first
    without_noul = [q for q in passing if q != "noul"]
    cap_without_noul = (combine_capability(od, without_noul)["capability"]
                        if without_noul else cap_t1)
    c_pass_without_noul = (cap_t1 is not None and cap_without_noul is not None
                           and cap_without_noul >= cap_t1)
    if c_pass:
        ships = list(passing)
    elif c_pass_without_noul:
        ships = without_noul
    else:
        ships = []
    rows["_overall"] = {
        "passing_types": passing,
        "od235_capability_t1": cap_t1,
        "od235_capability_candidate": cap_candidate,
        "c_pass": c_pass,
        "od235_capability_without_noul": cap_without_noul,
        "c_pass_without_noul": c_pass_without_noul,
        "jevtest_tvd_t1": jev["T = 1"]["scorer"].get("calibration_items_tvd"),
        "jevtest_tvd_all_three": (jev["affine (all three)"]["scorer"]
                                  .get("calibration_items_tvd")),
    }
    return {"rows": rows, "verdict": {
        "passing_a_and_b": passing,
        "c_pass": c_pass,
        "c_pass_without_noul": c_pass_without_noul,
        "ships": ships,
        "noul_judged_separately": True,
        "note": ("(c) is a population-level guard: the Capability of the arm made of the "
                 "types that pass (a) and (b). A type that passes (a)/(b) but whose "
                 "addition costs the 235 its Capability does not ship. The noul is judged "
                 "separately: if the set fails (c), it is removed first and (c) is "
                 "re-checked for the others (§20.2)."),
    }}


def render(out: dict) -> str:
    """The report, in the house style of temperature_pool_report.md."""
    A: list[str] = []
    add = A.append
    add("# The per-type affine calibration: the fit")
    add("")
    add(f"**Fitted {out['fitted_on']}, model `{out['model']}`.**  Roadmap item 1.4.  The")
    add("constants live in `affine_calibration.json` next to this file; **they are not in")
    add("any launch config**, and the map is not in the server - this is the fit and the")
    add("measurement, not a shipped feature.")
    add("")
    add(f"**The pool:** {out['source']['dataset']} - {out['source']['arm']}.")
    add("")
    add(f"**The objective:** {out['objective_text']}.  The control is the TVD objective.")
    add("")
    add("## The map")
    add("")
    add("One scale and, where the type has an axis to shift along, one shift:")
    add("")
    add("```")
    add("p'_l = normalize( p_l ** (1/T) * exp(b * s_l) )")
    add("```")
    add("")
    add("| type | parameters | the coordinate `s_l` | what `b` means |")
    add("|---|---|---|---|")
    add("| noul | T, b | 0 for no, 1 for yes | `logit(P'(yes)) = logit(P(yes))/T + b`: the shift, in log-odds |")
    add("| score | T, b | the level's index 0..K-1 | a tilt along the ordinal axis, in log-odds per level: a location parameter |")
    add("| choice | T | none | **not identified**: a shift that is the same for every option is cancelled by the normalizer |")
    add("")
    add("The map is invertible: `undo(q) = apply(q, 1/T, -bT)`, and the round trip is")
    add("checked on every item of every population (the worst difference is in the JSON).")
    add("With `b = 0` it is the server's own `apply_temperature`, called, not copied.")
    add("Unlike a temperature it can move the pick; every accuracy below is reported")
    add("because of that.")
    add("")
    add("## The fit")
    add("")
    add(f"5-fold cross-validation, stratified by type, seed {out['seed']}; the fold rule is")
    add("`temperature.stratified_folds`, the temperature fit's own function.  A fold's")
    add("(T, b) is fitted on the other folds' items of the type that carry a human")
    add("distribution, and applied to the fold's own items.  The full-pool fit (what the")
    add("two yardsticks are measured with) is below; the per-fold fits are in the JSON.")
    add("")
    add("| objective | type | items fitted | T | b | the objective at (T, b) | at T = 1 |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for objective in OBJECTIVES:
        for qtype in TYPES:
            f = out["fits"][objective][qtype]
            add("| %s | %s | %d | %.2f | %s | %.4f | %.4f |"
                % (objective, qtype, f["items"], f["temperature"],
                   ("%.3f" % f["shift"]) if f["shift_identified"] else "n/a",
                   f["objective_value"], f["objective_at_t1"]))
    add("")
    add("The grid: T from %.2f to %.2f (coarse %.2f, then fine %.2f), b from %.1f to %.1f"
        % (out["grid"]["t_min"], out["grid"]["t_max"], out["grid"]["coarse_t"],
           out["grid"]["fine_t"], out["grid"]["b_min"], out["grid"]["b_max"]))
    add("(coarse %.1f, then fine %.2f).  The objective is convex in (1/T, b), so the best"
        % (out["grid"]["coarse_b"], out["grid"]["fine_b"]))
    add("grid point is the global minimum up to the grid, not a local one.")
    add("")
    add("## The populations, paired")
    add("")
    add("Every arm of a population comes out of that population's ONE stored run: the")
    add("T = 1 distribution is the stored one, and the affine arm is the same distribution")
    add("put through the map.  The engine, the picks, the logits and the state are shared;")
    add("only the arithmetic differs.  `TVD` is the mean total variation distance to the")
    add("human distribution over the items that carry one; `bias` is the type's own signed")
    add("measure (the noul's P(yes) gap, the score's expected-level gap in levels, the")
    add("choice's top-label mass gap); `AUC` is the probability that the gold answer")
    add("outranks a random other (binary for the noul, multiclass for the choice and the")
    add("score).")
    add("")
    for name, label in (("pool", "The pool, out of fold (the fit's own population)"),
                        ("od235", "The OpenDecisions 235 (out of sample; never fitted on)"),
                        ("jevtest", "The jev-bench test sample, 683 items (out of sample)")):
        pop = out["populations"][name][out["objective"]]
        base = "affine (out of fold)" if name == "pool" else "affine (all three)"
        add(f"### {label}")
        add("")
        add("| arm | Capability | Intelligence | Calibration | mean TVD |")
        add("|---|---:|---:|---:|---:|")
        for arm in ("T = 1", base):
            s = pop["arms"][arm]["scorer"]
            add("| %s | %.2f | %.2f | %.2f | %s |"
                % (arm, s["capability"], s["intelligence"], s["calibration"],
                   fmt(s.get("calibration_items_tvd"))))
        add("")
        add("| type | n (with humans) | TVD T = 1 | TVD affine | bias T = 1 | bias affine | AUC T = 1 | AUC affine |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for qtype in TYPES:
            m1 = pop["arms"]["T = 1"]["metrics"][qtype]
            ma = pop["arms"][base]["metrics"][qtype]
            add("| %s | %d (%d) | %s | %s | %s | %s | %s | %s |"
                % (qtype, m1["n"], m1["with_human_distribution"],
                   fmt(m1["mean_tvd"]), fmt(ma["mean_tvd"]),
                   signed(m1["bias"]), signed(ma["bias"]),
                   fmt(m1["auc"]), fmt(ma["auc"])))
        add("")
        add("| type | calibration T = 1 | calibration affine | competence T = 1 | competence affine |")
        add("|---|---:|---:|---:|---:|")
        s1 = pop["arms"]["T = 1"]["scorer"]
        sa = pop["arms"][base]["scorer"]
        for qtype in TYPES:
            add("| %s | %s | %s | %s | %s |"
                % (qtype,
                   fmt((s1.get("calibration_per_type") or {}).get(qtype), 2),
                   fmt((sa.get("calibration_per_type") or {}).get(qtype), 2),
                   fmt((s1.get("competence_per_type") or {}).get(qtype), 2),
                   fmt((sa.get("competence_per_type") or {}).get(qtype), 2)))
        add("")
        add("The arms of one type at a time (the subset Capability is exact, from the")
        add("per-type components):")
        add("")
        add("| the types mapped | Capability | Intelligence | Calibration |")
        add("|---|---:|---:|---:|")
        for types, value in pop["subsets"].items():
            add("| %s | %s | %s | %s |"
                % (types, fmt(value["capability"], 2), fmt(value["intelligence"], 2),
                   fmt(value["calibration"], 2)))
        add("")
        boot = out["paired_bootstrap"].get(name) or {}
        if boot:
            add("The paired item bootstrap of the whole-population arm against T = 1")
            add("(the same items, resampled, scored in both arms):")
            add("")
            add("| metric | mean change | 95% interval | share of resamples better |")
            add("|---|---:|---|---:|")
            for key in ("capability", "intelligence", "calibration"):
                v = boot.get(key) or {}
                if v:
                    add("| %s | %s | [%s, %s] | %.2f |"
                        % (key, signed(v.get("mean"), 2), signed(v.get("p025"), 2),
                           signed(v.get("p975"), 2), v.get("share_better", 0.0)))
            add("")
        if name == "pool":
            b = pop["arms"][base].get("paired_tvd", {})
            if b:
                add("The paired item bootstrap of the TVD difference (affine - T = 1):")
                add("")
                add("| type | mean | 95% interval | share of items better |")
                add("|---|---:|---|---:|")
                for qtype in list(TYPES) + ["all"]:
                    v = b.get(qtype) or {}
                    if v:
                        add("| %s | %s | [%s, %s] | %.2f |"
                            % (qtype, signed(v["mean"]), signed(v["p025"]),
                               signed(v["p975"]), v["share_better"]))
                add("")
    add("## The control: the TVD objective")
    add("")
    add("The same map, fitted on the mean TVD to the human distribution instead of the")
    add("cross-entropy.  It is the control, not the shipped objective: TVD is a distance")
    add("and not a proper scoring rule, and its fit drives the noul to a near-degenerate")
    add("\"no\" (the table below).  The primary objective's numbers are the ones above.")
    add("")
    add("| type | pool OOF TVD, T = 1 | soft | tvd | jev test TVD, T = 1 | soft | tvd | the 235's competence, T = 1 | soft | tvd |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for qtype in TYPES:
        p1 = out["populations"]["pool"]["soft"]["arms"]["T = 1"]["metrics"][qtype]["mean_tvd"]
        ps = out["populations"]["pool"]["soft"]["arms"]["affine (out of fold)"]["metrics"][qtype]["mean_tvd"]
        pt = out["populations"]["pool"]["tvd"]["arms"]["affine (out of fold)"]["metrics"][qtype]["mean_tvd"]
        j1 = out["populations"]["jevtest"]["soft"]["arms"]["T = 1"]["metrics"][qtype]["mean_tvd"]
        js = out["populations"]["jevtest"]["soft"]["arms"]["affine (all three)"]["metrics"][qtype]["mean_tvd"]
        jt = out["populations"]["jevtest"]["tvd"]["arms"]["affine (all three)"]["metrics"][qtype]["mean_tvd"]
        o1 = out["populations"]["od235"]["soft"]["arms"]["T = 1"]["scorer"].get("competence_per_type", {}).get(qtype)
        os_ = out["populations"]["od235"]["soft"]["arms"]["only " + qtype]["scorer"].get("competence_per_type", {}).get(qtype)
        ot = out["populations"]["od235"]["tvd"]["arms"]["only " + qtype]["scorer"].get("competence_per_type", {}).get(qtype)
        add("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (qtype, fmt(p1), fmt(ps), fmt(pt), fmt(j1), fmt(js), fmt(jt),
               fmt(o1, 2), fmt(os_, 2), fmt(ot, 2)))
    add("")
    add("The whole-population Capability of the all-three arm: the 235 %.2f (soft) and"
        % out["populations"]["od235"]["soft"]["arms"]["affine (all three)"]["scorer"]["capability"])
    add("%.2f (tvd); the jev test %.2f and %.2f; the pool out of fold %.2f and %.2f (T = 1:"
        % (out["populations"]["od235"]["tvd"]["arms"]["affine (all three)"]["scorer"]["capability"],
           out["populations"]["jevtest"]["soft"]["arms"]["affine (all three)"]["scorer"]["capability"],
           out["populations"]["jevtest"]["tvd"]["arms"]["affine (all three)"]["scorer"]["capability"],
           out["populations"]["pool"]["soft"]["arms"]["affine (out of fold)"]["scorer"]["capability"],
           out["populations"]["pool"]["tvd"]["arms"]["affine (out of fold)"]["scorer"]["capability"]))
    add("%.2f on the pool, %.2f on the 235, %.2f on the test)."
        % (out["populations"]["pool"]["soft"]["arms"]["T = 1"]["scorer"]["capability"],
           out["populations"]["od235"]["soft"]["arms"]["T = 1"]["scorer"]["capability"],
           out["populations"]["jevtest"]["soft"]["arms"]["T = 1"]["scorer"]["capability"]))
    add("")
    add("## The rule, applied")
    add("")
    add("The pre-registered rule (§20.2): (a) the pool's out-of-fold calibration improves")
    add("against its own T = 1 arm; (b) the jev-bench test TVD does not get worse by more")
    add("than 0.005; (c) the 235's Capability does not fall below its T = 1 arm.  The noul")
    add("is judged separately.")
    add("")
    rows = out["rule"]["rows"]
    add("| type | (a) pool calibration | (b) jev test TVD | the 235, this type's own competence | verdict |")
    add("|---|---|---|---|---|")
    for qtype in TYPES:
        r = rows[qtype]
        add("| %s | %s -> %s: %s | %s -> %s (%s): %s | %s -> %s | %s |"
            % (qtype,
               fmt(r["a_calibration_before"], 2), fmt(r["a_calibration_after"], 2),
               "PASS" if r["a_pass"] else "FAIL",
               fmt(r["b_jevtest_tvd_before"]), fmt(r["b_jevtest_tvd_after"]),
               signed(r["b_delta"]), "PASS" if r["b_pass"] else "FAIL",
               fmt(r["od235_competence_before"], 2),
               fmt(r["od235_competence_after_own_map"], 2),
               "ships" if qtype in out["rule"]["verdict"]["ships"] else "does not ship"))
    add("")
    o = rows["_overall"]
    add("(c), the whole-population guard: the 235's Capability is %.2f at T = 1 and %.2f"
        % (o["od235_capability_t1"], o["od235_capability_candidate"]))
    add("with the maps of %s applied: **%s**.  Without the noul it is %.2f: **%s**."
        % (", ".join(o["passing_types"]) or "no type",
           "PASS" if o["c_pass"] else "FAIL",
           o["od235_capability_without_noul"],
           "PASS" if o["c_pass_without_noul"] else "FAIL"))
    add("")
    add("**The verdict: %s.**" % (
        ("%s ship" % ", ".join(out["rule"]["verdict"]["ships"]))
        if len(out["rule"]["verdict"]["ships"]) > 1 else
        (("%s ships" % out["rule"]["verdict"]["ships"][0])
         if out["rule"]["verdict"]["ships"] else "no constant ships")))
    add("")
    add("## The limits")
    add("")
    add("- **The 235's and the test sample's arms are offline re-scalings of a stored")
    add("  run**, not a live run of the map: the arithmetic is exact and the inversion is")
    add("  checked, but the engine's own reading is the stored one.  A live run would add")
    add("  the engine's run-to-run wobble (about 0.0193 mean |change| in P(yes) on the")
    add("  jev noul slice, and 1.8 Capability points on the 235, §23.5) and nothing else.")
    add("- **The map is not in the server.**  Shipping one would be a change to")
    add("  `serve/logits_first_token.py` and a new launch-config key, with its own")
    add("  byte-identity test; this item measured the map.")
    add("- **The noul's two parameters are one number's worth of evidence**: the fit is on")
    add("  250 `civil_comments` items whose humans say yes 9.2% of the time.  The second")
    add("  population that carries humans in the pool (the 250 `paws`/`strategyqa` items)")
    add("  has no human distribution, so the fit cannot see its 45.2% base rate - that is")
    add("  what the 235 and the test sample are for.")
    add("- **The choice and the score are judged on fewer human items in the pool** (250")
    add("  each), and the score's tilt is fitted on 250 items of three configs only.")
    add("- **One map per type is one map for a whole family of questions.**  A per-question")
    add("  map would fit better and would not be honest.")
    add("- **No new dependency, no engine, no server, no GPU.**  Everything above comes")
    add("  from the stored runs.")
    add("")
    add("The files the fit read, with their digests:")
    add("")
    add("| file | sha256 |")
    add("|---|---|")
    for label, entry in out["source"]["files"].items():
        add("| %s (`%s`) | `%s` |" % (label, entry["path"], entry["sha256"]))
    add("")
    return "\n".join(A) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
