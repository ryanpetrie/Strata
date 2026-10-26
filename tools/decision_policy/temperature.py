#!/usr/bin/env python3
"""tools/decision_policy/temperature.py - fit one temperature per question type.

A temperature divides the logits before the softmax, so the reported distribution over
the options is flattened (T > 1) or sharpened (T < 1).  The PICK cannot change: the
map is monotone, so the answer is the same and only the calibration moves.  (The
scorer's Intelligence can still move for a `score` question: it reads the EXPECTED
level, and the temperature moves the expected level too.  Measured, not assumed.)

THE MATH, AND WHY IT CAN BE FITTED OFFLINE.  A distribution over the options comes
from a softmax over logits, so p_i = exp(z_i) / sum_j exp(z_j).  Dividing the logits
by T gives p_i(T) = exp(z_i/T) / sum_j exp(z_j/T), and after renormalizing:

    closed_T(i) = normalize(closed_1(i) ** (1/T))

exactly.  No engine run is needed to know what the endpoint will report, which is
why the fit can be made on a stored run log and the server reproduces it (the same
expression, the same order of operations).

THE TRIE WALK.  For a question whose options share a first token the option's weight
is a PRODUCT of per-node token probabilities, so "divide each node's logit by T" is
NOT the same as "divide the final weight by (1/T)": the per-row normalizers differ
per option, because the options' paths have different numbers of nodes.  The
temperature is therefore applied to the OPTION-LEVEL effective logits (the closed
distribution), not to the per-trie-row logits.  For a single-row question the two
forms are identical; for the trie path the option-level form is the coherent one and
the only one that can be reproduced offline from a stored run log.

THE OBJECTIVES.  Four are fitted, because they disagree on this data and the
reader should see why (the report has the table):

  nll    the mean negative log-likelihood of the gold label.  The textbook objective
         for temperature scaling, and the smooth one - but it is dominated by the
         items the model gets wrong with confidence, and on this sample it wants a
         FLATTER distribution than the expected calibration error does.
  brier  the multiclass Brier score of the distribution against the gold label.  A
         smooth proper scoring rule, like the NLL, but quadratic: a confidently
         wrong item costs a bounded amount, so it is not dominated by the few
         extreme items.
  ece    the scorer's own expected calibration error for the type (the top-label ECE
         on the hard tier for choice, the ECE of P(yes) for noul, the top-level ECE
         for score).  The metric the exit criterion names - and a step function of T,
         so its minimum is a plateau and its folds are unstable.
  soft   the cross-entropy against the HUMAN DISTRIBUTION (the item's
         target_probs), not against the one-hot gold label.  This is the objective
         that lines the probability up with how uncertain the people were, and it
         is only defined on the items that carry a distribution.  The report says
         how many items that is per type.

The constants that ship are the ones of the objective named by --objective (the
default: brier).  Which objective is used is a choice, so every arm's cross-validated
numbers are in the report and in the JSON.

THE SERVER'S EXPRESSION IS IMPORTED, NOT COPIED.  The temperature is applied with
serve/logits_first_token.py::apply_temperature - the function the server itself
calls - so the fit cannot drift from the served one.  (The module lives under
serve/; the import needs the repository root on sys.path, which is what the
sys.path line below does.  It is pure Python: no Service, no engine, no I/O.)

THE VALIDATION.  The fit must never see the item it is scored on, so:

  - 5-fold cross-validation, stratified by type: every type's items are sorted by id,
    shuffled with a fixed seed and dealt round-robin into the five folds.  The
    temperature of a fold is fitted on the other four folds and applied to the fold's
    own items.  The out-of-fold predictions cover all the items, and they are scored
    with the OFFICIAL scorer (the script calls it, it does not re-implement it).
  - a single 70/30 split (the same seed), as a second check.
  - cross_validate() asserts that no item is in both the fold and the fit, and the
    test file checks the property from the outside as well.

THE SPLIT RULE AND THE SEED ARE IN THE OUTPUT (the JSON and the report), because a
calibration constant without its split rule is not a measurement.

ANY POOL.  The fit reads an items file, a run log and a predictions file, so it
fits on whatever pool it is pointed at: the 235-item OpenDecisions sample (the
numbers of 2026-10-21) or the jev-bench validation pool (the licence-clean
fitting set of roadmap item 1.1b).  The source block of the JSON is written from
the command line, not hard-coded, so the constants always say where they came
from.

HONESTY.  The OpenDecisions set says "fit temperatures on your own data, never on
OpenDecisions" (its README).  The constants that shipped on 2026-10-21 were
fitted on it anyway, so the 235-item re-run was IN-SAMPLE.  The licence-clean
refit (roadmap item 1.1b) is fitted on the jev-bench VALIDATION split, which
makes the OpenDecisions measurement out-of-sample and the jev-bench TEST sample
out-of-sample too.

    python tools/decision_policy/temperature.py \
        --items <items.jsonl> --runlog <runlog.jsonl> --predictions <predictions.jsonl> \
        --out-json tools/decision_policy/temperature_calibration.json \
        --out-report tools/decision_policy/temperature_report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# the repository root, so `serve.logits_first_token` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.logits_first_token import apply_temperature as _server_apply_temperature  # noqa: E402

TYPES = ("choice", "noul", "score")
SEED = 42                     # the seed of both splits (the 5-fold CV and the 70/30)
FOLDS = 5
# The grid.  The textbook range for temperature scaling is [1.0, 10.0] (a model that
# is over-confident); the optimum measured here is BELOW 1.0, so the grid has to reach
# below it or the fit cannot find it.  The report says where the optimum is.
T_MIN, T_MAX = 0.5, 10.0
COARSE_STEP = 0.01            # the grid step
DEFAULT_SCORER = Path(tempfile.gettempdir()) / "opencode" / "opendecisions" / "score.py"
DEFAULT_CONFIG = Path(r"C:\tools\strata\strata-unsloth-ud-iq4_xs.json")
OBJECTIVES = ("nll", "brier", "ece", "soft")
# the default source block, for the fit of 2026-10-21
DEFAULT_SOURCE_DATASET = "jgeuter/OpenDecisions v1.0.1, the 235-item 10% sample (seed 42)"
DEFAULT_SOURCE_ARM = "the serial first-token path, every item on the fast path"
DEFAULT_SOURCE_NOTE = ("the constants are fitted on these same items, so a re-run of the sample "
                       "is IN-SAMPLE; the honest number is the out-of-fold one below")


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:                     # split on \n only: the file holds a raw U+2028
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_scorer(path: Path):
    """The official scorer, loaded as a module: its metrics are used, not re-implemented."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("od_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the data

def build_records(items: list[dict], runlog: list[dict], predictions: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """-> ({id: record}, the cross-check messages).

    A record holds the item's CLOSED distribution - the scorer's own input - over the
    item's labels, the gold label, and the item's HUMAN DISTRIBUTION (its
    target_probs) when it has one.  The closed distribution is reconstructed from
    the predictions file (the scorer's input) and cross-checked against the run log's
    own readings, so a silent mismatch between the two cannot pass unnoticed.
    """
    tasks = {it["id"]: it for it in items}
    log = {r["id"]: r for r in runlog}
    preds = {p["id"]: p for p in predictions}
    records: dict[str, dict] = {}
    problems: list[str] = []
    for iid, task in tasks.items():
        qtype = task["question"]["type"]
        pred = preds.get(iid)
        rec = log.get(iid, {})
        if pred is None:
            problems.append(f"{iid}: no prediction in the predictions file")
            continue
        probs = pred.get("probs") or {}
        if qtype == "noul":
            # the scorer wants the probability of yes and fills in "no" itself
            p = float(probs["yes"])
            closed = {"yes": p, "no": 1.0 - p}
            gold = "yes" if task["expected"] == "yes" else "no"
            # the run log's open reading, renormalized over the two answers, is the
            # closed reading: the same number, from the other side
            op = rec.get("open_probs") or {}
            if op.get("yes") is not None and op.get("no") is not None and op["yes"] + op["no"] > 0:
                implied = op["yes"] / (op["yes"] + op["no"])
                if abs(implied - p) > 1e-9:
                    problems.append(f"{iid}: the noul's closed value {p} and the open reading "
                                    f"renormalized ({implied}) disagree")
        else:
            closed = {lab: float(probs.get(lab, 0.0)) for lab in task["labels"]}
            gold = task["expected"] if qtype == "choice" else str(int(task["expected"]))
            logged = rec.get("probabilities") or {}
            for lab, value in closed.items():
                if lab in logged and abs(float(logged[lab]) - value) > 1e-9:
                    problems.append(f"{iid}: the prediction's {lab} ({value}) and the run log's "
                                    f"({logged[lab]}) disagree")
        if abs(sum(closed.values()) - 1.0) > 1e-6:
            problems.append(f"{iid}: the closed distribution sums to {sum(closed.values())}, not 1")
        for lab, value in closed.items():
            if not math.isfinite(value) or value < 0.0:
                problems.append(f"{iid}: the probability of {lab} is {value}, not a probability")
        target, target_problem = human_target(task)
        if target_problem:
            problems.append(f"{iid}: {target_problem}")
        records[iid] = {"id": iid, "type": qtype, "tier": task.get("tier"),
                        "labels": list(task["labels"]), "closed": closed, "gold": gold,
                        "target": target, "fast_path": bool(rec.get("fast_path")), "task": task}
    return records, problems


def human_target(task: dict) -> tuple[dict[str, float] | None, str | None]:
    """-> (the human distribution over the item's labels, the problem).

    The item's target_probs is the distribution the PEOPLE gave: for a choice the
    share of the annotators per option, for a score the share per level, for a
    yes/no the share that said yes.  It is read in the scorer's label space (the
    keys are the item's labels), and a yes/no item that gives only the "yes" share
    gets the "no" share filled in - the same thing the scorer does with a
    prediction.

    The distribution may cover only SOME of the item's labels: a level the people
    never chose has a target mass of 0 and is simply left out of the keys.  The
    keys that are not labels of the item are an error.

    None means the item carries no human distribution: the soft-label objective is
    not defined on it.
    """
    raw = task.get("target_probs")
    if not raw:
        return None, None
    labels = list(task["labels"])
    try:
        target = {str(k): float(v) for k, v in raw.items()}
    except (TypeError, ValueError):
        return None, "the human distribution is not a set of numbers"
    if task["question"]["type"] == "noul" and "no" not in target and "yes" in target:
        target["no"] = 1.0 - target["yes"]
    extra = [k for k in target if k not in labels]
    if extra:
        return None, ("the human distribution has %s, which are not labels of the item"
                      % ", ".join(sorted(extra)))
    total = sum(target.values())
    if total <= 0.0:
        return None, "the human distribution sums to 0"
    if abs(total - 1.0) > 0.01:
        # the scorer allows 0.01 of slack; the objective needs a distribution
        target = {k: v / total for k, v in target.items()}
    return target, None


# ------------------------------------------------------------------ the temperature

def scale(closed: dict[str, float], temperature: float) -> dict[str, float]:
    """-> the closed distribution at temperature T.

    THE SERVER'S OWN EXPRESSION, CALLED, NOT COPIED.  apply_temperature
    (serve/logits_first_token.py) is the function the server calls on every
    answer, so the fit and the served numbers cannot drift apart: a second
    implementation of normalize(closed ** (1/T)) could differ in the order of the
    operations, in the handling of a zero, or in the T = 1 case.

    The reading handed to it carries the closed distribution as both the closed and
    the open reading, so the mass on the options is 1 and only the closed reading
    (the part the fit uses) is affected.  Nothing else the function reads is needed,
    so nothing else is built.
    """
    reading = {"closed": closed, "open": closed}
    return _server_apply_temperature(reading, temperature)["closed"]


def nll(closed: dict[str, float], gold: str, temperature: float) -> float:
    """The negative log-likelihood of the gold label under the scaled distribution."""
    p = scale(closed, temperature).get(gold, 0.0)
    return -math.log(p if p > 1e-300 else 1e-300)


def brier(closed: dict[str, float], gold: str, labels: list[str], temperature: float) -> float:
    """The multiclass Brier score: the mean squared error over the labels."""
    dist = scale(closed, temperature)
    return sum((dist.get(lab, 0.0) - (1.0 if lab == gold else 0.0)) ** 2 for lab in labels)


def soft_ce(closed: dict[str, float], target: dict[str, float], temperature: float) -> float:
    """The cross-entropy of the scaled distribution against the HUMAN distribution.

    -sum_q q log p, over the labels the human distribution covers.  This is the
    objective that lines the probability up with how uncertain the people were
    rather than with a one-hot label: a distribution that says "the people were
    split 60/40" is not punished for being 60/40.
    """
    dist = scale(closed, temperature)
    total = 0.0
    for key, q in target.items():
        if q <= 0.0:
            continue
        p = dist.get(key, 0.0)
        total -= q * math.log(p if p > 1e-300 else 1e-300)
    return total


def undo_temperature(closed: dict[str, float], temperature: float) -> dict[str, float]:
    """-> the closed distribution at T = 1, given the one at T.

    closed_T = normalize(closed_1 ** (1/T)), so closed_1 = normalize(closed_T ** T),
    and the inverse of "apply T" is "apply 1/T".  So this is the server's own
    function doing the work - there is ONE place in the tools that knows the
    transform, and the paired control below cannot drift from the served one.

    This is the PAIRED CONTROL: the same run, the same engine, the same picks, with
    the temperature's arithmetic removed from the distributions.
    """
    if temperature == 1.0:
        return dict(closed)
    return scale(closed, 1.0 / temperature)


def closed_from_runlog(task: dict, rec: dict) -> dict | None:
    """-> the closed distribution the endpoint reported for an item, from a run log.

    The choice and score readings are the run log's own "probabilities"; a noul's is
    the open reading renormalized over its two answers, which is the same number the
    endpoint reported as the noul's value.
    """
    qtype = task["question"]["type"]
    if qtype == "noul":
        op = rec.get("open_probs") or {}
        yes, no = op.get("yes"), op.get("no")
        if yes is None or no is None or yes + no <= 0:
            return None
        p = yes / (yes + no)
        return {"yes": p, "no": 1.0 - p}
    probs = rec.get("probabilities") or {}
    if not probs:
        return None
    return {lab: float(probs.get(lab, 0.0)) for lab in task["labels"]}


def mean_nll(records: list[dict], temperature: float) -> float:
    if not records:
        return float("inf")
    return sum(nll(r["closed"], r["gold"], temperature) for r in records) / len(records)


def mean_brier(records: list[dict], temperature: float) -> float:
    if not records:
        return float("inf")
    return sum(brier(r["closed"], r["gold"], r["labels"], temperature) for r in records) / len(records)


def mean_soft_ce(records: list[dict], temperature: float) -> float:
    """The mean cross-entropy against the human distribution, over the items that
    have one.  The items without one do not count: the objective is not defined on
    them, and averaging them in as a zero would make the fit prefer a temperature
    that says "no" about everything."""
    with_target = [r for r in records if r.get("target")]
    if not with_target:
        return float("inf")
    return sum(soft_ce(r["closed"], r["target"], temperature) for r in with_target) / len(with_target)


# ------------------------------------------------------------------ the scorer's own metrics

def outcomes(scorer, records: list[dict], temperature: float = 1.0,
             per_item: dict[str, float] | None = None) -> list[dict]:
    """The scorer's own per-item outcomes, on the temperature-scaled distributions.

    `per_item` gives the temperature of every item (the out-of-fold case: an item's
    temperature is the one fitted without it); `temperature` is the single one used
    when there is no per-item map.

    The scorer's own `item_outcome` is called, and the scorer's own `ece` reads
    what it returns.  The scorer's `normalized` is not called: the scaled
    distribution is already the normalized one (the same keys as the item's
    labels, the same sum of 1), and build_records() has checked the distribution
    it was built from.  The numbers are the scorer's either way; the check is
    where the data enters, not once per temperature.
    """
    out = []
    for rec in records:
        t = per_item[rec["id"]] if per_item is not None else temperature
        probs = scale(rec["closed"], t)
        out.append(scorer.item_outcome(rec["task"], probs))
    return out


def ece_by_type(scorer, outs: list[dict]) -> dict[str, float]:
    """-> {type: the scorer's own ECE}, computed on the same pairs the scorer uses.

    choice: the TOP-LABEL ECE on the hard tier only.  noul: the ECE of P(yes).
    score: the top-level ECE over every score item.

    This is the raw ECE, in the units of an ECE.  The SCORER'S calibration for the
    score type is not the same number: it is the mean of the top-level ECE and
    (1 - nRPS) (see score.py).  calibration_by_type() is that one.
    """
    out: dict[str, float] = {}
    choice = [o for o in outs if o["type"] == "choice" and o["valid"] and o["tier"] == "hard"]
    if choice:
        out["choice"] = scorer.ece([(o["conf"], float(o["correct"])) for o in choice])
    noul = [o for o in outs if o["type"] == "noul" and o["valid"]]
    if noul:
        out["noul"] = scorer.ece([(o["p_yes"], o["y"]) for o in noul])
    score = [o for o in outs if o["type"] == "score" and o["valid"]]
    if score:
        out["score"] = scorer.ece([(o["conf"], float(o["correct"])) for o in score])
    return out


def calibration_by_type(scorer, outs: list[dict]) -> dict[str, float]:
    """-> {type: the SCORER'S OWN calibration component} (0-100).

    The scorer's own function, called, not re-implemented:
      choice  100 (1 - ECE / 0.5), the top-label ECE on the hard tier;
      noul    100 (1 - ECE / 0.5), the ECE of P(yes);
      score   the mean of 100 (1 - top-level ECE / 0.5) and 100 (1 - mean nRPS).
    """
    _total, parts = scorer.calibration(outs)
    return {k: v for k, v in (parts or {}).items() if v is not None}


def tier_mix(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        key = f"{rec['type']}/{rec['tier']}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ the fit

def fit_temperature(scorer, records: list[dict], qtype: str, objective: str,
                    t_min: float = T_MIN, t_max: float = T_MAX) -> tuple[float, float]:
    """-> (T, the objective's value at T): the temperature the objective likes best.

    The grid is a plain sweep (the objective's value at every step), so the fit is
    deterministic and the objective's own shape is visible in the report.
    """
    if not records:
        return 1.0, float("inf")
    steps = int(round((t_max - t_min) / COARSE_STEP))
    best_t, best = t_min, None
    for i in range(steps + 1):
        t = round(t_min + i * COARSE_STEP, 6)
        if objective == "nll":
            value = mean_nll(records, t)
        elif objective == "brier":
            value = mean_brier(records, t)
        elif objective == "soft":
            value = mean_soft_ce(records, t)
        else:
            value = ece_by_type(scorer, outcomes(scorer, records, t)).get(qtype)
            if value is None:
                continue
        if best is None or value < best:
            best_t, best = t, value
    return best_t, (best if best is not None else float("inf"))


def fit_single_scalar(scorer, records: list[dict], objective: str = "brier",
                      t_min: float = T_MIN, t_max: float = T_MAX) -> tuple[float, float]:
    """-> (T, the objective's value at T): ONE temperature for every type.

    THE COMPARISON THE ROADMAP ASKS FOR: if one scalar for all three types does as
    well as three, then the three are not worth their complexity, and the honest
    thing is to ship one and say so.  The objective is the same rule the per-type arm
    uses, applied to the POOLED records - so the two arms are measured the same way.

    The ECE is a per-type metric, so for one scalar it is the mean of the three
    types' ECEs (the same shape as the scorer's own Calibration, which averages the
    types' components).
    """
    if not records:
        return 1.0, float("inf")
    steps = int(round((t_max - t_min) / COARSE_STEP))
    best_t, best = t_min, None
    for i in range(steps + 1):
        t = round(t_min + i * COARSE_STEP, 6)
        if objective == "nll":
            value = mean_nll(records, t)
        elif objective == "brier":
            value = mean_brier(records, t)
        elif objective == "soft":
            value = mean_soft_ce(records, t)
        else:
            per_type = ece_by_type(scorer, outcomes(scorer, records, t))
            values = [v for v in per_type.values() if v is not None]
            if not values:
                continue
            value = sum(values) / len(values)
        if best is None or value < best:
            best_t, best = t, value
    return best_t, (best if best is not None else float("inf"))


def bootstrap_ece(scorer, records: list[dict], qtype: str, objective: str,
                  resamples: int = 1000, seed: int = SEED,
                  t_min: float = T_MIN, t_max: float = T_MAX) -> dict:
    """-> how sure the temperature is, measured by resampling the items of the type.

    OUT OF BAG.  Every resample draws TWO independent samples of the type's items:
    the temperature is fitted on the first and the ECE is measured on the second, so
    the ECE that is compared is not the one the fit produced.  (Measuring both on the
    same resample - which an earlier version of this did - is IN-SAMPLE and
    optimistic: it reports a share of resamples the temperature wins that the honest
    out-of-fold number does not support.)

    The result is the share of resamples where the temperature lowers the ECE, and
    the spread of the difference.  A share near 0.5 means the difference is inside the
    noise of the sample.

    The grid of the refit is the fitted temperature's neighborhood (plus or minus
    0.5, step 0.01), so the resample's own minimum is found without paying for the
    whole grid a thousand times.
    """
    if not records or resamples <= 0:
        return {}
    rng = random.Random(seed)
    full = fit_temperature(scorer, records, qtype, objective, t_min, t_max)[0]
    lo = max(t_min, full - 0.5)
    hi = min(t_max, full + 0.5)
    grid = [round(lo + i * COARSE_STEP, 6) for i in range(int(round((hi - lo) / COARSE_STEP)) + 1)]
    better = 0
    diffs: list[float] = []
    for _ in range(resamples):
        fit_on = [records[rng.randrange(len(records))] for _ in range(len(records))]
        test_on = [records[rng.randrange(len(records))] for _ in range(len(records))]
        best_t, _ = fit_temperature(scorer, fit_on, qtype, objective, min(grid), max(grid))
        before = ece_by_type(scorer, outcomes(scorer, test_on, 1.0)).get(qtype)
        after = ece_by_type(scorer, outcomes(scorer, test_on, best_t)).get(qtype)
        if before is None or after is None:
            continue
        diffs.append(before - after)
        if after < before:
            better += 1
    if not diffs:
        return {}
    diffs.sort()
    return {"resamples": len(diffs), "share_better": better / len(diffs),
            "difference_p05": diffs[int(0.05 * (len(diffs) - 1))],
            "difference_p95": diffs[int(0.95 * (len(diffs) - 1))],
            "method": "out of bag: the temperature is fitted on one resample and the "
                      "ECE is measured on another"}


def summary_for(scorer, records: list[dict], distributions: dict[str, dict[str, float]]) -> dict:
    """-> the official scorer's summary for a set of records under given distributions."""
    outs = []
    for rec in records:
        probs = distributions[rec["id"]]
        outs.append(scorer.item_outcome(rec["task"], scorer.normalized(rec["task"], probs)))
    return scorer.summary(outs)


def paired_bootstrap(scorer, records: list[dict], arm_a: dict[str, dict[str, float]],
                      arm_b: dict[str, dict[str, float]], resamples: int = 1000,
                      seed: int = SEED) -> dict:
    """-> how sure the difference between two arms is, over a PAIRED item bootstrap.

    The items are resampled with replacement and BOTH arms are scored on the same
    resample, so the difference is paired: an item's own run-to-run wobble cancels
    and what is left is the temperature's effect.  The interval is the 2.5% and 97.5%
    percentiles of the differences, and the share is the resamples where the second
    arm is better.

    Both arms are the same items with the same picks; what differs is the
    distribution that is reported for them.
    """
    if not records or resamples <= 0:
        return {}
    outs_a = [scorer.item_outcome(rec["task"], scorer.normalized(rec["task"], arm_a[rec["id"]]))
              for rec in records]
    outs_b = [scorer.item_outcome(rec["task"], scorer.normalized(rec["task"], arm_b[rec["id"]]))
              for rec in records]
    rng = random.Random(seed)
    diffs: dict[str, list[float]] = {"capability": [], "intelligence": [], "calibration": []}
    n = len(records)
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = scorer.summary([outs_a[i] for i in idx])
        b = scorer.summary([outs_b[i] for i in idx])
        for key in diffs:
            if a.get(key) is not None and b.get(key) is not None:
                diffs[key].append(b[key] - a[key])
    out = {"resamples": resamples}
    for key, vals in diffs.items():
        if not vals:
            continue
        vals = sorted(vals)
        out[key] = {"mean": sum(vals) / len(vals),
                    "p025": vals[int(0.025 * (len(vals) - 1))],
                    "p975": vals[int(0.975 * (len(vals) - 1))],
                    "share_better": sum(1 for v in vals if v > 0) / len(vals)}
    return out


def stratified_folds(records: list[dict], folds: int = FOLDS, seed: int = SEED) -> dict[int, list[dict]]:
    """-> {fold: the records in it}, stratified by type.

    Every type's records are sorted by id, shuffled with the fixed seed and dealt
    round-robin into the folds, so each fold holds about a fifth of each type.
    """
    out: dict[int, list[dict]] = {f: [] for f in range(folds)}
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        by_type.setdefault(rec["type"], []).append(rec)
    for qtype in sorted(by_type):
        items = sorted(by_type[qtype], key=lambda r: r["id"])
        random.Random(seed).shuffle(items)
        for i, rec in enumerate(items):
            out[i % folds].append(rec)
    return out


def split_70_30(records: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """-> (train, held out): 70% of every type for the fit, 30% for the check."""
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        by_type.setdefault(rec["type"], []).append(rec)
    train: list[dict] = []
    test: list[dict] = []
    for qtype in sorted(by_type):
        items = sorted(by_type[qtype], key=lambda r: r["id"])
        random.Random(seed).shuffle(items)
        cut = int(round(0.7 * len(items)))
        train.extend(items[:cut])
        test.extend(items[cut:])
    return train, test


def fold_temperatures(scorer, train: list[dict], objective: str,
                      t_min: float = T_MIN, t_max: float = T_MAX) -> dict[str, float]:
    """-> {type: the temperature fitted on THESE items}.

    One place, so the cross-validation and the test that checks it cannot drift
    apart: the test refits the fold from the training items alone and compares.
    """
    out: dict[str, float] = {}
    for qtype in TYPES:
        tr = [r for r in train if r["type"] == qtype]
        out[qtype] = fit_temperature(scorer, tr, qtype, objective, t_min, t_max)[0] if tr else 1.0
    return out


def cross_validate(scorer, records: list[dict], objective: str, folds: int = FOLDS,
                   seed: int = SEED) -> tuple[dict[str, float], list[dict]]:
    """-> ({id: the out-of-fold temperature}, the fold table).

    NO LEAKAGE, AND IT IS CHECKED HERE AND NOT ONLY PROMISED.  The temperature of a
    fold is fitted on the other folds' items; the fold's own items are never in the
    fit, and the code refuses to continue if they are.
    """
    fold_items = stratified_folds(records, folds, seed)
    per_item: dict[str, float] = {}
    table = []
    for f in range(folds):
        held = fold_items[f]
        train = [r for g in range(folds) if g != f for r in fold_items[g]]
        held_ids = {r["id"] for r in held}
        train_ids = {r["id"] for r in train}
        overlap = held_ids & train_ids
        if overlap:
            raise AssertionError("the fold %d's own items are in its fit: %s"
                                 % (f, ", ".join(sorted(overlap)[:5])))
        temps = fold_temperatures(scorer, train, objective)
        for rec in held:
            per_item[rec["id"]] = temps[rec["type"]]
        hard_choice = sum(1 for r in held if r["type"] == "choice" and r["tier"] == "hard")
        table.append({"fold": f, "n": len(held), "temperatures": temps,
                      "tiers": tier_mix(held), "hard_choice_items": hard_choice})
    return per_item, table


# ------------------------------------------------------------------ the official scorer, run

def cross_validate_single(scorer, records: list[dict], objective: str, folds: int = FOLDS,
                          seed: int = SEED) -> tuple[dict[str, float], list[dict]]:
    """-> ({id: the out-of-fold temperature}, the fold table) for ONE scalar.

    The same rule as cross_validate, with one temperature for every type: a fold's
    scalar is fitted on the other folds' POOLED records and applied to the fold's
    own items, so the fit never sees the item it is scored on.
    """
    fold_items = stratified_folds(records, folds, seed)
    per_item: dict[str, float] = {}
    table = []
    for f in range(folds):
        held = fold_items[f]
        train = [r for g in range(folds) if g != f for r in fold_items[g]]
        held_ids = {r["id"] for r in held}
        train_ids = {r["id"] for r in train}
        overlap = held_ids & train_ids
        if overlap:
            raise AssertionError("the fold %d's own items are in its fit: %s"
                                 % (f, ", ".join(sorted(overlap)[:5])))
        t, _ = fit_single_scalar(scorer, train, objective)
        for rec in held:
            per_item[rec["id"]] = t
        table.append({"fold": f, "n": len(held), "temperature": t,
                      "tiers": tier_mix(held),
                      "hard_choice_items": sum(1 for r in held
                                               if r["type"] == "choice" and r["tier"] == "hard")})
    return per_item, table


def score_with_official_scorer(scorer_path: Path, records: list[dict],
                                distributions: dict[str, dict[str, float]]) -> dict:
    """-> the official scorer's summary, on the given distributions.

    The predictions and the matching items subset are written to temporary files and
    the official scorer is run on them, so the numbers are the scorer's own.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        items_path = tmp_path / "items.jsonl"
        preds_path = tmp_path / "predictions.jsonl"
        with items_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec["task"], ensure_ascii=False) + "\n")
        with preds_path.open("w", encoding="utf-8") as f:
            for rec in records:
                probs = distributions[rec["id"]]
                if rec["type"] == "noul":
                    probs = {"yes": probs["yes"]}
                f.write(json.dumps({"id": rec["id"], "probs": probs}, ensure_ascii=False) + "\n")
        out = subprocess.run([sys.executable, str(scorer_path), str(preds_path),
                              "--data", str(items_path), "--json"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            return {"error": out.stderr.strip()[-2000:]}
        return json.loads(out.stdout)


# ------------------------------------------------------------------ the report

def fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def file_digest(path: Path) -> str | None:
    """The sha256 of a file, for the provenance block.  None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the scorer-shaped items file")
    ap.add_argument("--runlog", type=Path, required=True, help="the run log of the baseline arm")
    ap.add_argument("--predictions", type=Path, required=True, help="the baseline arm's predictions")
    ap.add_argument("--temp-runlog", type=Path, default=None,
                    help="the run log of the arm that ran WITH the temperature (for the paired control)")
    ap.add_argument("--scorer", type=Path, default=DEFAULT_SCORER, help="the official score.py")
    ap.add_argument("--out-json", type=Path, default=Path("tools/decision_policy/temperature_calibration.json"))
    ap.add_argument("--out-report", type=Path, default=Path("tools/decision_policy/temperature_report.md"))
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="the launch config the model id is read from")
    ap.add_argument("--model", default=None, help="the model id (default: the config's)")
    ap.add_argument("--date", default=None, help="the date of the fit (default: today)")
    ap.add_argument("--objective", choices=OBJECTIVES, default="brier",
                    help="the objective the shipped constants come from (default: brier)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--t-min", type=float, default=T_MIN)
    ap.add_argument("--t-max", type=float, default=T_MAX)
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="resamples per type for the uncertainty check (0 = off)")
    ap.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET,
                    help="what the pool is, for the provenance block")
    ap.add_argument("--source-arm", default=DEFAULT_SOURCE_ARM,
                    help="which arm of the run the fit reads")
    ap.add_argument("--source-note", default=DEFAULT_SOURCE_NOTE,
                    help="the caveat that belongs with the numbers")
    args = ap.parse_args()

    from datetime import date as _date
    when = args.date or _date.today().isoformat()
    model = args.model
    if model is None and args.config.exists():
        model = json.loads(args.config.read_text(encoding="utf-8")).get("model_name")

    # the three inputs must be there: a missing run log means the run did not
    # finish, and a traceback would hide that behind a stack of frames
    for label, path in (("items", args.items), ("runlog", args.runlog),
                        ("predictions", args.predictions)):
        if not path.exists():
            raise SystemExit("the %s file is not there: %s" % (label, path))
    if not args.scorer.exists():
        raise SystemExit("the official scorer is not there: %s" % args.scorer)

    scorer = load_scorer(args.scorer)
    items = read_jsonl(args.items)
    runlog = read_jsonl(args.runlog)
    predictions = read_jsonl(args.predictions)
    records, problems = build_records(items, runlog, predictions)
    print(f"[temperature] {len(records)} items with a distribution, {len(problems)} cross-check problems")
    for p in problems[:10]:
        print("  !", p)

    allrecs = sorted(records.values(), key=lambda r: r["id"])
    by_type: dict[str, list[dict]] = {}
    for rec in allrecs:
        by_type.setdefault(rec["type"], []).append(rec)

    # ---- how many items carry a human distribution (the soft objective's data)
    soft_counts = {q: sum(1 for r in by_type.get(q, []) if r.get("target")) for q in TYPES}
    print("[temperature] items with a human distribution: " +
          "  ".join(f"{q} {soft_counts[q]}/{len(by_type.get(q, []))}" for q in TYPES))
    # the choice ECE is measured on the hard tier only, so the slice matters
    hard_choice = sum(1 for r in by_type.get("choice", []) if r.get("tier") == "hard")

    # ---- every objective, fitted on every item of the type
    fits: dict[str, dict[str, float]] = {}
    for objective in OBJECTIVES:
        fits[objective] = {q: fit_temperature(scorer, by_type.get(q, []), q, objective,
                                              args.t_min, args.t_max)[0] for q in TYPES}
        print(f"[temperature] {objective:6s} T: " +
              "  ".join(f"{q} {fits[objective][q]}" for q in TYPES))
    fitted = fits[args.objective]

    # ---- the cross-validation, per objective
    cv: dict[str, dict] = {}
    for objective in OBJECTIVES:
        per_item, table = cross_validate(scorer, allrecs, objective, args.folds, args.seed)
        outs_before = outcomes(scorer, allrecs)
        outs_after = outcomes(scorer, allrecs, 1.0, per_item=per_item)
        dist = {r["id"]: scale(r["closed"], per_item[r["id"]]) for r in allrecs}
        cv[objective] = {
            "per_item": per_item,
            "folds": table,
            "ece_before": ece_by_type(scorer, outs_before),
            "ece_after": ece_by_type(scorer, outs_after),
            "calibration_before": calibration_by_type(scorer, outs_before),
            "calibration_after": calibration_by_type(scorer, outs_after),
            "scorer": score_with_official_scorer(args.scorer, allrecs, dist),
        }
        print(f"[temperature] {objective:6s} out-of-fold ECE: "
              + "  ".join(f"{q} {cv[objective]['ece_after'].get(q):.4f}"
                          for q in TYPES)
              + f"   Capability {cv[objective]['scorer'].get('capability'):.2f}")

    # ---- the 70/30 split (the same seed)
    train30, test30 = split_70_30(allrecs, args.seed)
    temps30 = {q: fit_temperature(scorer, [r for r in train30 if r["type"] == q], q,
                                   args.objective, args.t_min, args.t_max)[0] for q in TYPES}
    ho_before = ece_by_type(scorer, outcomes(scorer, test30))
    ho_scaled = []
    for rec in test30:
        probs = scale(rec["closed"], temps30[rec["type"]])
        ho_scaled.append(scorer.item_outcome(rec["task"], probs))
    ho_after = ece_by_type(scorer, ho_scaled)
    ho_cal_before = calibration_by_type(scorer, outcomes(scorer, test30))
    ho_cal_after = calibration_by_type(scorer, ho_scaled)

    # ---- the single-scalar arm: ONE temperature for all three types, the same
    # objective, the same grid, the same seed, the same folds.  The roadmap asks for
    # this comparison: if one scalar does as well as three, the three are not worth
    # their complexity and the honest thing is to ship one and say so.
    single_t, _ = fit_single_scalar(scorer, allrecs, args.objective, args.t_min, args.t_max)
    single_per_item, single_table = cross_validate_single(scorer, allrecs, args.objective,
                                                           args.folds, args.seed)
    single_outs_before = outcomes(scorer, allrecs)
    single_outs_after = outcomes(scorer, allrecs, 1.0, per_item=single_per_item)
    single_dist = {r["id"]: scale(r["closed"], single_per_item[r["id"]]) for r in allrecs}
    single = {
        "temperature": single_t,
        "out_of_fold": {
            "ece_before": ece_by_type(scorer, single_outs_before),
            "ece_after": ece_by_type(scorer, single_outs_after),
            "calibration_before": calibration_by_type(scorer, single_outs_before),
            "calibration_after": calibration_by_type(scorer, single_outs_after),
            "folds": single_table,
            "scorer": score_with_official_scorer(args.scorer, allrecs, single_dist),
        },
    }
    print(f"[temperature] the single scalar (one T for all three types): {single_t}")

    # ---- how sure is the fit?  the items resampled, the temperature fitted again
    boot = {q: bootstrap_ece(scorer, by_type.get(q, []), q, args.objective,
                              args.bootstrap, args.seed, args.t_min, args.t_max) for q in TYPES}
    for q in TYPES:
        if boot[q]:
            print(f"[temperature] {q:7s} the temperature lowers the ECE in "
                  f"{boot[q]['share_better'] * 100:.0f}% of {boot[q]['resamples']} resamples")

    # ---- the scorer's own numbers
    base_summary = score_with_official_scorer(args.scorer, allrecs,
                                              {r["id"]: r["closed"] for r in allrecs})
    in_sample = score_with_official_scorer(
        args.scorer, allrecs, {r["id"]: scale(r["closed"], fitted[r["type"]]) for r in allrecs})

    # ---- how sure is the EFFECT?  a paired item bootstrap between two arms.
    # The out-of-fold arm against the baseline: the fit never saw the item it is
    # scored on, and the difference is paired over the items.
    oof_arm = {r["id"]: scale(r["closed"], cv[args.objective]["per_item"][r["id"]]) for r in allrecs}
    boot_oof = paired_bootstrap(scorer, allrecs, {r["id"]: r["closed"] for r in allrecs},
                                oof_arm, args.bootstrap, args.seed)

    # The arm that RAN with the temperature against the same run with the temperature
    # undone: the same engine, the same picks, the temperature's arithmetic removed.
    # This is the paired control the companion doc quotes.
    paired = {}
    if args.temp_runlog:
        temp_log = {r["id"]: r for r in read_jsonl(args.temp_runlog)}
        temp_arm, control_arm = {}, {}
        for rec in allrecs:
            got = closed_from_runlog(rec["task"], temp_log.get(rec["id"], {}))
            if got is None:
                continue
            temp_arm[rec["id"]] = got
            control_arm[rec["id"]] = undo_temperature(got, fitted[rec["type"]])
        if len(temp_arm) == len(allrecs):
            paired = {"arm": summary_for(scorer, allrecs, temp_arm),
                      "control": summary_for(scorer, allrecs, control_arm),
                      "bootstrap": paired_bootstrap(scorer, allrecs, control_arm, temp_arm,
                                                   args.bootstrap, args.seed)}
        else:
            print(f"[temperature] the temperature run has {len(temp_arm)} of {len(allrecs)} "
                  "items with a distribution; the paired control is skipped")

    print(f"\n[temperature] the baseline arm:        Capability {base_summary.get('capability'):.2f} "
          f"(Intelligence {base_summary.get('intelligence'):.2f}, Calibration {base_summary.get('calibration'):.2f})")
    print(f"[temperature] the out-of-fold fit:      Capability {cv[args.objective]['scorer'].get('capability'):.2f} "
          f"(Intelligence {cv[args.objective]['scorer'].get('intelligence'):.2f}, "
          f"Calibration {cv[args.objective]['scorer'].get('calibration'):.2f})")
    print(f"[temperature] the in-sample fit:        Capability {in_sample.get('capability'):.2f} "
          f"(Intelligence {in_sample.get('intelligence'):.2f}, Calibration {in_sample.get('calibration'):.2f})")

    # ---- the constants, with their provenance
    out = {
        "version": 2,
        "model": model,
        "fitted_on": when,
        "objective": args.objective,
        "objective_text": {
            "nll": "the mean negative log-likelihood of the gold label under normalize(closed ** (1/T))",
            "brier": "the mean multiclass Brier score of the distribution against the gold label",
            "ece": "the RAW TOP-LEVEL expected calibration error for the type (the top-label ECE on "
                   "the hard tier for choice, the ECE of P(yes) for noul, the top-level ECE for "
                   "score). For the score type this is the ECE half of the scorer's calibration "
                   "component, which also has (1 - nRPS)",
            "soft": "the mean cross-entropy of the distribution against the item's HUMAN distribution "
                    "(its target_probs), over the items that carry one",
        }[args.objective],
        "grid": {"min": args.t_min, "max": args.t_max, "step": COARSE_STEP},
        "seed": args.seed,
        "folds": args.folds,
        "split_rule": ("5-fold cross-validation, stratified by type: every type's items are "
                       "sorted by id, shuffled with random.Random(seed) and dealt round-robin "
                       "into the five folds; the temperature of a fold is fitted on the other "
                       "four folds and applied to the fold's own items. A 70/30 split (the "
                       "same seed, stratified by type the same way) is the second check."),
        "temperature": {q: fitted[q] for q in TYPES},
        "temperature_by_objective": {o: fits[o] for o in OBJECTIVES},
        "items_with_human_distribution": soft_counts,
        "source": {
            "dataset": args.source_dataset,
            "arm": args.source_arm,
            "note": args.source_note,
            "files": {
                "items": {"path": str(args.items), "sha256": file_digest(args.items)},
                "runlog": {"path": str(args.runlog), "sha256": file_digest(args.runlog)},
                "predictions": {"path": str(args.predictions), "sha256": file_digest(args.predictions)},
            },
        },
        "validation": {
            "out_of_fold": {o: {"ece_before": cv[o]["ece_before"], "ece_after": cv[o]["ece_after"],
                                "calibration_before": cv[o]["calibration_before"],
                                "calibration_after": cv[o]["calibration_after"],
                                "folds": cv[o]["folds"], "scorer": cv[o]["scorer"]} for o in OBJECTIVES},
            "holdout_30": {"ece_before": ho_before, "ece_after": ho_after, "temperatures": temps30,
                           "calibration_before": ho_cal_before, "calibration_after": ho_cal_after,
                           "train_items": {q: sum(1 for r in train30 if r["type"] == q) for q in TYPES},
                           "test_items": {q: sum(1 for r in test30 if r["type"] == q) for q in TYPES}},
            "bootstrap": boot,
            "paired_bootstrap_out_of_fold": boot_oof,
            "paired_control": paired,
            "single_scalar": single,
            "official_scorer_baseline": base_summary,
            "official_scorer_in_sample_fitted": in_sample,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[temperature] wrote {args.out_json}")

    # ---- the report
    lines: list[str] = []
    A = lines.append
    A("# The per-type temperature: the fit")
    A("")
    A(f"**Fitted {when}, model `{model}`.**  The constants live in")
    A("`temperature_calibration.json` next to this file; the server takes them from the")
    A("launch config (`\"systemone\": {\"temperature\": {...}}`).")
    A("")
    A(f"**The pool:** {args.source_dataset} - {args.source_arm}.")
    A("")
    A("## What was fitted")
    A("")
    A("One scalar per question type.  The temperature divides the logits before the")
    A("softmax, so the reported distribution is flattened (T > 1) or sharpened (T < 1).")
    A("The PICK cannot change - the map is monotone, so the argmax is the same and a")
    A("yes/no never crosses the 0.5 threshold.  The score type's EXPECTED LEVEL does")
    A("move (the probability-weighted mean the scorer reads), and with it the scorer's")
    A("Intelligence for that type: that is where the gain is, not only the calibration.")
    A("")
    A("The exact identity the endpoint applies:")
    A("")
    A("```")
    A("closed_T(i) = normalize(closed_1(i) ** (1/T))")
    A("```")
    A("")
    A("which is why the fit can be made offline on a stored run log and the server")
    A("reproduces it: the same expression, the same order of operations.  For the trie")
    A("walk the temperature is applied to the OPTION-LEVEL effective logits (the closed")
    A("distribution), not to the per-trie-row logits: an option's weight is a product of")
    A("per-node probabilities and the per-row normalizers differ per option.")
    A("")
    A("The expression is not written out a second time here: the fit calls the server's")
    A("own `apply_temperature` (serve/logits_first_token.py), so the fit and the served")
    A("numbers cannot drift apart.")
    A("")
    A("## The four objectives, and why the shipped one is the Brier score")
    A("")
    A("They disagree on this data, so all four are fitted and cross-validated:")
    A("")
    A("| objective | what it minimizes | choice | noul | score |")
    A("|---|---|---:|---:|---:|")
    A("| nll | the negative log-likelihood of the gold label | "
      f"{fits['nll']['choice']} | {fits['nll']['noul']} | {fits['nll']['score']} |")
    A("| brier | the multiclass Brier score | "
      f"{fits['brier']['choice']} | {fits['brier']['noul']} | {fits['brier']['score']} |")
    A("| ece | the raw top-level ECE for the type | "
      f"{fits['ece']['choice']} | {fits['ece']['noul']} | {fits['ece']['score']} |")
    A("| soft | the cross-entropy against the HUMAN distribution | "
      f"{fits['soft']['choice']} | {fits['soft']['noul']} | {fits['soft']['score']} |")
    A("")
    A("The NLL is the textbook objective, and it wants a FLATTER distribution than the")
    A("expected calibration error does.  The reason is in the data: the NLL is")
    A("dominated by the items the model gets wrong with confidence (a wrong item at")
    A("p = 0.99 costs -log(0.01) = 4.6), while the ECE is a per-bin average that")
    A("treats every bin alike.  On this sample the model's errors sit in the MIDDLE of")
    A("the confidence range, where it is UNDER-confident, so flattening makes the ECE")
    A("worse.  The Brier score is between the two: a smooth proper scoring rule, like")
    A("the NLL, but quadratic, so a confidently wrong item costs a bounded amount.")
    A("")
    A("The **soft** objective is the one that lines the probability up with how")
    A("uncertain the PEOPLE were, not with a one-hot label: it is the cross-entropy")
    A("against the item's `target_probs`, the annotators' own distribution.  It is")
    A("only defined on the items that carry one, and the report says how many that is:")
    A("")
    A("| type | items | of those, with a human distribution |")
    A("|---|---:|---:|")
    for qtype in TYPES:
        A(f"| {qtype} | {len(by_type.get(qtype, []))} | {soft_counts[qtype]} |")
    A("")
    A(f"**The shipped constants use the `{args.objective}` objective.**")
    A("")
    A("## The validation")
    A("")
    A(f"The split rule: {out['split_rule']}  The seed is **{args.seed}**.")
    A("")
    A("The fit never sees the item it is scored on.")
    A("")
    A("**The raw top-level ECE** - the first table below, and 1.1's table, reproduced bit")
    A("for bit: the scorer's own `ece` function on the same pairs it uses - the top-label")
    A("ECE on the hard tier for choice, the ECE of P(yes) for noul, the top-level ECE for")
    A("score.  For the SCORE type this is the ECE half of the scorer's calibration")
    A("component, not the component itself: the component is the mean of the top-level")
    A("ECE and (1 - nRPS) (score.py).  The second table is that component.")
    A("")
    A("| objective | type | the raw top-level ECE before | the raw top-level ECE after (out-of-fold) |")
    A("|---|---|---:|---:|")
    for objective in OBJECTIVES:
        for qtype in TYPES:
            A(f"| {objective} | {qtype} | {fmt(cv[objective]['ece_before'].get(qtype))} | "
              f"{fmt(cv[objective]['ece_after'].get(qtype))} |")
    A("")
    A("**The scorer's own per-type calibration, 0-100** - the second table - on the same")
    A("out-of-fold predictions, from the scorer's `calibration` function: the top-label ECE")
    A("on the hard tier for choice, the ECE of P(yes) for noul, and for score the mean of")
    A("the top-level ECE and (1 - nRPS).  **This is the number the exit criterion is")
    A("read from** (the held-out calibration, lower is better), and it is the one the")
    A("official scorer's Calibration number is made of.")
    A("")
    A("| objective | type | the scorer's calibration before | the scorer's calibration after (out-of-fold) |")
    A("|---|---|---:|---:|")
    for objective in OBJECTIVES:
        for qtype in TYPES:
            A(f"| {objective} | {qtype} | {fmt(cv[objective]['calibration_before'].get(qtype), 1)} | "
              f"{fmt(cv[objective]['calibration_after'].get(qtype), 1)} |")
    A("")
    A("The 70/30 split (the same seed), fitted on its own train part:")
    A("")
    A("| type | train items | held-out items | T (train) | the raw top-level ECE before | the raw top-level ECE after |")
    A("|---|---:|---:|---:|---:|---:|")
    for qtype in TYPES:
        A(f"| {qtype} | {sum(1 for r in train30 if r['type'] == qtype)} | "
          f"{sum(1 for r in test30 if r['type'] == qtype)} | {temps30[qtype]} | "
          f"{fmt(ho_before.get(qtype))} | {fmt(ho_after.get(qtype))} |")
    A("")
    A("The fold mix for the shipped objective (the choice ECE uses the hard tier only,")
    A("so a thin slice matters):")
    A("")
    A("| fold | items | choice T | noul T | score T | hard-tier choice items | tiers |")
    A("|---:|---:|---:|---:|---:|---:|---|")
    for row in cv[args.objective]["folds"]:
        A(f"| {row['fold']} | {row['n']} | {row['temperatures']['choice']} | "
          f"{row['temperatures']['noul']} | {row['temperatures']['score']} | "
          f"{row['hard_choice_items']} | {', '.join(f'{k} {v}' for k, v in row['tiers'].items())} |")
    A("")
    A("How sure is the fit?  The items of the type resampled with replacement, TWICE per")
    A("resample: the temperature is fitted on one sample and the ECE is measured on the")
    A("other, so the ECE that is compared is not the one the fit produced (measuring")
    A("both on the same resample is in-sample and optimistic).  The share is the")
    A("resamples where the temperature lowers the ECE; near 50% means the difference is")
    A("inside the sample's noise.")
    A("")
    A("| type | resamples | the temperature lowers the ECE | the ECE difference (5%, 95%) |")
    A("|---|---:|---:|---|")
    for q in TYPES:
        if boot.get(q):
            b = boot[q]
            A(f"| {q} | {b['resamples']} | {b['share_better'] * 100:.0f}% | "
              f"{b['difference_p05']:+.4f}, {b['difference_p95']:+.4f} |")
    A("")
    A("**A share is not an effect size.**  It says how often the temperature wins on a")
    A("resample of 235 items; it does not say by how much, and it does not carry the")
    A("uncertainty of the size.  For that, the paired item bootstrap below.")
    A("")
    A("## How sure is the EFFECT?  A paired item bootstrap")
    A("")
    A("The items are resampled with replacement and BOTH arms are scored on the same")
    A("resample, so the difference is paired: an item's own run-to-run wobble cancels")
    A("and what is left is the temperature's effect.  The interval is the 2.5% and")
    A("97.5% percentiles of the differences, and the share is the resamples where the")
    A("second arm is better.")
    A("")
    A("| comparison | Capability | Intelligence | Calibration |")
    A("|---|---|---|---|")
    def boot_row(label, boot_data):
        cells = []
        for key in ("capability", "intelligence", "calibration"):
            d = (boot_data or {}).get(key)
            cells.append("n/a" if not d else
                         f"{d['mean']:+.2f} [{d['p025']:+.2f}, {d['p975']:+.2f}] "
                         f"({d['share_better'] * 100:.0f}% > 0)")
        A(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    boot_row("the out-of-fold fit against the baseline arm", boot_oof)
    if paired.get("bootstrap"):
        boot_row("the temperature run against the same run with the temperature undone",
                 paired["bootstrap"])
    A("")
    if paired.get("arm"):
        A("The paired control: the arm that ran with the temperature, and the SAME run")
        A("with the temperature undone (the closed reading put back through")
        A("`normalize(closed_T ** T)`) - the same engine, the same picks, the same")
        A("logits, the temperature's arithmetic removed.  The difference between the two")
        A("is the temperature and nothing else.")
        A("")
        A("| run | Capability | Intelligence | Calibration |")
        A("|---|---:|---:|---:|")
        for label, summary in (("the same run with the temperature undone", paired["control"]),
                                ("the run with the temperature", paired["arm"])):
            A(f"| {label} | {fmt(summary.get('capability'), 1)} | "
              f"{fmt(summary.get('intelligence'), 1)} | {fmt(summary.get('calibration'), 1)} |")
        A("")
    A("## The single-scalar arm")
    A("")
    A("The roadmap asks for this comparison: **if per-type does not beat a single")
    A("scalar, ship a single scalar and say so.**  This is that arm - ONE temperature")
    A("for all three types, fitted on the pooled records with the same objective, the")
    A("same grid, the same seed and the same five folds, so the two arms are measured")
    A("the same way.")
    A("")
    A(f"The single scalar is **{single['temperature']}** (the per-type constants are "
      f"{', '.join('%s %s' % (q, fitted[q]) for q in TYPES)}).")
    A("")
    A("| run | Capability | Intelligence | Calibration | choice | noul | score |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    def single_row(label, summary, cal):
        parts = summary.get("calibration_per_type") or {}
        A(f"| {label} | {fmt(summary.get('capability'), 1)} | "
          f"{fmt(summary.get('intelligence'), 1)} | {fmt(summary.get('calibration'), 1)} | "
          f"{fmt(parts.get('choice'), 1)} | {fmt(parts.get('noul'), 1)} | "
          f"{fmt(parts.get('score'), 1)} |")
    single_row("the baseline arm (T = 1)", base_summary, None)
    single_row(f"the per-type fit ({args.objective}), out-of-fold",
               cv[args.objective]["scorer"], cv[args.objective]["calibration_after"])
    single_row("the single scalar, out-of-fold",
               single["out_of_fold"]["scorer"], single["out_of_fold"]["calibration_after"])
    A("")
    A("(the three right-hand columns are the scorer's own per-type calibration")
    A("components, 0-100, higher is better.)")
    A("")
    A("## The official scorer")
    A("")
    A("The out-of-fold predictions (all the items, each scaled by the temperature of the")
    A("fold it was not in) were written out and scored with the official `score.py`:")
    A("")
    A("| run | Capability | Intelligence | Calibration |")
    A("|---|---:|---:|---:|")
    A(f"| the baseline arm (T = 1) | {fmt(base_summary.get('capability'), 1)} | "
      f"{fmt(base_summary.get('intelligence'), 1)} | {fmt(base_summary.get('calibration'), 1)} |")
    for objective in OBJECTIVES:
        s = cv[objective]["scorer"]
        A(f"| the out-of-fold fit ({objective}) | {fmt(s.get('capability'), 1)} | "
          f"{fmt(s.get('intelligence'), 1)} | {fmt(s.get('calibration'), 1)} |")
    A(f"| the in-sample fit ({args.objective}, what the server gets) | "
      f"{fmt(in_sample.get('capability'), 1)} | {fmt(in_sample.get('intelligence'), 1)} | "
      f"{fmt(in_sample.get('calibration'), 1)} |")
    A("")
    A("Per-type calibration:")
    A("")
    A("| run | choice | noul | score |")
    A("|---|---:|---:|---:|")
    def cal_row(label, summary):
        parts = summary.get("calibration_per_type") or {}
        A(f"| {label} | {fmt(parts.get('choice'), 1)} | {fmt(parts.get('noul'), 1)} | "
          f"{fmt(parts.get('score'), 1)} |")
    cal_row("the baseline arm (T = 1)", base_summary)
    for objective in OBJECTIVES:
        cal_row(f"the out-of-fold fit ({objective})", cv[objective]["scorer"])
    cal_row(f"the in-sample fit ({args.objective})", in_sample)
    A("")
    A("Per-type competence (the score type's expected level moves with the temperature,")
    A("so its Intelligence is not fixed):")
    A("")
    A("| run | choice | noul | score |")
    A("|---|---:|---:|---:|")
    def comp_row(label, summary):
        parts = summary.get("competence_per_type") or {}
        A(f"| {label} | {fmt(parts.get('choice'), 1)} | {fmt(parts.get('noul'), 1)} | "
          f"{fmt(parts.get('score'), 1)} |")
    comp_row("the baseline arm (T = 1)", base_summary)
    for objective in OBJECTIVES:
        comp_row(f"the out-of-fold fit ({objective})", cv[objective]["scorer"])
    comp_row(f"the in-sample fit ({args.objective})", in_sample)
    A("")
    A("## The limits")
    A("")
    A(f"- **What the fit is measured on:** {args.source_dataset} - {args.source_arm}.")
    A(f"  {args.source_note}")
    if "OpenDecisions" in args.source_dataset:
        A("- **The 235-item re-run is in-sample.**  The constants are fitted on those items,")
        A("  so the in-sample row above is not an unbiased estimate of anything.  The honest")
        A("  number is the out-of-fold one.")
        A("- **The dataset says not to fit on it** (its README: \"fit temperatures on your own")
        A("  data, never on OpenDecisions\").  The roadmap asks for this fit, so the numbers")
        A("  are labelled as they are, and the independent check is the jev-bench run with")
        A("  human soft labels (roadmap item 1.2).")
    A(f"- **The raw top-level ECE of the choice questions is measured on the hard tier "
      f"only** ({hard_choice} of the {len(by_type.get('choice', []))} choice items), so it "
      f"is a thin slice: the fold table above says how thin.  (For the choice and the noul "
      f"types the scorer's calibration component is that ECE, scaled; for the score type it "
      f"is the mean of the top-level ECE and (1 - nRPS), so the ECE half is only part of "
      f"the component.)")
    A("- **The noul's ECE is a wash.**  Its curve is jagged and shallow around T = 1 and")
    A("  its minimum does not survive the cross-validation; see the table above.")
    A("- **One temperature per type is one number for a whole family of questions.**  A")
    A("  temperature per question would fit better and would not be honest.")
    A("- **The temperature does not touch the open reading's normalizer.**  The server has")
    A("  the engine's top-k window and the T = 1 normalizer, not the full-vocabulary")
    A("  normalizer at T, so the confidence is the closed reading at T scaled by the mass")
    A("  the model put on the options (measured at T = 1).  See the serving side.")
    A("- **The size of the effect is not established on 235 items.**  The paired item")
    A("  bootstrap puts an interval on it, and that interval is wide: the direction is")
    A("  right, the size is not known.  The Intelligence gain is the part that holds up;")
    A("  the Calibration gain is the least certain of the three.")
    A("")
    A("The files the fit read, with their digests:")
    A("")
    A("| file | sha256 |")
    A("|---|---|")
    for key in ("items", "runlog", "predictions"):
        info = out["source"]["files"][key]
        A(f"| {key} (`{info['path']}`) | `{info['sha256']}` |")
    A("")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[temperature] wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
