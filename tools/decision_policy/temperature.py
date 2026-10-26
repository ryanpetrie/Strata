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

THE OBJECTIVES.  Three are fitted, because they disagree on this data and the
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

The constants that ship are the ones of the objective named by --objective (the
default: brier).  Which objective is used is a choice, so every arm's cross-validated
numbers are in the report and in the JSON.

THE VALIDATION.  The fit must never see the item it is scored on, so:

  - 5-fold cross-validation, stratified by type: every type's items are sorted by id,
    shuffled with a fixed seed and dealt round-robin into the five folds.  The
    temperature of a fold is fitted on the other four folds and applied to the fold's
    own items.  The out-of-fold predictions cover all 235 items, and they are scored
    with the OFFICIAL scorer (the script calls it, it does not re-implement it).
  - a single 70/30 split (the same seed), as a second check.

THE SPLIT RULE AND THE SEED ARE IN THE OUTPUT (the JSON and the report), because a
calibration constant without its split rule is not a measurement.

HONESTY.  The OpenDecisions set says "fit temperatures on your own data, never on
OpenDecisions" (its README).  This script fits on OpenDecisions because the roadmap
asks for it, so the 235-item re-run is IN-SAMPLE and the honest headline is the
out-of-fold (cross-validated) calibration improvement.  The independent check is the
jev-bench run with human soft labels (roadmap item 1.2).

    python tools/decision_policy/temperature.py \
        --items <items.jsonl> --runlog <runlog.jsonl> --predictions <predictions.jsonl> \
        --out-json tools/decision_policy/temperature_calibration.json \
        --out-report tools/decision_policy/temperature_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

TYPES = ("choice", "noul", "score")
SEED = 42                     # the seed of both splits (the 5-fold CV and the 70/30)
FOLDS = 5
# The grid.  The textbook range for temperature scaling is [1.0, 10.0] (a model that
# is over-confident); the optimum measured here is BELOW 1.0, so the grid has to reach
# below it or the fit cannot find it.  The report says where the optimum is.
T_MIN, T_MAX = 0.5, 10.0
COARSE_STEP = 0.01            # the grid step
DEFAULT_SCORER = Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\score.py")
DEFAULT_CONFIG = Path(r"C:\tools\strata\strata-unsloth-ud-iq4_xs.json")
OBJECTIVES = ("nll", "brier", "ece")


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
    item's labels, and the gold label.  The closed distribution is reconstructed from
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
        records[iid] = {"id": iid, "type": qtype, "tier": task.get("tier"),
                        "labels": list(task["labels"]), "closed": closed, "gold": gold,
                        "fast_path": bool(rec.get("fast_path")), "task": task}
    return records, problems


# ------------------------------------------------------------------ the temperature

def scale(closed: dict[str, float], temperature: float) -> dict[str, float]:
    """-> the closed distribution at temperature T: normalize(closed ** (1/T)).

    The exact identity the server applies, in the same order of operations, so the
    offline fit predicts the served distribution.
    """
    if temperature == 1.0:
        return dict(closed)
    inv = 1.0 / temperature
    weights = {k: (v ** inv if v > 0.0 else 0.0) for k, v in closed.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return dict(closed)
    return {k: v / total for k, v in weights.items()}


def nll(closed: dict[str, float], gold: str, temperature: float) -> float:
    """The negative log-likelihood of the gold label under the scaled distribution."""
    p = scale(closed, temperature).get(gold, 0.0)
    return -math.log(p if p > 1e-300 else 1e-300)


def brier(closed: dict[str, float], gold: str, labels: list[str], temperature: float) -> float:
    """The multiclass Brier score: the mean squared error over the labels."""
    dist = scale(closed, temperature)
    return sum((dist.get(lab, 0.0) - (1.0 if lab == gold else 0.0)) ** 2 for lab in labels)


def undo_temperature(closed: dict[str, float], temperature: float) -> dict[str, float]:
    """-> the closed distribution at T = 1, given the one at T.

    closed_T = normalize(closed_1 ** (1/T)), so closed_1 = normalize(closed_T ** T).
    This is the PAIRED CONTROL: the same run, the same engine, the same picks, with
    the temperature's arithmetic removed from the distributions.
    """
    if temperature == 1.0:
        return dict(closed)
    weights = {k: (v ** temperature if v > 0.0 else 0.0) for k, v in closed.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return dict(closed)
    return {k: v / total for k, v in weights.items()}


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


# ------------------------------------------------------------------ the scorer's own metrics

def outcomes(scorer, records: list[dict], temperature: float = 1.0,
             per_item: dict[str, float] | None = None) -> list[dict]:
    """The scorer's own per-item outcomes, on the temperature-scaled distributions.

    `per_item` gives the temperature of every item (the out-of-fold case: an item's
    temperature is the one fitted without it); `temperature` is the single one used
    when there is no per-item map.
    """
    out = []
    for rec in records:
        t = per_item[rec["id"]] if per_item is not None else temperature
        probs = scale(rec["closed"], t)
        out.append(scorer.item_outcome(rec["task"], scorer.normalized(rec["task"], probs)))
    return out


def ece_by_type(scorer, outs: list[dict]) -> dict[str, float]:
    """-> {type: the scorer's own ECE}, computed on the same pairs the scorer uses.

    choice: the TOP-LABEL ECE on the hard tier only.  noul: the ECE of P(yes).
    score: the top-level ECE over every score item.
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
        else:
            value = ece_by_type(scorer, outcomes(scorer, records, t)).get(qtype)
            if value is None:
                continue
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


def cross_validate(scorer, records: list[dict], objective: str, folds: int = FOLDS,
                   seed: int = SEED) -> tuple[dict[str, float], list[dict]]:
    """-> ({id: the out-of-fold temperature}, the fold table)."""
    fold_items = stratified_folds(records, folds, seed)
    per_item: dict[str, float] = {}
    table = []
    for f in range(folds):
        held = fold_items[f]
        train = [r for g in range(folds) if g != f for r in fold_items[g]]
        temps = {}
        for qtype in TYPES:
            tr = [r for r in train if r["type"] == qtype]
            temps[qtype] = fit_temperature(scorer, tr, qtype, objective)[0] if tr else 1.0
        for rec in held:
            per_item[rec["id"]] = temps[rec["type"]]
        hard_choice = sum(1 for r in held if r["type"] == "choice" and r["tier"] == "hard")
        table.append({"fold": f, "n": len(held), "temperatures": temps,
                      "tiers": tier_mix(held), "hard_choice_items": hard_choice})
    return per_item, table


# ------------------------------------------------------------------ the official scorer, run

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
    args = ap.parse_args()

    from datetime import date as _date
    when = args.date or _date.today().isoformat()
    model = args.model
    if model is None and args.config.exists():
        model = json.loads(args.config.read_text(encoding="utf-8")).get("model_name")

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
        ho_scaled.append(scorer.item_outcome(rec["task"], scorer.normalized(rec["task"], probs)))
    ho_after = ece_by_type(scorer, ho_scaled)

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
        "version": 1,
        "model": model,
        "fitted_on": when,
        "objective": args.objective,
        "objective_text": {
            "nll": "the mean negative log-likelihood of the gold label under normalize(closed ** (1/T))",
            "brier": "the mean multiclass Brier score of the distribution against the gold label",
            "ece": "the scorer's own expected calibration error for the type (the top-label ECE on the "
                   "hard tier for choice, the ECE of P(yes) for noul, the top-level ECE for score)",
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
        "source": {
            "dataset": "jgeuter/OpenDecisions v1.0.1, the 235-item 10% sample (seed 42)",
            "arm": "the serial first-token path, every item on the fast path",
            "note": ("the constants are fitted on these same items, so a re-run of the sample "
                     "is IN-SAMPLE; the honest number is the out-of-fold one below"),
        },
        "validation": {
            "out_of_fold": {o: {"ece_before": cv[o]["ece_before"], "ece_after": cv[o]["ece_after"],
                                "folds": cv[o]["folds"], "scorer": cv[o]["scorer"]} for o in OBJECTIVES},
            "holdout_30": {"ece_before": ho_before, "ece_after": ho_after, "temperatures": temps30,
                           "train_items": {q: sum(1 for r in train30 if r["type"] == q) for q in TYPES},
                           "test_items": {q: sum(1 for r in test30 if r["type"] == q) for q in TYPES}},
            "bootstrap": boot,
            "paired_bootstrap_out_of_fold": boot_oof,
            "paired_control": paired,
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
    A("## The three objectives, and why the shipped one is the Brier score")
    A("")
    A("They disagree on this data, so all three are fitted and cross-validated:")
    A("")
    A("| objective | what it minimizes | choice | noul | score |")
    A("|---|---|---:|---:|---:|")
    A("| nll | the negative log-likelihood of the gold label | "
      f"{fits['nll']['choice']} | {fits['nll']['noul']} | {fits['nll']['score']} |")
    A("| brier | the multiclass Brier score | "
      f"{fits['brier']['choice']} | {fits['brier']['noul']} | {fits['brier']['score']} |")
    A("| ece | the scorer's own ECE for the type | "
      f"{fits['ece']['choice']} | {fits['ece']['noul']} | {fits['ece']['score']} |")
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
    A(f"**The shipped constants use the `{args.objective}` objective.**")
    A("")
    A("## The validation")
    A("")
    A(f"The split rule: {out['split_rule']}  The seed is **{args.seed}**.")
    A("")
    A("The fit never sees the item it is scored on.  The ECEs are the scorer's own (its")
    A("`ece` function, on the same pairs it uses: the top-label ECE on the hard tier for")
    A("choice, the ECE of P(yes) for noul, the top-level ECE for score).")
    A("")
    A("| objective | type | ECE before | ECE after (out-of-fold) |")
    A("|---|---|---:|---:|")
    for objective in OBJECTIVES:
        for qtype in TYPES:
            A(f"| {objective} | {qtype} | {fmt(cv[objective]['ece_before'].get(qtype))} | "
              f"{fmt(cv[objective]['ece_after'].get(qtype))} |")
    A("")
    A("The 70/30 split (the same seed), fitted on its own train part:")
    A("")
    A("| type | train items | held-out items | T (train) | ECE before | ECE after |")
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
    A("- **The 235-item re-run is in-sample.**  The constants are fitted on those items,")
    A("  so the in-sample row above is not an unbiased estimate of anything.  The honest")
    A("  number is the out-of-fold one.")
    A("- **The dataset says not to fit on it** (its README: \"fit temperatures on your own")
    A("  data, never on OpenDecisions\").  The roadmap asks for this fit, so the numbers")
    A("  are labelled as they are, and the independent check is the jev-bench run with")
    A("  human soft labels (roadmap item 1.2).")
    A("- **The ECE of the choice questions is measured on the hard tier only** (53 of the")
    A("  107 choice items), so it is a thin slice: the fold table above says how thin.")
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
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[temperature] wrote {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
