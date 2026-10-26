"""tools/decision_policy/score_location.py - roadmap 1.4b: one LOCATION per score config, on the normalised level.

    python tools/decision_policy/score_location.py --out-json score_location.json --out-report score_location.md

WHAT IT DOES.  For a score question with K levels the map is one parameter per config:

    p'_k = normalize( p_k * exp(b_c * (u_k - 0.5)) ),      u_k = k / (K - 1),  T = 1

and `b_c` is fitted on the licence-clean pool's stored T = 1 run - against the human distribution where the
config has one (Arm H, `measuring_hate_speech`), against the one-hot gold level where it does not (Arm G, the
two `helpsteer2` configs).  The map is then measured the way 1.1b and 1.4 were: 5-fold leak-free CV on the pool,
and a paired transfer to the jev-bench sample's items of the SAME config, re-scaling the stored run so both arms
share the engine, the picks, the logits and the state.

WHY THIS IS NOT 1.4's AFFINE FIT AGAIN.  1.4 fitted one (T, b) per TYPE with the shift on the raw level index,
which is not the same size on a 3-level and a 5-level question; its score arm was also a scale and a location at
once.  This is the location alone, fitted per config, on the normalised axis.  Note what that does and does not
buy (companion doc 26.1): inside ONE config the normalised axis and the raw index are the same one-parameter
family, so the normalisation does no work there - it matters when one b would be shared across configs, and for
comparing b's across them.

NO ENGINE.  No server, no GPU, no network.  Pure standard library plus numpy-free arithmetic, importing the
programme's own fold rule, TVD, level and bootstrap helpers.

THE RULE WAS COMMITTED BEFORE THIS RAN: docs/research/shared_backbone_decision_policy.md 26.1 (Arm H's (a) and
(b), Arm G's (c) and (d), the 0.005 and 0.10-level margins).  The results go into 26.2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import affine as af  # noqa: E402  (the records, the metrics, the bootstrap)
import temperature as tp  # noqa: E402  (the fold rule)

HERE = Path(__file__).resolve().parent

#: The data root: the tools never carry this machine's absolute paths into the repository
#: (CONTRIBUTING: no private paths).  Set STRATA_DECISION_DATA to the directory that holds
#: `decision_policy_training/` and `opendecisions/`.
DATA = Path(os.environ.get("STRATA_DECISION_DATA")
            or (Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Temp" / "opencode"))
POOL_ITEMS = DATA / "decision_policy_training" / "items" / "p1b_pool.jsonl"
POOL_RUN = DATA / "decision_policy_training" / "run" / "p1b_pool_T1.jsonl"
JEV_ITEMS = DATA / "opendecisions" / "jevbench_sample3_items.jsonl"
JEV_RUN = DATA / "decision_policy_training" / "run" / "p1b_jevbench_test_soft_T1.jsonl"
OD_ITEMS = DATA / "opendecisions" / "sample10_items.jsonl"
OD_RUN = DATA / "decision_policy_training" / "run" / "p1b_235_T1_from_current.jsonl"

FOLDS = 5
SEED = 42
B_GRID = (-6.0, 6.0, 0.01)
RESAMPLES = 1000
#: 26.1's fixed margins.
JEV_TVD_MARGIN = 0.005
JEV_LEVEL_MARGIN = 0.10


# ------------------------------------------------------------------------------------------------ the map
def unit(labels: list[str]) -> dict[str, float]:
    """The normalised level of every label: u = k / (K - 1), k the label's index (the labels are 0..K-1)."""
    k = len(labels)
    if k < 2:
        raise SystemExit("a score question needs at least two levels, got %r" % (labels,))
    return {lab: float(int(lab)) / (k - 1) for lab in labels}


def apply_location(closed: dict[str, float], labels: list[str], b: float) -> dict[str, float]:
    """-> the closed distribution under one location.  b = 0 returns the reading's own values, exactly."""
    if b == 0.0:
        return dict(closed)
    coords = unit(labels)
    weights = {}
    for lab in labels:
        p = float(closed.get(lab, 0.0))
        weights[lab] = 0.0 if p <= 0.0 else p * math.exp(b * (coords[lab] - 0.5))
    total = sum(weights.values())
    if total <= 0.0:
        return dict(closed)          # every weight underflowed: a silent no-op, like the server's own guard
    return {lab: w / total for lab, w in weights.items()}


def fit_location(records: list[dict], objective: str) -> float:
    """-> the b that minimises the mean objective over `records` (a coarse-to-fine grid, deterministic).

    objective: "soft" - the cross-entropy against the human distribution (`target`);
               "gold" - the cross-entropy against the one-hot gold level.
    """
    best_b, best = 0.0, None
    lo, hi, step = B_GRID
    steps = int(round((hi - lo) / step))
    for i in range(steps + 1):
        b = lo + i * step
        value = mean_objective(records, b, objective)
        if best is None or value < best:
            best, best_b = value, b
    # one refinement pass around the winner, to a thousandth
    for i in range(-10, 11):
        b = best_b + i * 0.001
        value = mean_objective(records, b, objective)
        if value < best:
            best, best_b = value, b
    return round(best_b, 3)


def mean_objective(records: list[dict], b: float, objective: str) -> float:
    total, n = 0.0, 0
    for rec in records:
        target = rec["target"] if objective == "soft" else {
            lab: (1.0 if lab == rec["gold"] else 0.0) for lab in rec["labels"]}
        if not target:
            continue
        p = apply_location(rec["closed"], rec["labels"], b)
        total += cross_entropy(p, target, rec["labels"])
        n += 1
    return total / n if n else float("inf")


def cross_entropy(p: dict[str, float], q: dict[str, float], labels: list[str]) -> float:
    out = 0.0
    for lab in labels:
        w = float(q.get(lab, 0.0))
        if w > 0.0:
            out -= w * math.log(max(p.get(lab, 0.0), 1e-12))
    return out


# ------------------------------------------------------------------------------------------------ the measurements
def level_error(rec: dict, dist: dict[str, float]) -> float:
    """|expected level - gold level|, the score's own error unit (1.4 used it)."""
    return abs(af.expected_level(dist, rec["labels"]) - float(rec["gold"]))


def paired_mean_bootstrap(a: list[float], b: list[float], resamples: int = RESAMPLES,
                          seed: int = SEED) -> dict:
    """The paired item bootstrap of mean(b - a); both lists are the same items in the same order."""
    n = len(a)
    if not n:
        return {}
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {"n": n, "mean": sum(diffs) / n,
            "p025": means[int(0.025 * (len(means) - 1))], "p975": means[int(0.975 * (len(means) - 1))],
            "share_better": sum(1 for d in diffs if d < 0) / n, "resamples": resamples}


def score_records(items: list[dict], predictions: list[dict], with_target: bool) -> list[dict]:
    """The score records of one population, grouped by config, with their stored T = 1 readings."""
    records, problems = af.records_from_predictions(items, predictions)
    out = [r for r in records if r["type"] == "score"
           and (bool(r.get("target")) if with_target else not r.get("target"))]
    if problems:
        print("[score-location] %d problem(s) in the records, first: %s" % (len(problems), problems[0]),
              file=sys.stderr)
    return out


def configs_of(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for rec in records:
        out.setdefault(rec["task"].get("config") or "?", []).append(rec)
    return {k: out[k] for k in sorted(out)}


def out_of_fold(records: list[dict], objective: str) -> tuple[dict[str, float], list[dict]]:
    """-> ({id: the out-of-fold distribution}, the fold table, so a reader can see every b)."""
    fold_items = tp.stratified_folds(records, FOLDS, SEED)
    dists, table = {}, []
    for f in range(FOLDS):
        held = fold_items[f]
        train = [r for g in range(FOLDS) if g != f for r in fold_items[g]]
        b = fit_location(train, objective)
        table.append({"fold": f, "b": b, "fitted_on": len(train), "held": len(held)})
        for rec in held:
            dists[rec["id"]] = apply_location(rec["closed"], rec["labels"], b)
    return dists, table


def arm_report(name: str, pool: dict[str, list[dict]], jev: dict[str, list[dict]],
               objective: str, metric: str, originals: dict[str, dict]) -> dict:
    """One config's two measurements: the pool's out-of-fold result and the jev-bench transfer, both paired."""
    recs = pool[name]
    identity = {r["id"]: dict(r["closed"]) for r in recs}
    oof, table = out_of_fold(recs, objective)
    full_b = fit_location(recs, objective)

    out = {"config": name, "k": len(recs[0]["labels"]), "n_pool": len(recs),
           "b": full_b, "folds": table, "objective": objective, "metric": metric}
    if metric == "tvd":
        out["pool"] = {"t1_mean": af.mean([af.tvd(identity[r["id"]], r["target"], r["labels"]) for r in recs]),
                       "oof_mean": af.mean([af.tvd(oof[r["id"]], r["target"], r["labels"]) for r in recs]),
                       "paired": af.paired_tvd_bootstrap(recs, identity, oof, resamples=RESAMPLES)}
    else:
        t1 = [level_error(r, identity[r["id"]]) for r in recs]
        arm = [level_error(r, oof[r["id"]]) for r in recs]
        out["pool"] = {"t1_mean": af.mean(t1), "oof_mean": af.mean(arm),
                       "paired": paired_mean_bootstrap(t1, arm)}
        out["pool"]["t1_accuracy"] = sum(1 for r in recs
                                         if max(identity[r["id"]], key=identity[r["id"]].get) == r["gold"]) / len(recs)
        out["pool"]["oof_accuracy"] = sum(1 for r in recs
                                          if max(oof[r["id"]], key=oof[r["id"]].get) == r["gold"]) / len(recs)

    transfer = jev.get(name) or []
    if transfer:
        b = full_b
        ident = {r["id"]: dict(r["closed"]) for r in transfer}
        moved = {r["id"]: apply_location(r["closed"], r["labels"], b) for r in transfer}
        if metric == "tvd":
            out["jev"] = {"n": len(transfer),
                          "t1_mean": af.mean([af.tvd(ident[r["id"]], r["target"], r["labels"]) for r in transfer]),
                          "arm_mean": af.mean([af.tvd(moved[r["id"]], r["target"], r["labels"]) for r in transfer]),
                          "paired": af.paired_tvd_bootstrap(transfer, ident, moved, resamples=RESAMPLES),
                          "b": b}
        else:
            t1 = [level_error(r, ident[r["id"]]) for r in transfer]
            arm = [level_error(r, moved[r["id"]]) for r in transfer]
            out["jev"] = {"n": len(transfer), "t1_mean": af.mean(t1), "arm_mean": af.mean(arm),
                          "paired": paired_mean_bootstrap(t1, arm), "b": b}
    return out


# ------------------------------------------------------------------------------------------------ the report
def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def ci(paired, nd=4):
    if not paired:
        return "n/a"
    return "%s [%s, %s]" % (fmt(paired["mean"], nd), fmt(paired.get("p025"), nd), fmt(paired.get("p975"), nd))


def verdict(arm: dict) -> dict:
    """26.1's rule, applied mechanically.  A margin is a bound, not a hope: it is in the doc before the run."""
    out = {}
    if arm["metric"] == "tvd":
        out["a_pool_improves"] = arm["pool"]["oof_mean"] < arm["pool"]["t1_mean"]
        if arm.get("jev"):
            out["b_jev_not_worse"] = arm["jev"]["arm_mean"] - arm["jev"]["t1_mean"] <= JEV_TVD_MARGIN
        out["passes"] = out["a_pool_improves"] and out.get("b_jev_not_worse", True)
    else:
        out["c_pool_level_falls"] = arm["pool"]["oof_mean"] < arm["pool"]["t1_mean"]
        if arm.get("jev"):
            out["d_jev_not_worse"] = arm["jev"]["arm_mean"] - arm["jev"]["t1_mean"] <= JEV_LEVEL_MARGIN
        out["passes"] = out["c_pool_level_falls"] and out.get("d_jev_not_worse", True)
    return out


def report_md(arms: list[dict], od_note: dict, when: str) -> str:
    lines = ["# 1.4b: one location per score config (the run of %s)" % when, "",
             "The map: `p'_k = normalize(p_k * exp(b_c * (u_k - 0.5)))`, `u_k = k/(K-1)`, `T = 1`; one `b_c` per",
             "config, fitted on the licence-clean pool's stored T = 1 run and transferred to the jev-bench",
             "sample's items of the same config.  The rule is companion doc 26.1, committed before this ran.", "",
             "| config | K | n (pool) | b_c | pool T=1 | pool out-of-fold | the pool, paired |",
             "|---|---:|---:|---:|---|---|---|"]
    for arm in arms:
        if arm["metric"] == "tvd":
            lines.append("| `%s` | %d | %d | %s | TVD %s | TVD %s | %s |" % (
                arm["config"], arm["k"], arm["n_pool"], fmt(arm["b"], 3), fmt(arm["pool"]["t1_mean"]),
                fmt(arm["pool"]["oof_mean"]), ci(arm["pool"]["paired"])))
        else:
            lines.append("| `%s` | %d | %d | %s | level %s | level %s | %s |" % (
                arm["config"], arm["k"], arm["n_pool"], fmt(arm["b"], 3), fmt(arm["pool"]["t1_mean"], 3),
                fmt(arm["pool"]["oof_mean"], 3), ci(arm["pool"]["paired"], 3)))
    lines += ["", "| config | n (jev sample) | T=1 | the map | the sample, paired | passes the rule? |",
              "|---|---:|---|---|---|---|"]
    for arm in arms:
        v = arm["verdict"]
        if arm.get("jev"):
            lines.append("| `%s` | %d | %s | %s | %s | %s |" % (
                arm["config"], arm["jev"]["n"], fmt(arm["jev"]["t1_mean"], 3), fmt(arm["jev"]["arm_mean"], 3),
                ci(arm["jev"]["paired"], 3), "**yes**" if v["passes"] else "no"))
        else:
            lines.append("| `%s` | - | - | - | - | no jev items of this config |" % arm["config"])
    lines += ["", "## The 235, for information", "",
              od_note.get("text", ""), "",
              "## Every fold's b", "",
              "| config | " + " | ".join("fold %d" % i for i in range(FOLDS)) + " |",
              "|---|" + "---|" * FOLDS]
    for arm in arms:
        lines.append("| `%s` | %s |" % (arm["config"], " | ".join(fmt(f["b"], 3) for f in arm["folds"])))
    return "\n".join(lines) + "\n"


def od_235_note(items_path: Path, run_path: Path) -> dict:
    """The 235's score items at T = 1: reported, not judged (26.1: none of its configs is a jev config)."""
    if not (items_path.exists() and run_path.exists()):
        return {"text": "The 235's stored run was not found, so nothing is reported here."}
    records, _ = af.records_from_predictions(af.read_jsonl(items_path), af.read_jsonl(run_path))
    sub = [r for r in records if r["type"] == "score"]
    if not sub:
        return {"text": "The 235's stored run has no score items."}
    err = af.mean([abs(af.expected_level(r["closed"], r["labels"]) - float(r["gold"])) for r in sub])
    acc = sum(1 for r in sub if max(r["closed"], key=r["closed"].get) == r["gold"]) / len(sub)
    return {"n": len(sub), "mean_abs_level_error": err, "accuracy": acc,
            "text": ("The 235's %d score items at T = 1: mean |expected level - gold| %.3f levels, argmax "
                     "accuracy %.3f. No arm is applied to them: none of their configs is a jev config, so a "
                     "fitted b would arrive there as a foreign constant (26.1)." % (len(sub), err, acc))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool-items", type=Path, default=POOL_ITEMS)
    ap.add_argument("--pool-run", type=Path, default=POOL_RUN)
    ap.add_argument("--jev-items", type=Path, default=JEV_ITEMS)
    ap.add_argument("--jev-run", type=Path, default=JEV_RUN)
    ap.add_argument("--od-items", type=Path, default=OD_ITEMS)
    ap.add_argument("--od-run", type=Path, default=OD_RUN)
    ap.add_argument("--out-json", type=Path, default=HERE / "score_location.json")
    ap.add_argument("--out-report", type=Path, default=HERE / "score_location_report.md")
    args = ap.parse_args(argv)

    for p in (args.pool_items, args.pool_run, args.jev_items, args.jev_run):
        if not p.exists():
            raise SystemExit("missing %s (set STRATA_DECISION_DATA)" % p)

    pool_items, pool_run = af.read_jsonl(args.pool_items), af.read_jsonl(args.pool_run)
    jev_items, jev_run = af.read_jsonl(args.jev_items), af.read_jsonl(args.jev_run)
    pool_h = configs_of(score_records(pool_items, pool_run, with_target=True))
    pool_g = configs_of(score_records(pool_items, pool_run, with_target=False))
    jev = configs_of(score_records(jev_items, jev_run, with_target=True))
    jev_g = configs_of(score_records(jev_items, jev_run, with_target=False))

    arms = []
    for name, recs in pool_h.items():
        arm = arm_report(name, pool_h, jev, "soft", "tvd", {})
        arm["verdict"] = verdict(arm)
        arms.append(arm)
        print("[score-location] Arm H %s: b %.3f, pool out-of-fold TVD %.4f (T=1 %.4f), jev %s -> %s" % (
            name, arm["b"], arm["pool"]["oof_mean"], arm["pool"]["t1_mean"],
            fmt(arm.get("jev", {}).get("t1_mean")), fmt(arm.get("jev", {}).get("arm_mean"))))
    for name, recs in pool_g.items():
        arm = arm_report(name, pool_g, jev_g, "gold", "level", {})
        arm["verdict"] = verdict(arm)
        arms.append(arm)
        print("[score-location] Arm G %s: b %.3f, pool out-of-fold level %.3f (T=1 %.3f), jev %s -> %s" % (
            name, arm["b"], arm["pool"]["oof_mean"], arm["pool"]["t1_mean"],
            fmt(arm.get("jev", {}).get("t1_mean"), 3), fmt(arm.get("jev", {}).get("arm_mean"), 3)))

    od = od_235_note(args.od_items, args.od_run)
    when = date.today().isoformat()
    report = {"when": when, "map": "p'_k = normalize(p_k * exp(b_c * (u_k - 0.5))), u_k = k/(K-1), T = 1",
              "folds": FOLDS, "seed": SEED, "margins": {"jev_tvd": JEV_TVD_MARGIN, "jev_level": JEV_LEVEL_MARGIN},
              "arms": arms, "od_235": od}
    args.out_json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    args.out_report.write_text(report_md(arms, od, when), encoding="utf-8")
    print("[score-location] wrote %s and %s" % (args.out_json, args.out_report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
