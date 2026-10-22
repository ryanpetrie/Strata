#!/usr/bin/env python3
"""tools/decision_policy/jevbench_paired.py - the paired temperature comparison on jev-bench.

The temperature the endpoint applies to the CLOSED reading is invertible:

    closed_T(i) = normalize(closed_1(i) ** (1/T))   ->   closed_1(i) = normalize(closed_T(i) ** T)

so a run made with a temperature carries the T = 1 distribution inside it.  That is
what this script uses to make a PAIRED comparison: the T = 1 arm and the tempered
arm are the SAME run, the same engine, the same items - the only difference is
whether the temperature's arithmetic is applied.

It also writes the recovered T = 1 arm as a predictions file, so the OFFICIAL
scorer (score.py) can score it: the paired Capability on identical items.

The inversion is VALIDATED, not assumed:
  - the tempered arm's TVD is recomputed from the run log and compared with the TVD
    the harness stored per item (they must agree to ~1e-6);
  - the forward transform is applied to the recovered T = 1 arm and compared with
    the tempered arm the run actually reported (the max absolute difference);
  - the noul's recovered P(yes) is cross-checked against the run log's own open
    reading (closed[yes] = open[yes] / mass);
  - the recovered T = 1 arm is compared with the STORED T = 1 run's distribution for
    the same items (the run-to-run wobble the engine has, which is the yardstick for
    how far the recovered arm may sit from the stored one).

The per-type ECEs are the OFFICIAL scorer's own (`score.py`'s `ece` and
`item_outcome`), on the same pairs the scorer's calibration component uses, so the
numbers here and the numbers in the scorer's own output are the same measurement.

Nothing but this script is written into the repository; the data stays outside.

USAGE
  python tools/decision_policy/jevbench_paired.py \
    --items <items.jsonl> \
    --runlog <new run's runlog.jsonl> \
    --baseline-runlog <stored T = 1 run's runlog.jsonl> \
    --scorer <path to score.py> \
    --out-predictions <recovered T = 1 predictions.jsonl> \
    --out-report <report.md> [--json <report.json>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TYPES = ("choice", "noul", "score")
TEMPERATURE = {"choice": 0.62, "noul": 0.67, "score": 0.77}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_scorer(path: Path):
    """The official scorer, loaded from wherever it is (it is not in this repo)."""
    spec = importlib.util.spec_from_file_location("jevbench_official_scorer", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load the scorer from %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def temper(closed: dict, t: float) -> dict:
    """The server's own transform: closed_T(i) = normalize(closed_1(i) ** (1/T))."""
    if t == 1.0:
        return dict(closed)
    weights = {k: (v ** (1.0 / t) if v > 0.0 else 0.0) for k, v in closed.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return dict(closed)
    return {k: v / total for k, v in weights.items()}


def untemper(closed_t: dict, t: float) -> dict:
    """The inverse: closed_1(i) = normalize(closed_T(i) ** T)."""
    if t == 1.0:
        return dict(closed_t)
    weights = {k: (v ** t if v > 0.0 else 0.0) for k, v in closed_t.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return dict(closed_t)
    return {k: v / total for k, v in weights.items()}


def closed_of(it: dict, probs: dict) -> dict:
    """The closed reading over the item's labels, from the scorer's probabilities.

    The harness's `probs` ARE the closed reading (the options renormalized over
    themselves): the choice and score arms take the endpoint's `probabilities` when
    they add up to 1, and the noul arm takes the endpoint's noul value, which is the
    closed P(yes).  The noul's "no" is filled in here, so the noul is inverted over
    BOTH answers, exactly as the server scales it.
    """
    labels = list(it["labels"])
    if it["question"]["type"] == "noul" and "no" not in probs:
        probs = {"yes": probs["yes"], "no": 1.0 - probs["yes"]}
    return {lab: float(probs.get(lab, 0.0)) for lab in labels}


def tvd(probs: dict, target: dict, labels: list[str]) -> float:
    return 0.5 * sum(abs(float(probs.get(lab, 0.0)) - float(target.get(lab, 0.0)))
                     for lab in labels)


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def entropy(probs: dict) -> float:
    total = sum(float(v) for v in probs.values())
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in probs.values() if v > 0)


def probs_from_record(rec: dict, it: dict) -> dict:
    """The scorer's probabilities for one run log record, rebuilt from the record.

    The harness's `to_probs` is the rule: the choice and score arms take the
    endpoint's `probabilities` when they add up to 1, and the noul arm takes the
    endpoint's noul value (the closed P(yes)).  The noul's value is not stored
    directly; it is recovered from the record's `pick` and `closed_confidence` (the
    closed reading of the pick), and cross-checked against the record's own open
    reading in the caller.
    """
    qtype = it["question"]["type"]
    labels = list(it["labels"])
    if qtype == "noul":
        pick = rec.get("pick")
        cc = rec.get("closed_confidence")
        if cc is None:
            raise ValueError("no closed confidence for %s" % rec["id"])
        p = float(cc) if pick == "yes" else 1.0 - float(cc)
        return {"yes": p}
    probs = rec.get("probabilities")
    if probs and abs(sum(float(v) for v in probs.values()) - 1.0) < 0.01:
        return {lab: float(probs.get(lab, 0.0)) for lab in labels}
    # the decode path: the degenerate distribution
    if qtype == "choice":
        pick = rec.get("pick")
        return {lab: (1.0 if lab == pick else 0.0) for lab in labels}
    idx = rec.get("pick")
    return {str(i): (1.0 if i == idx else 0.0) for i in range(len(labels))}


def scorer_probs(it: dict, probs: dict) -> dict:
    """The shape the official scorer takes: the noul carries only "yes"."""
    if it["question"]["type"] == "noul":
        return {"yes": probs["yes"]}
    return {lab: probs[lab] for lab in it["labels"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--runlog", type=Path, required=True, help="the new (tempered) run's run log")
    ap.add_argument("--baseline-runlog", type=Path, required=True, help="the stored T = 1 run's run log")
    ap.add_argument("--scorer", type=Path, default=None, help="the official score.py")
    ap.add_argument("--out-predictions", type=Path, required=True,
                    help="the recovered T = 1 arm, as a predictions file for the official scorer")
    ap.add_argument("--out-report", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    scorer = load_scorer(args.scorer) if args.scorer else None
    items_list = read_jsonl(args.items)
    items = {it["id"]: it for it in items_list}
    log = read_jsonl(args.runlog)
    base = {r["id"]: r for r in read_jsonl(args.baseline_runlog)}

    report: dict = {"temperature": TEMPERATURE, "items_in_file": len(items),
                    "runlog_records": len(log)}

    arms: list[dict] = []          # one record per item that has a target distribution
    recovered: list[dict] = []     # the recovered T = 1 predictions, every predicted item
    fitted: list[dict] = []        # the fitted arm's predictions, every predicted item
    n_fast = n_fail = n_notfast = 0
    latencies: list[float] = []
    stored_tvd_mismatch: list[dict] = []
    fwd_max_diff = 0.0
    noul_crosscheck_max = 0.0
    vs_stored_max = 0.0
    vs_stored_by_type: dict[str, list[float]] = defaultdict(list)
    picks_moved = 0
    picks_moved_ids: list[str] = []

    for rec in log:
        it = items.get(rec["id"])
        if it is None:
            continue
        if rec.get("latency_ms") is not None:
            latencies.append(float(rec["latency_ms"]))
        if rec.get("fast_path"):
            n_fast += 1
        else:
            n_notfast += 1
        if not rec.get("predicted"):
            n_fail += 1
            # the scorer's rule: a missing prediction is wrong, and the TVD is 1
            if it.get("target_probs"):
                arms.append({"id": rec["id"], "type": it["question"]["type"],
                             "config": it.get("config"), "labels": list(it["labels"]),
                             "tvd_1": 1.0, "tvd_t": 1.0, "failed": True,
                             "human_entropy": entropy(it["target_probs"])})
            continue

        qtype = it["question"]["type"]
        labels = list(it["labels"])
        t = TEMPERATURE[qtype]
        probs_t = closed_of(it, probs_from_record(rec, it))
        probs_1 = untemper(probs_t, t)

        # the noul's cross-check: closed[yes] = open[yes] / mass
        if qtype == "noul" and rec.get("open_probs"):
            op = rec["open_probs"]
            mass = sum(float(v) for v in op.values())
            if mass > 0:
                noul_crosscheck_max = max(noul_crosscheck_max,
                                          abs(op["yes"] / mass - probs_t["yes"]))

        recovered.append({"id": rec["id"], "probs": scorer_probs(it, probs_1)})
        fitted.append({"id": rec["id"], "probs": scorer_probs(it, probs_t)})

        # the pick: the temperature cannot move it, so a moved pick is the engine's wobble
        b = base.get(rec["id"])
        if b and b.get("pick") is not None and rec.get("pick") is not None \
                and rec["pick"] != b["pick"]:
            picks_moved += 1
            picks_moved_ids.append(rec["id"])

        target = it.get("target_probs")
        if not target:
            continue
        tvd_t = tvd(probs_t, target, labels)
        tvd_1 = tvd(probs_1, target, labels)
        stored = rec.get("tvd")
        if stored is not None and abs(tvd_t - float(stored)) > 1e-6:
            stored_tvd_mismatch.append({"id": rec["id"], "stored": stored, "recomputed": tvd_t})
        fwd = temper(probs_1, t)
        fwd_max_diff = max(fwd_max_diff, max(abs(fwd[k] - probs_t[k]) for k in labels))
        if b and b.get("predicted"):
            bp = closed_of(it, probs_from_record(b, it))
            d = max(abs(bp[k] - probs_1[k]) for k in labels)
            vs_stored_max = max(vs_stored_max, d)
            vs_stored_by_type[qtype].append(d)
        arms.append({"id": rec["id"], "type": qtype, "config": it.get("config"),
                     "labels": labels, "tvd_1": tvd_1, "tvd_t": tvd_t, "failed": False,
                     "human_entropy": entropy(target)})

    report["fast_path"] = n_fast
    report["not_fast_path"] = n_notfast
    report["failures"] = n_fail
    report["latency_median_ms"] = statistics.median(latencies) if latencies else None
    report["latency_p95_ms"] = pct(latencies, 0.95)
    report["latency_mean_ms"] = mean(latencies)
    report["stored_tvd_mismatches"] = len(stored_tvd_mismatch)
    report["stored_tvd_mismatch_examples"] = stored_tvd_mismatch[:5]
    report["forward_check_max_abs_diff"] = fwd_max_diff
    report["noul_crosscheck_max_abs_diff"] = noul_crosscheck_max
    report["recovered_vs_stored_max_abs_diff"] = vs_stored_max
    report["recovered_vs_stored_mean_abs_diff_by_type"] = {
        t: mean(vs_stored_by_type[t]) for t in TYPES if vs_stored_by_type[t]}
    report["picks_moved_against_the_stored_run"] = picks_moved
    report["picks_moved_ids"] = picks_moved_ids

    report["arms"] = {
        "t1": {"n": len(arms), "mean_tvd": mean([a["tvd_1"] for a in arms])},
        "fitted": {"n": len(arms), "mean_tvd": mean([a["tvd_t"] for a in arms])},
    }
    report["per_type"] = {}
    for t in TYPES:
        sub = [a for a in arms if a["type"] == t]
        m1 = mean([a["tvd_1"] for a in sub])
        mt = mean([a["tvd_t"] for a in sub])
        report["per_type"][t] = {"n": len(sub), "tvd_t1": m1, "tvd_fitted": mt,
                                 "delta": (mt - m1) if (m1 is not None and mt is not None) else None,
                                 "better": sum(1 for a in sub if a["tvd_t"] < a["tvd_1"]),
                                 "worse": sum(1 for a in sub if a["tvd_t"] > a["tvd_1"]),
                                 "same": sum(1 for a in sub if a["tvd_t"] == a["tvd_1"])}
    report["per_config"] = {}
    for cfg in sorted({a["config"] for a in arms}):
        sub = [a for a in arms if a["config"] == cfg]
        m1 = mean([a["tvd_1"] for a in sub])
        mt = mean([a["tvd_t"] for a in sub])
        report["per_config"][cfg] = {"n": len(sub), "tvd_t1": m1, "tvd_fitted": mt,
                                     "delta": (mt - m1) if (m1 is not None and mt is not None) else None}

    # ------------------------------------------- the humans' uncertainty, split in half
    report["per_type_by_human_uncertainty"] = {}
    for t in TYPES:
        sub = sorted([a for a in arms if a["type"] == t], key=lambda a: a["human_entropy"])
        k = len(sub) // 2
        halves = {}
        for name, ss in (("the humans least uncertain (half)", sub[:k]),
                         ("the humans most uncertain (half)", sub[k:])):
            m1 = mean([a["tvd_1"] for a in ss])
            mt = mean([a["tvd_t"] for a in ss])
            halves[name] = {"n": len(ss), "tvd_t1": m1, "tvd_fitted": mt,
                            "delta": (mt - m1) if (m1 is not None and mt is not None) else None}
        report["per_type_by_human_uncertainty"][t] = halves

    # ------------------------------------------------ the recovered T = 1 predictions
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.out_predictions.open("w", encoding="utf-8") as f:
        for r in recovered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["recovered_predictions"] = str(args.out_predictions)
    report["recovered_predictions_items"] = len(recovered)
    report["fitted_predictions_items"] = len(fitted)

    # ------------------------------------------------ the official scorer's numbers
    if scorer is not None:
        by_id = {it["id"]: it for it in items_list}
        def score_arm(preds: list[dict]) -> dict:
            probs = {p["id"]: scorer.normalized(by_id[p["id"]], p["probs"]) for p in preds}
            outs = [scorer.item_outcome(it, probs.get(it["id"])) for it in items_list]
            res = scorer.summary(outs)
            # the per-type ECE, on the scorer's own pairs
            eces = {}
            for t in TYPES:
                sub = [o for o in outs if o["type"] == t and o["valid"]]
                if t == "choice":
                    sub = [o for o in sub if o["tier"] == "hard"]
                    pairs = [(o["conf"], float(o["correct"])) for o in sub]
                elif t == "noul":
                    pairs = [(o["p_yes"], o["y"]) for o in sub]
                else:
                    pairs = [(o["conf"], float(o["correct"])) for o in sub]
                eces[t] = scorer.ece(pairs) if pairs else None
            res["ece_per_type"] = eces
            return res
        report["scorer_t1"] = score_arm(recovered)
        report["scorer_fitted"] = score_arm(fitted)

    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if args.out_report:
        args.out_report.write_text(render(report), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))
    return 0


def render(report: dict) -> str:
    out: list[str] = []
    a = report["arms"]
    out.append("# The paired temperature comparison on jev-bench\n")
    out.append("| | n | mean TVD to the humans |\n|---|---:|---:|")
    out.append("| the T = 1 arm (the temperature's arithmetic removed) | %d | %.6f |"
               % (a["t1"]["n"], a["t1"]["mean_tvd"]))
    out.append("| the fitted arm (the temperature applied) | %d | %.6f |"
               % (a["fitted"]["n"], a["fitted"]["mean_tvd"]))
    out.append("")
    out.append("## per type\n")
    out.append("| type | n | T = 1 | fitted | change | better | worse | same |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in TYPES:
        p = report["per_type"][t]
        out.append("| %s | %d | %.6f | %.6f | %+.6f | %d | %d | %d |"
                   % (t, p["n"], p["tvd_t1"], p["tvd_fitted"], p["delta"],
                      p["better"], p["worse"], p["same"]))
    out.append("")
    out.append("## per config\n")
    out.append("| config | n | T = 1 | fitted | change |\n|---|---:|---:|---:|---:|")
    for cfg, p in report["per_config"].items():
        out.append("| %s | %d | %.6f | %.6f | %+.6f |"
                   % (cfg, p["n"], p["tvd_t1"], p["tvd_fitted"], p["delta"]))
    out.append("")
    out.append("## the humans' uncertainty\n")
    out.append("| type | half | n | T = 1 | fitted | change |\n|---|---|---:|---:|---:|---:|")
    for t in TYPES:
        for name, p in report["per_type_by_human_uncertainty"][t].items():
            out.append("| %s | %s | %d | %.6f | %.6f | %+.6f |"
                       % (t, name, p["n"], p["tvd_t1"], p["tvd_fitted"], p["delta"]))
    out.append("")
    out.append("## the checks\n")
    out.append("* the stored per-item TVD vs the recomputed one: %d mismatches (over 1e-6)"
               % report["stored_tvd_mismatches"])
    out.append("* the forward transform's max absolute difference: %.3e"
               % report["forward_check_max_abs_diff"])
    out.append("* the noul's cross-check against the open reading: max %.3e"
               % report["noul_crosscheck_max_abs_diff"])
    out.append("* the recovered T = 1 arm vs the STORED T = 1 run: max %.6f, mean %s"
               % (report["recovered_vs_stored_max_abs_diff"],
                  json.dumps(report["recovered_vs_stored_mean_abs_diff_by_type"])))
    out.append("* fast path %d, not fast path %d, failures %d, latency median %.0f ms, p95 %.0f ms"
               % (report["fast_path"], report["not_fast_path"], report["failures"],
                  report["latency_median_ms"] or 0, report["latency_p95_ms"] or 0))
    out.append("* the picks that moved against the stored run: %d (the temperature cannot move a pick)"
               % report["picks_moved_against_the_stored_run"])
    if "scorer_t1" in report:
        out.append("")
        out.append("## the official scorer\n")
        out.append("| arm | Capability | Intelligence | Calibration | choice ECE | noul ECE | score ECE | TVD |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, key in (("the T = 1 arm (recovered)", "scorer_t1"),
                          ("the fitted arm", "scorer_fitted")):
            s = report[key]
            e = s["ece_per_type"]
            out.append("| %s | %.1f | %.1f | %.1f | %.4f | %.4f | %.4f | %.6f |"
                       % (name, s["capability"], s["intelligence"], s["calibration"],
                          e["choice"] or 0, e["noul"] or 0, e["score"] or 0,
                          s["calibration_items_tvd"] or 0))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
