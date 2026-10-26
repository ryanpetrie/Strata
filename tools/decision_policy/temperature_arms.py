#!/usr/bin/env python3
"""tools/decision_policy/temperature_arms.py - every candidate arm of one run.

A temperature's transform is invertible: the T = 1 distribution is inside a tempered
run, and any other temperature can be re-applied to it.  So ONE run carries every
arm: the T = 1 arm (recovered), the temperature the run was served with (the live
one), and any candidate.

That is what makes the comparison PAIRED.  Every arm comes out of the same run, so
the engine, the picks, the logits and the state are shared, and the only thing that
differs is the arithmetic - which is the whole point when the run-to-run spread
between two engines is larger than the effect being measured.

    python tools/decision_policy/temperature_arms.py --items <items.jsonl> \
        --runlog <runlog.jsonl> --served 3.65,3.75,3.73 \
        --arm soft=3.65,3.75,3.73 --arm brier=2.27,2.93,3.29 --arm single=2.47,2.47,2.47 \
        --scorer <path to score.py> --out-report <report.md> [--out-predictions <T = 1 arm>]

For every arm the script reports the mean total variation distance to the HUMAN
distribution (over the items that carry one), overall and per type and per config,
and the official scorer's Capability, Intelligence and Calibration.  The inversion is
VALIDATED, not assumed: the recovered T = 1 arm is put back through the served
temperature and compared with the arm the run actually reported.

The transform is the server's own: the fit's scale() calls
serve/logits_first_token.py::apply_temperature, and this script calls the fit's.

Pure standard library.  Nothing is written inside the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import temperature as tp  # noqa: E402  (the one place that knows the transform)
import jevbench_paired as jp  # noqa: E402  (the harness's own rules for a run log record)

TYPES = ("choice", "noul", "score")
DEFAULT_SCORER = Path(tempfile.gettempdir()) / "opencode" / "opendecisions" / "score.py"


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_temperatures(text: str) -> dict[str, float]:
    """'choice=0.62,noul=0.67,score=0.77' or '0.62,0.67,0.77' -> {type: T}."""
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, value = part.split("=", 1)
            name = name.strip()
        else:
            name, value = None, part
        try:
            t = float(value)
        except ValueError:
            raise SystemExit("not a temperature: %r" % part)
        if t <= 0.0:
            raise SystemExit("a temperature must be above 0: %r" % part)
        if name is None:
            types = [x for x in TYPES if x not in out]
            if not types:
                raise SystemExit("more temperatures than question types: %r" % text)
            out[types[0]] = t
        else:
            if name not in TYPES:
                raise SystemExit("not a question type: %r" % name)
            out[name] = t
    return out


def parse_arms(values: list[str]) -> list[tuple[str, dict[str, float]]]:
    """--arm 'name=choice=0.62,noul=0.67,score=0.77' -> [(name, {type: T})].

    The temperatures may also be positional ('name=0.62,0.67,0.77'), in the fixed
    order choice, noul, score.
    """
    arms = []
    for value in values:
        name, _, rest = value.partition("=")
        if not rest:
            raise SystemExit("an arm needs a name and temperatures: %r" % value)
        temps = parse_temperatures(rest)
        if not temps:
            raise SystemExit("an arm needs at least one temperature: %r" % value)
        arms.append((name.strip(), temps))
    return arms


def closed_of(item: dict, probs: dict) -> dict:
    """The closed reading over the item's labels, from the harness's probabilities."""
    labels = list(item["labels"])
    if item["question"]["type"] == "noul" and "no" not in probs:
        probs = {"yes": probs["yes"], "no": 1.0 - probs["yes"]}
    return {lab: float(probs.get(lab, 0.0)) for lab in labels}


def tvd(probs: dict, target: dict, labels: list[str]) -> float:
    return 0.5 * sum(abs(float(probs.get(lab, 0.0)) - float(target.get(lab, 0.0)))
                     for lab in labels)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_arm(scorer_path: Path, items: list[dict], distributions: dict[str, dict]) -> dict:
    """The official scorer's summary for one arm, over a temporary predictions file."""
    scorer = tp.load_scorer(scorer_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        items_path = tmp_path / "items.jsonl"
        preds_path = tmp_path / "predictions.jsonl"
        with items_path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with preds_path.open("w", encoding="utf-8") as f:
            for it in items:
                probs = distributions[it["id"]]
                if it["question"]["type"] == "noul":
                    probs = {"yes": probs["yes"]}
                f.write(json.dumps({"id": it["id"], "probs": probs}, ensure_ascii=False) + "\n")
        summary = scorer.score(read_jsonl(items_path), read_jsonl(preds_path))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the scorer-shaped items file")
    ap.add_argument("--runlog", type=Path, required=True, help="the run's run log")
    ap.add_argument("--served", required=True,
                    help="the temperature the run was served with: choice=0.62,noul=0.67,score=0.77")
    ap.add_argument("--arm", action="append", default=[], dest="arms",
                    help="an arm: name=choice=0.62,noul=0.67,score=0.77. Repeatable")
    ap.add_argument("--scorer", type=Path, default=DEFAULT_SCORER, help="the official score.py")
    ap.add_argument("--out-report", type=Path, default=None, help="the report to write")
    ap.add_argument("--out-predictions", type=Path, default=None,
                    help="the recovered T = 1 arm, as a predictions file")
    args = ap.parse_args()

    served = parse_temperatures(args.served)
    arms = parse_arms(args.arms)
    items = {it["id"]: it for it in read_jsonl(args.items)}
    runlog = {r["id"]: r for r in read_jsonl(args.runlog)}

    # ---- the arms, all out of this one run
    live: dict[str, dict] = {}          # the closed reading the run reported
    at_one: dict[str, dict] = {}        # the recovered T = 1 reading
    forward_max_diff = 0.0
    noul_crosscheck_max = 0.0
    for iid, rec in runlog.items():
        item = items.get(iid)
        if item is None or not rec.get("predicted"):
            continue
        # the harness's own rule for what the run reported (the noul's value is
        # recovered from the record's pick and closed confidence)
        probs = jp.probs_from_record(rec, item)
        closed = jp.closed_of(item, probs)
        if sum(closed.values()) <= 0.0:
            continue
        one = tp.undo_temperature(closed, served.get(item["question"]["type"], 1.0))
        at_one[iid] = one
        live[iid] = closed
        # the inversion, checked: the recovered T = 1 arm put back through the served
        # temperature must give the arm the run reported
        back = tp.scale(one, served.get(item["question"]["type"], 1.0))
        for lab in closed:
            forward_max_diff = max(forward_max_diff, abs(back[lab] - closed[lab]))
        # the noul's recovered P(yes), against the run log's own open reading: the
        # open reading is the SERVED closed reading scaled by the mass on the
        # options, so the T = 1 value is its T-th power, normalized
        if item["question"]["type"] == "noul":
            op = rec.get("open_probs") or {}
            if op.get("yes") is not None and op.get("no") is not None:
                mass = op["yes"] + op["no"]
                if mass > 0.0:
                    t = served.get("noul", 1.0)
                    served_yes = op["yes"] / mass
                    served_no = op["no"] / mass
                    wy, wn = served_yes ** t, served_no ** t
                    if wy + wn > 0.0:
                        noul_crosscheck_max = max(noul_crosscheck_max,
                                                  abs(one["yes"] - wy / (wy + wn)))

    if not at_one:
        raise SystemExit("the run log has no predictions to work with")

    def arm_distributions(temps: dict[str, float]) -> dict[str, dict]:
        out = {}
        for iid, one in at_one.items():
            qtype = items[iid]["question"]["type"]
            out[iid] = tp.scale(one, temps.get(qtype, 1.0))
        return out

    # ---- the T = 1 arm (the paired baseline) and every candidate
    rows = []
    for name, temps in [("T = 1 (recovered)", {t: 1.0 for t in TYPES})] + arms:
        dists = at_one if name.startswith("T = 1") else arm_distributions(temps)
        tvds, by_type, by_config = [], defaultdict(list), defaultdict(list)
        for iid, dist in dists.items():
            item = items[iid]
            target = item.get("target_probs")
            if not target:
                continue
            d = tvd(dist, target, list(item["labels"]))
            tvds.append(d)
            by_type[item["question"]["type"]].append(d)
            by_config[item.get("config") or "?"].append(d)
        scored = score_arm(args.scorer, [items[i] for i in sorted(dists)], dists)
        rows.append({
            "arm": name,
            "temperatures": temps,
            "items": len(dists),
            "with_human_distribution": len(tvds),
            "mean_tvd": mean(tvds),
            "mean_tvd_by_type": {k: mean(v) for k, v in sorted(by_type.items())},
            "mean_tvd_by_config": {k: mean(v) for k, v in sorted(by_config.items())},
            "scorer": scored,
        })
        print("[arms] %-22s T=%-28s mean TVD %s over %d items | Capability %s"
              % (name, ",".join("%s %s" % (k, temps[k]) for k in TYPES if k in temps),
                 ("%.4f" % mean(tvds)) if tvds else "n/a", len(tvds),
                 ("%.2f" % scored["capability"]) if scored.get("capability") is not None else "n/a"))

    print("[arms] the inversion, checked: the recovered T = 1 arm put back through the "
          "served temperature, max |difference| %.3e over %d items" % (forward_max_diff, len(at_one)))
    if noul_crosscheck_max:
        print("[arms] the noul's recovered P(yes) against the run log's own open reading, "
              "max |difference| %.3e" % noul_crosscheck_max)

    if args.out_predictions:
        with args.out_predictions.open("w", encoding="utf-8") as f:
            for iid in sorted(at_one):
                item = items[iid]
                probs = at_one[iid]
                if item["question"]["type"] == "noul":
                    probs = {"yes": probs["yes"]}
                f.write(json.dumps({"id": iid, "probs": probs}, ensure_ascii=False) + "\n")
        print("[arms] wrote the recovered T = 1 arm to %s" % args.out_predictions)

    if args.out_report:
        lines = ["# The arms of one run, paired", "",
                 "Every arm below comes out of ONE run, so they share the engine, the picks,",
                 "the logits and the state; only the arithmetic differs.  The inversion is",
                 "checked: the recovered T = 1 arm put back through the served temperature",
                 "reproduces the arm the run reported (max |difference| %.3e)." % forward_max_diff, "",
                 "| arm | temperatures | mean TVD to the humans | items | Capability | Intelligence | Calibration |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            s = r["scorer"]
            lines.append("| %s | %s | %s | %d | %s | %s | %s |" % (
                r["arm"],
                ", ".join("%s %s" % (k, r["temperatures"][k]) for k in TYPES if k in r["temperatures"]),
                ("%.4f" % r["mean_tvd"]) if r["mean_tvd"] is not None else "n/a",
                r["with_human_distribution"],
                ("%.1f" % s["capability"]) if s.get("capability") is not None else "n/a",
                ("%.1f" % s["intelligence"]) if s.get("intelligence") is not None else "n/a",
                ("%.1f" % s["calibration"]) if s.get("calibration") is not None else "n/a"))
        lines.append("")
        for r in rows:
            if not r["mean_tvd_by_type"]:
                continue
            lines.append("### %s" % r["arm"])
            lines.append("")
            lines.append("| type | mean TVD |")
            lines.append("|---|---:|")
            for k, v in r["mean_tvd_by_type"].items():
                lines.append("| %s | %.4f |" % (k, v))
            lines.append("")
            lines.append("| config | mean TVD |")
            lines.append("|---|---:|")
            for k, v in r["mean_tvd_by_config"].items():
                lines.append("| %s | %.4f |" % (k, v))
            lines.append("")
        args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("[arms] wrote %s" % args.out_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
