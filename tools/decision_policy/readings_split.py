"""tools/decision_policy/readings_split.py - the two readings, per question type.

Which reading should the endpoint report?  The closed reading (the options
renormalized over themselves) and the open reading (the model's own probability) do
not answer the same question, and the official scorer measures them differently:

  choice  the calibration component is the ECE of (confidence, correct) on the hard
          tier, and the confidence comes from the probability distribution, which
          has to add up to 1 over the labels - so it is the closed reading.
  noul    the scorer's rule is (p_yes >= 0.5) == (y == 1.0) and its calibration is
          the ECE of (p_yes, y), so here the endpoint's number IS the reading.
  score   the expected level comes from the probability vector, and the
          calibration is the ECE of (confidence, correct) plus the NRMSE.

So the decision is: which reading, for each type, ranks a correct answer above a
wrong one (the AUC) and is calibrated (the ECE)?  This measures both, per type.

    python tools/decision_policy/readings_split.py --items <items.jsonl> \
        --runlog <runlog.jsonl>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:                       # split on \n only: the file holds a raw U+2028
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    """The area under the ROC of a score against correctness (ties count as 0.5)."""
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return None
    total = 0.0
    for s in pos:
        for t in neg:
            total += 1.0 if s > t else (0.5 if s == t else 0.0)
    return total / (len(pos) * len(neg))


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """The expected calibration error: bucket by confidence, compare the mean
    confidence with the accuracy in each bucket.  0 is perfect."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, correct in pairs:
        buckets[min(bins - 1, max(0, int(conf * bins)))].append((conf, correct))
    total = len(pairs)
    if not total:
        return None
    err = 0.0
    for b in buckets:
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, c in b if c) / len(b)
        err += len(b) / total * abs(mean_conf - acc)
    return err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--runlog", type=Path, required=True)
    args = ap.parse_args()

    items = {it["id"]: it for it in read_jsonl(args.items)}
    log = {r["id"]: r for r in read_jsonl(args.runlog)}

    # the per-item correctness, from the official scorer's own rule
    scorer_path = Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\score.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("od_score", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    rows = []
    for iid, task in items.items():
        rec = log.get(iid) or {}
        if rec.get("closed_confidence") is None:
            continue                       # the decode path: no probabilities
        op = rec.get("open_probs") or {}
        if not op:
            continue
        # the correctness: the scorer's own rule, on the reported distribution
        if task["question"]["type"] == "noul":
            # the scorer fills in the "no" itself, so only the "yes" is needed
            probs = {"yes": op["yes"] / (op["yes"] + op["no"])} if op.get("yes") is not None \
                and op.get("no") is not None and op["yes"] + op["no"] > 0 else {"yes": 0.0}
        else:
            probs = rec.get("probabilities") or {}
            if not probs:
                continue
        out = scorer.item_outcome(task, scorer.normalized(task, probs))
        closed_yes = (op["yes"] / (op["yes"] + op["no"])
                      if "yes" in op and "no" in op and op["yes"] + op["no"] > 0 else None)
        rows.append({"task": task, "rec": rec, "op": op, "correct": out["correct"],
                     "closed_yes": closed_yes})

    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["task"]["question"]["type"], []).append(r)
    for qtype, rs in sorted(by_type.items()):
        print(f"\n=== {qtype} ({len(rs)} items on the fast path)")
        if qtype == "noul":
            for label in ("closed", "open"):
                pairs = []
                for r in rs:
                    p = r["closed_yes"] if label == "closed" else r["op"]["yes"]
                    y = 1.0 if r["task"]["expected"] == "yes" else 0.0
                    pairs.append((p, y))
                a = auc(pairs)
                e = ece(pairs)
                mean_p = sum(p for p, _ in pairs) / len(pairs)
                side = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / len(pairs)
                print(f"  {label:7s} AUC {a if a is None else round(a, 4)}   "
                      f"ECE {e if e is None else round(e, 4)}   mean p {mean_p:.4f}   "
                      f"side accuracy {side:.4f}")
            continue
        for label in ("closed", "open"):
            pairs = []
            for r in rs:
                if label == "closed":
                    p = r["rec"]["closed_confidence"]
                else:
                    op = r["op"]
                    p = max(op.values())
                pairs.append((p, r["correct"]))
            a = auc(pairs)
            e = ece(pairs)
            mean_p = sum(p for p, _ in pairs) / len(pairs)
            print(f"  {label:7s} AUC {a if a is None else round(a, 4)}   "
                  f"ECE {e if e is None else round(e, 4)}   mean conf {mean_p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
