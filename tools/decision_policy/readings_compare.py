"""tools/decision_policy/readings_compare.py - compare the two readings of the confidence.

The endpoint can report the confidence of a pick two ways:

  closed  the softmax over the option tokens alone - the FORCED-CHOICE reading.  The
          weight the model put on anything else (ending the turn, a digit) is not
          counted, so the options always add up to 1.
  open    the model's OWN probability of the option, P = exp(logit - Z), over the whole
          vocabulary - so the options add up to less than 1 and the rest is what it
          wanted to say instead.

For a choice or a score the ANSWER is the same under both (a renormalization does not
reorder them), so what differs is only the confidence: how well does each one separate
the right answers from the wrong ones?  That is the calibration question, and it is
what decides whether the end-of-turn weight is noise (closed) or uncertainty (open).

For a yes/no question the two can even disagree on the SIDE: a closed P(yes) above 0.5
with an open P(yes) below it means the model was torn, and the two readings answer
differently.  Those are counted separately.

    python tools/decision_policy/readings_compare.py --items <items.jsonl> \
        --predictions <predictions.jsonl> --runlog <runlog.jsonl>

The per-item correctness comes from the official scorer (score.py), imported so the
rules are exactly the ones the leaderboard uses.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

#: The data root: no tool in the repository carries this machine's absolute paths
#: (CONTRIBUTING: no private paths).  Set STRATA_DECISION_DATA to the directory that holds
#: `decision_policy_training/` and `opendecisions/`.
DATA = Path(os.environ.get("STRATA_DECISION_DATA")
            or (Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Temp" / "opencode"))
DEFAULT_SCORE = DATA / "opendecisions" / "score.py"


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("od_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:                       # split on \n only: the file holds a raw U+2028
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    """The area under the ROC of a score against correctness (ties count as 0.5).
    0.5 is chance; above 0.5 the score ranks the right answers higher."""
    pos = [s for s, c in pairs if c]
    neg = [s for s, c in pairs if not c]
    if not pos or not neg:
        return None
    total = 0.0
    for s in pos:
        for t in neg:
            total += 1.0 if s > t else (0.5 if s == t else 0.0)
    return total / (len(pos) * len(neg))


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> tuple[float | None, list]:
    """The expected calibration error: bucket by confidence, compare the mean
    confidence with the accuracy in each bucket.  0 is perfect."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, correct in pairs:
        idx = min(bins - 1, max(0, int(conf * bins)))
        buckets[idx].append((conf, correct))
    total = len(pairs)
    if not total:
        return None, []
    err = 0.0
    rows = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, c in b if c) / len(b)
        err += len(b) / total * abs(mean_conf - acc)
        rows.append((i, len(b), mean_conf, acc))
    return err, rows


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--runlog", type=Path, required=True)
    ap.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    scorer = load_scorer(args.score)
    items = {it["id"]: it for it in read_jsonl(args.items)}
    preds = {p["id"]: p["probs"] for p in read_jsonl(args.predictions)}
    log = {r["id"]: r for r in read_jsonl(args.runlog)}

    rows = []
    for iid, task in items.items():
        rec = log.get(iid, {})
        diag = rec.get("closed_confidence") is not None
        probs = preds.get(iid)
        if probs is not None:
            probs = scorer.normalized(task, probs)      # the scorer's own normalization
        outcome = scorer.item_outcome(task, probs) if probs is not None else None
        rows.append({"id": iid, "task": task, "rec": rec, "diag": diag,
                     "probs": probs, "outcome": outcome})

    n = len(rows)
    predicted = [r for r in rows if r["outcome"] is not None]
    fast = [r for r in predicted if r["diag"]]
    slow = [r for r in predicted if not r["diag"]]
    print(f"items {n}  predicted {len(predicted)}  fast path {len(fast)}  decode path {len(slow)}")
    if not fast:
        print("no fast-path items: the runlog carries no diagnostics (the single-question path "
              "is the one that reports them)")
        return 1

    fast_acc = mean([1.0 if r["outcome"]["correct"] else 0.0 for r in fast])
    line = f"\naccuracy: fast {fast_acc:.3f}"
    if slow:
        line += f"   decode {mean([1.0 if r['outcome']['correct'] else 0.0 for r in slow]):.3f}"
    print(line)

    # --- why the decode path was used: the reasons say different things (a collision
    #     is what the trie walk fixes; the others are not)
    if slow:
        reasons = {}
        for r in slow:
            key = (r["rec"].get("fallback_reason") or "not recorded").split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print("\nwhy the decode path was used:")
        for key, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {key:40s} {n:4d} items")

    # --- the two readings, over the items that took the fast path
    closed_pairs, open_pairs = [], []
    for r in fast:
        correct = r["outcome"]["correct"]
        closed_pairs.append((r["rec"]["closed_confidence"], correct))
        open_pairs.append((r["rec"]["open_confidence"], correct))

    print("\n--- calibration over the fast-path items (the pick's confidence)")
    for name, pairs in (("closed", closed_pairs), ("open", open_pairs)):
        a = auc(pairs)
        e, rows_ece = ece(pairs, args.bins)
        confs = [conf for conf, _ in pairs]
        corr = [conf for conf, ok in pairs if ok]
        wrong = [conf for conf, ok in pairs if not ok]
        line = (f"{name:7s} AUC {a if a is None else round(a, 4)}   "
                f"ECE {e if e is None else round(e, 4)}   mean conf {mean(confs):.3f}")
        if corr and wrong:
            line += f"   correct {mean(corr):.3f} / wrong {mean(wrong):.3f}"
        print(line)

    # --- the stop mass: does the weight the model put on ending the turn predict a
    #     wrong answer?  (below 0.5 means "the more it wanted to stop, the more often
    #     it was wrong")
    stop_pairs = [(r["rec"]["stop_probability"], r["outcome"]["correct"]) for r in fast]
    out_pairs = [(r["rec"]["outside_probability"], r["outcome"]["correct"]) for r in fast]
    a_stop, a_out = auc(stop_pairs), auc(out_pairs)
    print(f"\nstop mass:   AUC vs correctness {a_stop if a_stop is None else round(a_stop, 4)}"
          f"   mean {mean([s for s, _ in stop_pairs]):.4f}")
    print(f"outside mass: AUC vs correctness {a_out if a_out is None else round(a_out, 4)}"
          f"   mean {mean([s for s, _ in out_pairs]):.4f}")

    # --- the hypothesis: among the confident picks, the ones the model wanted to end
    #     the turn on are the wrong ones
    confident = [r for r in fast if r["rec"]["closed_confidence"] >= 0.8]
    if confident:
        hi = [r for r in confident if r["rec"]["stop_probability"] >= 0.1]
        lo = [r for r in confident if r["rec"]["stop_probability"] < 0.1]
        def acc(rs):
            return mean([1.0 if r["outcome"]["correct"] else 0.0 for r in rs])
        print(f"\nconfident (closed >= 0.8): {len(confident)} items")
        print(f"  stop >= 0.10 : {len(hi):4d} items  accuracy {acc(hi) if hi else float('nan'):.3f}")
        print(f"  stop <  0.10 : {len(lo):4d} items  accuracy {acc(lo) if lo else float('nan'):.3f}")

    # --- yes/no: the two readings can disagree on the side
    noul = [r for r in fast if r["task"]["question"]["type"] == "noul"]
    if noul:
        def closed_yes(r):
            """The closed reading's P(yes): the options renormalized over themselves.
            The runlog's closed_confidence is the MAX of that distribution, not
            P(yes), so it cannot be used for the side (it is always >= 0.5)."""
            op = r["rec"].get("open_probs") or {}
            if "yes" not in op or "no" not in op:
                return None
            total = op["yes"] + op["no"]
            return op["yes"] / total if total > 0 else None
        sides, disagree = {}, 0
        for r in noul:
            op = r["rec"].get("open_probs") or {}
            cy = closed_yes(r)
            if "yes" not in op or cy is None:
                continue
            if (cy >= 0.5) != (op["yes"] >= 0.5):
                disagree += 1
            gold = r["task"]["expected"] == "yes"
            for label, side in (("closed", cy >= 0.5), ("open", op["yes"] >= 0.5)):
                sides.setdefault(label, []).append(side == gold)
        print(f"\nyes/no items on the fast path: {len(noul)}"
              f"   the readings disagree on the side for {disagree} of them")
        for label, oks in sides.items():
            print(f"  {label:7s} accuracy on the side it answers: {mean(oks):.3f}"
                  f"   ({len(oks)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
