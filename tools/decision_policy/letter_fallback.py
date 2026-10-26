#!/usr/bin/env python3
"""tools/decision_policy/letter_fallback.py - what the >26-option fallback costs the letter readout.

The letter readout can serve at most 26 options; a question with more is answered by the
decode path, whose reported distribution is a degenerate 1.0/0.0.  On jev-bench that is 144
of 683 questions.  The fix under test routes such a request to the DIRECT readout instead,
so its prompt and reading are byte-identical to the direct arm's - which means the fixed
arm's numbers are the stored direct run for those items and the stored letter run for the
others, and no new engine run is needed.

    python tools/decision_policy/letter_fallback.py \
        --items <jevbench_sample3_items.jsonl> \
        --direct <jevbench_sample3_predictions.jsonl> \
        --letters <jevbench_sample3_letter_predictions.jsonl> \
        --scorer <score.py> --out <letters_fixed.jsonl>

The rule and its result are docs/research/letter_readout.md's "The >26-option fallback".
Nothing here changes the readout; it only measures.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MAX_LETTERED = 26              # serve/systemone_letters.py's MAX_LETTERED_OPTIONS
MARGIN_CAPABILITY = 1.8        # the measured run-to-run spread
MARGIN_TVD = 0.005


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load(path: Path) -> dict[str, dict]:
    """The predictions file: {id: {"probs": {...}}}, as every harness writes it."""
    return {r["id"]: r["probs"] for r in read_jsonl(path)}


def n_options(item: dict) -> int:
    """The number of answerable options, as the readout sees them."""
    q = item["question"]
    if q["type"] == "noul":
        return 2
    return len(q["criteria"])


def ece(pairs: list[tuple[float, int]]) -> float:
    """Expected calibration error, 10 equal-width bins (the programme's own rule)."""
    if not pairs:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for p, y in pairs:
        bins[min(9, max(0, int(10 * p)))].append((p, y))
    total = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(y for _, y in b) / len(b)
            total += len(b) * abs(conf - acc)
    return total / len(pairs)


def tvd(p: dict, h: dict) -> float:
    keys = set(p) | set(h)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(h.get(k, 0.0))) for k in keys)


def correctness(item: dict, probs: dict) -> int | None:
    """1/0 when the item has a hard label, else None."""
    q = item["question"]
    if q["type"] == "choice":
        pick = max(probs, key=lambda k: float(probs[k]))
        return 1 if pick == str(item.get("expected")) else 0
    if q["type"] == "score":
        pick = max(probs, key=lambda k: float(probs[k]))
        return 1 if str(pick) == str(item.get("expected")) else 0
    if q["type"] == "noul" and "yes" in probs:
        pick = "yes" if float(probs["yes"]) >= 0.5 else "no"
        return 1 if pick == item.get("expected") else 0
    return None


def subset_stats(ids: list[str], items: dict, arms: dict[str, dict]) -> dict:
    out: dict = {"n": len(ids)}
    for tag, preds in arms.items():
        good = [i for i in ids if i in preds]
        hits = [correctness(items[i], preds[i]) for i in good]
        hits = [h for h in hits if h is not None]
        out[f"acc_{tag}"] = (sum(hits) / len(hits)) if hits else None
        conf = [(max(preds[i].values()), correctness(items[i], preds[i]))
                for i in good if correctness(items[i], preds[i]) is not None]
        out[f"ece_{tag}"] = ece(conf)
        soft = [i for i in good if items[i].get("target_probs")]
        out[f"n_soft_{tag}"] = len(soft)
        out[f"tvd_{tag}"] = (sum(tvd(preds[i], items[i]["target_probs"])
                                 for i in soft) / len(soft)) if soft else None
        if soft:
            out[f"mass_{tag}"] = sum(max(preds[i].values()) for i in soft) / len(soft)
    return out


def score(preds: Path, items: Path, scorer: Path) -> str:
    r = subprocess.run([sys.executable, str(scorer), str(preds), "--data", str(items)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in (r.stdout or "").splitlines()
             if ln.startswith("Capability") or ln.startswith("Distance")]
    return "\n".join("    " + ln for ln in lines) or (r.stdout + r.stderr)


def capability_of(text: str) -> float | None:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Capability"):
            try:
                return float(s.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def distance_of(text: str) -> float | None:
    """The scorer's 'Distance to the true probabilities' (the mean TVD on the soft items)."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Distance"):
            try:
                return float(s.rsplit(":", 1)[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--direct", type=Path, required=True)
    ap.add_argument("--letters", type=Path, required=True)
    ap.add_argument("--letters-runlog", type=Path, default=None,
                    help="optional: the letter run's log, to show the fallback reason")
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="write the recombined predictions here")
    args = ap.parse_args()

    items = {it["id"]: it for it in read_jsonl(args.items)}
    direct, letters = load(args.direct), load(args.letters)
    common = [i for i in items if i in direct and i in letters]
    long_ids = [i for i in common if n_options(items[i]) > MAX_LETTERED]
    short_ids = [i for i in common if n_options(items[i]) <= MAX_LETTERED]
    fixed = {i: (direct[i] if i in set(long_ids) else letters[i]) for i in common}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in common:
            f.write(json.dumps({"id": i, "probs": fixed[i]},
                               ensure_ascii=False) + "\n")

    print(f"the jev-bench sample: {len(items)} items, {len(common)} in both arms")
    print(f"  more than {MAX_LETTERED} options: {len(long_ids)};  served by letters: {len(short_ids)}")
    if args.letters_runlog and args.letters_runlog.exists():
        log = {r["id"]: r for r in read_jsonl(args.letters_runlog)}
        reasons: dict[str, int] = {}
        for i in long_ids:
            why = (log.get(i) or {}).get("fallback_reason") or "?"
            reasons[why] = reasons.get(why, 0) + 1
        print(f"  the letter arm's own reason on those items: {reasons}")

    arms = {"direct": direct, "letters": letters, "fixed": fixed}
    print("\nthe official scorer:")
    caps, dists = {}, {}
    for tag, preds in (("direct", args.direct), ("letters", args.letters), ("fixed", args.out)):
        text = score(preds, args.items, args.scorer)
        caps[tag] = capability_of(text)
        dists[tag] = distance_of(text)
        print(f"  {tag}:")
        print(text)

    print("\nthe two subsets, per arm (the confidence's ECE, the human TVD, the mode mass):")
    for label, ids in (("the >26-option items", long_ids), ("the letters' own items", short_ids)):
        s = subset_stats(ids, items, arms)
        print(f"  {label} ({s['n']}):")
        for tag in arms:
            tvd_s = (f"{s[f'tvd_{tag}']:.4f}" if s.get(f"tvd_{tag}") is not None else "-")
            acc_s = (f"{100 * s[f'acc_{tag}']:.1f}%" if s.get(f"acc_{tag}") is not None else "-")
            mass_s = (f"{s[f'mass_{tag}']:.4f}" if s.get(f"mass_{tag}") is not None else "-")
            print(f"    {tag:8s} accuracy {acc_s:>6s}  ECE {s[f'ece_{tag}']:.4f}  "
                  f"human TVD {tvd_s:>6s} (n {s[f'n_soft_{tag}']})  mode mass {mass_s}")

    if caps["direct"] is not None and caps["fixed"] is not None:
        d_cap = caps["fixed"] - caps["direct"]
        d_tvd = ((dists["fixed"] - dists["direct"])
                 if dists["direct"] is not None and dists["fixed"] is not None else None)
        print("\nthe rule (letter_readout.md):")
        print(f"  Capability: direct {caps['direct']:.1f}, letters {caps['letters']:.1f}, "
              f"fixed {caps['fixed']:.1f} (fixed - direct {d_cap:+.1f}; the margin is "
              f"{MARGIN_CAPABILITY})")
        rule_cap = d_cap >= -MARGIN_CAPABILITY
        print(f"  -> the Capability half: {'PASS' if rule_cap else 'FAIL'}")
        if d_tvd is not None:
            print(f"  the human TVD: direct {dists['direct']:.4f}, letters "
                  f"{dists['letters']:.4f}, fixed {dists['fixed']:.4f} "
                  f"(fixed - direct {d_tvd:+.4f}; the margin is {MARGIN_TVD})")
            rule_tvd = d_tvd <= MARGIN_TVD
            print(f"  -> the TVD half: {'PASS' if rule_tvd else 'FAIL'}")
            print(f"\n  VERDICT: the fallback "
                  f"{'is' if (rule_cap and rule_tvd) else 'is NOT'} the whole loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
