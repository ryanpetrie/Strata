#!/usr/bin/env python3
"""tools/decision_policy/prompt_design_set.py - what the design set is.

The prompt experiment's design set is the licence-clean pool
(decision_policy_training/items/p1b_pool.jsonl).  Its NOUL slice is 500 items in
TWO halves, and the two halves are the point: the base-rate check lives in the
difference between them.

    half A   250 `civil_comments` items   a human distribution (the mean P(yes) is
             about 0.11) and the true/false criteria.  The bias is measured here.
    half B   250 `paws` + `strategyqa`     a hard label only, at a 45.2% "yes" gold
             items                          rate.  A variant that only shifts the
                                              base rate improves half A and wrecks
                                              half B, in the same run.

This script prints the composition, so the operator can confirm what they are about
to run.  It reads a file and needs no server, no engine and no GPU.

USAGE
  python tools/decision_policy/prompt_design_set.py --items <pool.jsonl>
  python tools/decision_policy/prompt_design_set.py --items <pool.jsonl> --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def describe(items: list[dict]) -> dict:
    """The composition: per type, per half, and the two base rates."""
    out = {"items": len(items), "types": {}, "halves": {}}
    by_type: dict[str, list[dict]] = {}
    for it in items:
        by_type.setdefault(it["question"]["type"], []).append(it)
    for qtype, rows in sorted(by_type.items()):
        soft = [r for r in rows if r.get("target_probs")]
        configs: dict[str, int] = {}
        for r in rows:
            cfg = r.get("config") or "?"
            configs[cfg] = configs.get(cfg, 0) + 1
        entry = {"n": len(rows), "with_a_human_distribution": len(soft), "configs": configs,
                 "with_true_false_criteria": sum(1 for r in rows
                                                if r["question"].get("criteria"))}
        if soft and all("yes" in r["target_probs"] for r in soft):
            ys = [float(r["target_probs"]["yes"]) for r in soft]
            entry["human_mean_p_yes"] = sum(ys) / len(ys)
            entry["human_p_yes_exactly_zero"] = sum(1 for y in ys if y == 0.0)
            entry["human_p_yes_between"] = sum(1 for y in ys if 0.0 < y < 1.0)
            entry["human_p_yes_exactly_one"] = sum(1 for y in ys if y == 1.0)
        golds = [r for r in rows if r.get("expected") in ("yes", "no")]
        if golds and qtype == "noul":
            yes = sum(1 for r in golds if r["expected"] == "yes")
            entry["gold_yes_rate"] = yes / len(golds)
            entry["gold_yes"] = f"{yes}/{len(golds)}"
        out["types"][qtype] = entry

    noul = by_type.get("noul", [])
    half_a = [r for r in noul if r.get("target_probs")]
    half_b = [r for r in noul if not r.get("target_probs")]
    for name, rows in (("A", half_a), ("B", half_b)):
        if not rows:
            continue
        configs: dict[str, int] = {}
        for r in rows:
            cfg = r.get("config") or "?"
            configs[cfg] = configs.get(cfg, 0) + 1
        golds = [r for r in rows if r.get("expected") in ("yes", "no")]
        yes = sum(1 for r in golds if r["expected"] == "yes")
        entry = {"n": len(rows), "configs": configs,
                 "with_true_false_criteria": sum(1 for r in rows if r["question"].get("criteria")),
                 "gold_yes_rate": (yes / len(golds)) if golds else None,
                 "gold_yes": f"{yes}/{len(golds)}" if golds else "-"}
        if any(r.get("target_probs") for r in rows):
            ys = [float(r["target_probs"]["yes"]) for r in rows
                  if r.get("target_probs") and "yes" in r["target_probs"]]
            if ys:
                entry["human_mean_p_yes"] = sum(ys) / len(ys)
        out["halves"][name] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the pool's items file (p1b_pool.jsonl)")
    ap.add_argument("--json", action="store_true", help="print the composition as JSON")
    args = ap.parse_args()
    if not args.items.exists():
        print(f"{args.items} is not there", file=sys.stderr)
        return 2
    items = read_jsonl(args.items)
    d = describe(items)
    if args.json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return 0
    print(f"the design set: {args.items}")
    print(f"  {d['items']} items")
    for qtype, e in d["types"].items():
        print(f"\n  the {qtype} slice: {e['n']} items")
        for cfg, n in sorted(e["configs"].items()):
            print(f"    {cfg:26s} {n}")
        if "human_mean_p_yes" in e:
            print(f"    with a human distribution: {e['with_a_human_distribution']}, "
                  f"the human's mean P(yes) {e['human_mean_p_yes']:.4f} "
                  f"({e['human_p_yes_exactly_zero']} at 0, {e['human_p_yes_between']} between, "
                  f"{e['human_p_yes_exactly_one']} at 1)")
        if e.get("gold_yes_rate") is not None and qtype == "noul":
            print(f"    the gold 'yes' rate: {e['gold_yes_rate']:.1%} ({e['gold_yes']})")
        print(f"    with the true/false criteria: {e['with_true_false_criteria']}")
    print("\n  the noul's two halves - the base-rate check:")
    for name in ("A", "B"):
        e = d["halves"].get(name)
        if not e:
            continue
        print(f"    half {name}: {e['n']} items, {e['configs']}")
        print(f"      the gold 'yes' rate: {e['gold_yes_rate']:.1%} ({e['gold_yes']})"
              if e.get("gold_yes_rate") is not None else "      the gold 'yes' rate: -")
        if "human_mean_p_yes" in e:
            print(f"      the human's mean P(yes): {e['human_mean_p_yes']:.4f}")
        print(f"      with the true/false criteria: {e['with_true_false_criteria']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
