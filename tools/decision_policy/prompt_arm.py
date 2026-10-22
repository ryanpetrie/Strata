#!/usr/bin/env python3
"""tools/decision_policy/prompt_arm.py - the pre-registered rule, applied to one arm.

The prompt experiment's decision rule (docs/research/prompt_variants.md §6) is
written against the DESIGN SET: the licence-clean pool's 500 noul items, in two
halves.  This script applies rules (a), (b) and (c) to one arm and prints the
numbers and a pass or fail for each.  The yardstick rules ((d), (e), (f)) need the
235 and the jev-bench test sample and are applied by the scorer and by hand.

    python tools/decision_policy/prompt_arm.py --items <p1b_pool.jsonl> \
        --arm <the arm's predictions.jsonl> --base <the base arm's predictions.jsonl>

THE MARGINS ARE PRE-REGISTERED.  They are written here, not chosen per arm:

    (a) half A's mean TVD to the human distribution falls by more than 0.007,
        AND half A's mean signed difference (the bias) moves toward zero by more
        than 0.007.
    (b) half A's AUC against the humans' hard label does not fall by more than
        0.05.  A non-inferiority check: the AUC's MDE at n = 250 is about 0.13,
        so an improvement is not measurable and is not asked for.
    (c) half B's accuracy does not fall by more than 3 items (of 250), and half B's
        ECE of P(yes) does not rise by more than 0.02.  VACUOUS for a variant
        that does not change the prompt on half B (the criteria's variants are
        inert there): those arms' base-rate check is rule (d), the 235.

Honesty rules: the comparison is item by item, only the items in BOTH arms are
compared, and the number compared is said.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.decision_policy import prompt_probe as P  # noqa: E402

# the pre-registered margins
MARGIN_TVD = 0.007          # (a) the mean TVD and the mean signed difference
MARGIN_AUC = 0.05           # (b) the AUC's non-inferiority
MARGIN_ITEMS = 3            # (c) half B's accuracy, in items of 250
MARGIN_ECE = 0.02           # (c) half B's ECE

ece = P.ece                  # the official scorer's own rule, from the probe module


def arm_report(items: list[dict], base: dict[str, dict], arm: dict[str, dict]) -> dict:
    """The design set's two halves, the arm against the base arm."""
    out: dict = {"n_items": len(items)}
    for half, rows in (("A", [it for it in items if it.get("target_probs")]),
                       ("B", [it for it in items if not it.get("target_probs")])):
        both = [it for it in rows if it["id"] in base and it["id"] in arm]
        entry = {"n": len(rows), "n_both": len(both)}
        if not both:
            out[half] = entry
            continue
        bp = [P.p_yes(base[it["id"]]) for it in both]
        ap = [P.p_yes(arm[it["id"]]) for it in both]
        if half == "A":
            # half A carries the human distribution: the LEVEL and the RANKING
            entry["mean_tvd_base"] = sum(P.tvd(it, p) for it, p in zip(both, bp)) / len(both)
            entry["mean_tvd_arm"] = sum(P.tvd(it, p) for it, p in zip(both, ap)) / len(both)
            entry["d_tvd"] = entry["mean_tvd_base"] - entry["mean_tvd_arm"]
            entry["bias_base"] = sum(p - float(it["target_probs"]["yes"]) for it, p in zip(both, bp)) / len(both)
            entry["bias_arm"] = sum(p - float(it["target_probs"]["yes"]) for it, p in zip(both, ap)) / len(both)
            entry["d_bias_toward_zero"] = abs(entry["bias_base"]) - abs(entry["bias_arm"])
            hard = [1 if float(it["target_probs"]["yes"]) >= 0.5 else 0 for it in both]
            entry["auc_base"] = P.auc(bp, hard)
            entry["auc_arm"] = P.auc(ap, hard)
            entry["d_auc"] = entry["auc_arm"] - entry["auc_base"]
            entry["mean_p_yes_base"] = sum(bp) / len(both)
            entry["mean_p_yes_arm"] = sum(ap) / len(both)
            entry["flips"] = sum(1 for b, a in zip(bp, ap) if P.pick_yes(b) != P.pick_yes(a))
            entry["flip_rate"] = entry["flips"] / len(both)
        else:
            # half B: the hard label and the accuracy, the base rate's own check
            gold = [1 if it.get("expected") == "yes" else 0 for it in both]
            acc_b = sum(1 for p, g in zip(bp, gold) if (p >= 0.5) == (g == 1))
            acc_a = sum(1 for p, g in zip(ap, gold) if (p >= 0.5) == (g == 1))
            entry["gold_yes_rate"] = sum(gold) / len(gold)
            entry["acc_base"] = acc_b / len(both)
            entry["acc_arm"] = acc_a / len(both)
            entry["acc_items_lost"] = acc_b - acc_a
            entry["ece_base"] = ece([(p, g) for p, g in zip(bp, gold)])
            entry["ece_arm"] = ece([(p, g) for p, g in zip(ap, gold)])
            entry["d_ece"] = entry["ece_arm"] - entry["ece_base"]
            entry["mean_p_yes_base"] = sum(bp) / len(both)
            entry["mean_p_yes_arm"] = sum(ap) / len(both)
        out[half] = entry
    return out


def verdict(r: dict) -> list[str]:
    """Rules (a), (b) and (c), each with its number and its pass or fail."""
    lines = []
    a, b = r.get("A", {}), r.get("B", {})
    if a.get("n_both"):
        ok_a = a["d_tvd"] > MARGIN_TVD and a["d_bias_toward_zero"] > MARGIN_TVD
        lines.append(f"(a) calibration: the TVD {a['mean_tvd_base']:.4f} -> {a['mean_tvd_arm']:.4f} "
                     f"({a['d_tvd']:+.4f}), the bias {a['bias_base']:+.4f} -> {a['bias_arm']:+.4f} "
                     f"(toward zero by {a['d_bias_toward_zero']:+.4f}); the margin is {MARGIN_TVD} "
                     f"-> {'PASS' if ok_a else 'FAIL'}")
        ok_b = a["d_auc"] >= -MARGIN_AUC
        lines.append(f"(b) ranking: the AUC {a['auc_base']:.4f} -> {a['auc_arm']:.4f} "
                     f"({a['d_auc']:+.4f}); the margin is -{MARGIN_AUC} -> {'PASS' if ok_b else 'FAIL'}")
    if b.get("n_both"):
        ok_c = b["acc_items_lost"] <= MARGIN_ITEMS and b["d_ece"] <= MARGIN_ECE
        lines.append(f"(c) the base rate, on half B: the accuracy {b['acc_base']:.1%} -> "
                     f"{b['acc_arm']:.1%} ({b['acc_items_lost']:+d} items), the ECE "
                     f"{b['ece_base']:.4f} -> {b['ece_arm']:.4f} ({b['d_ece']:+.4f}); the margins are "
                     f"{MARGIN_ITEMS} items and {MARGIN_ECE} -> {'PASS' if ok_c else 'FAIL'}")
    return lines


def yard_235(arm_path: Path, base_path: Path, items_path: Path | None,
             scorer: Path | None) -> dict:
    """Rule (d): the OpenDecisions 235.

    The noul's accuracy on the 69 noul items (the base arm's is 61 of 69) and the
    235's overall Capability (the official scorer's), against the base arm's own
    run of the same set.
    """
    out: dict = {}
    items = {r["id"]: r for r in P.read_jsonl(items_path)} if items_path else {}
    for tag, path in (("arm", arm_path), ("base", base_path)):
        preds = {r["id"]: r for r in P.read_jsonl(path)}
        noul = [(iid, p) for iid, p in preds.items() if "yes" in (p.get("probs") or {})]
        if not noul:
            continue
        acc = 0
        for iid, p in noul:
            gold = items.get(iid, {}).get("expected")
            if gold in ("yes", "no"):
                acc += 1 if (float(p["probs"]["yes"]) >= 0.5) == (gold == "yes") else 0
        out[f"od235_noul_acc_{tag}"] = acc / len(noul)
        out[f"od235_noul_correct_{tag}"] = acc
        out[f"od235_noul_n_{tag}"] = len(noul)
        out[f"od235_mean_p_yes_{tag}"] = sum(float(p["probs"]["yes"]) for _, p in noul) / len(noul)
    if "od235_noul_acc_arm" in out:
        out["od235_noul_items_lost"] = (out["od235_noul_correct_base"]
                                        - out["od235_noul_correct_arm"])
    if scorer and items_path:
        out["od235_capability"] = capability(arm_path, scorer, items_path)
        out["od235_capability_base"] = capability(base_path, scorer, items_path)
        if out["od235_capability"] is not None and out["od235_capability_base"] is not None:
            out["od235_d_capability"] = out["od235_capability"] - out["od235_capability_base"]
    return out


def capability(preds_path: Path, scorer: Path, items_path: Path) -> float | None:
    """The official scorer's Capability on a predictions file."""
    import subprocess
    import tempfile
    try:
        r = subprocess.run([sys.executable, str(scorer), str(preds_path), "--data", str(items_path)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in (r.stdout or "").splitlines():
        if line.startswith("Capability"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def yard_jevbench(arm_path: Path, base_path: Path, items_path: Path) -> dict:
    """Rule (e): the jev-bench test sample's noul items with a human distribution.

    The mean TVD to the human distribution and the ECE of P(yes), against the base
    arm's own run of the same items.
    """
    out: dict = {}
    items = {r["id"]: r for r in P.read_jsonl(items_path)}
    for tag, path in (("arm", arm_path), ("base", base_path)):
        preds = {r["id"]: r for r in P.read_jsonl(path)}
        both = [(iid, p) for iid, p in preds.items()
                if items.get(iid, {}).get("question", {}).get("type") == "noul"
                and items.get(iid, {}).get("target_probs")]
        if not both:
            continue
        ps = [float(p["probs"]["yes"]) for _, p in both]
        hs = [float(items[iid]["target_probs"]["yes"]) for iid, _ in both]
        hard = [1 if h >= 0.5 else 0 for h in hs]
        out[f"jev_noul_n_{tag}"] = len(both)
        out[f"jev_tvd_{tag}"] = sum(abs(p - h) for p, h in zip(ps, hs)) / len(ps)
        out[f"jev_ece_{tag}"] = ece(list(zip(ps, hard)))
        out[f"jev_auc_{tag}"] = P.auc(ps, hard)
        out[f"jev_mean_p_yes_{tag}"] = sum(ps) / len(ps)
    if "jev_tvd_arm" in out:
        out["jev_d_tvd"] = out["jev_tvd_arm"] - out["jev_tvd_base"]
        out["jev_d_ece"] = out["jev_ece_arm"] - out["jev_ece_base"]
    return out


def yardstick_verdict(y: dict) -> list[str]:
    """Rules (d) and (e), with their numbers and their pass or fail."""
    lines = []
    if "od235_noul_acc_arm" in y:
        lost = y.get("od235_noul_items_lost", 0)
        ok = lost <= 2
        lines.append(f"(d) the 235: the noul's accuracy {y['od235_noul_acc_base']:.1%} -> "
                     f"{y['od235_noul_acc_arm']:.1%} ({lost:+d} items of {y['od235_noul_n_arm']}), "
                     f"the margin is 2 -> {'PASS' if ok else 'FAIL'}")
        if y.get("od235_capability") is not None:
            d = y.get("od235_d_capability", 0.0)
            okc = d >= -0.6
            lines.append(f"    the 235's Capability {y['od235_capability_base']:.1f} -> "
                         f"{y['od235_capability']:.1f} ({d:+.1f}); the margin is the 0.6-point "
                         f"run-to-run spread -> {'PASS' if okc else 'FAIL'}")
    if "jev_tvd_arm" in y:
        dt = y.get("jev_d_tvd", 0.0)
        de = y.get("jev_d_ece", 0.0)
        ok = dt <= 0.005 and de <= 0.02
        lines.append(f"(e) the jev-bench noul ({y['jev_noul_n_arm']} items with a human "
                     f"distribution): the mean TVD {y['jev_tvd_base']:.4f} -> {y['jev_tvd_arm']:.4f} "
                     f"({dt:+.4f}), the ECE {y['jev_ece_base']:.4f} -> {y['jev_ece_arm']:.4f} "
                     f"({de:+.4f}); the margins are 0.005 and 0.02 -> {'PASS' if ok else 'FAIL'}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the pool's items file")
    ap.add_argument("--arm", type=Path, required=True, help="the arm's predictions.jsonl")
    ap.add_argument("--base", type=Path, required=True, help="the base arm's predictions.jsonl")
    ap.add_argument("--name", default="the arm", help="the arm's name, for the report")
    ap.add_argument("--json", type=Path, default=None, help="write the numbers as JSON here")
    ap.add_argument("--od235", type=Path, default=None,
                    help="the arm's OpenDecisions 235 predictions (rule (d))")
    ap.add_argument("--od235-base", type=Path, default=None,
                    help="the base arm's OpenDecisions 235 predictions")
    ap.add_argument("--jevbench", type=Path, default=None,
                    help="the arm's jev-bench noul predictions (rule (e))")
    ap.add_argument("--jevbench-base", type=Path, default=None,
                    help="the base arm's jev-bench noul predictions")
    ap.add_argument("--jevbench-items", type=Path, default=None,
                    help="the jev-bench items file, for the human distribution")
    ap.add_argument("--scorer", type=Path, default=None,
                    help="the official score.py, for the 235's Capability")
    ap.add_argument("--od235-items", type=Path, default=None,
                    help="the OpenDecisions 235 items file, for the scorer")
    args = ap.parse_args()

    items = P.read_jsonl(args.items)
    noul = [it for it in items if it["question"]["type"] == "noul"]
    base = {r["id"]: r for r in P.read_jsonl(args.base)}
    arm = {r["id"]: r for r in P.read_jsonl(args.arm)}
    r = arm_report(noul, base, arm)
    yard: dict = {}
    if args.od235 and args.od235_base:
        yard.update(yard_235(args.od235, args.od235_base, args.od235_items, args.scorer))
    if args.jevbench and args.jevbench_base and args.jevbench_items:
        yard.update(yard_jevbench(args.jevbench, args.jevbench_base, args.jevbench_items))

    print(f"the arm: {args.name}")
    print(f"  the arm's file: {args.arm}")
    print(f"  the base arm's file: {args.base}")
    a, b = r.get("A", {}), r.get("B", {})
    print(f"\n  half A ({a.get('n')} items, {a.get('n_both')} in both arms)")
    if a.get("n_both"):
        print(f"    the mean P(yes): {a['mean_p_yes_base']:.4f} -> {a['mean_p_yes_arm']:.4f}")
        print(f"    the mean TVD to the human distribution: {a['mean_tvd_base']:.4f} -> {a['mean_tvd_arm']:.4f}")
        print(f"    the bias (the mean signed difference): {a['bias_base']:+.4f} -> {a['bias_arm']:+.4f}")
        print(f"    the AUC against the humans' hard label: {a['auc_base']:.4f} -> {a['auc_arm']:.4f}")
        print(f"    the picks that moved: {a['flips']} of {a['n_both']} ({a['flip_rate']:.1%})")
    print(f"\n  half B ({b.get('n')} items, {b.get('n_both')} in both arms)")
    if b.get("n_both"):
        print(f"    the gold 'yes' rate: {b['gold_yes_rate']:.1%}")
        print(f"    the mean P(yes): {b['mean_p_yes_base']:.4f} -> {b['mean_p_yes_arm']:.4f}")
        print(f"    the accuracy: {b['acc_base']:.1%} -> {b['acc_arm']:.1%}")
        print(f"    the ECE of P(yes): {b['ece_base']:.4f} -> {b['ece_arm']:.4f}")
    print("\n  the pre-registered rules:")
    for line in verdict(r):
        print("    " + line)
    if yard:
        print("\n  the yardstick rules:")
        for line in yardstick_verdict(yard):
            print("    " + line)
    if args.json:
        args.json.write_text(json.dumps({"design": r, "yardstick": yard}, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\n  the numbers are at {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
