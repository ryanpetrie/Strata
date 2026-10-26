#!/usr/bin/env python3
"""tools/decision_policy/prompt_arm_choice_score.py - the choice's or the score's rule, applied.

The pre-registration is docs/research/shared_backbone_decision_policy.md §27.  This
script applies rules (a), (b), (c) to one arm on the design set and (d), (e) to the two
yardsticks when their files are given.  `prompt_arm.py` is the noul's equivalent; this one
is separate because the metrics are not the noul's: the choice's are the distance to the
human distribution and the top-label mass, the score's are the expected level and the
distance to the human distribution, and the hard half is judged per config.

    python tools/decision_policy/prompt_arm_choice_score.py --type choice \
        --items <p1b_pool.jsonl> --base <base.jsonl> --arm <C1.jsonl> \
        [--od235 <arm_235.jsonl> --od235-base <base_235.jsonl> \
         --od235-items <sample10_items.jsonl> --scorer <score.py>] \
        [--jev <arm_jev.jsonl> --jev-base <base_jev.jsonl> \
         --jev-items <jevbench_sample3_items.jsonl>]

THE MARGINS ARE PRE-REGISTERED (§27.3); they are constants here, not options.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.decision_policy.prompt_probe import ece  # noqa: E402

# the pre-registered margins (§27.3)
MARGIN_TVD_CHOICE = 0.0074
MARGIN_TVD_SCORE = 0.006
MARGIN_LEVEL = 0.006          # the score's mean signed level difference, toward zero
MARGIN_ACC_ITEMS = 3          # of 250, both types
MARGIN_ECE = 0.02
MARGIN_HARD_LEVEL = 0.05      # levels, per 5-level config
MARGIN_HARD_ITEMS = 5         # of 125, per 5-level config
MARGIN_235_LEVEL = 0.05
MARGIN_235_ITEMS = 2
MARGIN_CAPABILITY = 1.8       # the measured base spread, both types
MARGIN_JEV_TVD = 0.005
MARGIN_JEV_LEVEL = 0.05


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_probs(path: Path) -> dict[str, dict]:
    return {r["id"]: r["probs"] for r in read_jsonl(path)}


def tvd(p: dict, h: dict) -> float:
    """Total variation distance between two distributions over the same keys."""
    keys = set(p) | set(h)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(h.get(k, 0.0))) for k in keys)


def argmax(d: dict, keys: list[str] | None = None) -> str:
    use = keys if keys is not None else list(d)
    return max(use, key=lambda k: float(d.get(k, 0.0)))


def paired_ci(deltas: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float]:
    """The 95% interval of the mean of a paired per-item change, by an item bootstrap."""
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def expected_level(p: dict, k: int) -> float:
    """The distribution's expected level on the item's own lowest-first order."""
    return sum(i * float(p.get(str(i), 0.0)) for i in range(k))


def choice_block(items: list[dict], base: dict, arm: dict) -> dict:
    soft = [it for it in items if it.get("target_probs")]
    hard = [it for it in items if not it.get("target_probs")]
    out: dict = {"n_soft": len(soft), "n_hard": len(hard)}

    both = [it for it in soft if it["id"] in base and it["id"] in arm]
    if both:
        bp = [base[it["id"]] for it in both]
        ap = [arm[it["id"]] for it in both]
        hs = [it["target_probs"] for it in both]
        out["soft"] = {
            "n_both": len(both),
            "tvd_base": sum(tvd(p, h) for p, h in zip(bp, hs)) / len(both),
            "tvd_arm": sum(tvd(p, h) for p, h in zip(ap, hs)) / len(both),
            "top_mass_base": sum(max(p.values()) for p in bp) / len(both),
            "top_mass_arm": sum(max(p.values()) for p in ap) / len(both),
            "top_mass_humans": sum(max(h.values()) for h in hs) / len(both),
            "pick_matches_base": sum(1 for p, h in zip(bp, hs) if argmax(p) == argmax(h)),
            "pick_matches_arm": sum(1 for p, h in zip(ap, hs) if argmax(p) == argmax(h)),
            "picks_moved": sum(1 for p, a in zip(bp, ap) if argmax(p) != argmax(a)),
        }
        out["soft"]["d_tvd"] = out["soft"]["tvd_base"] - out["soft"]["tvd_arm"]
        out["soft"]["tvd_ci"] = list(paired_ci(
            [tvd(b, h) - tvd(a, h) for b, a, h in zip(bp, ap, hs)]))
        out["soft"]["acc_items_lost"] = (out["soft"]["pick_matches_base"]
                                         - out["soft"]["pick_matches_arm"])

    both_h = [it for it in hard if it["id"] in base and it["id"] in arm]
    if both_h:
        per_config: dict[str, dict] = {}
        for cfg in sorted({it.get("config") or "?" for it in both_h}):
            rows = [it for it in both_h if (it.get("config") or "?") == cfg]
            entry = {"n": len(rows)}
            for tag, preds in (("base", base), ("arm", arm)):
                correct = [1 if argmax(preds[it["id"]]) == str(it.get("expected")) else 0
                           for it in rows]
                conf = [max(preds[it["id"]].values()) for it in rows]
                entry[f"acc_{tag}"] = sum(correct) / len(rows)
                entry[f"ece_{tag}"] = ece(list(zip(conf, correct)))
                entry[f"correct_{tag}"] = sum(correct)
            entry["items_lost"] = entry["correct_base"] - entry["correct_arm"]
            entry["d_ece"] = entry["ece_arm"] - entry["ece_base"]
            per_config[cfg] = entry
        out["hard_per_config"] = per_config
        out["hard"] = {
            "n_both": len(both_h),
            "acc_base": sum(e["acc_base"] * e["n"] for e in per_config.values()) / len(both_h),
            "acc_arm": sum(e["acc_arm"] * e["n"] for e in per_config.values()) / len(both_h),
            "items_lost": sum(e["items_lost"] for e in per_config.values()),
            "d_ece_max": max(e["d_ece"] for e in per_config.values()),
        }
    return out


def score_block(items: list[dict], base: dict, arm: dict) -> dict:
    soft = [it for it in items if it.get("target_probs")]
    hard = [it for it in items if not it.get("target_probs")]
    out: dict = {"n_soft": len(soft), "n_hard": len(hard)}

    both = [it for it in soft if it["id"] in base and it["id"] in arm]
    if both:
        k = len(both[0]["target_probs"])
        hs = [it["target_probs"] for it in both]
        human_levels = [expected_level(h, k) for h in hs]
        base_levels = [expected_level(base[it["id"]], k) for it in both]
        arm_levels = [expected_level(arm[it["id"]], k) for it in both]
        out["soft"] = {
            "n_both": len(both), "k": k,
            "tvd_base": sum(tvd(base[it["id"]], h) for it, h in zip(both, hs)) / len(both),
            "tvd_arm": sum(tvd(arm[it["id"]], h) for it, h in zip(both, hs)) / len(both),
            "level_base": sum(base_levels) / len(both),
            "level_arm": sum(arm_levels) / len(both),
            "level_humans": sum(human_levels) / len(both),
            "top_matches_base": sum(1 for it in both
                                    if argmax(base[it["id"]]) == argmax(it["target_probs"])),
            "top_matches_arm": sum(1 for it in both
                                   if argmax(arm[it["id"]]) == argmax(it["target_probs"])),
        }
        s = out["soft"]
        s["signed_base"] = s["level_base"] - s["level_humans"]
        s["signed_arm"] = s["level_arm"] - s["level_humans"]
        s["d_signed_toward_zero"] = abs(s["signed_base"]) - abs(s["signed_arm"])
        s["d_tvd"] = s["tvd_base"] - s["tvd_arm"]
        s["tvd_ci"] = list(paired_ci(
            [tvd(base[it["id"]], h) - tvd(arm[it["id"]], h) for it, h in zip(both, hs)]))
        s["signed_ci"] = list(paired_ci(
            [abs(b - hl) - abs(a - hl)
             for b, a, hl in zip(base_levels, arm_levels, human_levels)]))
        s["top_items_lost"] = s["top_matches_base"] - s["top_matches_arm"]
        # the baseline the rule names: one point prediction at the humans' mean level
        mean_level = sum(human_levels) / len(both)
        s["mean_level_prior"] = mean_level
        s["mean_level_prior_mae"] = sum(abs(mean_level - h) for h in human_levels) / len(both)
        s["mae_base"] = sum(abs(b - h) for b, h in zip(base_levels, human_levels)) / len(both)
        s["mae_arm"] = sum(abs(a - h) for a, h in zip(arm_levels, human_levels)) / len(both)

    both_h = [it for it in hard if it["id"] in base and it["id"] in arm]
    if both_h:
        per_config: dict[str, dict] = {}
        for cfg in sorted({it.get("config") or "?" for it in both_h}):
            rows = [it for it in both_h if (it.get("config") or "?") == cfg]
            k = len(rows[0]["question"]["criteria"])
            entry = {"n": len(rows), "k": k}
            for tag, preds in (("base", base), ("arm", arm)):
                levels = [expected_level(preds[it["id"]], k) for it in rows]
                gold = [float(it["expected"]) for it in rows]
                entry[f"signed_{tag}"] = sum(l - g for l, g in zip(levels, gold)) / len(rows)
                entry[f"mae_{tag}"] = sum(abs(l - g) for l, g in zip(levels, gold)) / len(rows)
                entry[f"exact_{tag}"] = sum(1 for l, g in zip(levels, gold)
                                            if round(l) == int(g)) / len(rows)
                entry[f"correct_{tag}"] = sum(1 for l, g in zip(levels, gold)
                                              if round(l) == int(g))
            entry["d_mae"] = entry["mae_arm"] - entry["mae_base"]
            entry["items_lost"] = entry["correct_base"] - entry["correct_arm"]
            per_config[cfg] = entry
        out["hard_per_config"] = per_config
    return out


def rule_choice(r: dict) -> list[str]:
    lines, ok_all = [], True
    s, h = r.get("soft"), r.get("hard")
    if s:
        ok = s["d_tvd"] > MARGIN_TVD_CHOICE
        ok_all &= ok
        lines.append(f"(a) the mean TVD {s['tvd_base']:.4f} -> {s['tvd_arm']:.4f} "
                     f"({s['d_tvd']:+.4f} [{s['tvd_ci'][0]:+.4f}, {s['tvd_ci'][1]:+.4f}]); "
                     f"the margin is {MARGIN_TVD_CHOICE} -> "
                     f"{'PASS' if ok else 'FAIL'}")
        lines.append(f"    the top-label mass: the model {s['top_mass_base']:.4f} -> "
                     f"{s['top_mass_arm']:.4f}, the humans {s['top_mass_humans']:.4f} "
                     f"(reported, not tested); the picks moved {s['picks_moved']}/{s['n_both']}")
        ok = s["acc_items_lost"] <= MARGIN_ACC_ITEMS
        ok_all &= ok
        lines.append(f"(b) the humans' majority: {s['pick_matches_base']} -> "
                     f"{s['pick_matches_arm']} of {s['n_both']} ({s['acc_items_lost']:+d} items); "
                     f"the margin is {MARGIN_ACC_ITEMS} -> {'PASS' if ok else 'FAIL'}")
    if h:
        ok = h["items_lost"] <= MARGIN_ACC_ITEMS and h["d_ece_max"] <= MARGIN_ECE
        ok_all &= ok
        lines.append(f"(c) the hard half: {h['acc_base']:.1%} -> {h['acc_arm']:.1%} "
                     f"({h['items_lost']:+d} items of {h['n_both']}), the ECE's largest rise "
                     f"{h['d_ece_max']:+.4f}; the margins are {MARGIN_ACC_ITEMS} and "
                     f"{MARGIN_ECE} -> {'PASS' if ok else 'FAIL'}")
        for cfg, e in r.get("hard_per_config", {}).items():
            lines.append(f"      {cfg}: {e['acc_base']:.1%} -> {e['acc_arm']:.1%} "
                         f"({e['items_lost']:+d} of {e['n']}), ECE {e['ece_base']:.4f} -> "
                         f"{e['ece_arm']:.4f}")
    return lines + [f"  -> the design set: {'PASS' if ok_all else 'FAIL'}"]


def rule_score(r: dict) -> list[str]:
    lines, ok_all = [], True
    s = r.get("soft")
    if s:
        ok = (s["d_signed_toward_zero"] > MARGIN_LEVEL) and (s["d_tvd"] > MARGIN_TVD_SCORE)
        ok_all &= ok
        lines.append(f"(a) the location: the mean expected level {s['level_base']:.4f} -> "
                     f"{s['level_arm']:.4f}, the humans {s['level_humans']:.4f} "
                     f"({s['signed_base']:+.4f} -> {s['signed_arm']:+.4f}, toward zero by "
                     f"{s['d_signed_toward_zero']:+.4f} "
                     f"[{s['signed_ci'][0]:+.4f}, {s['signed_ci'][1]:+.4f}]); the TVD "
                     f"{s['tvd_base']:.4f} -> {s['tvd_arm']:.4f} ({s['d_tvd']:+.4f} "
                     f"[{s['tvd_ci'][0]:+.4f}, {s['tvd_ci'][1]:+.4f}]); the margins are "
                     f"{MARGIN_LEVEL} and {MARGIN_TVD_SCORE} -> {'PASS' if ok else 'FAIL'}")
        lines.append(f"    the mean-level prior: predicting {s['mean_level_prior']:.4f} levels "
                     f"for every item gives a mean |error| of {s['mean_level_prior_mae']:.4f}; "
                     f"the base arm is {s['mae_base']:.4f}, this arm {s['mae_arm']:.4f}")
        ok = s["top_items_lost"] <= MARGIN_ACC_ITEMS
        ok_all &= ok
        lines.append(f"(b) the humans' top level: {s['top_matches_base']} -> "
                     f"{s['top_matches_arm']} of {s['n_both']} "
                     f"({s['top_items_lost']:+d} items); the margin is {MARGIN_ACC_ITEMS} -> "
                     f"{'PASS' if ok else 'FAIL'}")
    hard = r.get("hard_per_config", {})
    if hard:
        ok = all(e["d_mae"] <= MARGIN_HARD_LEVEL and e["items_lost"] <= MARGIN_HARD_ITEMS
                 for e in hard.values())
        ok_all &= ok
        lines.append(f"(c) the hard half, per config: the mean |level error| may rise by at "
                     f"most {MARGIN_HARD_LEVEL} levels and the exact accuracy lose at most "
                     f"{MARGIN_HARD_ITEMS} of 125 -> {'PASS' if ok else 'FAIL'}")
        for cfg, e in hard.items():
            lines.append(f"      {cfg}: the signed error {e['signed_base']:+.3f} -> "
                         f"{e['signed_arm']:+.3f}, the mean |error| {e['mae_base']:.3f} -> "
                         f"{e['mae_arm']:.3f} ({e['d_mae']:+.3f}), the exact accuracy "
                         f"{e['exact_base']:.1%} -> {e['exact_arm']:.1%} ({e['items_lost']:+d})")
    return lines + [f"  -> the design set: {'PASS' if ok_all else 'FAIL'}"]


def yardstick_235(qtype: str, items_path: Path, arm_path: Path, base_path: Path,
                  scorer: Path | None) -> dict:
    items = {it["id"]: it for it in read_jsonl(items_path)}
    base, arm = load_probs(base_path), load_probs(arm_path)
    rows = [iid for iid, it in items.items()
            if it["question"]["type"] == qtype and iid in base and iid in arm]
    out: dict = {"n": len(rows)}
    if qtype == "choice":
        for tag, preds in (("base", base), ("arm", arm)):
            out[f"correct_{tag}"] = sum(1 for iid in rows
                                        if argmax(preds[iid]) == str(items[iid]["expected"]))
        out["items_lost"] = out["correct_base"] - out["correct_arm"]
    else:
        for tag, preds in (("base", base), ("arm", arm)):
            signed, hits = [], 0
            for iid in rows:
                p = preds[iid]
                k = len(p)
                level = expected_level(p, k)
                gold = int(items[iid]["expected"])
                signed.append(level - gold)
                hits += 1 if round(level) == gold else 0
            out[f"signed_{tag}"] = sum(signed) / len(rows)
            out[f"mae_{tag}"] = sum(abs(s) for s in signed) / len(rows)
            out[f"correct_{tag}"] = hits
        out["items_lost"] = out["correct_base"] - out["correct_arm"]
        out["d_mae"] = out["mae_arm"] - out["mae_base"]
    if scorer:
        out["capability"] = capability(arm_path, scorer, items_path)
        out["capability_base"] = capability(base_path, scorer, items_path)
        if out["capability"] is not None and out["capability_base"] is not None:
            out["d_capability"] = out["capability"] - out["capability_base"]
    return out


def capability(preds: Path, scorer: Path, items: Path) -> float | None:
    try:
        r = subprocess.run([sys.executable, str(scorer), str(preds), "--data", str(items)],
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


def yardstick_jev(qtype: str, items_path: Path, arm_path: Path, base_path: Path) -> dict:
    items = {it["id"]: it for it in read_jsonl(items_path)}
    base, arm = load_probs(base_path), load_probs(arm_path)
    rows = [iid for iid, it in items.items()
            if it["question"]["type"] == qtype and iid in base and iid in arm]
    out: dict = {"n": len(rows)}
    if qtype == "choice":
        for tag, preds in (("base", base), ("arm", arm)):
            out[f"correct_{tag}"] = sum(1 for iid in rows
                                        if argmax(preds[iid]) == str(items[iid]["expected"]))
        out["items_lost"] = out["correct_base"] - out["correct_arm"]
        out["acc_base"] = out["correct_base"] / len(rows)
        out["acc_arm"] = out["correct_arm"] / len(rows)
    else:
        for tag, preds in (("base", base), ("arm", arm)):
            signed = []
            for iid in rows:
                p = preds[iid]
                signed.append(expected_level(p, len(p)) - int(items[iid]["expected"]))
            out[f"mae_{tag}"] = sum(abs(s) for s in signed) / len(rows)
            out[f"signed_{tag}"] = sum(signed) / len(rows)
        out["d_mae"] = out["mae_arm"] - out["mae_base"]
        soft = [iid for iid in rows if items[iid].get("target_probs")]
        if soft:
            for tag, preds in (("base", base), ("arm", arm)):
                out[f"tvd_{tag}"] = sum(tvd(preds[iid], items[iid]["target_probs"])
                                        for iid in soft) / len(soft)
            out["n_soft"] = len(soft)
            out["d_tvd"] = out["tvd_arm"] - out["tvd_base"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=("choice", "score"))
    ap.add_argument("--items", type=Path, required=True, help="the pool's items file")
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--name", default="the arm")
    ap.add_argument("--od235", type=Path, default=None)
    ap.add_argument("--od235-base", type=Path, default=None)
    ap.add_argument("--od235-items", type=Path, default=None)
    ap.add_argument("--scorer", type=Path, default=None)
    ap.add_argument("--jev", type=Path, default=None)
    ap.add_argument("--jev-base", type=Path, default=None)
    ap.add_argument("--jev-items", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    items = [it for it in read_jsonl(args.items)
             if it["question"]["type"] == args.type]
    base, arm = load_probs(args.base), load_probs(args.arm)
    r = (choice_block if args.type == "choice" else score_block)(items, base, arm)
    print(f"the arm: {args.name}  (type {args.type})")
    print(f"  the arm's file: {args.arm}")
    print(f"  the base arm's file: {args.base}")
    print()
    for line in (rule_choice if args.type == "choice" else rule_score)(r):
        print("  " + line)

    yard: dict = {}
    if args.od235 and args.od235_base and args.od235_items:
        yard["od235"] = yardstick_235(args.type, args.od235_items, args.od235,
                                      args.od235_base, args.scorer)
    if args.jev and args.jev_base and args.jev_items:
        yard["jev"] = yardstick_jev(args.type, args.jev_items, args.jev, args.jev_base)
    if yard:
        print("\n  the yardsticks:")
        y = yard.get("od235")
        if y:
            if args.type == "choice":
                ok = y["items_lost"] <= MARGIN_235_ITEMS
                print(f"    (d) the 235's {y['n']} choice items: {y['correct_base']} -> "
                      f"{y['correct_arm']} ({y['items_lost']:+d}); the margin is "
                      f"{MARGIN_235_ITEMS} -> {'PASS' if ok else 'FAIL'}")
            else:
                ok = (y["items_lost"] <= MARGIN_235_ITEMS
                      and y["d_mae"] <= MARGIN_235_LEVEL)
                print(f"    (d) the 235's {y['n']} score items: the mean |level error| "
                      f"{y['mae_base']:.4f} -> {y['mae_arm']:.4f} ({y['d_mae']:+.4f}), the "
                      f"argmax {y['correct_base']} -> {y['correct_arm']} "
                      f"({y['items_lost']:+d}); the margins are {MARGIN_235_LEVEL} and "
                      f"{MARGIN_235_ITEMS} -> {'PASS' if ok else 'FAIL'}")
            if y.get("capability") is not None:
                okc = y["d_capability"] >= -MARGIN_CAPABILITY
                print(f"        the 235's Capability {y['capability_base']:.1f} -> "
                      f"{y['capability']:.1f} ({y['d_capability']:+.1f}); the margin is "
                      f"-{MARGIN_CAPABILITY} -> {'PASS' if okc else 'FAIL'}")
                yard["od235"]["capability_ok"] = okc
        j = yard.get("jev")
        if j:
            if args.type == "choice":
                ok = j["items_lost"] <= MARGIN_235_ITEMS
                print(f"    (e) the jev-bench {j['n']} choice items: {j['acc_base']:.1%} -> "
                      f"{j['acc_arm']:.1%} ({j['items_lost']:+d}); the margin is "
                      f"{MARGIN_235_ITEMS} -> {'PASS' if ok else 'FAIL'}")
            else:
                ok = j["d_mae"] <= MARGIN_JEV_LEVEL and j.get("d_tvd", 0.0) <= MARGIN_JEV_TVD
                print(f"    (e) the jev-bench {j['n']} score items: the mean |level error| "
                      f"{j['mae_base']:.4f} -> {j['mae_arm']:.4f} ({j['d_mae']:+.4f})"
                      + (f", the {j['n_soft']} with a human distribution give a TVD "
                         f"{j['tvd_base']:.4f} -> {j['tvd_arm']:.4f} ({j['d_tvd']:+.4f})"
                         if j.get("n_soft") else "")
                      + f"; the margins are {MARGIN_JEV_LEVEL} and {MARGIN_JEV_TVD} -> "
                        f"{'PASS' if ok else 'FAIL'}")
    if args.json:
        args.json.write_text(json.dumps({"design": r, "yardstick": yard}, indent=2,
                                        ensure_ascii=False), encoding="utf-8")
        print(f"\n  the numbers are at {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
