#!/usr/bin/env python3
"""tools/decision_policy/prompt_probe.py - the PROBE of the prompt experiment: is the
noul's yes-bias in the TEMPLATE or in the TASK?

The probe is the "swap" arm (P1 in docs/research/prompt_variants.md §3.4): the noul's
criteria line names the two sides in the other order AND the answer line reads
"answer no or yes".  The answer TOKENS do not change - the model still writes " yes"
or " no" and "yes" still means the same thing - so the two arms are two PRESENTATIONS
of the same question, and the readout is untouched.

Three readouts, and what each one means:

- the FLIP RATE: the share of items whose pick changes between the two arms.  The
  engine's own run-to-run wobble flips about 1 in 100, so that is the floor.  A flip
  rate near 1% means the order does not change the model's decisions.
- the SWAP GAP: d = mean(P_yes(base) - P_yes(swap)).  d > 0 means the first-listed
  side carries a prior; d = 0 means the two presentations produce the same judgement.
- the AUC of P(yes) against the humans' hard label, in each arm.  A presentation
  change should not change the ranking, so the AUC should not move.

WHAT IT RUNS.  The design set's half A: the licence-clean pool's noul items that
carry a human distribution (the 250 `civil_comments` items).  Half A is the
`validation` split of civil_comments, so it is disjoint from the jev-bench test
split that the yardstick uses.

THE BASE ARM IS FREE.  Roadmap item 1.1b's pool run is the base arm: the pool at
T = 1, the direct readout, today's template.  Point --base at that run's
predictions file and the probe runs the SWAP arm only, and computes the gap against
the stored base.  If the base arm's artifacts are not there, run the base arm
first (--arm base) with the server on the default template, and then point --base
at it.

THE SERVER MUST BE STARTED WITH THE VARIANT.  The variant is a launch-config value,
so the two arms are two server starts:

    "systemone": {"enabled": true, "diagnostics": true, "prompt": {"noul": "swap"}}

and the default template for the base arm.  The script says which one it expects.

Honesty rules, the same as the other harnesses:
- One attempt per item.  A non-200 reply is OMITTED from the predictions file, which
  the scorer counts as wrong.  Nothing is retried, reworded or repaired.
- The probabilities are the endpoint's own (the closed reading over the options).
- The comparison is item by item: only the items in BOTH arms are compared, and the
  script says how many that is.

USAGE
  # the swap arm, against the base arm the pool run left behind
  python tools/decision_policy/prompt_probe.py --items <pool.jsonl> --out <swap.jsonl> \
      --base <the pool run's predictions.jsonl> --report <report.md>

  # the base arm, when the pool run's artifacts are not there
  python tools/decision_policy/prompt_probe.py --items <pool.jsonl> --out <base.jsonl> --arm base

  # what would run, and the prompts it would send
  python tools/decision_policy/prompt_probe.py --items <pool.jsonl> --out <swap.jsonl> --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import systemone as sysone  # noqa: E402

DEFAULT_CONFIG = ROOT / "strata-unsloth-ud-iq4_xs.json"
# The probe's arm.  The criteria line's two sides are swapped AND the answer line
# names "no" first: the full mirror of the noul's template.
PROBE_VARIANT = "swap"
BASE_VARIANT = "base"


def load_config(path: Path) -> dict:
    """The launch config's port and API key.  The config is per machine and is not in
    the repository, so a missing file is not an error: the defaults are used and
    --port / --api-key carry the rest."""
    if not path.exists():
        print(f"[prompt-probe] the launch config {path} is not there; using the default port "
              f"and STRATA_API_KEY.  Pass --config for this machine's config.", flush=True)
        return {"port": 8080, "model": None, "api_key": ""}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"port": int(cfg.get("port", 8080)), "model": cfg.get("model_name"),
            "api_key": cfg.get("api_key", "")}


def post(url: str, key: str, payload: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), \
                (time.perf_counter() - t0) * 1000.0
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:                                       # noqa: BLE001
            body = {}
        return e.code, body, (time.perf_counter() - t0) * 1000.0
    except Exception as e:                                      # noqa: BLE001 - connection loss
        return -1, {"error": str(e)}, (time.perf_counter() - t0) * 1000.0


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def design_set(path: Path, half: str = "A") -> list[dict]:
    """The design set's noul slice, one half or both.

    The pool's noul is 500 items in two halves: half A (the `civil_comments` items)
    carries the human distribution and the true/false criteria, and half B (the
    `paws` and `strategyqa` items) carries a hard label at a 45.2% "yes" rate.  The
    probe runs half A by default, because that is the half the bias is measured on;
    half B is the base rate's own check, so a variant that only moves the level
    shows up there as a loss.
    """
    items = read_jsonl(path)
    noul = [it for it in items if it["question"]["type"] == "noul"]
    if half == "A":
        out = [it for it in noul if it.get("target_probs")]
    elif half == "B":
        out = [it for it in noul if not it.get("target_probs")]
    else:
        out = noul
    if not out:
        raise SystemExit(f"{path}: no noul item in the half {half!r}")
    return out


def p_yes(pred: dict) -> float:
    """The prediction's P(yes).  The harness's own normalization: a yes/no
    prediction fills in the "no" itself."""
    probs = pred.get("probs") or {}
    return float(probs.get("yes", 0.0))


def pick_yes(p: float) -> str:
    """The scorer's own rule: a yes/no answer is "yes" when P(yes) >= 0.5."""
    return "yes" if p >= 0.5 else "no"


def auc(scores: list[float], labels: list[int]) -> float:
    """The AUC of a score against a binary label, ties counted as half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(pos) * len(neg))


def tvd(item: dict, p: float) -> float:
    """The distance to the human distribution: for a yes/no it is |P(yes) - P_human(yes)|."""
    h = float(item["target_probs"]["yes"])
    return abs(p - h)


def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float:
    """The expected calibration error, the official scorer's own rule: 10 equal-width
    bins, the weighted absolute difference between the bin's mean forecast and its
    mean outcome."""
    sums: dict[int, list] = {}
    for p, y in pairs:
        p = min(max(float(p), 0.0), 1.0)
        b = sums.setdefault(min(int(p * bins), bins - 1), [0, 0.0, 0.0])
        b[0] += 1
        b[1] += p
        b[2] += float(y)
    return sum(n / len(pairs) * abs(o / n - f / n) for n, f, o in sums.values())


def compare(items: list[dict], base: dict[str, dict], swap: dict[str, dict]) -> dict:
    """The readouts, on the items both arms answered.

    The design set's noul slice is two halves, and each half answers a different
    question:

    - half A (the items with a human distribution): the LEVEL and the RANKING.  The
      swap gap, the TVD to the human distribution, the bias, and the AUC of P(yes)
      against the humans' hard label.
    - half B (the hard label only, a 45.2% "yes" population): the BASE RATE's own
      check.  The accuracy against the hard label and the ECE of P(yes).

    The flip rate and the swap gap are computed over both halves.
    """
    both = [it for it in items if it["id"] in base and it["id"] in swap]
    out = {"n_items": len(items), "n_base": len(base), "n_swap": len(swap), "n_both": len(both)}
    if not both:
        return out
    bp = [p_yes(base[it["id"]]) for it in both]
    sp = [p_yes(swap[it["id"]]) for it in both]
    flips = sum(1 for b, s in zip(bp, sp) if pick_yes(b) != pick_yes(s))
    out.update({
        "flip_rate": flips / len(both),
        "flips": flips,
        "swap_gap": sum(b - s for b, s in zip(bp, sp)) / len(both),
        "mean_p_yes_base": sum(bp) / len(both),
        "mean_p_yes_swap": sum(sp) / len(both),
    })
    with_human = [it for it in both if it.get("target_probs")]
    if with_human:
        wh = {it["id"] for it in with_human}
        idx = [i for i, it in enumerate(both) if it["id"] in wh]
        hb = [bp[i] for i in idx]
        hs = [sp[i] for i in idx]
        human = [float(both[i]["target_probs"]["yes"]) for i in idx]
        hard = [1 if h >= 0.5 else 0 for h in human]
        out.update({
            "n_A": len(idx),
            "mean_human_yes": sum(human) / len(idx),
            "auc_base": auc(hb, hard),
            "auc_swap": auc(hs, hard),
            "tvd_base": sum(abs(b - h) for b, h in zip(hb, human)) / len(idx),
            "tvd_swap": sum(abs(s - h) for s, h in zip(hs, human)) / len(idx),
            "bias_base": sum(b - h for b, h in zip(hb, human)) / len(idx),
            "bias_swap": sum(s - h for s, h in zip(hs, human)) / len(idx),
        })
    without = [it for it in both if not it.get("target_probs")]
    if without:
        wo = {it["id"] for it in without}
        idx = [i for i, it in enumerate(both) if it["id"] in wo]
        hb = [bp[i] for i in idx]
        hs = [sp[i] for i in idx]
        gold = [1 if both[i].get("expected") == "yes" else 0 for i in idx]
        acc_b = sum(1 for p, g in zip(hb, gold) if (p >= 0.5) == (g == 1))
        acc_s = sum(1 for p, g in zip(hs, gold) if (p >= 0.5) == (g == 1))
        out.update({
            "n_B": len(idx),
            "gold_yes_rate": sum(gold) / len(gold),
            "acc_base": acc_b / len(idx),
            "acc_swap": acc_s / len(idx),
            "acc_items_lost": acc_b - acc_s,
            "ece_base": ece(list(zip(hb, gold))),
            "ece_swap": ece(list(zip(hs, gold))),
        })
    return out
    return out


def verdict(m: dict) -> str:
    """The pre-registered reading of the probe (prompt_variants.md §6)."""
    if not m.get("n_both"):
        return "NOTHING TO COMPARE: the two arms have no item in common."
    if "auc_base" not in m:
        # half B: the base rate's own check, not the level
        lost = m.get("acc_items_lost", 0)
        return ("half B (the base rate's own check): the accuracy %.1f%% -> %.1f%% (%+d items of %d), "
                "the ECE %.4f -> %.4f.  The pre-registered margins are 3 items and 0.02: %s"
                % (100 * m.get("acc_base", 0.0), 100 * m.get("acc_swap", 0.0), lost, m["n_both"],
                   m.get("ece_base", 0.0), m.get("ece_swap", 0.0),
                   "PASS" if lost <= 3 and m.get("ece_swap", 0.0) - m.get("ece_base", 0.0) <= 0.02
                   else "FAIL"))
    gap = m["swap_gap"]
    flip = m["flip_rate"]
    if abs(gap) < 0.05 and flip <= 0.03:
        return ("the order axis is DEAD: the swap gap is below 0.05 and the flip rate is at "
                "the engine's own wobble. The over-yes is in the criteria's meaning, the "
                "question or the task - not in the template's order. N1 and N2 are not run.")
    if abs(gap) < 0.05:
        return ("the order axis is dead on the LEVEL (the swap gap is below 0.05) but the "
                "picks moved: something other than the order changed. Look at the AUC.")
    return ("the order axis is ALIVE: the swap gap is %.4f and the flip rate is %.1f%%. "
            "N1 and N2 are worth their runs." % (gap, 100.0 * flip))


def render_prompt(item: dict, variant: str) -> str:
    """The prompt the endpoint will send for this item, rendered by the shipped code."""
    state = sysone.serialize_state(item["state"])
    _, questions = sysone.validate_request({"state": item["state"], "questions": {"q": item["question"]}})
    name, q = questions[0]
    return sysone.render_answer_prompt(state, name, q, variant)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the pool's items file (p1b_pool.jsonl)")
    ap.add_argument("--out", type=Path, required=True, help="this arm's predictions.jsonl")
    ap.add_argument("--log", type=Path, default=None, help="per-item run log (default: <out>.runlog.jsonl)")
    ap.add_argument("--arm", default="swap",
                    help="which template this run is: a variant name (swap, rule-form, ...) or "
                         "'base' for today's template")
    ap.add_argument("--base", type=Path, default=None,
                    help="the base arm's predictions file (the pool run's); compared against")
    ap.add_argument("--swap", type=Path, default=None,
                    help="the arm's predictions file; defaults to --out when --arm is not base")
    ap.add_argument("--report", type=Path, default=None, help="write the comparison report here")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="print what would run, send nothing")
    ap.add_argument("--half", choices=("A", "B", "both"), default="A",
                    help="which half of the design set's noul slice to run: A (the items with a "
                         "human distribution), B (the hard label only, the base rate's own check), "
                         "or both")
    args = ap.parse_args()

    items = design_set(args.items, args.half)
    if args.limit:
        items = items[:args.limit]
    variant = BASE_VARIANT if args.arm == "base" else args.arm
    if variant != BASE_VARIANT:
        from serve.systemone import PROMPT_VARIANTS
        known = PROMPT_VARIANTS.get("noul", ())
        if variant not in known:
            print(f"unknown noul prompt variant {variant!r}: use one of {', '.join(known)}",
                  file=sys.stderr)
            return 2
    print(f"[prompt-probe] the design set's half {args.half}: {len(items)} noul items, from "
          f"{args.items}", flush=True)
    print(f"[prompt-probe] this run is the {args.arm!r} arm (the {variant!r} template).  The server "
          f"must be started with \"systemone\": {{\"prompt\": {{\"noul\": \"{variant}\"}}}}"
          if variant != BASE_VARIANT else
          f"[prompt-probe] this run is the base arm (today's template).  The server must be "
          f"started with the default template (no \"prompt\" key).", flush=True)

    if args.dry_run:
        print("  the first item's prompt, as the endpoint would send it:")
        for line in render_prompt(items[0], variant).split("\n"):
            print("    | " + line)
        print(f"  the item count: {len(items)}")
        counts = {}
        for it in items:
            counts[it.get("config") or "?"] = counts.get(it.get("config") or "?", 0) + 1
        print(f"  per config: {counts}")
        print(f"  the human's mean P(yes): "
              f"{sum(float(it['target_probs']['yes']) for it in items) / len(items):.6f}")
        if args.arm == "swap" and not (args.base and args.base.exists()):
            print("  the base arm's predictions were not given: pass --base <the pool run's "
                  "predictions.jsonl> to compare against, or run the base arm first "
                  "(--arm base).")
        return 0

    cfg = load_config(args.config)
    port = args.port or cfg["port"]
    key = args.api_key or cfg["api_key"] or os.environ.get("STRATA_API_KEY", "")
    url = f"http://{args.host}:{port}/v1/systemone"
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [it for it in items if it["id"] not in done]
    print(f"[prompt-probe] {len(todo)} items to run (+{len(done)} already done), {url}", flush=True)
    if todo and not key:
        print("no API key: pass --api-key or set STRATA_API_KEY", file=sys.stderr)
        return 2

    log_path = args.log or args.out.with_suffix(".runlog.jsonl")
    n_ok = n_fail = 0
    t_start = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as fout, log_path.open("a", encoding="utf-8") as flog:
        for n, it in enumerate(todo, 1):
            payload = {"state": it["state"], "questions": {"q": it["question"]}}
            status, body, ms = post(url, key, payload, args.timeout)
            got = body.get("answers", {}) if status == 200 else {}
            record = {"id": it["id"], "config": it.get("config"), "type": "noul",
                      "tier": it.get("tier"), "arm": args.arm, "variant": variant,
                      "http": status, "latency_ms": round(ms, 1),
                      "fast_path": bool(body.get("calibrated"))}
            if status == 200 and "q" in got:
                p = float(got["q"].get("noul", 0.0))
                fout.write(json.dumps({"id": it["id"], "probs": {"yes": p, "no": 1.0 - p}},
                                      ensure_ascii=False) + "\n")
                fout.flush()
                record["predicted"] = True
                record["p_yes"] = p
                record["pick"] = pick_yes(p)
                if it.get("target_probs"):
                    record["target_probs"] = it["target_probs"]
                    record["tvd"] = round(tvd(it, p), 6)
                else:
                    # half B carries a hard label only: the distance to a distribution
                    # is not defined, and the base rate's own check is the accuracy and
                    # the ECE, which the report computes from the predictions.
                    record["expected"] = it.get("expected")
                usage = body.get("usage") or {}
                record["prompt_tokens"] = usage.get("prompt_tokens")
                diag = (body.get("diagnostics") or {}).get("q")
                if diag:
                    record["closed_confidence"] = diag.get("closed_confidence")
                    record["open_confidence"] = diag.get("open_confidence")
                    record["open_probs"] = diag.get("open")
                n_ok += 1
            else:
                n_fail += 1
            flog.write(json.dumps(record, ensure_ascii=False) + "\n")
            flog.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ok={n_ok} fail={n_fail} "
                      f"elapsed={time.perf_counter() - t_start:.0f}s", flush=True)

    print(f"[prompt-probe] done: ok={n_ok} fail={n_fail}, {time.perf_counter() - t_start:.0f}s",
          flush=True)

    # ------------------------------------------------------------ the comparison
    base_path = args.base
    swap_path = args.swap or (args.out if variant != BASE_VARIANT else None)
    if base_path and not base_path.exists():
        print(f"[prompt-probe] the base arm's predictions are not there ({base_path}).  Run the "
              f"base arm first (--arm base) or point --base at the pool run's file.",
              file=sys.stderr)
        return 3
    if not base_path or not swap_path or not swap_path.exists():
        print("[prompt-probe] the run is done, and there is nothing to compare against yet.")
        print("[prompt-probe] next: run the other arm and point --base (or --swap) at it:")
        print(f"[prompt-probe]   base arm: --arm base --out <base.jsonl>")
        print(f"[prompt-probe]   probe:    --arm swap --out <swap.jsonl> --base <base.jsonl>")
        return 0
    base = {p["id"]: p for p in read_jsonl(base_path)}
    swap = {p["id"]: p for p in read_jsonl(swap_path)}
    m = compare(items, base, swap)
    arm_name = f"the {args.arm} arm"
    if "auc_base" in m:
        lines = [
            f"# The prompt experiment: the noul's {args.arm} arm, against the base arm",
            "",
            f"the design set's half {args.half}: {m['n_items']} items; the base arm answered "
            f"{m['n_base']}, the arm {m['n_swap']}, and **{m['n_both']}** are in both.",
            "",
            "| | base | the arm |",
            "|---|---:|---:|",
            f"| the mean P(yes) | {m.get('mean_p_yes_base', float('nan')):.4f} | "
            f"{m.get('mean_p_yes_swap', float('nan')):.4f} |",
            f"| the bias (P(yes) - the human's) | {m.get('bias_base', float('nan')):+.4f} | "
            f"{m.get('bias_swap', float('nan')):+.4f} |",
            f"| the mean TVD to the human distribution | {m.get('tvd_base', float('nan')):.4f} | "
            f"{m.get('tvd_swap', float('nan')):.4f} |",
            f"| the AUC against the humans' hard label | {m.get('auc_base', float('nan')):.4f} | "
            f"{m.get('auc_swap', float('nan')):.4f} |",
            "",
            f"the humans' mean P(yes): {m.get('mean_human_yes', float('nan')):.4f}",
            "",
            "## the three readouts",
            "",
            f"- the **flip rate**: {m.get('flips', 0)} of {m['n_both']} items change their pick "
            f"= **{100.0 * m.get('flip_rate', 0.0):.1f}%** (the engine's own run-to-run wobble "
            f"flips about 1%)",
            f"- the **swap gap**: mean(P_yes(base) - P_yes(swap)) = "
            f"**{m.get('swap_gap', float('nan')):+.4f}**",
            f"- the **AUC**: {m.get('auc_base', float('nan')):.4f} against "
            f"{m.get('auc_swap', float('nan')):.4f}",
            "",
            "## the reading",
            "",
            verdict(m),
            "",
        ]
    else:
        lines = [
            f"# The prompt experiment: the noul's {args.arm} arm, half B (the base rate's check)",
            "",
            f"the design set's half {args.half}: {m['n_items']} items; the base arm answered "
            f"{m['n_base']}, the arm {m['n_swap']}, and **{m['n_both']}** are in both.",
            "",
            "| | base | the arm |",
            "|---|---:|---:|",
            f"| the mean P(yes) | {m.get('mean_p_yes_base', float('nan')):.4f} | "
            f"{m.get('mean_p_yes_swap', float('nan')):.4f} |",
            f"| the accuracy against the hard label | {100 * m.get('acc_base', 0.0):.1f}% | "
            f"{100 * m.get('acc_swap', 0.0):.1f}% |",
            f"| the ECE of P(yes) | {m.get('ece_base', float('nan')):.4f} | "
            f"{m.get('ece_swap', float('nan')):.4f} |",
            f"| the picks that moved | - | {m.get('flips', 0)} |",
            "",
            f"the gold 'yes' rate on this half: {m.get('gold_yes_rate', float('nan')):.1%}",
            "",
            "## the reading",
            "",
            verdict(m),
            "",
        ]
    report = "\n".join(lines)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
        print(f"[prompt-probe] the report is at {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
