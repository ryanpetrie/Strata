#!/usr/bin/env python3
"""tools/decision_policy/prompt_arm_run.py - one prompt arm, for any question type.

`prompt_probe.py` is the noul's probe: it filters the pool to the noul's 500 items and
its report is the noul's bias.  The choice's and the score's arms (C1-C4, S1-S4) need the
same run loop over their own 500 items, so this tool is that loop, type-parameterised.
The pre-registration is docs/research/shared_backbone_decision_policy.md §27; the variant
text is docs/research/prompt_variants.md §3.6.

    python tools/decision_policy/prompt_arm_run.py \
        --items <p1b_pool.jsonl> --type choice --arm base \
        --out <base.jsonl> --config <the arm's launch config>

The server must be started with `"systemone": {"prompt": {"<type>": "<variant>"}}`; the
script prints the arm's variant and refuses a name the registry does not know.  It is
resume-safe: the ids already in --out are skipped, so an interrupted arm continues.

The predictions file is `{"id", "probs": {...}}` per line, keyed as the endpoint reports
them (a choice's option names, a score's level index as a string, in the item's own
lowest-first order whatever S1 renders).  The runlog is beside it and carries the arm,
the variant, the latency, the fast-path flag, the prompt tokens and the labels.
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

DEFAULT_CONFIG = Path(r"strata-unsloth-ud-iq4_xs.json")


def load_config(path: Path) -> dict:
    """The launch config, or {} when it is absent (a fresh checkout has none: it is gitignored)."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        print(f"[prompt-arm] no config at {path}: using the default port and $STRATA_API_KEY",
              file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"[prompt-arm] {path} is not JSON: {e}", file=sys.stderr)
        raise SystemExit(2)


def post(url: str, key: str, payload: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")}
    except Exception as e:                                  # noqa: BLE001 - reported, not raised
        return 0, {"error": f"{type(e).__name__}: {e}"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def arm_probs(qtype: str, answer: dict) -> dict | None:
    """The endpoint's distribution for one answer, in the shape the analysis reads."""
    if qtype == "choice":
        probs = answer.get("probabilities")
        return {k: float(v) for k, v in probs.items()} if probs else None
    if qtype == "score":
        probs = answer.get("probabilities")
        return {str(k): float(v) for k, v in probs.items()} if probs else None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the pool's items file")
    ap.add_argument("--type", required=True, choices=("choice", "score"),
                    help="which question type's 500 items to run")
    ap.add_argument("--arm", required=True,
                    help="the variant name, or 'base' for today's template")
    ap.add_argument("--out", type=Path, required=True, help="this arm's predictions.jsonl")
    ap.add_argument("--log", type=Path, default=None, help="per-item run log (default: <out>.runlog.jsonl)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="print the arm and the first prompt, send nothing")
    args = ap.parse_args()

    from serve.systemone import PROMPT_VARIANTS, render_answer_prompt, serialize_state, \
        validate_request

    variant = "base" if args.arm == "base" else args.arm
    known = PROMPT_VARIANTS.get(args.type, ())
    if variant not in known:
        print(f"unknown {args.type} prompt variant {variant!r}: use one of {', '.join(known)}",
              file=sys.stderr)
        return 2

    items = [it for it in read_jsonl(args.items) if it["question"]["type"] == args.type]
    if args.limit:
        items = items[:args.limit]
    print(f"[prompt-arm] {args.type}: {len(items)} items from {args.items}", flush=True)
    print(f"[prompt-arm] this run is the {args.arm!r} arm (the {variant!r} template).  The server "
          f"must be started with \"systemone\": {{\"prompt\": {{\"{args.type}\": \"{variant}\"}}}}",
          flush=True)

    if args.dry_run:
        it = items[0]
        state = serialize_state(it["state"])
        _, questions = validate_request({"state": it["state"],
                                         "questions": {"q": it["question"]}})
        name, q = questions[0]
        print("  the first item's prompt, as the endpoint would send it:")
        for line in render_answer_prompt(state, name, q, variant).split("\n"):
            print("    | " + line)
        counts: dict[str, int] = {}
        for it in items:
            counts[it.get("config") or "?"] = counts.get(it.get("config") or "?", 0) + 1
        print(f"  per config: {counts}")
        return 0

    cfg = load_config(args.config)
    port = args.port or cfg.get("port") or 8080
    key = args.api_key or cfg.get("api_key") or os.environ.get("STRATA_API_KEY", "")
    url = f"http://{args.host}:{port}/v1/systemone"

    done: set[str] = set()
    if args.out.exists():
        for row in read_jsonl(args.out):
            done.add(row["id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"[prompt-arm] {len(todo)} items to run (+{len(done)} already done), {url}", flush=True)
    if todo and not key:
        print("no API key: pass --api-key or set STRATA_API_KEY", file=sys.stderr)
        return 2

    log_path = args.log or args.out.with_suffix(".runlog.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    t0 = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as fout, log_path.open("a", encoding="utf-8") as flog:
        for n, it in enumerate(todo, 1):
            payload = {"state": it["state"], "questions": {"q": it["question"]}}
            t1 = time.perf_counter()
            status, body = post(url, key, payload, args.timeout)
            ms = 1000.0 * (time.perf_counter() - t1)
            answers = body.get("answers", {}) if status == 200 else {}
            rec = {"id": it["id"], "config": it.get("config"), "type": args.type,
                   "tier": it.get("tier"), "arm": args.arm, "variant": variant,
                   "http": status, "latency_ms": round(ms, 1),
                   "fast_path": bool(body.get("calibrated"))}
            if status == 200 and "q" in answers:
                answer = answers["q"]
                probs = arm_probs(args.type, answer)
                if probs:
                    fout.write(json.dumps({"id": it["id"], "probs": probs},
                                          ensure_ascii=False) + "\n")
                    fout.flush()
                    rec["predicted"] = True
                    rec["pick"] = (answer.get("choice") if args.type == "choice"
                                   else answer.get("score"))
                    rec["confidence"] = answer.get("confidence")
                    usage = body.get("usage") or {}
                    rec["prompt_tokens"] = usage.get("prompt_tokens")
                    if it.get("target_probs"):
                        rec["target_probs"] = it["target_probs"]
                    if "expected" in it:
                        rec["expected"] = it["expected"]
                    n_ok += 1
                else:
                    rec["why"] = "no probabilities in the answer"
                    n_fail += 1
            else:
                rec["why"] = str(body.get("error") or "no answer")[:300]
                n_fail += 1
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
            flog.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ok={n_ok} fail={n_fail} "
                      f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)

    print(f"[prompt-arm] done: ok={n_ok} fail={n_fail}, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    if n_fail:
        print("[prompt-arm] a failed item is retried by re-running the same command (the ids "
              "already written are skipped)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
