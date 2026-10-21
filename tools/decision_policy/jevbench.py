#!/usr/bin/env python3
"""tools/decision_policy/jevbench.py - run the jev-bench sample through /v1/systemone.

jev-bench (Praveenrajus/jev-bench) is 22 configs of System One questions in the same
shape as OpenDecisions: a state, a typed question (choice / score / noul) and a gold
answer.  Four configs carry a human label DISTRIBUTION, so a prediction can be scored
against how uncertain the humans were, not only against one right answer.  This
harness sends one item per request as a single question named "q" and writes the
official predictions file ({"id", "probs"} per line) for score.py.

Honesty rules, enforced by the code:
- One attempt per item. A non-200 reply (502 on unparseable output, truncation
  included) is OMITTED from the predictions file, which score.py counts as wrong -
  the dataset's own rule for "inputs too long for a model". Nothing is retried,
  reworded, or repaired.
- Probabilities are the endpoint's own (the closed reading over the options, which
  adds up to 1); the decode path's degenerate 1.0/0.0 is passed on as it is.
- Resume-safe: ids already in the predictions file are skipped, so an interrupted
  run continues without redoing work.
- The run log carries the human distribution and the distance to it (the TVD) for the
  items that have one, so the calibration can be reported offline.

One question per request: the endpoint's answer position is per question, so a
batched request would leave the LOGITS fast path (the decode path instead).

USAGE
  python tools/decision_policy/jevbench.py --items <items.jsonl> --out <predictions.jsonl>
  python tools/decision_policy/jevbench.py --items ... --out ... --limit 24 --dry-run
  python tools/decision_policy/jevbench.py --items ... --out ... --ids-file <ids.txt>
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
DEFAULT_CONFIG = ROOT / "strata-unsloth-ud-iq4_xs.json"


def load_config(path: Path) -> dict:
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


def normalize_probs(item: dict, probs: dict) -> dict:
    """The scorer's own normalization: a yes/no prediction fills in the "no" itself.

    The official scorer rejects a distribution that does not add up to 1 over the
    item's labels, so the harness checks it here too and reports the mismatch
    instead of writing a file the scorer will refuse.
    """
    if item["question"]["type"] == "noul" and "no" not in probs:
        probs = {"yes": probs["yes"], "no": 1.0 - probs["yes"]}
    values = {lab: float(probs.get(lab, 0.0)) for lab in item["labels"]}
    total = sum(values.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError("the probabilities over %d labels sum to %.4f, not 1" % (len(values), total))
    return values


def tvd(item: dict, probs: dict) -> float:
    """The distance to the human distribution: half the sum of the absolute
    differences over the labels (0 = exact, 1 = disjoint). The scorer's own rule."""
    target = item.get("target_probs") or {}
    labels = item["labels"]
    return 0.5 * sum(abs(probs.get(lab, 0.0) - float(target.get(lab, 0.0))) for lab in labels)


def to_probs(item: dict, answer: dict) -> dict:
    """The endpoint's answer -> the scorer's probs.

    The scorer wants a distribution over the labels: the option names for a choice,
    the level indices for a score, the probability of yes for a yes/no.  The
    fast path sends real probabilities (the closed reading: the options renormalized
    over themselves), and the scorer's calibration component is computed from exactly
    that - so they are passed on, not flattened to 1.0/0.0.  The decode path sends
    the degenerate distribution, and so does an answer with none.
    """
    qtype, labels = item["question"]["type"], item["labels"]
    if qtype == "choice":
        pick = answer["choice"]
        probs = answer.get("probabilities") or {}
        if probs and set(labels) <= set(probs) and abs(sum(float(v) for v in probs.values()) - 1.0) < 0.01:
            return {lab: float(probs[lab]) for lab in labels}
        return {lab: 1.0 if lab == pick else 0.0 for lab in labels}
    if qtype == "noul":
        return {"yes": float(answer["noul"])}
    idx = int(answer["score"])
    probs = answer.get("probabilities") or {}
    if probs and abs(sum(float(v) for v in probs.values()) - 1.0) < 0.01:
        return {str(i): float(probs.get(str(i), 0.0)) for i in range(len(labels))}
    return {str(i): 1.0 if i == idx else 0.0 for i in range(len(labels))}


MAX_ID_LEN = 40


def level_candidates(text: str) -> list[tuple[str, str | None]]:
    """The (id, description) candidates for one score level, best first.

    The dataset writes its levels as "Very negative: strongly unfavorable, ...": the
    id is the part before the delimiter and the description is the rest. The
    delimiter is not the same in every config (": " in most, "; " in one level of
    helpsteer2_helpfulness, ", " in stsb's middle levels), so every delimiter is
    tried; a level with none falls back to its first two words.
    """
    out = []
    for delim in (": ", "; ", ", "):
        head, sep, tail = text.partition(delim)
        if sep and head.strip():
            out.append((head.strip(), tail.strip()))
    words = text.split()
    if len(words) > 2:
        out.append((" ".join(words[:2]), " ".join(words[2:])))
    out.append((text.strip(), None))
    return out


def shape_score(criteria: list) -> dict | None:
    """The score levels -> {id: description}, or None when they cannot be shaped.

    The ids have to be distinct, and no id may be a prefix of another: the model
    answers one id exactly, and a prefix pair ("Not equivalent" and "Not
    equivalent, but share some details") would let the shorter one answer for the
    longer. A collision is resolved by widening the level to its whole text, which
    keeps every id a piece of the level it names.
    """
    ids: list[str] = []
    descs: list[str | None] = []
    for level in criteria:
        if not isinstance(level, str) or not level.strip():
            return None
        for cand_id, cand_desc in level_candidates(level):
            if not cand_id or len(cand_id) > MAX_ID_LEN:
                continue
            if ":" in cand_id or "\n" in cand_id or "\r" in cand_id:
                continue                       # the endpoint rejects those in a name
            ids.append(cand_id)
            descs.append(cand_desc)
            break
        else:
            return None
    for _ in range(len(ids) + 2):
        widen = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] == ids[j]:
                    widen = j                  # the same id twice: widen the later one
                elif ids[j].startswith(ids[i]):
                    widen = i                  # the shorter one is a prefix: widen it
                elif ids[i].startswith(ids[j]):
                    widen = j
                if widen is not None:
                    break
            if widen is not None:
                break
        if widen is None:
            break
        whole = criteria[widen].strip().rstrip(".")
        if whole == ids[widen] or len(whole) > MAX_ID_LEN:
            return None                       # widening does not help: do not shape
        ids[widen] = whole
        descs[widen] = None
    else:
        return None
    if len(set(ids)) != len(ids):
        return None
    return dict(zip(ids, descs))


def shape_question(q: dict) -> tuple[dict, bool]:
    """Split 'id: description' score levels into {id: description} so the model
    answers the short id instead of reproducing a ~100-character string exactly.
    Index order is unchanged, so scoring is unaffected; the full text stays visible
    as the description (the same information, reorganized like choice options).

    Falls back to (raw list, False) when the levels cannot be shaped - the endpoint
    rejects a level name with a colon, and the ids have to stay distinct."""
    if q.get("type") != "score" or not isinstance(q.get("criteria"), list):
        return dict(q), False
    shaped = shape_score(q["criteria"])
    if not shaped:
        return dict(q), False
    out = dict(q)
    out["criteria"] = shaped
    return out, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="jevbench_sample3_items.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="predictions.jsonl to append")
    ap.add_argument("--log", type=Path, default=None, help="per-item JSONL log (default: <out>.runlog.jsonl)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", type=Path, default=None, help="only run these item ids")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="list what would run, send nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    port = args.port or cfg["port"]
    key = args.api_key or cfg["api_key"] or os.environ.get("STRATA_API_KEY", "")
    url = f"http://{args.host}:{port}/v1/systemone"
    if not key:
        print("no API key: pass --api-key or set STRATA_API_KEY", file=sys.stderr)
        return 2

    items = [json.loads(l) for l in args.items.read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.ids_file:
        want = args.ids_file.read_text(encoding="utf-8").split()
        rank = {v: i for i, v in enumerate(want)}
        items = [it for it in items if it["id"] in rank]
        items.sort(key=lambda it: rank[it["id"]])   # the file's order
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    items = [it for it in items if it["id"] not in done]
    if args.limit:
        items = items[:args.limit]
    print(f"[jevbench] {len(items)} items to run (+{len(done)} already done), {url}", flush=True)
    if args.dry_run:
        per_config: dict[str, int] = {}
        per_type: dict[str, int] = {}
        shaped = 0
        for it in items:
            per_config[it.get("config") or "?"] = per_config.get(it.get("config") or "?", 0) + 1
            t = it["question"]["type"]
            per_type[t] = per_type.get(t, 0) + 1
            shaped += 1 if shape_question(it["question"])[1] else 0
        print("  per config: %s" % ", ".join(f"{k} {v}" for k, v in sorted(per_config.items())))
        print("  per type: %s" % per_type)
        print("  score questions shaped into short ids: %d" % shaped)
        for it in items[:10]:
            q, was_shaped = shape_question(it["question"])
            print("  %-28s %-6s %-22s shaped=%s" % (it["id"], it["question"]["type"],
                                                    it.get("config"), was_shaped))
        if len(items) > 10:
            print("  ... %d more" % (len(items) - 10))
        return 0

    log_path = args.log or args.out.with_suffix(".runlog.jsonl")
    n_ok = n_fail = 0
    t_start = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as fout, log_path.open("a", encoding="utf-8") as flog:
        for n, it in enumerate(items, 1):
            shaped_q, was_shaped = shape_question(it["question"])
            payload = {"state": it["state"], "questions": {"q": shaped_q}}
            try:
                status, body, ms = post(url, key, payload, args.timeout)
            except Exception as e:                                # noqa: BLE001 - connection loss etc.
                status, body, ms = -1, {"error": str(e)}, 0.0
            got = body.get("answers", {}) if status == 200 else {}
            record = {"id": it["id"], "config": it.get("config"), "type": it["question"]["type"],
                      "primitive": it.get("primitive"), "tier": it.get("tier"),
                      "http": status, "latency_ms": round(ms, 1), "shaped": was_shaped,
                      "prompt_chars": len(json.dumps(payload, ensure_ascii=False)),
                      # the fast path is the one that reports a real probability
                      "fast_path": bool(body.get("calibrated"))}
            if status == 200 and "q" in got:
                try:
                    probs = to_probs(it, got["q"])
                    probs = normalize_probs(it, probs)
                    fout.write(json.dumps({"id": it["id"], "probs": probs}, ensure_ascii=False) + "\n")
                    fout.flush()
                    record["predicted"] = True
                    usage = body.get("usage") or {}
                    record["prompt_tokens"] = usage.get("prompt_tokens")
                    record["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    # both readings of the confidence, when the server sent them (the
                    # single-question path): the closed one (renormalized over the
                    # options) and the open one (the model's own probability), with the
                    # mass it put on ending the turn and on everything else
                    diag = (body.get("diagnostics") or {}).get("q")
                    if diag:
                        record["closed_confidence"] = diag.get("closed_confidence")
                        record["open_confidence"] = diag.get("open_confidence")
                        record["stop_probability"] = diag.get("stop")
                        record["outside_probability"] = diag.get("outside")
                        record["open_probs"] = diag.get("open")
                    if diag and diag.get("reason"):
                        # why the fast path did not fire (the server's own words)
                        record["fallback_reason"] = diag["reason"]
                        record["fast_path"] = False
                    # the whole distribution the endpoint reported, so the run can be
                    # re-scored with the probabilities (the closed reading) offline
                    record["probabilities"] = got["q"].get("probabilities")
                    record["confidence"] = got["q"].get("confidence")
                    if it["question"]["type"] == "choice":
                        record["pick"] = got["q"].get("choice")
                    elif it["question"]["type"] == "noul":
                        p = float(got["q"].get("noul", 0.0))
                        record["pick"] = "yes" if p >= 0.5 else "no"
                    else:
                        record["pick"] = got["q"].get("score")
                    # the human distribution, where the dataset has one, and the
                    # distance to it (the scorer's TVD)
                    if it.get("target_probs"):
                        record["target_probs"] = it["target_probs"]
                        record["tvd"] = round(tvd(it, probs), 6)
                    n_ok += 1
                except (KeyError, TypeError, ValueError) as e:
                    record["predicted"] = False
                    record["error"] = f"unmappable answer: {e}"
                    if it.get("target_probs"):
                        record["target_probs"] = it["target_probs"]
                        record["tvd"] = 1.0      # the scorer's rule: a missing prediction is 1
                    n_fail += 1
            else:
                record["predicted"] = False
                if status == 200:
                    record["error"] = "no answer line for q (the reply did not parse)"
                else:
                    record["error"] = json.dumps(body.get("error", {}), ensure_ascii=False)[:300]
                if it.get("target_probs"):
                    record["target_probs"] = it["target_probs"]
                    record["tvd"] = 1.0          # the scorer's rule: a missing prediction is 1
                n_fail += 1
            flog.write(json.dumps(record, ensure_ascii=False) + "\n")
            flog.flush()
            if n % 25 == 0 or n == len(items):
                el = time.perf_counter() - t_start
                print(f"[jevbench] {n}/{len(items)} items ok={n_ok} fail={n_fail} "
                      f"elapsed={el:.0f}s", flush=True)
    print(f"[jevbench] done: ok={n_ok} fail={n_fail}, predictions={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
