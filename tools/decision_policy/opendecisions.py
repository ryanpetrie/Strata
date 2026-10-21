#!/usr/bin/env python3
"""tools/decision_policy/opendecisions.py - run OpenDecisions through /v1/systemone.

OpenDecisions (jgeuter/OpenDecisions, CC-BY-SA-4.0, "do not use for training") is a
validation set of 2,352 Jev-style decisions whose task format maps 1:1 onto the
endpoint: state + typed question (choice/noul/score) + labels + expected. This
harness sends one item per request as a single question named "q" and writes the
official predictions file ({"id", "probs"} per line) for score.py.

Honesty rules, enforced by the code:
- One attempt per item. A non-200 reply (502 on unparseable output, truncation
  included) is OMITTED from the predictions file, which score.py counts as wrong -
  the dataset's own rule for "inputs too long for a model". Nothing is retried,
  reworded, or repaired.
- Probabilities are degenerate (1.0 on the pick): the endpoint is uncalibrated by
  design ("calibrated": false), so Calibration will be poor and Intelligence is
  the meaningful number. Reported as-is, never massaged.
- Resume-safe: ids already in the predictions file are skipped, so an
  interrupted run continues without redoing work.
- Batch mode (--batch-k N, --batch-seed S): groups N items per request to measure
  batched-vs-serial behavior. Each request concatenates the N states as numbered
  sections with one question per section (each instruction gains an "answer only
  from Section i" scoping line, a documented deviation from the reference
  presentation). A failed batch omits ALL its items (this correlated failure is
  the point: it is what batching risks). Per-question latency is the batch total
  divided by N. Compare against the serial run of the same ids for agreement,
  correctness, and format-failure rate.

USAGE
  python tools/decision_policy/opendecisions.py --items <opendecisions.jsonl> --out <predictions.jsonl>
  python tools/decision_policy/opendecisions.py --items ... --out ... --limit 24 --stratify
  python tools/decision_policy/opendecisions.py --items ... --out ... --ids-file <ids.txt>
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

DEFAULT_CONFIG = Path(r"C:\tools\strata\strata-unsloth-ud-iq4_xs.json")


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


def to_probs(item: dict, answer: dict) -> dict:
    """The endpoint's answer -> the scorer's probs.

    The scorer wants a distribution over the labels: the option names for a choice,
    the level indices for a score, the probability of yes for a yes/no.  The
    first-token path sends real probabilities (the closed reading: the options
    renormalized over themselves), and the scorer's calibration component is computed
    from exactly that - so they are passed on, not flattened to 1.0/0.0.  The decode
    path sends the degenerate distribution, and so does an answer with none.
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


def shape_question(q: dict) -> tuple[dict, bool]:
    """Split 'id: description' score levels into {id: description} so the model
    answers the short id instead of reproducing a ~100-character string exactly.
    Index order is unchanged, so scoring is unaffected; the full text stays visible
    as the description (the same information, reorganized like choice options).
    Falls back to (raw list, False) when any level does not fit the pattern."""
    if q.get("type") == "score" and isinstance(q.get("criteria"), list):
        levels = q["criteria"]
        ids: list[str] = []
        ok = True
        for lv in levels:
            head, sep, tail = lv.partition(": ")
            head, tail = head.strip(), tail.strip()
            if not sep or not head or not tail or len(head) > 40 or head.isdigit() \
                    or ":" in head or "\n" in head or "\r" in head:
                ok = False
                break
            ids.append(head)
        if ok and len(set(ids)) == len(ids):
            out = dict(q)
            out["criteria"] = {i: lv.partition(": ")[2].strip() for i, lv in zip(ids, levels)}
            return out, True
    return dict(q), False


def serialize_state_for_batch(state) -> str:
    """Canonical state text, matching the endpoint's own serialization."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def make_batches(items: list[dict], k: int, seed: int, order: str = "shuffle") -> list[list[dict]]:
    """Groups of k. shuffle: deterministic shuffle then consecutive groups (mixed
    batches). file: consecutive groups of the input order (same-question batches
    when the caller lists cluster-mates adjacently)."""
    if order == "file":
        return [items[i:i + k] for i in range(0, len(items), k)]
    import random
    sequence = list(items)
    rng = random.Random(seed)
    rng.shuffle(sequence)
    return [sequence[i:i + k] for i in range(0, len(sequence), k)]


def batch_payload(batch: list[dict]) -> tuple[dict, dict]:
    """-> (request payload, {question_name: (item, shaped)}). Each item's state is
    a numbered section; each question is scoped to its section."""
    sections, questions, mapping = [], {}, {}
    for i, it in enumerate(batch, 1):
        sections.append(f"--- Section {i} ---\n" + serialize_state_for_batch(it["state"]))
        shaped_q, was_shaped = shape_question(it["question"])
        scoped = dict(shaped_q)
        scoped["instructions"] = shaped_q["instructions"] + f" Answer only from Section {i}; ignore the other sections."
        name = f"q{i}"
        questions[name] = scoped
        mapping[name] = (it, was_shaped)
    return {"state": "\n\n".join(sections), "questions": questions}, mapping


def passes_filter(item: dict, args) -> bool:
    if args.types and item["question"]["type"] not in args.types:
        return False
    if args.tiers and item.get("tier") not in args.tiers:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="opendecisions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="predictions.jsonl to append")
    ap.add_argument("--log", type=Path, default=None, help="per-item JSONL log (default: <out>.runlog.jsonl)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratify", action="store_true", help="spread --limit across type x tier")
    ap.add_argument("--ids-file", type=Path, default=None, help="only run these item ids")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="list what would run, send nothing")
    ap.add_argument("--batch-k", type=int, default=1,
                    help="items per request (1 = serial, one question per request)")
    ap.add_argument("--batch-seed", type=int, default=43, help="seed for batch grouping")
    ap.add_argument("--batch-order", choices=("shuffle", "file"), default="shuffle",
                    help="shuffle: random groups (mixed batches); file: consecutive groups of "
                         "the input order (same-question batches when the ids file lists "
                         "cluster-mates adjacently)")
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
        items.sort(key=lambda it: rank[it["id"]])   # the file's order: cluster-mates stay adjacent
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    items = [it for it in items if it["id"] not in done]
    if args.stratify and args.limit:
        buckets: dict[tuple, list] = {}
        for it in items:
            buckets.setdefault((it["question"]["type"], it.get("tier")), []).append(it)
        keys = sorted(buckets)
        picked, i = [], 0
        while len(picked) < args.limit:
            progressed = False
            for k in keys:
                if i < len(buckets[k]) and len(picked) < args.limit:
                    picked.append(buckets[k][i])
                    progressed = True
            if not progressed:
                break
            i += 1
        items = picked
    elif args.limit:
        items = items[:args.limit]
    print(f"[opendecisions] {len(items)} items to run (+{len(done)} already done), {url}", flush=True)
    if args.dry_run:
        k = max(1, int(args.batch_k or 1))
        groups = make_batches(items, k, args.batch_seed, args.batch_order) if k > 1 else [[it] for it in items[:20]]
        for g in groups[:8]:
            print("  batch of %d: %s" % (len(g), ", ".join(f"{it['id']} [{it['question']['type']}]" for it in g)))
        print(f"  ... {len(groups)} batches total")
        return 0

    log_path = args.log or args.out.with_suffix(".runlog.jsonl")
    k = max(1, int(args.batch_k or 1))
    if k == 1:
        units: list[tuple[dict, dict]] = []
        for it in items:
            shaped_q, was_shaped = shape_question(it["question"])
            units.append(({"state": it["state"], "questions": {"q": shaped_q}},
                          {"q": (it, was_shaped)}))
    else:
        units = []
        for batch in make_batches(items, k, args.batch_seed, args.batch_order):
            payload, mapping = batch_payload(batch)
            units.append((payload, mapping))
    n_ok = n_fail = 0
    t_start = time.perf_counter()
    total_questions = sum(len(m) for _, m in units)
    print(f"[opendecisions] {len(units)} requests x up to {k} questions "
          f"({total_questions} questions over {len(items)} items)", flush=True)
    with args.out.open("a", encoding="utf-8") as fout, log_path.open("a", encoding="utf-8") as flog:
        for n, (payload, mapping) in enumerate(units):
            try:
                status, body, ms = post(url, key, payload, args.timeout)
            except Exception as e:                                # noqa: BLE001 - connection loss etc.
                status, body, ms = -1, {"error": str(e)}, 0.0
            per_q_ms = ms / max(1, len(mapping))
            got = body.get("answers", {}) if status == 200 else {}
            for name, (it, was_shaped) in mapping.items():
                record = {"id": it["id"], "type": it["question"]["type"], "tier": it.get("tier"),
                          "http": status, "latency_ms": round(per_q_ms, 1),
                          "batch_latency_ms": round(ms, 1), "batch_k": len(mapping),
                          "shaped": was_shaped}
                if status == 200 and name in got:
                    try:
                        fout.write(json.dumps({"id": it["id"],
                                               "probs": to_probs(it, got[name])}, ensure_ascii=False) + "\n")
                        fout.flush()
                        record["predicted"] = True
                        usage = body.get("usage") or {}
                        record["prompt_tokens"] = usage.get("prompt_tokens")
                        record["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        # both readings of the confidence, when the server sent them (the
                        # single-question path): the closed one (renormalized over the
                        # options) and the open one (the model's own probability), with
                        # the mass it put on ending the turn and on everything else
                        diag = (body.get("diagnostics") or {}).get(name)
                        if diag:
                            record["closed_confidence"] = diag.get("closed_confidence")
                            record["open_confidence"] = diag.get("open_confidence")
                            record["stop_probability"] = diag.get("stop")
                            record["outside_probability"] = diag.get("outside")
                            record["open_probs"] = diag.get("open")
                            # the question is on the fast path when its own diagnostics
                            # carry the two readings.  The body's "calibrated" is the AND
                            # over the questions of a merged answer, so it cannot say
                            # which one of them fell back.
                            record["fast_path"] = "closed_confidence" in diag
                        if diag and diag.get("reason"):
                            # why the fast path did not fire (the server's own words)
                            record["fallback_reason"] = diag["reason"]
                            record["fast_path"] = False
                        # the whole distribution the endpoint reported, so the run can be
                        # re-scored with the probabilities (the closed reading) offline
                        record["probabilities"] = got[name].get("probabilities")
                        record["confidence"] = got[name].get("confidence")
                        n_ok += 1
                    except (KeyError, TypeError, ValueError) as e:
                        record["predicted"] = False
                        record["error"] = f"unmappable answer: {e}"
                        n_fail += 1
                else:
                    record["predicted"] = False
                    if status == 200:
                        record["error"] = f"no answer for {name} (the question could not be answered)"
                    else:
                        record["error"] = json.dumps(body.get("error", {}), ensure_ascii=False)[:300]
                    n_fail += 1
                flog.write(json.dumps(record, ensure_ascii=False) + "\n")
                flog.flush()
            if (n + 1) % 25 == 0 or (n + 1) == len(units):
                el = time.perf_counter() - t_start
                print(f"[opendecisions] {n + 1}/{len(units)} requests ok={n_ok} fail={n_fail} "
                      f"elapsed={el:.0f}s", flush=True)
    print(f"[opendecisions] done: ok={n_ok} fail={n_fail}, predictions={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
