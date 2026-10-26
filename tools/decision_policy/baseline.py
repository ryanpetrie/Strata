#!/usr/bin/env python3
"""tools/decision_policy/baseline.py - the API-only decision baseline.

This is Phase 1 of the shared-backbone decision-policy exploration. It changes nothing in the engine or the
server: it only calls the already-running Strata server's OpenAI-compatible endpoint with a compact decision
prompt and measures what comes back.

WHAT IT ANSWERS
  * Is the prompt route fast enough on its own? (median and p95 end-to-end latency, time to first token)
  * How accurate is it on a small, explicitly labelled set? (accuracy, balanced accuracy, per-action
    precision/recall, confusion matrix, invalid-output rate)

WHAT IT DOES NOT DO
  * It does not start a second server or load a second model. It reads the running server's address, model id
    and API key from the launch config (or from flags / the STRATA_API_KEY environment variable) and uses them.
  * It does not claim a speedup. It only records the prompt route's own numbers, as the reference every later
    stage must beat.

LABELS. The example set is synthetic and rule-based (see make_examples.py). Accuracy here measures agreement
with that rule, not with a learned policy on real trajectories.

USAGE
  python tools/decision_policy/baseline.py                 # defaults, writes baseline_report.json/.md beside this file
  python tools/decision_policy/baseline.py --runs 3 --warmup 2 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIONS = ["CONTINUE", "CALL_TOOL", "ANSWER", "ASK_USER", "STOP"]
DEFAULT_CONFIG = Path(r"C:\tools\Strata\strata-unsloth-ud-iq4_xs.json")

SYSTEM = ("You are an agent decision policy. Choose exactly one next action from the legal list. "
          "Reply with the action name only, nothing else.")


def build_prompt(state: dict) -> str:
    """The decision prompt: the state as compact JSON, then the legal actions."""
    compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return (f"State:\n{compact}\n\n"
            f"Legal actions: {', '.join(ACTIONS)}\n\n"
            "Next action:")


def normalize_action(text: str):
    """Return the action name if the reply is exactly one legal action, else None.

    Strict on purpose: a malformed or unknown action is a failure, never a guess. Whitespace and a single
    trailing punctuation mark are tolerated because a chat model often adds them; anything else is not.
    """
    if not text:
        return None
    t = text.strip().strip(".").strip().upper()
    t = re.sub(r"\s+", " ", t)
    return t if t in ACTIONS else None


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "host": cfg.get("host", "127.0.0.1"), "port": int(cfg.get("port", 8080)),
            "model": cfg.get("model_name"), "api_key": cfg.get("api_key", "")}


def post(url: str, key: str, payload: dict, timeout: float):
    """One POST, returning (status, body_bytes). Raises urllib.error.HTTPError on a non-2xx status."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def post_stream(url: str, key: str, payload: dict, timeout: float):
    """One streaming POST. Returns (lines, ttft_ms): raw SSE lines and the time to the first content delta.

    The clock starts when the request is opened, so ttft_ms includes the server's prefill, not just the first
    token's decode. It is the honest number for a decision the caller has to wait for.
    """
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.perf_counter()
    ttft_ms = None
    lines = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            lines.append(line)
            if ttft_ms is None and line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                        break
    return lines, ttft_ms


def parse_stream(lines):
    """(content, usage, ttft_seen) from SSE lines; ttft_seen is True once a content delta arrived."""
    content_parts, usage, seen_content = [], None, False
    for ln in lines:
        if not ln.startswith("data:"):
            continue
        data = ln[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = delta.get("content")
            if piece:
                seen_content = True
                content_parts.append(piece)
    return "".join(content_parts), usage, seen_content


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def confusion(rows):
    """rows: list of (target, predicted|None). Returns matrix, per-action precision/recall, balanced accuracy."""
    matrix = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
    for target, pred in rows:
        key = pred if pred in ACTIONS else "INVALID"
        matrix[target][key] = matrix[target].get(key, 0) + 1
    per = {}
    recalls = []
    for a in ACTIONS:
        tp = matrix[a].get(a, 0)
        fn = sum(matrix[a].values()) - tp
        fp = sum(matrix[b].get(a, 0) for b in ACTIONS if b != a)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per[a] = {"precision": round(prec, 4), "recall": round(rec, 4), "support": tp + fn}
        if tp + fn:
            recalls.append(rec)
    balanced = sum(recalls) / len(recalls) if recalls else 0.0
    return matrix, per, round(balanced, 4)


def env_snapshot(cfg, url_base, key):
    snap = {"python": sys.version.split()[0], "platform": sys.platform, "config_path": cfg["path"],
            "model": cfg["model"], "endpoint": url_base, "api_key_present": bool(key)}
    # models + props are GETs
    for name, path in (("models", "/v1/models"), ("props", "/props")):
        try:
            req = urllib.request.Request(url_base + path)
            if key:
                req.add_header("Authorization", "Bearer " + key)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if name == "props":                 # the whole Jinja template is not useful in a report
                data = {k: v for k, v in data.items() if k not in ("chat_template", "chat_template_caps")}
                data["chat_template_trimmed"] = True
            snap[name] = data
        except Exception as e:                                    # noqa: BLE001 - diagnostics only
            snap[name] = {"error": str(e)}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            snap["gpu"] = {"name": parts[0], "memory_total_mib": parts[1], "memory_used_mib": parts[2],
                           "driver": parts[3]}
    except Exception as e:                                        # noqa: BLE001
        snap["gpu"] = {"error": str(e)}
    return snap


def _machine_sample(url_base: str, key: str):
    """One /v1/status machine reading (it queries the driver on demand)."""
    req = urllib.request.Request(url_base + "/v1/status")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=10) as resp:
        st = json.loads(resp.read().decode("utf-8"))
    m = st.get("machine") or {}
    return {"gpu_used_mib": (m.get("gpu") or {}).get("used_mib"),
            "ram_used_gib": (m.get("ram") or {}).get("used_gib")}


class MachineSampler:
    """Polls VRAM/RAM in the background while the timed requests run, and keeps the peak.

    A decision request is short, so the peak is a coarse number: it shows the model's footprint, not the
    per-request increment. That increment is what matters, and it is small (a few MB of KV), so the sampler's
    value is confirming that nothing unexpected was loaded.
    """

    def __init__(self, url_base: str, key: str, interval: float = 0.5):
        self.url_base, self.key, self.interval = url_base, key, interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(_machine_sample(self.url_base, self.key))
            except Exception:                                       # noqa: BLE001 - diagnostics only
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def finish(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        gpu = [s["gpu_used_mib"] for s in self.samples if s["gpu_used_mib"] is not None]
        ram = [s["ram_used_gib"] for s in self.samples if s["ram_used_gib"] is not None]
        return {"gpu_used_mib": {"min": min(gpu) if gpu else None, "max": max(gpu) if gpu else None,
                                 "n": len(gpu)},
                "ram_used_gib": {"min": min(ram) if ram else None, "max": max(ram) if ram else None,
                                 "n": len(ram)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="launch config with port/model/api_key")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None, help="defaults to the config's api_key, then STRATA_API_KEY")
    ap.add_argument("--examples", type=Path, default=HERE / "examples.jsonl")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--warmup", type=int, default=3, help="warm-up requests per example, not timed")
    ap.add_argument("--runs", type=int, default=3, help="timed requests per example")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = args.host or "127.0.0.1"
    port = args.port or cfg["port"]
    model = args.model or cfg["model"]
    key = args.api_key or cfg["api_key"] or os.environ.get("STRATA_API_KEY", "")
    url_base = f"http://{host}:{port}"
    url = url_base + "/v1/chat/completions"
    if not key:
        print("no API key: pass --api-key or set STRATA_API_KEY", file=sys.stderr)
        return 2
    if not model:
        print("no model id: pass --model or fix the config", file=sys.stderr)
        return 2

    examples = [json.loads(ln) for ln in args.examples.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[baseline] {len(examples)} examples, model={model}, {url}", flush=True)
    sampler = MachineSampler(url_base, key)
    sampler.start()

    def payload(state, stream):
        p = {"model": model,
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(state)}],
             "max_tokens": args.max_tokens, "temperature": args.temperature,
             "reasoning_effort": args.reasoning_effort}
        if stream:
            p["stream"] = True
            p["stream_options"] = {"include_usage": True}
        return p

    rows = []
    timed, warm = [], []
    warm_rows = []
    cold_first = []
    row_latency = []
    ttfts = []
    prompt_tokens, completion_tokens = [], []
    invalid = 0
    errors = 0

    for i, ex in enumerate(examples):
        state, target = ex["state"], ex["target_action"]
        for w in range(args.warmup):
            t0 = time.perf_counter()
            try:
                post(url, key, payload(state, stream=False), args.timeout)
            except Exception as e:                                # noqa: BLE001
                errors += 1
                print(f"[baseline] warmup error {ex['id']}: {e}", file=sys.stderr)
            dt = (time.perf_counter() - t0) * 1000.0
            warm.append(dt)
            warm_rows.append({"id": ex["id"], "run": w, "latency_ms": round(dt, 2)})
            if w == 0:
                cold_first.append(dt)
        for r in range(args.runs):
            t0 = time.perf_counter()
            content, usage, ttft = "", None, None
            try:
                lines, ttft = post_stream(url, key, payload(state, stream=True), args.timeout)
                content, usage, _ = parse_stream(lines)
            except Exception as e:                                # noqa: BLE001
                errors += 1
                print(f"[baseline] request error {ex['id']}: {e}", file=sys.stderr)
                rows.append((target, None))
                timed.append((time.perf_counter() - t0) * 1000.0)
                continue
            dt = (time.perf_counter() - t0) * 1000.0
            timed.append(dt)
            pred = normalize_action(content)
            if pred is None:
                invalid += 1
            rows.append((target, pred))
            if usage:
                prompt_tokens.append(int(usage.get("prompt_tokens", 0)))
                completion_tokens.append(int(usage.get("completion_tokens", 0)))
            if ttft is not None:
                ttfts.append(ttft)
            row_latency.append({"id": ex["id"], "run": r, "target": target, "predicted": pred,
                                "latency_ms": round(dt, 2), "ttft_ms": round(ttft, 2) if ttft is not None else None})
        if (i + 1) % 10 == 0:
            print(f"[baseline] {i + 1}/{len(examples)} examples done", flush=True)

    n = len(rows)
    correct = sum(1 for t, p in rows if t == p)
    accuracy = round(correct / n, 4) if n else 0.0
    matrix, per_action, balanced = confusion(rows)
    machine = sampler.finish()
    report = {
        "schema_version": 1,
        "task": "shared_backbone_decision_policy_phase1_baseline",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": env_snapshot(cfg, url_base, key),
        "settings": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                     "reasoning_effort": args.reasoning_effort, "warmup_per_example": args.warmup,
                     "timed_runs_per_example": args.runs, "endpoint": url,
                     "examples": str(args.examples), "n_examples": len(examples)},
        "label_provenance": "synthetic, rule-based (tools/decision_policy/make_examples.py); not real trajectories",
        "machine_peak_during_run": machine,
        "results": {
            "n_requests": n, "n_errors": errors, "n_invalid_output": invalid,
            "invalid_rate": round(invalid / n, 4) if n else 0.0,
            "accuracy": accuracy, "balanced_accuracy": balanced,
            "per_action": per_action, "confusion": matrix,
            "latency_ms": {"median": round(statistics.median(timed), 2) if timed else None,
                           "p95": round(percentile(timed, 95), 2) if timed else None,
                           "min": round(min(timed), 2) if timed else None,
                           "max": round(max(timed), 2) if timed else None},
            "ttft_ms": {"median": round(statistics.median(ttfts), 2) if ttfts else None,
                        "p95": round(percentile(ttfts, 95), 2) if ttfts else None,
                        "n_measured": len(ttfts)},
            "warmup_latency_ms_median": round(statistics.median(warm), 2) if warm else None,
            "cold_first_request_ms": {"median": round(statistics.median(cold_first), 2) if cold_first else None,
                                     "p95": round(percentile(cold_first, 95), 2) if cold_first else None,
                                     "min": round(min(cold_first), 2) if cold_first else None,
                                     "max": round(max(cold_first), 2) if cold_first else None},
            "prompt_tokens_median": statistics.median(prompt_tokens) if prompt_tokens else None,
            "completion_tokens_median": statistics.median(completion_tokens) if completion_tokens else None,
        },
        "rows": [{"id": ex["id"], "target": t, "predicted": p} for ex, (t, p) in zip(examples, rows)],
        "per_row_latency_ms": row_latency,
        "per_warmup_latency_ms": warm_rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    md = [f"# Phase 1 decision baseline",
          "",
          f"- measured: {report['measured_at']}",
          f"- model: `{model}`",
          f"- endpoint: `{url}`",
          f"- labels: {report['label_provenance']}",
          f"- settings: {json.dumps(report['settings'])}",
          "",
          "## Results",
          "",
          f"| metric | value |",
          f"| --- | --- |",
          f"| requests | {n} |",
          f"| errors | {errors} |",
          f"| invalid output rate | {report['results']['invalid_rate']} |",
          f"| accuracy | {accuracy} |",
          f"| balanced accuracy | {balanced} |",
          f"| latency median (ms) | {report['results']['latency_ms']['median']} |",
          f"| latency p95 (ms) | {report['results']['latency_ms']['p95']} |",
          f"| ttft median (ms) | {report['results']['ttft_ms']['median']} |",
          f"| ttft p95 (ms) | {report['results']['ttft_ms']['p95']} |",
          f"| prompt tokens (median) | {report['results']['prompt_tokens_median']} |",
          f"| completion tokens (median) | {report['results']['completion_tokens_median']} |",
          f"| first-request (cold) median (ms) | {report['results']['cold_first_request_ms']['median']} |",
          f"| first-request (cold) p95 (ms) | {report['results']['cold_first_request_ms']['p95']} |",
          "", "## Cold vs warm", "",
          "A request for a prompt the engine has seen before is fast; a request for a NEW prompt is slow. The warmup "
          "median above is the cold number (each example's first request); the timed median is the warm number "
          "(the same prompt again). The engine's own log separates them: a new prompt reads every token "
          "(`0 reused + N read`), a repeat reads only the tail (`~118 reused + 7 read`). This is the single most "
          "important number for this task, because a decision head saves the decode and the vocabulary projection, "
          "not the prefill - and the prefill is the slow part for a new decision state.", "",
          "## Per-action precision / recall",
          "",
          "| action | precision | recall | support |",
          "| --- | --- | --- | --- |"]
    for a in ACTIONS:
        d = per_action[a]
        md.append(f"| {a} | {d['precision']} | {d['recall']} | {d['support']} |")
    md += ["", "## Confusion (rows = target, cols = predicted; INVALID = malformed/unknown)", "",
           "| target \\ predicted | " + " | ".join(ACTIONS) + " | INVALID |", "| --- | " + " | ".join(["---"] * (len(ACTIONS) + 1)) + " |"]
    for a in ACTIONS:
        md.append(f"| {a} | " + " | ".join(str(matrix[a].get(b, 0)) for b in ACTIONS) + f" | {matrix[a].get('INVALID', 0)} |")
    md += ["", "## Note", "",
           "Latency is end-to-end wall clock for one request, measured on the client. It includes the server's "
           "prefill and decode, not just decoding. Warm-up requests are recorded separately and excluded from the "
           "timed statistics.", ""]
    (args.out_dir / "baseline_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[baseline] accuracy={accuracy} balanced={balanced} invalid={invalid}/{n} "
          f"median={report['results']['latency_ms']['median']}ms p95={report['results']['latency_ms']['p95']}ms")
    print(f"[baseline] wrote {args.out_dir / 'baseline_report.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
