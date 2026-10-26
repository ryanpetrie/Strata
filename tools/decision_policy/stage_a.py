#!/usr/bin/env python3
"""tools/decision_policy/stage_a.py - Stage A: a fixed long preamble with the state at the end.

THE HYPOTHESIS. The Phase 1 baseline showed that a NEW decision state costs ~1.6 s because the engine reads every
prompt token (its prefill is expert-I/O bound, ~70-84 tok/s). It also showed a REPEATED prompt costs ~160 ms, because
the engine resumes from a checkpoint. The engine checkpoints the end of a system prompt once that prompt is
2,048 tokens or more (`--prompt-cache-root`, default 2048; `docs/DETAILS.md:1096-1106`), and keeps that root for
good. So a decision client that sends the SAME long preamble every time, with the variable state at the end, should
read only the state's tail - and a NEW state should be nearly as cheap as a repeat.

This script measures exactly that. It is API-only: it uses the already-running server and starts nothing.

WHAT IS COMPARED
  * `preamble` arm: system = a fixed ~2,048+ token decision spec, user = the compact state.
  * `short` arm (optional): the Phase 1 prompt with no preamble, for contrast.

The preamble's worked examples are DISJOINT from the 60 test examples, so the accuracy number is not inflated by
being handed the answers. Each test state is unique, so every request is a new prompt (the cold path) - which is the
whole point.

USAGE
  python tools/decision_policy/stage_a.py
  python tools/decision_policy/stage_a.py --with-short-arm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_examples import decide  # noqa: E402  (the label rule, reused to build disjoint preamble cases)

ACTIONS = ["CONTINUE", "CALL_TOOL", "ANSWER", "ASK_USER", "STOP"]
DEFAULT_CONFIG = Path(r"C:\tools\Strata\strata-unsloth-ud-iq4_xs.json")
LEGAL = "Legal actions: CONTINUE, CALL_TOOL, ANSWER, ASK_USER, STOP"

INTRO = """AGENT DECISION POLICY - SYSTEM ONE

ROLE
You are the decision module of an autonomous agent. Your only job is to choose the single next action for the
agent's orchestrator, given a compact description of the current state. You do not write prose. You do not explain
yourself. You do not run tools. You return exactly one action name.

CONTEXT
The agent runs in a loop: it observes state, chooses an action, the orchestrator executes that action and reports
the result, and the loop repeats. Your choice is made once per iteration. The orchestrator is responsible for
executing actions safely, for validating tool arguments, and for enforcing permissions. You are responsible only
for choosing the action well. The orchestrator never executes an action you did not choose.

ACTION VOCABULARY
You must choose exactly one of these five actions. No other output is valid.

  CONTINUE   Keep working toward the goal when there is no new input to act on. Use it while the agent is drafting,
             refining, planning, organising, reviewing, or waiting. CONTINUE never calls a tool and never ends the
             task. It is the default when nothing else applies.

  CALL_TOOL  Invoke a tool from the available_tools list. Use it when a tool result added new information that
             changes what should happen next, or when the previous tool call failed and the same or another tool can
             be retried. The orchestrator selects which tool and validates its arguments; you only choose the action.

  ANSWER     Produce the final answer to the goal. Use it only when the agent's confidence in the answer is high AND
             the evidence already gathered is sufficient to support it. ANSWER ends the task. Do not answer to avoid
             work that still has budget.

  ASK_USER   Ask the user a clarifying question. Use it when the goal is ambiguous, the constraints are unclear, or
             the correct interpretation cannot be determined from the state. ASK_USER pauses the task and hands
             control back to the user.

  STOP       Stop the task without answering. Use it when the step budget is exhausted, that is when
             steps_remaining is zero or less, regardless of how complete the work looks. STOP is a safety action.

DECISION RULES
Apply these rules in order. The first rule that applies decides the action. Do not skip ahead.

  1. If steps_remaining is zero or less, choose STOP. Never spend budget that does not exist.
  2. If user_intent_ambiguous is true, choose ASK_USER. Do not guess what the user wants.
  3. If answer_confidence is "high" AND evidence_sufficient is true, choose ANSWER. Answer only when you are
     confident and the evidence is complete.
  4. If available_tools is not empty AND (new_information_available is true OR last_tool_succeeded is false),
     choose CALL_TOOL. New information, or a failed call that can be retried, is a reason to act.
  5. Otherwise choose CONTINUE. The agent is still working and there is nothing new to act on.

SAFETY RULES
  - Never choose an action outside the five in the vocabulary.
  - Never name a tool. CALL_TOOL is the only tool-related action; the orchestrator picks the tool.
  - Never use STOP to avoid finishing work that still has budget.
  - Never use ANSWER to skip evidence that is still missing.
  - If the state is contradictory, choose the safer action: STOP rather than ANSWER, ASK_USER rather than CALL_TOOL.
  - The state is the only evidence you have. Do not assume facts that are not in it.

OUTPUT FORMAT
Return only the action name, in upper case, with no punctuation, no explanation, no quotes, and no trailing text.
Valid outputs are exactly:
  CONTINUE
  CALL_TOOL
  ANSWER
  ASK_USER
  STOP
"""

RULES_TAIL = """WORKED EXAMPLES
The examples below show the shape of the state and the action the rules produce. The state you are given is in the
same shape. The examples are illustrations of the rules, not a lookup table.

READING A STATE FIELD BY FIELD
  goal: what the agent is trying to achieve. Context only; it never selects the action by itself.
  last_action: what the agent did last. A tool name means the last step called a tool; CONTINUE or COMPOSE means
    the agent was working without calling one.
  new_information_available: whether the last tool result added something the agent did not know before. A fresh
    fact to act on points toward CALL_TOOL; silence points toward CONTINUE.
  answer_confidence and evidence_sufficient: the two halves of ANSWER. Confidence without evidence is not enough,
    and evidence without confidence is not enough. Both must hold.
  steps_remaining: the budget. Zero or less ends the task at once, whatever else is true.
  available_tools: what the agent may call. Empty means CALL_TOOL is impossible, so the choice is between
    CONTINUE, ANSWER, ASK_USER and STOP.
  last_tool_succeeded: a failed call that can be retried points toward CALL_TOOL.
  user_intent_ambiguous: an unclear request points toward ASK_USER before anything else is tried.

"""


def S(goal, last_action, new_info, confidence, steps, tools, tool_ok, sufficient, ambiguous):
    return {"goal": goal, "last_action": last_action, "new_information_available": bool(new_info),
            "answer_confidence": confidence, "steps_remaining": int(steps),
            "available_tools": list(tools), "last_tool_succeeded": bool(tool_ok),
            "evidence_sufficient": bool(sufficient), "user_intent_ambiguous": bool(ambiguous)}


# Disjoint from the 60 test examples: different goals and states, so the model is not handed the answers.
PREAMBLE_CASES = [
    ("Clarify the requirements", "READ_NOTES", False, "low", 3, ["SEARCH"], True, False, True, "ASK_USER"),
    ("Clarify the requirements", "SEARCH", True, "medium", 2, ["SEARCH"], True, False, True, "ASK_USER"),
    ("Fix the crash", "READ_LOG", True, "low", 4, ["READ_LOG", "SEARCH"], True, False, False, "CALL_TOOL"),
    ("Fix the crash", "READ_LOG", False, "low", 4, ["READ_LOG"], False, False, False, "CALL_TOOL"),
    ("Fix the crash", "RUN_TESTS", True, "high", 2, ["RUN_TESTS"], True, True, False, "ANSWER"),
    ("Compare the two designs", "READ_DOCUMENT", True, "medium", 3, ["READ_DOCUMENT"], True, False, False, "CALL_TOOL"),
    ("Compare the two designs", "READ_DOCUMENT", True, "high", 1, ["READ_DOCUMENT"], True, True, False, "ANSWER"),
    ("Update the configuration", "READ_FILE", False, "medium", 2, ["READ_FILE"], False, False, False, "CALL_TOOL"),
    ("Update the configuration", "READ_FILE", False, "medium", 2, ["READ_FILE"], False, False, True, "ASK_USER"),
    ("Write the summary", "COMPOSE", False, "medium", 2, [], True, False, False, "CONTINUE"),
    ("Write the summary", "COMPOSE", False, "high", 1, [], True, True, False, "ANSWER"),
    ("Write the summary", "COMPOSE", False, "medium", 0, [], True, False, False, "STOP"),
    ("Search for the error", "SEARCH", True, "low", 4, ["SEARCH"], True, False, False, "CALL_TOOL"),
    ("Search for the error", "SEARCH", False, "low", 3, ["SEARCH"], False, False, False, "CALL_TOOL"),
    ("Search for the error", "SEARCH", True, "high", 1, ["SEARCH"], True, True, False, "ANSWER"),
    ("Check the dependency", "FETCH", False, "medium", 2, ["FETCH"], False, False, False, "CALL_TOOL"),
    ("Check the dependency", "FETCH", True, "medium", 2, ["FETCH"], True, False, False, "CALL_TOOL"),
    ("Check the dependency", "FETCH", True, "high", 1, ["FETCH"], True, True, False, "ANSWER"),
    ("Verify the signature", "PROBE", True, "medium", 3, ["PROBE"], True, False, False, "CALL_TOOL"),
    ("Verify the signature", "PROBE", False, "medium", 2, ["PROBE"], False, False, False, "CALL_TOOL"),
    ("Verify the signature", "PROBE", True, "high", 1, ["PROBE"], True, True, False, "ANSWER"),
    ("Revise the draft", "COMPOSE", False, "low", 3, [], True, False, False, "CONTINUE"),
    ("Revise the draft", "COMPOSE", False, "medium", 2, [], True, False, False, "CONTINUE"),
    ("Revise the draft", "COMPOSE", False, "medium", 0, [], True, False, False, "STOP"),
    ("Revise the draft", "COMPOSE", False, "high", 1, [], True, True, False, "ANSWER"),
    ("Diagnose the warning", "READ_LOG", True, "medium", 2, ["READ_LOG"], True, False, False, "CALL_TOOL"),
    ("Diagnose the warning", "READ_LOG", True, "low", 2, ["READ_LOG"], True, False, True, "ASK_USER"),
    ("Diagnose the warning", "READ_LOG", True, "high", 1, ["READ_LOG"], True, True, False, "ANSWER"),
    ("Plan the migration", "CONTINUE", False, "low", 4, [], True, False, False, "CONTINUE"),
    ("Plan the migration", "CONTINUE", False, "medium", 3, [], True, False, False, "CONTINUE"),
    ("Plan the migration", "SCAN", True, "medium", 2, ["SCAN"], True, False, False, "CALL_TOOL"),
    ("Plan the migration", "SCAN", False, "medium", 2, ["SCAN"], False, False, False, "CALL_TOOL"),
    ("Plan the migration", "SCAN", True, "high", 1, ["SCAN"], True, True, False, "ANSWER"),
    # ---- extra edge cases: rule priority and empty-tool corners ----
    ("Archive the logs", "ARCHIVE", True, "medium", 2, ["ARCHIVE"], True, False, False, "CALL_TOOL"),
    ("Archive the logs", "ARCHIVE", False, "medium", 2, ["ARCHIVE"], True, False, True, "ASK_USER"),
    ("Summarize the thread", "READ_THREAD", True, "high", 2, ["READ_THREAD"], True, False, False, "CALL_TOOL"),
    ("Summarize the thread", "READ_THREAD", False, "high", 2, ["READ_THREAD"], True, False, False, "CONTINUE"),
    ("Summarize the thread", "READ_THREAD", False, "high", 1, [], True, True, False, "ANSWER"),
    ("Migrate the data", "MIGRATE", True, "low", 0, ["MIGRATE"], True, False, False, "STOP"),
    ("Migrate the data", "MIGRATE", True, "medium", 1, ["MIGRATE"], False, False, True, "ASK_USER"),
    ("Review the patch", "READ_PATCH", False, "medium", 2, [], True, False, False, "CONTINUE"),
    ("Review the patch", "READ_PATCH", True, "medium", 2, [], True, False, False, "CONTINUE"),
    ("Deploy the release", "DEPLOY", False, "low", 1, ["DEPLOY"], False, False, False, "CALL_TOOL"),
    ("Deploy the release", "DEPLOY", True, "high", 0, ["DEPLOY"], True, True, False, "STOP"),
    ("Triage the inbox", "SCAN", True, "low", 3, ["SCAN", "READ"], True, False, False, "CALL_TOOL"),
]


def build_preamble(compact: bool = False) -> str:
    if not compact:
        out = [INTRO, RULES_TAIL]
        for i, (goal, last, new, conf, steps, tools, ok, suff, amb, action) in enumerate(PREAMBLE_CASES, 1):
            st = S(goal, last, new, conf, steps, tools, ok, suff, amb)
            assert decide(st) == action, f"preamble case {i}: rule says {decide(st)}, stated {action}"
            out.append(f"Example {i}.\n")
            out.append("State: goal = " + repr(goal) + "; last_action = " + repr(last) + ";\n")
            out.append("        new_information_available = " + str(bool(new)).lower()
                       + "; answer_confidence = " + repr(conf) + "; steps_remaining = " + str(steps) + ";\n")
            out.append("        available_tools = " + repr(list(tools)) + "; last_tool_succeeded = " + str(bool(ok)).lower()
                       + ";\n")
            out.append("        evidence_sufficient = " + str(bool(suff)).lower()
                       + "; user_intent_ambiguous = " + str(bool(amb)).lower() + "\n")
            out.append("Action: " + action + "\n\n")
        return "".join(out)
    out = [INTRO, COMPACT_SYNTAX, RULES_TAIL]
    for i, (goal, last, new, conf, steps, tools, ok, suff, amb, action) in enumerate(PREAMBLE_CASES, 1):
        st = S(goal, last, new, conf, steps, tools, ok, suff, amb)
        assert decide(st) == action, f"preamble case {i}: rule says {decide(st)}, stated {action}"
        out.append(f"Example {i}.\n{compact_state(st)}\nAction: {action}\n\n")
    return "".join(out)


def state_text(state: dict) -> str:
    compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return f"State:\n{compact}\n\nNext action:"


COMPACT_KEYS = (("goal", "G"), ("last_action", "LA"), ("new_information_available", "NI"),
                ("answer_confidence", "AC"), ("steps_remaining", "S"), ("available_tools", "T"),
                ("last_tool_succeeded", "OK"), ("evidence_sufficient", "SUF"),
                ("user_intent_ambiguous", "AMB"))


def compact_value(v) -> str:
    if v is True:
        return "1"
    if v is False:
        return "0"
    if isinstance(v, list):
        return ",".join(v) if v else "-"
    if v == "high":
        return "h"
    if v == "medium":
        return "m"
    if v == "low":
        return "l"
    return str(v)


def compact_state(state: dict) -> str:
    parts = []
    for long_k, short_k in COMPACT_KEYS:
        parts.append(f"{short_k}:{compact_value(state[long_k])}")
    return "|".join(parts)


def state_text_compact(state: dict) -> str:
    return compact_state(state) + "|Next:"


COMPACT_SYNTAX = """STATE SYNTAX (compact form)
The state is one line of short fields separated by `|`. The fields, in order, are:

  G   the goal (free text, may contain spaces but never `|`)
  LA  the last action (a tool name, or CONTINUE/COMPOSE)
  NI  1 when a tool result added new information, else 0
  AC  answer confidence: h (high), m (medium) or l (low)
  S   steps remaining (a whole number)
  T   the usable tools, comma-separated, or `-` when there are none
  OK  1 when the previous tool call succeeded, else 0
  SUF 1 when the evidence is sufficient to answer, else 0
  AMB 1 when the request is ambiguous, else 0

Example: `G:Find the docs|LA:SEARCH|NI:1|AC:m|S:3|T:SEARCH,READ_DOCUMENT|OK:1|SUF:0|AMB:0|Next:`
means the verbose state with goal "Find the docs", last_action "SEARCH", new_information_available true,
answer_confidence "medium", steps_remaining 3, available_tools ["SEARCH", "READ_DOCUMENT"],
last_tool_succeeded true, evidence_sufficient false, user_intent_ambiguous false.
The decision rules apply to these fields exactly as written above.

"""


def normalize_action(text: str):
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.strip().strip(".").strip().upper())
    return t if t in ACTIONS else None


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "port": int(cfg.get("port", 8080)), "model": cfg.get("model_name"),
            "api_key": cfg.get("api_key", "")}


def call(url, key, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.perf_counter()
    ttft = None
    usage = None
    content = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            if ttft is None:
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        ttft = (time.perf_counter() - t0) * 1000.0
                        break
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    content.append(piece)
    return (time.perf_counter() - t0) * 1000.0, ttft, "".join(content), usage


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_arm(url, key, model, examples, preamble, args, compact=False):
    render = state_text_compact if compact else state_text

    def payload(state):
        return {"model": model,
                "messages": ([{"role": "system", "content": preamble}] if preamble is not None else [])
                           + [{"role": "user", "content": render(state) if preamble is not None
                                else _short_user(state)}],
                "max_tokens": args.max_tokens, "temperature": 0.0, "reasoning_effort": "none",
                "stream": True, "stream_options": {"include_usage": True}}
    # one warm request with a dummy state (not in the test set) to force the root checkpoint,
    # so every measured example is a NEW state sharing the same preamble
    dummy = {"goal": "Warm the cache", "last_action": "CONTINUE",
             "new_information_available": False, "answer_confidence": "low", "steps_remaining": 5,
             "available_tools": [], "last_tool_succeeded": True,
             "evidence_sufficient": False, "user_intent_ambiguous": False}
    t0 = time.perf_counter()
    _, _, _, warm_usage = call(url, key, payload(dummy), args.timeout)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    warm_prompt_tokens = (warm_usage or {}).get("prompt_tokens")
    lat, ttfts, rows = [], [], []
    per_row, prompt_toks, comp_toks = [], [], []
    for r in range(args.runs):
        for ex in examples:
            try:
                ms, ttft, content, usage = call(url, key, payload(ex["state"]), args.timeout)
            except Exception as e:                                # noqa: BLE001
                print(f"[stage_a] error {ex['id']} run {r}: {e}", file=sys.stderr)
                rows.append((ex["target_action"], None))
                lat.append(None)
                per_row.append({"id": ex["id"], "run": r, "latency_ms": None})
                continue
            lat.append(ms)
            if ttft is not None:
                ttfts.append(ttft)
            if usage:
                prompt_toks.append(int(usage.get("prompt_tokens", 0)))
                comp_toks.append(int(usage.get("completion_tokens", 0)))
            rows.append((ex["target_action"], normalize_action(content)))
            per_row.append({"id": ex["id"], "run": r, "latency_ms": round(ms, 2),
                            "ttft_ms": round(ttft, 2) if ttft is not None else None})
    ok = [x for x in lat if x is not None]
    n = len(rows)
    correct = sum(1 for t, p in rows if t == p)
    matrix = {a: {b: 0 for b in ACTIONS} for a in ACTIONS}
    for t, p in rows:
        matrix[t][p if p in ACTIONS else "INVALID"] = matrix[t].get(p if p in ACTIONS else "INVALID", 0) + 1
    new_lat = [r["latency_ms"] for r in per_row if r["run"] == 0 and r["latency_ms"] is not None]
    rep_lat = [r["latency_ms"] for r in per_row if r["run"] > 0 and r["latency_ms"] is not None]
    return {"n": n, "warm_first_ms": round(warm_ms, 2), "warm_prompt_tokens": warm_prompt_tokens,
            "latency_ms": {"median": round(statistics.median(ok), 2) if ok else None,
                           "p95": round(percentile(ok, 95), 2) if ok else None,
                           "min": round(min(ok), 2) if ok else None, "max": round(max(ok), 2) if ok else None},
            "new_state_latency_ms": {"median": round(statistics.median(new_lat), 2) if new_lat else None,
                                     "p95": round(percentile(new_lat, 95), 2) if new_lat else None,
                                     "n": len(new_lat)},
            "repeat_latency_ms": {"median": round(statistics.median(rep_lat), 2) if rep_lat else None,
                                  "p95": round(percentile(rep_lat, 95), 2) if rep_lat else None,
                                  "n": len(rep_lat)},
            "ttft_ms": {"median": round(statistics.median(ttfts), 2) if ttfts else None},
            "prompt_tokens_median": round(statistics.median(prompt_toks), 2) if prompt_toks else None,
            "completion_tokens_median": round(statistics.median(comp_toks), 2) if comp_toks else None,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "invalid": sum(1 for _, p in rows if p is None), "confusion": matrix, "rows": rows,
            "per_row": per_row}


def _short_user(state):
    compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return (f"You are an agent decision policy. Choose exactly one next action. Reply with the action name only.\n\n"
            f"State:\n{compact}\n\n{LEGAL}\n\nNext action:")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--examples", type=Path, default=HERE / "examples.jsonl")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--warmup", type=int, default=1, help="warm requests before timing (root checkpoint)")
    ap.add_argument("--runs", type=int, default=2, help="timed requests per example")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--with-short-arm", action="store_true", help="also run the Phase 1 prompt for contrast")
    ap.add_argument("--with-compact-arm", action="store_true",
                    help="also run the compact state encoding (matching compact preamble)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    port, model, key = cfg["port"], cfg["model"], cfg["api_key"] or os.environ.get("STRATA_API_KEY", "")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    if not key or not model:
        print("need --config with api_key and model_name, or STRATA_API_KEY", file=sys.stderr)
        return 2

    examples = [json.loads(ln) for ln in args.examples.read_text(encoding="utf-8").splitlines() if ln.strip()]
    preamble = build_preamble()
    print(f"[stage_a] {len(examples)} examples, preamble {len(preamble)} chars, model={model}", flush=True)

    report = {"schema_version": 1, "task": "stage_a_fixed_preamble", "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "model": model, "endpoint": url, "preamble_chars": len(preamble),
              "label_provenance": "synthetic, rule-based; preamble worked examples are disjoint from the test set"}

    # preamble arm, `runs` passes so each state is new the first time and repeated after
    arm = run_arm(url, key, model, examples, preamble, args)
    report["preamble_arm"] = arm
    print(f"[stage_a] preamble arm: warm={arm['warm_first_ms']}ms median={arm['latency_ms']['median']}ms "
          f"p95={arm['latency_ms']['p95']}ms acc={arm['accuracy']} invalid={arm['invalid']} "
          f"prompt_tok={arm['prompt_tokens_median']}", flush=True)

    if args.with_compact_arm:
        cpre = build_preamble(compact=True)
        print(f"[stage_a] compact preamble {len(cpre)} chars", flush=True)
        compact = run_arm(url, key, model, examples, cpre, args, compact=True)
        report["compact_arm"] = compact
        print(f"[stage_a] compact arm:  warm={compact['warm_first_ms']}ms median={compact['latency_ms']['median']}ms "
              f"p95={compact['latency_ms']['p95']}ms acc={compact['accuracy']} invalid={compact['invalid']} "
              f"prompt_tok={compact['prompt_tokens_median']}", flush=True)

    if args.with_short_arm:
        short = run_arm(url, key, model, examples, None, args)
        report["short_arm"] = short
        print(f"[stage_a] short arm:    warm={short['warm_first_ms']}ms median={short['latency_ms']['median']}ms "
              f"p95={short['latency_ms']['p95']}ms acc={short['accuracy']} invalid={short['invalid']}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stage_a_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    md = ["# Stage A - fixed long preamble with the state at the end", "",
          f"- measured: {report['measured_at']}", f"- model: `{model}`", f"- endpoint: `{url}`",
          f"- preamble: {len(preamble)} chars", f"- labels: {report['label_provenance']}", "",
          "## Preamble arm", "", "| metric | value |", "| --- | ---: |",
          f"| warm (first, forces the root checkpoint) | {arm['warm_first_ms']} ms |",
          f"| warm prompt tokens | {arm['warm_prompt_tokens']} |",
          f"| prompt tokens median | {arm['prompt_tokens_median']} |",
          f"| new-state (run 0) median | {arm['new_state_latency_ms']['median']} ms |",
          f"| new-state (run 0) p95 | {arm['new_state_latency_ms']['p95']} ms |",
          f"| repeat (run 1+) median | {arm['repeat_latency_ms']['median']} ms |",
          f"| repeat (run 1+) p95 | {arm['repeat_latency_ms']['p95']} ms |",
          f"| ttft median | {arm['ttft_ms']['median']} ms |",
          f"| accuracy | {arm['accuracy']} |", f"| invalid | {arm['invalid']} |", ""]
    if "short_arm" in report:
        s = report["short_arm"]
        md += ["## Short arm (Phase 1 prompt, for contrast)", "", "| metric | value |", "| --- | ---: |",
               f"| warm median | {s['warm_first_ms']} ms |",
               f"| median | {s['latency_ms']['median']} ms |", f"| p95 | {s['latency_ms']['p95']} ms |",
               f"| accuracy | {s['accuracy']} |", f"| invalid | {s['invalid']} |", ""]
    if "compact_arm" in report:
        c = report["compact_arm"]
        md += ["## Compact arm (compact preamble + compact states)", "", "| metric | value |", "| --- | ---: |",
               f"| warm (first, forces the root checkpoint) | {c['warm_first_ms']} ms |",
               f"| warm prompt tokens | {c['warm_prompt_tokens']} |",
               f"| prompt tokens median | {c['prompt_tokens_median']} |",
               f"| new-state (run 0) median | {c['new_state_latency_ms']['median']} ms |",
               f"| new-state (run 0) p95 | {c['new_state_latency_ms']['p95']} ms |",
               f"| repeat (run 1+) median | {c['repeat_latency_ms']['median']} ms |",
               f"| repeat (run 1+) p95 | {c['repeat_latency_ms']['p95']} ms |",
               f"| ttft median | {c['ttft_ms']['median']} ms |",
               f"| accuracy | {c['accuracy']} |", f"| invalid | {c['invalid']} |", ""]
    (args.out_dir / "stage_a_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[stage_a] wrote {args.out_dir / 'stage_a_report.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
