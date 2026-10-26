"""tools/decision_policy/logits_check.py - probe the first-token logits path (milestone 1).

Sends a few small requests to POST /v1/systemone and reports, per case, whether the
request took the LOGITS fast path (the body says "calibrated": true and carries a real
probability) or fell back to the decode path ("calibrated": false, 1.0/0.0).

    python tools/decision_policy/logits_check.py
    python tools/decision_policy/logits_check.py --config <launch config>

The cases are chosen so that one run is decisive:

  noul         yes/no are single tokens with distinct first tokens -> the fast path MUST fire
  choice       two options with distinct first tokens              -> the fast path MUST fire
  score        two levels with distinct first tokens               -> the fast path MUST fire
  collide      two options that share their first token (" rule_1"/" rule_2")
               -> the fast path MUST fire: the trie walk reads the row that
                  separates them (measured: closed 0.668, open 0.641)
  prefix       two options of different token lengths (" rule_1"/" rule_1 bis")
               -> the fast path MUST fire: the trie walk reads the row where the
                  shorter option ends, and the end-of-turn probability there is what
                  separates them
  partial      three options where only two share a first token ("No change"/"One file"/
               "One service") -> the fast path MUST fire: the trie walk reads the row
                  where the two that share are told apart
  multi        two questions in one request                        -> the fast path MUST NOT fire
               (the answer position is per question, so one question per request)

If `noul` comes back "calibrated": false, the fast path is not reaching the logits at all:
the most likely cause is the tokenization pairing (the answer's first token), which the
`collide` and `multi` controls then confirm. The API key is read from the launch config
and never printed.
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

CASES = [
    ("noul", {"state": {"plan": "enterprise", "days": 41},
              "questions": {"urgent": {"type": "noul", "instructions": "Is this an outage?"}}},
     True),
    ("choice", {"state": {"plan": "enterprise", "days": 41},
                "questions": {"dept": {"type": "choice", "instructions": "Which team owns it?",
                                       "criteria": {"billing": None, "technical": None}}}},
     True),
    ("score", {"state": {"plan": "enterprise", "days": 41},
               "questions": {"mood": {"type": "score", "instructions": "How happy?",
                                      "criteria": ["unhappy", "happy"]}}},
     True),
    ("collide", {"state": {"plan": "enterprise", "days": 41},
                 "questions": {"rule": {"type": "choice", "instructions": "Which rule applies?",
                                        "criteria": {"rule_1": None, "rule_2": None}}}},
     True),
    ("prefix", {"state": {"plan": "enterprise", "days": 41},
                "questions": {"rule": {"type": "choice", "instructions": "Which rule applies?",
                                       "criteria": {"rule_1": None, "rule_1 bis": None}}}},
     True),
    ("partial", {"state": {"plan": "enterprise", "days": 41},
                 "questions": {"who": {"type": "choice", "instructions": "Who owns it?",
                                       "criteria": {"No change": None, "One file": None,
                                                    "One service": None}}}},
     True),
    ("multi", {"state": {"plan": "enterprise", "days": 41},
               "questions": {"a": {"type": "noul", "instructions": "Is it x?"},
                             "b": {"type": "noul", "instructions": "Is it y?"}}},
     False),
]


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"host": cfg.get("host", "127.0.0.1"), "port": int(cfg.get("port", 8080)),
            "model": cfg.get("model_name"), "api_key": cfg.get("api_key", "")}


def request(url: str, key: str, payload, timeout: float, method: str = "POST"):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def post(url: str, key: str, payload: dict, timeout: float):
    return request(url, key, payload, timeout)


def get(url: str, key: str, timeout: float):
    return request(url, key, None, timeout, method="GET")


def summarize(body: dict) -> str:
    """One line per answer: the type, the pick and the top probability - and, when the
    server sends them, the two readings of that probability side by side."""
    parts = []
    for name, ans in (body.get("answers") or {}).items():
        if ans.get("type") == "noul":
            parts.append(f"{name}=noul {ans.get('noul')}")
        elif ans.get("type") == "choice":
            top = max((ans.get("probabilities") or {}).values(), default=0.0)
            parts.append(f"{name}=choice {ans.get('choice')!r} p={top:.3f} "
                         f"conf={ans.get('confidence')}")
        elif ans.get("type") == "score":
            top = max((ans.get("probabilities") or {}).values(), default=0.0)
            parts.append(f"{name}=score {ans.get('score')} p={top:.3f} conf={ans.get('confidence')}")
        else:
            parts.append(f"{name}=? {ans}")
    for name, d in (body.get("diagnostics") or {}).items():
        parts.append(f"{name}[closed={d.get('closed_confidence', 0):.3f} "
                     f"open={d.get('open_confidence', 0):.3f} "
                     f"stop={d.get('stop', 0):.3f} outside={d.get('outside', 0):.3f}]")
    return " | ".join(parts) if parts else "(no answers)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="launch config with port/model/api_key")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None, help="defaults to the config's api_key, then STRATA_API_KEY")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    key = args.api_key or cfg["api_key"] or os.environ.get("STRATA_API_KEY", "")
    host = args.host or cfg["host"]
    port = args.port or cfg["port"]
    if host in ("0.0.0.0", "::", ""):        # a bind address is not a connect target
        host = "127.0.0.1"
    base = f"http://{host}:{port}"
    print(f"endpoint {base}  model {cfg['model']}  api_key {'present' if key else 'ABSENT'}")

    # the engine build the server is running, and the dialects it serves
    try:
        _, raw = get(base + "/props", key, 20.0)
        props = json.loads(raw)
        print(f"engine build_info: {props.get('build_info') or props.get('version') or '?'}")
    except Exception as e:                                   # the props endpoint is optional
        print(f"engine props: unavailable ({e})")
    try:
        _, raw = get(base + "/v1/status", key, 20.0)
        status = json.loads(raw)
        dialects = status.get("dialects") or []
        print(f"dialects: {'/v1/systemone' in dialects}")
    except Exception as e:
        print(f"status: unavailable ({e})")

    print()
    ok = True
    for name, payload, want_fast in CASES:
        t0 = time.perf_counter()
        try:
            status, raw = post(base + "/v1/systemone", key, payload, args.timeout)
            ms = (time.perf_counter() - t0) * 1000.0
            body = json.loads(raw)
        except urllib.error.HTTPError as e:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"{name:8s} HTTP {e.code} after {ms:7.1f} ms  {e.read()[:200]!r}")
            ok = False
            continue
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"{name:8s} ERROR after {ms:7.1f} ms  {e}")
            ok = False
            continue
        fast = bool(body.get("calibrated"))
        good = fast == want_fast
        ok = ok and good
        print(f"{name:8s} {'OK ' if good else 'BAD'} {status} {ms:7.1f} ms  "
              f"fast_path={str(fast):5s} (want {want_fast})  {summarize(body)}")

    # a normal chat request must still work
    try:
        t0 = time.perf_counter()
        status, raw = post(base + "/v1/chat/completions", key,
                            {"model": cfg["model"], "messages": [{"role": "user", "content": "Say ok."}],
                             "max_tokens": 64, "temperature": 0.0}, args.timeout)
        ms = (time.perf_counter() - t0) * 1000.0
        body = json.loads(raw)
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content")
        if not isinstance(text, str):
            # the content can legitimately be null (thinking-only, a tool call, or an empty
            # completion): show the body so the case is diagnosed rather than crashing the run
            print(f"{'chat':8s} BAD {status} {ms:7.1f} ms  content={text!r} "
                  f"finish={choice.get('finish_reason')!r}  body={raw[:300]!r}")
            ok = False
        else:
            print(f"{'chat':8s} OK  {status} {ms:7.1f} ms  finish={choice.get('finish_reason')!r}  "
                  f"{text.strip()[:60]!r}")
    except Exception as e:
        print(f"chat     BAD  {e}")
        ok = False

    print()
    print("all cases as expected" if ok else "SOME CASES DIFFERED - see the table above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
