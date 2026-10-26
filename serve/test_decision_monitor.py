"""serve/test_decision_monitor.py - the decision path in the server's own telemetry.

    python -m unittest serve.test_decision_monitor -v

The decision endpoint (POST /v1/systemone) does its own engine turn, so it does not
go through Service.run.  Without the bookkeeping below it was invisible: the Monitor
said the model was idle while a decision ran, and the decision never appeared in the
request list.  These tests cover what the Monitor reads:

  * the live state (the model is busy, and it is a decision);
  * the request list (one row for the request, with what it answered);
  * the totals (a decision is a request that used the engine);
  * the timings on the answer.

No GPU: the engine is the mock.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import logits_first_token as lft   # noqa: E402
from serve import systemone as sysone         # noqa: E402
from serve.frontend import ChatTemplate       # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.structured import StructuredOutputError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096

TOKENS = {"a": 1000, "b": 1001, "yes": 1002, "no": 1003, "low": 1004, "high": 1005}


def svc_with(script: str | list[str] = "q: yes\n") -> Service:
    """A service whose mock engine replays `script`."""
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.systemone_enabled = True
    svc.model = "test-model"
    return svc


def req(**kw):
    base = {"state": {"goal": "x"},
            "questions": {"q": {"type": "choice", "instructions": "Which one?",
                                "criteria": {"a": None, "b": None}}}}
    base.update(kw)
    return base


def fast_path(svc: Service) -> None:
    """Make the mock engine answer the fast path: the options have their own first
    tokens and the model writes the first one."""
    svc.engine.logits_topk = {TOKENS["a"]: 3.0, TOKENS["b"]: 1.0}
    svc.engine.logits_logz = 4.0


def first_tokens(opts, tok, prefix=""):
    """The option's own first token, the way the engine would tokenize it."""
    return {opt: TOKENS[opt] for opt in opts}


def settled(svc: Service, timeout: float = 5.0) -> dict:
    """The API request monitor's row, once the server has closed it: the response is
    written before the record is settled, so a client can see it a moment earlier."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = svc.api_requests[-1]
        if record.get("state") in ("completed", "error"):
            return record
        time.sleep(0.01)
    return svc.api_requests[-1]


class DecisionRequestList(unittest.TestCase):
    """The request list: one row for the decision, with what it answered."""

    def test_a_decision_is_one_row_with_its_answers(self):
        svc = svc_with()
        fast_path(svc)
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(len(svc.history), 1)
        row = svc.history[0]
        self.assertEqual(row["kind"], "decision")
        self.assertEqual(row["finish"], "stop")
        self.assertEqual(row["questions"], 1)
        self.assertEqual(row["answers"], {"q": "a"})          # what it picked
        self.assertEqual(row["path"], "fast")                 # the model's own probability
        self.assertTrue(row["calibrated"])
        self.assertEqual(row["output_tokens"], body["usage"]["completion_tokens"])
        self.assertGreater(row["prompt_tokens"], 0)

    def test_the_picks_of_every_type(self):
        svc = svc_with()
        # one row per question: the merged path asks the engine once per question
        svc.engine.logits_rows = [({TOKENS["yes"]: 3.0, TOKENS["no"]: 1.0}, 4.0),
                                  ({TOKENS["a"]: 3.0, TOKENS["b"]: 1.0}, 4.0),
                                  ({TOKENS["low"]: 3.0, TOKENS["high"]: 1.0}, 4.0)]
        questions = {"yes_no": {"type": "noul", "instructions": "Is it?"},
                     "which": {"type": "choice", "instructions": "Which one?",
                               "criteria": {"a": None, "b": None}},
                     "how_bad": {"type": "score", "instructions": "How bad?",
                                 "criteria": ["low", "high"]}}
        with mock.patch.object(lft, "first_token_ids", first_tokens):
            code, body = svc.systemone(req(questions=questions), threading.Event())
        self.assertEqual(code, 200)
        row = svc.history[0]
        self.assertEqual(row["questions"], 3)
        self.assertEqual(sorted(row["answers"]), ["how_bad", "which", "yes_no"])
        self.assertEqual(row["answers"]["which"], "a")
        self.assertEqual(row["answers"]["yes_no"], "yes (0.88)")

    def test_the_decode_path_is_marked_as_a_decode(self):
        # no logits: the model writes the whole line and the answer comes from its text
        svc = svc_with("q: a\n")
        code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        row = svc.history[0]
        self.assertEqual(row["kind"], "decision")
        self.assertEqual(row["path"], "decode")
        self.assertEqual(row["answers"], {"q": "a"})
        self.assertFalse(row["calibrated"])

    def test_a_decision_that_could_not_be_answered_is_an_error_row(self):
        svc = svc_with("not an answer\n")
        with self.assertRaises(StructuredOutputError):
            svc.systemone(req(), threading.Event())
        self.assertEqual(len(svc.history), 1)
        row = svc.history[0]
        self.assertEqual(row["kind"], "decision")
        self.assertEqual(row["finish"], "error")
        self.assertIn("error", row)

    def test_a_bad_request_is_not_a_request(self):
        svc = svc_with()
        with self.assertRaises(ValueError):
            svc.systemone({"state": "x", "questions": {}}, threading.Event())
        self.assertEqual(len(svc.history), 0)


class DecisionTotals(unittest.TestCase):
    """The totals: a decision is a request that used the engine."""

    def test_the_totals_count_the_decision(self):
        svc = svc_with()
        fast_path(svc)
        before = dict(svc.totals)
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS):
            svc.systemone(req(), threading.Event())
        self.assertEqual(svc.totals["requests"], before["requests"] + 1)
        self.assertEqual(svc.totals["decisions"], before["decisions"] + 1)
        self.assertGreater(svc.totals["prompt_tokens"], before["prompt_tokens"])

    def test_the_chat_path_is_not_counted_as_a_decision(self):
        svc = svc_with()
        svc.totals["decisions"] = 0
        svc.engine.script = svc.tok.encode("Hello.", parse_special=True)
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
        chunks = __import__("serve.server", fromlist=["openai_chunks"]).openai_chunks(
            svc, chat, svc.tok.encode("hi", parse_special=True), False, None, 16, threading.Event())
        list(chunks)
        self.assertEqual(svc.totals["decisions"], 0)
        self.assertEqual(svc.totals["requests"], 1)
        self.assertNotIn("kind", svc.history[0])          # a chat row, not a decision


class DecisionLiveState(unittest.TestCase):
    """The live state: the Monitor says the model is working on a decision."""

    def test_the_model_is_busy_while_the_decision_runs(self):
        svc = svc_with()
        fast_path(svc)
        seen = {}

        def logits(ids, k, max_new, cancel, park=True):
            seen.update(svc.status)                  # the status while the engine works
            return svc.engine.logits_topk, svc.engine.logits_logz, []

        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS), \
                mock.patch.object(svc.engine, "logits", logits):
            svc.systemone(req(), threading.Event())
        self.assertTrue(seen.get("busy"), "the model must be busy while a decision runs")
        self.assertTrue(seen.get("decision"), "the Monitor must see it is a decision")
        self.assertEqual(seen.get("phase"), "deciding")
        # and the status is clean afterwards
        self.assertFalse(svc.status.get("busy"))
        self.assertNotIn("decision", svc.status)
        self.assertEqual(svc.metrics()["live"]["decision"], False)

    def test_the_monitor_is_told_it_is_a_decision(self):
        svc = svc_with()
        fast_path(svc)
        live = {}

        def logits(ids, k, max_new, cancel, park=True):
            live.update(svc.metrics()["live"])
            return svc.engine.logits_topk, svc.engine.logits_logz, []

        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS), \
                mock.patch.object(svc.engine, "logits", logits):
            svc.systemone(req(), threading.Event())
        self.assertTrue(live["decision"])
        self.assertEqual(live["state"], "reading")     # the prefill is the whole cost

    def test_a_waiting_decision_is_counted_as_queued(self):
        svc = svc_with()
        fast_path(svc)
        svc.fifo.acquire()                            # the engine's turn is taken
        try:
            done = threading.Event()

            def ask():
                with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS):
                    svc.systemone(req(), threading.Event())
                done.set()

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            deadline = time.monotonic() + 5
            while svc.status.get("queued", 0) < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(svc.status.get("queued"), 1, "a waiting decision is queued")
        finally:
            svc.fifo.release()
        worker.join(5)
        self.assertTrue(done.is_set())
        self.assertEqual(svc.status.get("queued"), 0)


class DecisionTimings(unittest.TestCase):
    """The answer carries the timing, so a client can see what the decision cost."""

    def test_the_answer_carries_the_timings(self):
        svc = svc_with()
        fast_path(svc)
        svc.engine.last = {"generated": 2, "prompt_tokens": 120, "prompt_ms": 1500.0,
                           "decode_ms": 30.0, "finish": "stop", "reused": 0, "prompt_read": 120}
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        timings = body["timings"]
        self.assertEqual(timings["prompt_ms"], 1500.0)
        self.assertEqual(timings["predicted_n"], 0)         # the fast path reads the answer
        self.assertIsNotNone(timings["prompt_per_second"])
        self.assertIsNotNone(svc.last_timings)              # and /v1/status shows it

    def test_the_row_carries_the_prompt_rate(self):
        svc = svc_with()
        fast_path(svc)
        svc.engine.last = {"generated": 2, "prompt_tokens": 120, "prompt_ms": 1500.0,
                           "decode_ms": 30.0, "finish": "stop", "reused": 0, "prompt_read": 120}
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS):
            svc.systemone(req(), threading.Event())
        row = svc.history[0]
        self.assertEqual(row["prompt_ms"], 1500.0)
        self.assertEqual(row["decode_ms"], 30.0)
        self.assertGreater(row["prompt_tokens"], 0)


class DecisionApiMonitor(unittest.TestCase):
    """The API request monitor (/api-monitor): the decision is a request there too."""

    def setUp(self):
        self.svc = svc_with()
        self.svc.api_monitor = True
        fast_path(self.svc)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_the_decision_is_in_the_request_monitor(self):
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": TOKENS):
            request = urllib.request.Request(
                self.base + "/v1/systemone", data=json.dumps(req()).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(response.status, 200)
        records = self.svc.api_requests
        self.assertEqual(len(records), 1)
        record = settled(self.svc)
        self.assertEqual(record["path"], "/v1/systemone")
        self.assertEqual(record["state"], "completed")
        self.assertIn("a", record["output"])            # what it answered
        self.assertIsNotNone(record["first_token_s"])
        self.assertGreaterEqual(record["queue_s"], 0)   # the wait for the engine's turn


class TheMonitorPage(unittest.TestCase):
    """The Monitor tab shows the decisions: the page reads /metrics, so the fields
    the table renders have to be there."""

    def test_the_page_knows_about_decisions(self):
        web = Path(__file__).parent / "web"
        js = (web / "app.js").read_text(encoding="utf-8")
        self.assertIn("live.decision", js)              # the state card and the pill
        self.assertIn('r.kind === "decision"', js)     # the request table's row
        self.assertIn('"Decision"', js)                # its badge
        self.assertIn("t.decisions", js)                # the totals line


if __name__ == "__main__":
    unittest.main()
