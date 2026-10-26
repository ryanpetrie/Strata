"""The decision path's turn on the engine.

Three things this branch's code got wrong, all of them about what happens to the
engine's lines and its turn:

  * `Engine.logits` (the decision path's engine call) wrote STOP and returned, but
    did NOT drain the engine's line queue.  The chat path's `Engine.generate` has
    drained to DONE since forever, and its own comment says why: the engine's lines
    for THIS request must not reach the NEXT request, which reads the same queue.
    Without the drain a stopped request left its tokens and its DONE behind, and the
    next request read them as its own answer - another question's tokens, another
    question's distribution.

  * the decision path read the tokens of a request the engine STOPPED (a client that
    went away, a STOP that arrived for another request) as if they were the model's
    answer.  A stopped request has a partial answer, or none.

  * the trie walk took the engine's turn AGAIN, so another request could take the
    turn between a question's base request and its walk.  The walk appends a node's
    tokens to the prompt it was given, so it needs that prompt still in the engine's
    cache - and another request's turn takes the cache away.

No GPU: the engine client is faked, the model is the mock.
"""
import os
import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import logits_first_token as lft   # noqa: E402
from serve import server                      # noqa: E402
from serve.frontend import ChatTemplate       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
PREFIX = "q:"


class FakeProc:
    """The engine process, as the client needs it: a stdin it writes to and a poll()."""

    def __init__(self):
        self.stdin = self
        self.written = []
        self.killed = False

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def fake_engine(lines=()):
    """A StrataEngine with no process behind it but a line queue and a fake stdin."""
    eng = object.__new__(server.StrataEngine)
    eng.lines = queue.Queue()
    for l in lines:
        eng.lines.put(l)
    eng.proc = FakeProc()
    eng.can_stop = True
    eng.silence_s = 5.0
    eng.ended = False
    eng.progress = None
    eng.progress_ms = None
    eng.prefill_tok_s_mean = None
    eng.reused = 0
    eng.last = {}
    return eng


class DrainAfterStop(unittest.TestCase):
    """A stopped request must not leave its lines in the queue the next request reads."""

    def test_a_stopped_request_leaves_the_queue_empty(self):
        eng = fake_engine()
        # the engine keeps working for a moment and then reports the stopped request.
        # The lines land AFTER the client gave up (the read gives up after 0.5 s of
        # silence), so they are the stopped request's leftovers.
        def late():
            time.sleep(0.9)
            eng.lines.put("T 5")
            eng.lines.put("DONE 1 3 1 1 stop")
        t = threading.Thread(target=late, daemon=True)
        t.start()

        cancel = threading.Event()
        cancel.set()                      # the client went away
        with self.assertRaises(server.QueueCancelled):
            eng.logits([1, 2, 3], 8, 4, cancel)

        self.assertIn("STOP\n", eng.proc.written)   # the engine was stopped
        t.join(timeout=5)                # the leftovers are in the queue by now
        self.assertTrue(eng.lines.empty(),
                        f"{eng.lines.qsize()} line(s) left for the next request: "
                        f"{list(eng.lines.queue)}")

    def test_a_drain_that_gives_up_ends_the_engine(self):
        # the engine never answers the stopped request: the sides lost step, so the
        # engine is ended rather than left with its lines in the queue.  The
        # cancellation that is already on its way out is not masked by that.
        eng = fake_engine()
        eng.silence_s = 0.5
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(server.QueueCancelled):
            eng.logits([1, 2, 3], 8, 4, cancel)
        self.assertTrue(eng.proc.killed, "the engine was not ended")
        self.assertTrue(eng.ended)

    def test_a_drain_that_gives_up_ends_the_engine_without_a_cancel(self):
        # the same, when nothing else is propagating: the engine is ended and the
        # loss of step is reported.  The call runs on its own thread with a join
        # timeout, so a version WITHOUT the watchdog fails the assertion instead of
        # hanging the suite.
        eng = fake_engine()
        eng.silence_s = 0.5
        outcome = []

        def call():
            try:
                eng.logits([1, 2, 3], 8, 4, threading.Event())   # never cancelled
                outcome.append("returned")
            except Exception as e:                                # noqa: BLE001
                outcome.append(e)
        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(timeout=10)
        self.assertFalse(t.is_alive(), "the request waited for an engine that said nothing")
        self.assertTrue(eng.proc.killed, "the engine was not ended")
        self.assertIsInstance(outcome[0], server.EngineSilent)

    def test_an_answer_that_arrives_is_the_answer(self):
        eng = fake_engine(["DONE 1 3 1 1 stop"])
        cancel = threading.Event()
        cancel.set()
        out, logz, tokens = eng.logits([1, 2, 3], 8, 4, cancel)
        self.assertEqual(tokens, [])
        self.assertTrue(eng.lines.empty())


class TheAnswerMustBeMine(unittest.TestCase):
    """The engine's DONE carries the prompt's own token count, so the server can ask
    whether the answer it just read belongs to the request it sent.  It does not, the
    sides lost step: the answer is not used, and the engine is ended so the next
    request starts on a fresh pipe."""

    def test_an_answer_for_another_request_is_not_used(self):
        # the engine answered with 330 prompt tokens; 3 were sent
        eng = fake_engine(["L 5 1.0", "T 7", "DONE 2 330 10.0 5.0 stop"])
        with self.assertRaises(server.EngineSilent):
            eng.logits([1, 2, 3], 8, 4, threading.Event())
        self.assertTrue(eng.proc.killed, "the engine was not ended")
        self.assertTrue(eng.ended)

    def test_the_answer_for_this_request_is_used(self):
        eng = fake_engine(["L 5 1.0", "T 7", "DONE 2 3 10.0 5.0 stop"])
        out, logz, tokens = eng.logits([1, 2, 3], 8, 4, threading.Event())
        self.assertEqual(out, {5: 1.0})
        self.assertEqual(tokens, [7])
        self.assertFalse(eng.proc.killed)

    def test_the_engine_s_last_is_not_stale(self):
        # a request that gave up leaves no finish behind for the next one to read
        eng = fake_engine()
        eng.last = {"finish": "cancel"}
        cancel = threading.Event()
        cancel.set()

        def late():
            time.sleep(0.9)
            eng.lines.put("DONE 1 3 1 1 stop")
        t = threading.Thread(target=late, daemon=True)
        t.start()
        with self.assertRaises(server.QueueCancelled):
            eng.logits([1, 2, 3], 8, 4, cancel)
        t.join(timeout=5)
        self.assertNotIn("finish", eng.last)


class CountingLock:
    """The FIFO, counting how many times it is taken."""

    def __init__(self, inner):
        self.inner = inner
        self.count = 0

    def acquire(self, *a, **k):
        self.count += 1
        return self.inner.acquire(*a, **k)

    def release(self):
        return self.inner.release()

    def locked(self):
        return self.inner.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class DecisionTurn(unittest.TestCase):
    """The decision path's use of the engine's turn and of the engine's answer."""

    def svc_with(self, script="q: yes\n"):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, script, max_context=CTX), tok,
                             ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.systemone_enabled = True
        return svc

    def req(self, **questions):
        base = {"state": {"goal": "x"},
                "questions": questions or {"q": {"type": "choice", "instructions": "Pick",
                                                  "criteria": {"creative_writing": None,
                                                               "data_analysis": None}}}}
        return base

    def test_a_stopped_request_does_not_answer_from_its_tokens(self):
        # the engine stopped the request: its tokens are a partial answer, not the
        # model's answer, so the endpoint must not read them as one - it falls back
        # to the decode path, which answers with the script's option
        svc = self.svc_with("q: data_analysis\n")
        svc.engine.logits_tokens = svc.tok.encode("creative_writing")
        svc.engine.last = {"finish": "cancel"}
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertNotEqual(body["answers"]["q"].get("choice"), "creative_writing")
        self.assertEqual(body["answers"]["q"].get("choice"), "data_analysis")

    def test_a_finished_request_answers_from_its_tokens(self):
        # the control: the engine finished the turn, so the tokens ARE the answer
        svc = self.svc_with("q: data_analysis\n")
        svc.engine.logits_tokens = svc.tok.encode("creative_writing")
        svc.engine.last = {"finish": "stop"}
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"].get("choice"), "creative_writing")

    def test_the_walk_shares_the_base_request_s_turn(self):
        # "x1" and "x2" share their first token, so the trie walk runs: it must not
        # take the engine's turn again, or another request can slip in between the
        # base request and the walk
        svc = self.svc_with()
        tok = svc.tok
        ids = lft.option_token_ids(["x1", "x2"], tok, PREFIX)
        a, b = ids["x1"], ids["x2"]
        shared = 0
        while shared < len(a) and shared < len(b) and a[shared] == b[shared]:
            shared += 1
        self.assertGreater(shared, 0)
        rows = [({a[i]: 3.0}, 3.0) for i in range(shared)]
        rows.append(({a[shared]: 2.0, b[shared]: 1.0}, 3.0))
        svc.engine.logits_rows = rows
        svc.fifo = CountingLock(svc.fifo)
        code, body = svc.systemone(self.req(q={"type": "choice", "instructions": "Pick",
                                               "criteria": {"x1": None, "x2": None}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(svc.fifo.count, 1,
                         f"the question took the engine's turn {svc.fifo.count} times")


if __name__ == "__main__":
    unittest.main()
