"""serve/test_logits_first_token.py - the first-token logits path, against mocks (no GPU).

    python -m unittest serve.test_logits_first_token -v
"""
from __future__ import annotations

import math
import queue
import re
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import logits_first_token as lft      # noqa: E402
from serve import systemone as sysone            # noqa: E402
from serve.frontend import ChatTemplate          # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402
from serve.structured import StructuredOutputError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
PREFIX = "q:"          # the answer line's prefix, ending at the colon and NOT at a space


class Tok:
    """A word-ish stand-in for the model's tokenizer: whitespace and underscores split
    words, each word is one id.  `"rule_1"` and `"rule_2"` share their first token, the
    way the model's BPE gives them both the token " rule"."""

    def __init__(self):
        self.ids = {}

    def _word(self, w):
        return self.ids.setdefault(w, 100 + len(self.ids))

    def encode(self, s, parse_special=False):
        out = []
        for w in s.split():
            for i, part in enumerate(re.split(r"(_)", w)):
                if part:
                    out.append(self._word(part))
        return out

    def decode(self, ids, errors="replace"):
        return " ".join(f"<{i}>" for i in ids)


def svc_with(script="yes\n", cycle=False):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve/chat_template.jinja"), )
    svc.engine.cycle = cycle
    svc.systemone_enabled = True
    svc.model = "test-model"
    return svc


def req(**kw):
    base = {"state": {"goal": "x"},
            "questions": {"q": {"type": "choice", "instructions": "Pick",
                                "criteria": {"a": None, "b": None}}}}
    base.update(kw)
    return base


class RenderAnswerPrompt(unittest.TestCase):
    def setUp(self):
        _, self.qs = sysone.validate_request(req())

    def test_no_prefix_in_the_user_prompt(self):
        # the answer line's prefix prefills the ASSISTANT's turn, so it is not here
        p = sysone.render_answer_prompt("STATE", "q", self.qs[0][1])
        self.assertFalse(p.endswith("q: "))
        self.assertIn("- a", p)
        self.assertIn("- b", p)
        self.assertIn('start your answer line with exactly "q:"', p)

    def test_render_prompt_unchanged(self):
        p = sysone.render_prompt("STATE", self.qs)
        self.assertTrue(p.endswith("Reply with exactly one line per question and nothing else."))
        self.assertIn("Valid line starts: q.", p)


class FirstTokenIds(unittest.TestCase):
    def setUp(self):
        self.tok = Tok()

    def test_token_follows_the_prefix(self):
        ids = lft.first_token_ids(["a", "b"], self.tok, PREFIX)
        self.assertEqual(ids["a"], self.tok.encode(lft.answer_text(PREFIX, "a"))[1])
        self.assertEqual(ids["b"], self.tok.encode(lft.answer_text(PREFIX, "b"))[1])
        self.assertNotEqual(ids["a"], ids["b"])

    def test_answer_text_puts_the_space_with_the_answer(self):
        # the space belongs to the answer's first token, not to the prefix
        self.assertEqual(lft.answer_text("q:", "a"), "q: a")
        self.assertEqual(lft.answer_text("q: ", "a"), "q: a")
        self.assertEqual(lft.answer_text("", "a"), "a")

    def test_shared_first_token(self):
        ids = lft.first_token_ids(["rule_1", "rule_2"], self.tok, PREFIX)
        self.assertEqual(ids["rule_1"], ids["rule_2"])      # the collision case

    def test_without_a_prefix(self):
        ids = lft.first_token_ids(["a", "b"], self.tok)
        self.assertEqual(ids["a"], self.tok.encode("a")[0])

    def test_option_that_adds_no_token_raises(self):
        class Empty:
            def encode(self, s, parse_special=False):
                return []
        with self.assertRaises(ValueError):
            lft.first_token_ids(["a"], Empty(), PREFIX)


class Readings(unittest.TestCase):
    def setUp(self):
        self.tok = Tok()
        _, self.qs = sysone.validate_request(req())

    def _logits(self, values, prefix=PREFIX):
        out = {}
        for opt, v in values.items():
            out[lft.first_token_ids([opt], self.tok, prefix)[opt]] = v
        return out

    def test_unique_tokens(self):
        readings = lft.first_token_probs(self.qs, self._logits({"a": 2.0, "b": 1.0}), self.tok, PREFIX)
        self.assertIsNotNone(readings)
        resp = lft.reading_response(readings["q"])
        self.assertEqual(resp["choice"], "a")
        p = resp["probabilities"]
        self.assertAlmostEqual(p["a"] + p["b"], 1.0)          # the closed reading adds up to 1
        self.assertAlmostEqual(resp["confidence"], p["a"])

    def test_the_two_readings_differ_by_the_mass_outside(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        r = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)["q"]
        # closed: renormalized over the options
        self.assertAlmostEqual(sum(r["closed"].values()), 1.0)
        # open: the model's own probability, so the options add up to less than 1
        self.assertAlmostEqual(r["open"]["a"], math.exp(2.0 - 3.0))
        self.assertLess(sum(r["open"].values()), 1.0)
        self.assertAlmostEqual(r["outside"], 1.0 - sum(r["open"].values()))
        # the answer is the same under both: a renormalization does not reorder them
        self.assertEqual(max(r["closed"], key=r["closed"].get), max(r["open"], key=r["open"].get))

    def test_stop_mass(self):
        # the end of the turn BELOW the top option: the model chose the option but is
        # close to stopping instead - the case the two readings disagree about
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        logits = self._logits({"a": 2.0, "b": 1.0})
        logits[stop_id] = 1.5
        r = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)["q"]
        self.assertAlmostEqual(r["stop"], math.exp(1.5 - 3.0))
        self.assertGreater(r["stop"], 0.0)

    def test_falls_back_when_the_model_choses_something_else(self):
        # the model's top token is not one of the options (it wants to write a digit,
        # or to end the turn): the option distribution is about a choice it is not
        # making, so the decode path answers it instead
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        logits = self._logits({"a": 2.0, "b": 1.0})
        logits[stop_id] = 5.0
        self.assertIsNone(lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0))

    def test_collision_falls_back(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": {"rule_1": None, "rule_2": None}}}})
        tok = self.tok
        shared = lft.first_token_ids(["rule_1"], tok, PREFIX)["rule_1"]
        self.assertIsNone(lft.first_token_probs(qs, {shared: 2.0}, tok, PREFIX))

    def test_missing_falls_back(self):
        self.assertIsNone(lft.first_token_probs(self.qs, self._logits({"a": 2.0}), self.tok, PREFIX))

    def test_noul(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "noul", "instructions": "Is it?"}}})
        logits = self._logits({"yes": 3.0, "no": 0.0})
        readings = lft.first_token_probs(qs, logits, self.tok, PREFIX)
        self.assertGreater(lft.reading_response(readings["q"])["noul"], 0.9)

    def test_score(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "score", "instructions": "How?", "criteria": ["lo", "hi"]}}})
        logits = self._logits({"lo": 0.0, "hi": 2.0})
        readings = lft.first_token_probs(qs, logits, self.tok, PREFIX)
        resp = lft.reading_response(readings["q"])
        self.assertEqual(resp["score"], 1.0)
        self.assertEqual(resp["legend"], {"0": "lo", "1": "hi"})


class DebugReport(unittest.TestCase):
    def setUp(self):
        self.tok = Tok()
        _, self.qs = sysone.validate_request(req())

    def test_top_k_is_the_engine_maximum(self):
        self.assertEqual(lft.LOGITS_TOP_K, 1024)

    def test_report_marks_missing_and_lists_top(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "topk.log")
            a = lft.first_token_ids(["a"], self.tok, PREFIX)["a"]
            b = lft.first_token_ids(["b"], self.tok, PREFIX)["b"]
            lft.debug_report(self.qs, {a: 2.0}, self.tok, path, PREFIX)
            text = Path(path).read_text(encoding="utf-8")
        self.assertIn("'a'=id%d" % a, text)
        self.assertIn("'b'=id%d MISSING" % b, text)   # out of the window, not unlikely
        self.assertIn("top 1", text)


class DecodeAnswer(unittest.TestCase):
    """The text the model wrote at the answer position."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.end = self.tok.encode("<|im_end|>", parse_special=True)

    def test_the_text_is_decoded(self):
        self.assertEqual(sysone.decode_answer(self.tok.encode("no"), self.tok), "no")

    def test_the_end_of_turn_token_is_cut(self):
        self.assertEqual(sysone.decode_answer(self.tok.encode("no") + self.end, self.tok), "no")

    def test_the_cut_uses_the_stop_ids(self):
        text = sysone.decode_answer(self.tok.encode("no") + self.end, self.tok, set(self.end))
        self.assertEqual(text, "no")

    def test_nothing_is_empty(self):
        self.assertEqual(sysone.decode_answer([], self.tok), "")


class ParseAnswerText(unittest.TestCase):
    """The reply of a single-question request: the answer alone, or the whole line."""

    def setUp(self):
        _, self.choice = sysone.validate_request(req())
        _, self.noul = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "noul", "instructions": "Is it?"}}})

    def test_bare_answer(self):
        self.assertEqual(sysone.parse_answer_text("a\n", self.choice), {"q": "a"})

    def test_whole_line(self):
        self.assertEqual(sysone.parse_answer_text("q: a\n", self.choice), {"q": "a"})

    def test_noul_forms(self):
        self.assertEqual(sysone.parse_answer_text("yes\n", self.noul), {"q": True})
        self.assertEqual(sysone.parse_answer_text("no\n", self.noul), {"q": False})
        self.assertEqual(sysone.parse_answer_text("0.7\n", self.noul), {"q": 0.7})

    def test_bad_answers_are_errors(self):
        for bad in ("", "maybe\n", "zzz\n", "0.7\n1.2\n"):
            with self.assertRaises(StructuredOutputError, msg=bad):
                sysone.parse_answer_text(bad, self.choice)

    def test_two_bare_lines_are_an_error(self):
        with self.assertRaises(StructuredOutputError):
            sysone.parse_answer_text("a\nb\n", self.choice)


class ServiceLogitsPath(unittest.TestCase):
    def test_single_question_uses_logits(self):
        svc = svc_with()
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0}
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        self.assertEqual(body["usage"]["completion_tokens"], 0)
        self.assertAlmostEqual(body["answers"]["q"]["probabilities"]["a"] +
                               body["answers"]["q"]["probabilities"]["b"], 1.0)

    def test_missing_token_falls_back_to_the_generated_tokens(self):
        # the fast path cannot score the options, but THIS request generated the answer
        svc = svc_with("q: b\n")
        svc.engine.logits_topk = {1000: 2.0}                       # 1001 missing
        svc.engine.logits_tokens = svc.tok.encode("q: b")
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "b")
        self.assertFalse(body["calibrated"])

    def test_the_end_of_turn_token_is_not_part_of_the_answer(self):
        # the model wrote the answer and then ended its turn.  The end token must not
        # be decoded into the answer: "rule_2<|im_end|>" is not one of the options, so
        # the request failed with a 502 (measured, on the collision case).
        svc = svc_with("q: b\n")
        svc.engine.logits_topk = {1000: 2.0}                       # 1001 missing
        svc.engine.logits_tokens = svc.tok.encode("q: b") + [next(iter(svc.stop_ids))]
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "b")

    def test_multi_question_uses_decode(self):
        svc = svc_with("q1: a\nq2: b\n")
        code, body = svc.systemone(req(questions={
            "q1": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}},
            "q2": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}}),
            threading.Event())
        self.assertEqual(code, 200)
        self.assertFalse(body["calibrated"])

    def test_engine_error_falls_back(self):
        svc = svc_with("q: a\n")

        def boom(ids, k, max_new, cancel):
            raise RuntimeError("boom")
        svc.engine.logits = boom
        code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "a")

    def test_diagnostics_are_opt_in(self):
        svc = svc_with()
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0}
        svc.engine.logits_logz = 3.0
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertNotIn("diagnostics", body)          # off by default: the shape is the reference one
        svc.systemone_diagnostics = True
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        diag = body["diagnostics"]["q"]
        self.assertAlmostEqual(diag["open_confidence"], math.exp(2.0 - 3.0))
        self.assertAlmostEqual(diag["closed_confidence"], body["answers"]["q"]["confidence"])
        self.assertGreaterEqual(diag["outside"], 0.0)

    def test_empty_reply_falls_back_to_the_classic_prompt(self):
        # the model wanted to end the turn at the answer position: the classic prompt,
        # where it writes the whole line, answers it instead of erroring
        svc = svc_with("q: a\n")
        svc.engine.logits_topk = {1000: 2.0}                       # 1001 missing
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        self.assertFalse(body["calibrated"])


class EngineWireFormat(unittest.TestCase):
    """The LOGITS command's wire format, against the real engine wrapper.

    The mock engine in the tests above answers from a script, so it never sees the
    line the server writes to the engine.  A protocol slip here is invisible to them
    and costs the whole fast path: the engine answers "ERR expected: LOGITS ..." and
    the request never runs (measured: every probe case fell back to the decode path
    and the L/Z rows never came).  So the format is pinned here, on both sides:
    the server's line and the engine's parser.
    """

    def _engine(self, answers):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc = mock.Mock()
        eng.proc.poll.return_value = None
        eng.proc.stdin = mock.Mock()
        eng.ended = False
        eng.lines = queue.Queue()
        eng.last = {}
        for line in answers:
            eng.lines.put(line)
        return eng

    def test_the_command_carries_k_max_new_and_the_ids(self):
        eng = self._engine(["Z 3.0", "L 1000 2.0", "T 7", "DONE 1 2 3 4 stop"])
        topk, logz, tokens = eng.logits([11, 22, 33], 64, 32, threading.Event())
        written = eng.proc.stdin.write.call_args[0][0]
        self.assertEqual(written, "LOGITS 64 32 11,22,33\n")
        self.assertEqual(topk, {1000: 2.0})
        self.assertEqual(logz, 3.0)
        self.assertEqual(tokens, [7])

    def test_the_engine_accepts_the_line_the_server_writes(self):
        # the engine's own parser (src/program/generate.cpp): LOGITS <k> <max_new> <ids>
        eng = self._engine(["L 1000 2.0", "DONE 0 2 3 4 stop"])
        eng.logits([11, 22], 1024, 32, threading.Event())
        written = eng.proc.stdin.write.call_args[0][0]
        m = re.fullmatch(r"LOGITS (\d+) (\d+) ([\d,]+)\n", written)
        self.assertIsNotNone(m, f"the server wrote {written!r}, which the engine cannot parse")
        k, max_new, ids = int(m[1]), int(m[2]), [int(t) for t in m[3].split(",")]
        self.assertTrue(1 <= k <= 1024)
        self.assertTrue(1 <= max_new <= 4096)
        self.assertEqual(ids, [11, 22])


if __name__ == "__main__":
    unittest.main()
