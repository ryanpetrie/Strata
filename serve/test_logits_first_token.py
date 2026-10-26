"""serve/test_logits_first_token.py - the first-token logits path, against mocks (no GPU).

    python -m unittest serve.test_logits_first_token -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import logits_first_token as lft      # noqa: E402
from serve import systemone as sysone            # noqa: E402
from serve.frontend import ChatTemplate          # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


class Tok:
    """One id per distinct string, so `" a"` and `" b"` differ (the byte tokenizer's space is shared)."""

    def __init__(self):
        self.ids = {}

    def encode(self, s, parse_special=False):
        return [self.ids.setdefault(s, 100 + len(self.ids))]


def svc_with(script="q: yes\n"):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.systemone_enabled = True
    svc.model = "test-model"
    return svc


def req(**kw):
    base = {"state": {"goal": "x"},
            "questions": {"q": {"type": "choice", "instructions": "Pick",
                                "criteria": {"a": None, "b": None}}}}
    base.update(kw)
    return base


class RenderLogitsPrompt(unittest.TestCase):
    def setUp(self):
        _, self.qs = sysone.validate_request(req())

    def test_ends_at_answer(self):
        p = sysone.render_logits_prompt("STATE", "q", self.qs[0][1])
        self.assertTrue(p.endswith("q: "))
        self.assertIn("- a", p)
        self.assertIn("- b", p)
        self.assertIn('start your answer line with exactly "q:"', p)

    def test_render_prompt_unchanged(self):
        p = sysone.render_prompt("STATE", self.qs)
        self.assertTrue(p.endswith("Reply with exactly one line per question and nothing else."))
        self.assertIn("Valid line starts: q.", p)


class FirstTokenProbs(unittest.TestCase):
    def setUp(self):
        self.tok = Tok()
        _, self.qs = sysone.validate_request(req())

    def _logits(self, values):
        return {self.tok.encode(" " + opt)[0]: v for opt, v in values.items()}

    def test_unique_tokens(self):
        probs = lft.try_first_token_probs(self.qs, self._logits({"a": 2.0, "b": 1.0}), self.tok)
        self.assertIsNotNone(probs)
        self.assertEqual(probs["q"]["choice"], "a")
        p = probs["q"]["probabilities"]
        self.assertAlmostEqual(p["a"] + p["b"], 1.0)
        self.assertAlmostEqual(probs["q"]["confidence"], p["a"])

    def test_collision_falls_back(self):
        class Same:
            def encode(self, s, parse_special=False):
                return [7]
        self.assertIsNone(lft.try_first_token_probs(self.qs, {7: 2.0}, Same()))

    def test_missing_falls_back(self):
        self.assertIsNone(lft.try_first_token_probs(self.qs, self._logits({"a": 2.0}), self.tok))

    def test_noul(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "noul", "instructions": "Is it?"}}})
        logits = {self.tok.encode(" yes")[0]: 3.0, self.tok.encode(" no")[0]: 0.0}
        probs = lft.try_first_token_probs(qs, logits, self.tok)
        self.assertGreater(probs["q"]["noul"], 0.9)

    def test_score(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "score", "instructions": "How?", "criteria": ["lo", "hi"]}}})
        logits = {self.tok.encode(" lo")[0]: 0.0, self.tok.encode(" hi")[0]: 2.0}
        probs = lft.try_first_token_probs(qs, logits, self.tok)
        self.assertEqual(probs["q"]["score"], 1.0)
        self.assertEqual(probs["q"]["legend"], {"0": "lo", "1": "hi"})


class ServiceLogitsPath(unittest.TestCase):
    def test_single_question_uses_logits(self):
        svc = svc_with()
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0}
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok: {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        self.assertEqual(body["usage"]["completion_tokens"], 0)
        self.assertAlmostEqual(body["answers"]["q"]["probabilities"]["a"] +
                               body["answers"]["q"]["probabilities"]["b"], 1.0)

    def test_missing_token_falls_back(self):
        svc = svc_with("q: b\n")
        svc.engine.logits_topk = {1000: 2.0}                       # 1001 missing
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok: {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "b")       # the decode path
        self.assertFalse(body["calibrated"])

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

        def boom(ids, k, cancel):
            raise RuntimeError("boom")
        svc.engine.logits = boom
        code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "a")


if __name__ == "__main__":
    unittest.main()
