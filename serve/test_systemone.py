"""serve/test_systemone.py - POST /v1/systemone, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_systemone -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import systemone as sysone  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402
from serve.structured import StructuredOutputError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


def svc_with(script: str) -> Service:
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.systemone_enabled = True
    svc.model = "test-model"
    return svc


def req(**kw):
    base = {"state": {"goal": "x"},
            "questions": {"q": {"type": "noul", "instructions": "Is it x?"}}}
    base.update(kw)
    return base


class Validate(unittest.TestCase):
    def test_ok_all_types(self):
        state, questions = sysone.validate_request({
            "state": {"plan": "enterprise", "days": 41},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Urgency?",
                           "criteria": {"true": "outage", "false": "questions"}},
                "dept": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "charges", "technical": None}},
                "mood": {"type": "score", "instructions": "How happy?",
                         "criteria": ["low", "high"]}}})
        self.assertEqual([n for n, _ in questions], ["urgent", "dept", "mood"])
        self.assertIn('"days":41', state)          # canonical serialization

    def test_state_forms(self):
        s, _ = sysone.validate_request(req(state="hello"))
        self.assertEqual(s, "hello")
        s, _ = sysone.validate_request(req(state=[1, 2]))
        self.assertEqual(s, "[1,2]")

    def test_missing_and_empty(self):
        for bad in ({}, {"state": "x"}, {"questions": {}}, {"state": "", "questions": {"q": 1}},
                    {"state": {}, "questions": {"q": {"type": "noul", "instructions": "i"}}},
                    {"state": [], "questions": {"q": {"type": "noul", "instructions": "i"}}},
                    {"state": 5, "questions": {"q": {"type": "noul", "instructions": "i"}}}):
            with self.assertRaises(ValueError, msg=bad):
                sysone.validate_request(bad)

    def test_question_rules(self):
        good_q = {"type": "noul", "instructions": "i"}
        bad_qs = [
            {"type": "maybe", "instructions": "i"},          # unknown type
            {"type": "noul"},                                # no instructions
            {"type": "noul", "instructions": "  "},          # blank instructions
            {"type": "noul", "instructions": "i", "criteria": {"yes": "y"}},   # bad noul keys
            {"type": "choice", "instructions": "i"},         # missing criteria
            {"type": "choice", "instructions": "i", "criteria": {}},          # empty criteria
            {"type": "score", "instructions": "i", "criteria": ["only"]},     # one level
            {"type": "score", "instructions": "i", "criteria": ["a"] * 11},   # too many
            {"type": "score", "instructions": "i", "criteria": ["a", "a"]},    # dup levels
            {"type": "score", "instructions": "i", "criteria": {"a": 1}},      # bad desc type
            {"type": "score", "instructions": "i", "criteria": {"a: b": "d"}}, # colon in level name
        ]
        for q in bad_qs:
            with self.assertRaises(ValueError, msg=q):
                sysone.validate_request(req(questions={"q": q}))
        too_many = {f"q{i}": dict(good_q) for i in range(11)}
        with self.assertRaises(ValueError):
            sysone.validate_request(req(questions=too_many))
        with self.assertRaises(ValueError):                 # name with a colon breaks `name: answer`
            sysone.validate_request(req(questions={"a:b": dict(good_q)}))

    def test_score_dict_criteria(self):
        state, questions = sysone.validate_request({
            "state": "x", "questions": {
                "mood": {"type": "score", "instructions": "How happy?",
                         "criteria": {"unhappy": "bad day", "happy": None}}}})
        self.assertEqual(questions[0][1]["criteria"],
                         {"levels": ["unhappy", "happy"], "descriptions": {"unhappy": "bad day"}})
        prompt = sysone.render_prompt(state, questions)
        self.assertIn('- unhappy ("bad day")', prompt)
        self.assertIn("- happy", prompt)
        answers = sysone.parse_answers("mood: happy\n", questions)
        self.assertEqual(answers, {"mood": 1})
        body = sysone.build_response(questions, answers, "m", {})
        self.assertEqual(body["answers"]["mood"]["legend"], {"0": "unhappy", "1": "happy"})

    def test_limits(self):
        with self.assertRaises(ValueError):
            sysone.validate_request(req(state="x" * (sysone.MAX_STATE_CHARS + 1)))


class RenderParse(unittest.TestCase):
    RAW = {"urgent": {"type": "noul", "instructions": "Urgency?",
                      "criteria": {"true": "outage", "false": "chit-chat"}},
           "dept": {"type": "choice", "instructions": "Which team?",
                    "criteria": {"billing": "charges", "technical": None}},
           "mood": {"type": "score", "instructions": "How happy?",
                    "criteria": ["unhappy", "neutral", "happy"]}}

    @classmethod
    def setUpClass(cls):
        # the server always validates before rendering/parsing, so the fixtures are normalized too
        _, cls.QUESTIONS = sysone.validate_request({"state": "x", "questions": cls.RAW})

    def test_render_has_everything(self):
        prompt = sysone.render_prompt("STATE", self.QUESTIONS)
        for needle in ("STATE", "[urgent | noul]", "Urgency?", 'yes means "outage"',
                       "[dept | choice]", '- billing ("charges")', "- technical",
                       "[mood | score]", "- unhappy", "- neutral", "- happy",
                       'start your answer line with exactly "urgent:"',
                       "Valid line starts: urgent, dept, mood."):
            self.assertIn(needle, prompt)
        self.assertNotIn("urgent (noul):", prompt)   # no colon-shaped headers: the reply's
        self.assertNotIn("options:", prompt)         # `name: answer` is the only colon shape

    def test_round_trip(self):
        text = "urgent: yes\ndept: technical\nmood: happy\n"
        answers = sysone.parse_answers(text, self.QUESTIONS)
        self.assertEqual(answers, {"urgent": True, "dept": "technical", "mood": 2})

    def test_noul_number(self):
        text = "urgent: 0.7\ndept: technical\nmood: happy\n"
        self.assertEqual(sysone.parse_answers(text, self.QUESTIONS),
                         {"urgent": 0.7, "dept": "technical", "mood": 2})
        self.assertEqual(sysone.parse_answers("urgent: 1\ndept: technical\nmood: happy\n",
                                              self.QUESTIONS)["urgent"], 1.0)
        self.assertEqual(sysone.parse_answers("urgent: 0\ndept: technical\nmood: happy\n",
                                              self.QUESTIONS)["urgent"], 0.0)
        for bad_value in ("1.5", "-0.1", "nan", "inf", "probably"):
            bad = f"urgent: {bad_value}\ndept: technical\nmood: happy\n"
            with self.assertRaises(StructuredOutputError, msg=bad):
                sysone.parse_answers(bad, self.QUESTIONS)
        body = sysone.build_response(self.QUESTIONS, {"urgent": 0.7, "dept": "technical", "mood": 2},
                                     "m", {})
        self.assertEqual(body["answers"]["urgent"], {"type": "noul", "noul": 0.7})

    def test_bad_replies(self):
        cases = [
            "urgent: yes\ndept: technical\n",                                  # missing one
            "urgent: yes\nurgent: no\ndept: technical\nmood: happy\n",          # duplicate
            "urgent: yes\ndept: technical\nmood: happy\nextra: stuff\n",        # unknown line
            "urgent: maybe\ndept: technical\nmood: happy\n",                    # bad noul
            "urgent: yes\ndept: legal\nmood: happy\n",                          # bad choice
            "urgent: yes\ndept: technical\nmood: ecstatic\n",                   # bad level
            "just some prose\n",                                               # no answers at all
        ]
        for text in cases:
            with self.assertRaises(StructuredOutputError, msg=text):
                sysone.parse_answers(text, self.QUESTIONS)
        # the control holds: the good reply parses
        self.assertEqual(sysone.parse_answers("urgent: yes\ndept: technical\nmood: happy\n",
                                              self.QUESTIONS)["mood"], 2)


class Response(unittest.TestCase):
    def test_shape(self):
        body = sysone.build_response(RenderParse.QUESTIONS,
                                     {"urgent": True, "dept": "technical", "mood": 2},
                                     "m", {"prompt_tokens": 10, "completion_tokens": 3,
                                           "total_tokens": 13,
                                           "prompt_tokens_details": {"cached_tokens": 4}})
        self.assertEqual(body["object"], "systemone.classification")
        self.assertTrue(body["id"].startswith("systemone-"))
        self.assertFalse(body["calibrated"])
        self.assertEqual(body["answers"]["urgent"], {"type": "noul", "noul": 1.0})
        dept = body["answers"]["dept"]
        self.assertEqual(dept["choice"], "technical")
        self.assertEqual(dept["probabilities"], {"billing": 0.0, "technical": 1.0})
        self.assertEqual(dept["confidence"], 1.0)
        mood = body["answers"]["mood"]
        self.assertEqual(mood["score"], 2.0)
        self.assertEqual(mood["legend"], {"0": "unhappy", "1": "neutral", "2": "happy"})
        self.assertEqual(body["usage"]["prompt_tokens_details"], {"cached_tokens": 4})


class ServiceRoute(unittest.TestCase):
    def test_disabled_is_404(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "q: yes", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 404)

    def test_noul_yes(self):
        svc = svc_with("q: yes\n")
        code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"], {"q": {"type": "noul", "noul": 1.0}})
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["calibrated"])
        self.assertGreater(body["usage"]["prompt_tokens"], 0)

    def test_choice_and_score(self):
        svc = svc_with("dept: technical\nmood: happy\n")
        code, body = svc.systemone(req(questions={
            "dept": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": None, "technical": None}},
            "mood": {"type": "score", "instructions": "How happy?",
                     "criteria": ["unhappy", "happy"]}}), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["dept"]["choice"], "technical")
        self.assertEqual(body["answers"]["mood"]["score"], 1.0)

    def test_garbage_is_502_not_an_answer(self):
        svc = svc_with("probably yeah\n")
        with self.assertRaises(StructuredOutputError):
            svc.systemone(req(), threading.Event())

    def test_bad_request_is_400(self):
        svc = svc_with("q: yes\n")
        with self.assertRaises(ValueError):
            svc.systemone({"state": "x", "questions": {}}, threading.Event())


if __name__ == "__main__":
    unittest.main()
