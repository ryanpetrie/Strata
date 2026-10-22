"""serve/test_systemone.py - POST /v1/systemone, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_systemone -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import logits_first_token as lft   # noqa: E402
from serve import systemone as sysone  # noqa: E402
from serve import systemone_letters as sl  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, systemone_multi_mode,  # noqa: E402
                          systemone_readout_mode, systemone_temperatures)
from serve.structured import StructuredOutputError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


def svc_with(script: str | list[str] = "q: yes\n") -> Service:
    """A service whose mock engine replays `script`.  A list gives one reply per
    request (the last one repeats), which is what a request with several questions
    needs: every question is its own request."""
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.systemone_enabled = True
    svc.model = "test-model"
    return svc


def count_requests(svc: Service) -> list:
    """Wrap the mock engine's generate() so a test can count the engine's requests."""
    calls = []
    original = svc.engine.generate

    def counting(ids, max_new, sampling, cancel, embeddings=None):
        calls.append(list(ids))
        return original(ids, max_new, sampling, cancel, embeddings)
    svc.engine.generate = counting
    return calls


def count_logits(svc: Service) -> list:
    """Wrap the mock engine's logits() so a test can count the engine's requests of
    the fast path: the answer position's row, and the trie walk's extra rows."""
    calls = []
    original = svc.engine.logits

    def counting(ids, k, max_new, cancel):
        calls.append(list(ids))
        return original(ids, k, max_new, cancel)
    svc.engine.logits = counting
    return calls


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
        # several questions: each one is its own request, so the model answers one
        # question per reply
        svc = svc_with(["dept: technical\n", "mood: happy\n"])
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


class MultiQuestion(unittest.TestCase):
    """Several questions: each one is answered on its own (the single-question path)
    and the answers are merged into one response."""

    Q1 = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}
    Q2 = {"type": "choice", "instructions": "Pick", "criteria": {"x": None, "y": None}}

    def req(self, **kw):
        base = {"state": {"goal": "x"}, "questions": {"q1": self.Q1, "q2": self.Q2}}
        base.update(kw)
        return base

    def test_each_question_is_answered_on_its_own(self):
        # two questions, two engine turns, one answer each
        svc = svc_with(["q1: a\n", "q2: y\n"])
        calls = count_requests(svc)
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q1"]["choice"], "a")
        self.assertEqual(body["answers"]["q2"]["choice"], "y")
        self.assertEqual(list(body["answers"]), ["q1", "q2"])     # the request's order
        self.assertEqual(len(calls), 2)                          # one turn per question

    def test_every_answer_carries_the_model_s_own_probability(self):
        # the fast path: the model's own next-token distribution at each answer
        # position, so the answers are real probabilities and not the degenerate 1.0/0.0
        svc = svc_with()
        svc.engine.logits_rows = [({1000: 3.0, 1001: 1.0}, 4.0), ({1002: 3.0, 1003: 1.0}, 4.0)]
        ids = {"a": 1000, "b": 1001, "x": 1002, "y": 1003}
        with mock.patch.object(lft, "first_token_ids",
                               lambda opts, tok, prefix="": {o: ids[o] for o in opts}):
            code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q1"]["choice"], "a")
        self.assertEqual(body["answers"]["q2"]["choice"], "x")
        probs = body["answers"]["q1"]["probabilities"]
        self.assertAlmostEqual(probs["a"] + probs["b"], 1.0)      # the closed reading adds up to 1
        self.assertGreater(probs["b"], 0.0)                     # and it is not the degenerate one
        self.assertLess(body["answers"]["q1"]["confidence"], 1.0)   # the model's own probability

    def test_the_engine_s_turn_is_taken_once_per_question(self):
        # the fast path holds the engine's turn while it runs, and gives it back
        # before the next question: the turn is not held across the questions, or the
        # requests would serialise wrongly
        svc = svc_with(["q1: a\n", "q2: y\n"])
        calls = []
        original = svc.engine.logits

        def counting(ids, k, max_new, cancel):
            calls.append(svc.fifo.locked())
            return original(ids, k, max_new, cancel)
        svc.engine.logits = counting
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(len(calls), 2)              # one engine request per question
        self.assertTrue(all(calls))                  # and the turn was held while it ran
        self.assertFalse(svc.fifo.locked())          # and given back afterwards

    def test_one_failing_question_does_not_take_the_others_down(self):
        # the model did not answer q1: q1 is left out, q2 keeps its answer, and the
        # request is still a 200
        svc = svc_with(["not an answer\n", "q2: y\n"])
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertNotIn("q1", body["answers"])
        self.assertEqual(body["answers"]["q2"]["choice"], "y")
        # q2 came from the decode path, so the merged answer is not calibrated
        self.assertFalse(body["calibrated"])

    def test_no_answer_at_all_is_a_502(self):
        # nothing could be answered: there is no answer to return, and a one-question
        # request in the same situation is a 502 too
        svc = svc_with(["not an answer\n"])
        with self.assertRaises(StructuredOutputError):
            svc.systemone(self.req(), threading.Event())

    def test_the_single_question_path_is_unchanged(self):
        # the same question answered alone and answered as part of a pair gives the
        # same answer: A1 is the single-question path, once per question
        alone = svc_with(["q1: a\n"]).systemone(
            self.req(questions={"q1": self.Q1}), threading.Event())[1]
        pair = svc_with(["q1: a\n", "q2: y\n"]).systemone(self.req(), threading.Event())[1]
        self.assertEqual(pair["answers"]["q1"], alone["answers"]["q1"])
        self.assertFalse(alone["calibrated"])        # the decode path: the degenerate 1.0/0.0

    def test_the_usage_adds_up(self):
        # the merged request costs what the two single-question requests cost
        one = svc_with(["q1: a\n"]).systemone(
            self.req(questions={"q1": self.Q1}), threading.Event())[1]
        two = svc_with(["q2: y\n"]).systemone(
            self.req(questions={"q2": self.Q2}), threading.Event())[1]
        pair = svc_with(["q1: a\n", "q2: y\n"]).systemone(self.req(), threading.Event())[1]
        self.assertEqual(pair["usage"]["prompt_tokens"],
                         one["usage"]["prompt_tokens"] + two["usage"]["prompt_tokens"])
        self.assertEqual(pair["usage"]["total_tokens"],
                         pair["usage"]["prompt_tokens"] + pair["usage"]["completion_tokens"])

    def test_the_merged_response_has_the_reference_shape(self):
        svc = svc_with(["q1: a\n", "q2: y\n"])
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["object"], "systemone.classification")
        self.assertTrue(body["id"].startswith("systemone-"))
        self.assertEqual(body["model"], "test-model")
        self.assertIsInstance(body["created"], int)
        for name in ("q1", "q2"):
            self.assertEqual(set(body["answers"][name]),
                             {"type", "choice", "probabilities", "confidence"})
        self.assertEqual(set(body["usage"]),
                         {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"})

    def test_a_noul_in_the_merged_answer(self):
        svc = svc_with(["q1: yes\n", "q2: no\n"])
        code, body = svc.systemone(self.req(questions={
            "q1": {"type": "noul", "instructions": "Is it?"},
            "q2": {"type": "noul", "instructions": "Is it not?"}}), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q1"], {"type": "noul", "noul": 1.0})
        self.assertEqual(body["answers"]["q2"], {"type": "noul", "noul": 0.0})

    def test_the_default_is_separate(self):
        self.assertEqual(svc_with("q: yes\n").systemone_multi, "separate")

    def test_the_shared_prompt_is_still_there(self):
        # the old path, behind the config flag: one prompt answers every question, so
        # the prefill is shared and the answers carry no probability
        svc = svc_with(["q1: a\nq2: y\n"])
        svc.systemone_multi = "shared"
        calls = count_requests(svc)
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q1"]["choice"], "a")
        self.assertEqual(body["answers"]["q2"]["choice"], "y")
        self.assertFalse(body["calibrated"])
        self.assertEqual(len(calls), 1)                # one prompt for both questions

    def test_the_diagnostics_name_the_question_that_failed(self):
        svc = svc_with(["not an answer\n", "q2: y\n"])
        svc.systemone_diagnostics = True
        code, body = svc.systemone(self.req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertIn("reason", body["diagnostics"]["q1"])
        self.assertIn("q2", body["answers"])


class ConfigFlag(unittest.TestCase):
    """The config's "systemone": {"multi": ...}."""

    def test_the_default_is_separate(self):
        self.assertEqual(systemone_multi_mode({}), "separate")
        self.assertEqual(systemone_multi_mode({"systemone": {"enabled": True}}), "separate")
        self.assertEqual(systemone_multi_mode({"systemone": None}), "separate")

    def test_the_two_modes(self):
        self.assertEqual(systemone_multi_mode({"systemone": {"multi": "separate"}}), "separate")
        self.assertEqual(systemone_multi_mode({"systemone": {"multi": "shared"}}), "shared")

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            systemone_multi_mode({"systemone": {"multi": "batch"}})


class TemperatureConfig(unittest.TestCase):
    """The config's "systemone": {"temperature": {...}}: one temperature per question
    type.  1.0 means "no temperature", and that is the default - the path before the
    feature reported the same numbers."""

    def test_the_default_is_no_temperature(self):
        self.assertEqual(systemone_temperatures({}), {"choice": 1.0, "noul": 1.0, "score": 1.0})
        self.assertEqual(systemone_temperatures({"systemone": {"enabled": True}}),
                         {"choice": 1.0, "noul": 1.0, "score": 1.0})
        self.assertEqual(systemone_temperatures({"systemone": None}),
                         {"choice": 1.0, "noul": 1.0, "score": 1.0})
        self.assertEqual(systemone_temperatures({"systemone": {"temperature": None}}),
                         {"choice": 1.0, "noul": 1.0, "score": 1.0})

    def test_the_three_types(self):
        got = systemone_temperatures({"systemone": {"temperature": {"choice": 0.62, "noul": 0.67,
                                                                    "score": 0.77}}})
        self.assertEqual(got, {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_one_type_leaves_the_others_alone(self):
        got = systemone_temperatures({"systemone": {"temperature": {"choice": 2.0}}})
        self.assertEqual(got["choice"], 2.0)
        self.assertEqual(got["noul"], 1.0)
        self.assertEqual(got["score"], 1.0)

    def test_a_temperature_below_one_is_allowed(self):
        # a temperature below 1 sharpens the reported distribution
        got = systemone_temperatures({"systemone": {"temperature": {"choice": 0.5}}})
        self.assertEqual(got["choice"], 0.5)

    def test_an_invalid_temperature_is_refused(self):
        for bad in (0.0, -1.0, "hot", None, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                systemone_temperatures({"systemone": {"temperature": {"choice": bad}}})

    def test_an_unknown_question_type_is_refused(self):
        with self.assertRaises(ValueError):
            systemone_temperatures({"systemone": {"temperature": {"guess": 2.0}}})

    def test_a_temperature_that_is_not_an_object_is_refused(self):
        with self.assertRaises(ValueError):
            systemone_temperatures({"systemone": {"temperature": 2.0}})


class OptionDescriptions(unittest.TestCase):
    """A model that answers with an option's DESCRIPTION instead of its name.

    The prompt shows an option as `- name ("description")` and asks for the name, but
    the JevBench rewrite cases (whose options are named option_1 ... option_4 with a
    description attached) drew the description: four of 643 requests answered 502
    because of it.  The description names the option just as well, so the endpoint
    maps it - but only when EXACTLY ONE option carries that description, so it can
    never be a guess between two options.
    """

    RAW = {"tier": {"type": "choice", "instructions": "Which plan?",
                    "criteria": {"option_1": "The paid pro plan",
                                 "option_2": "The free tier",
                                 "option_3": None}}}

    @classmethod
    def setUpClass(cls):
        _, cls.QUESTIONS = sysone.validate_request({"state": "x", "questions": cls.RAW})

    def test_a_description_names_its_option(self):
        answers = sysone.parse_answer_text("The paid pro plan", self.QUESTIONS)
        self.assertEqual(answers["tier"], "option_1")

    def test_the_whole_line_is_read_too(self):
        answers = sysone.parse_answers("tier: The free tier\n", self.QUESTIONS)
        self.assertEqual(answers["tier"], "option_2")

    def test_a_name_is_never_second_guessed(self):
        # the name is checked first, so a name that also reads as a description wins
        answers = sysone.parse_answer_text("option_3", self.QUESTIONS)
        self.assertEqual(answers["tier"], "option_3")

    def test_an_unknown_text_is_still_an_error(self):
        with self.assertRaises(StructuredOutputError):
            sysone.parse_answer_text("something else", self.QUESTIONS)

    def test_an_ambiguous_description_is_not_guessed(self):
        # two options carry the same description: there is no single answer, so the
        # text stays an error rather than a coin flip
        _, questions = sysone.validate_request({"state": "x", "questions": {
            "tier": {"type": "choice", "instructions": "Which?",
                     "criteria": {"option_1": "the same", "option_2": "the same"}}}})
        with self.assertRaises(StructuredOutputError):
            sysone.parse_answer_text("the same", questions)

    def test_an_option_without_a_description_is_never_matched(self):
        with self.assertRaises(StructuredOutputError):
            sysone.parse_answer_text("None", self.QUESTIONS)


class ReadoutFlag(unittest.TestCase):
    """The config's "systemone": {"readout": ...}: how the model's answer is read."""

    def test_the_default_is_the_tokens(self):
        self.assertEqual(systemone_readout_mode({}), "tokens")
        self.assertEqual(systemone_readout_mode({"systemone": {"enabled": True}}), "tokens")
        self.assertEqual(systemone_readout_mode({"systemone": None}), "tokens")

    def test_the_two_readouts(self):
        self.assertEqual(systemone_readout_mode({"systemone": {"readout": "tokens"}}), "tokens")
        self.assertEqual(systemone_readout_mode({"systemone": {"readout": "letters"}}), "letters")

    def test_an_unknown_readout_is_refused(self):
        with self.assertRaises(ValueError):
            systemone_readout_mode({"systemone": {"readout": "letter"}})


class LetterReadout(unittest.TestCase):
    """The letter readout: the options are shown as A, B, C ... and the model answers
    with the letter.

    The letter is one token and no two letters are the same token, so the options
    cannot collide and the trie walk is not needed.  The client never sees the
    letter: the response is the shape the endpoint already has.
    """

    RAW = {"tier": {"type": "choice", "instructions": "Which plan?",
                    "criteria": {"basic": "The free tier", "pro": "The paid pro plan"}},
           "mood": {"type": "score", "instructions": "How happy?",
                    "criteria": ["unhappy", "neutral", "happy"]},
           "urgent": {"type": "noul", "instructions": "Urgency?",
                      "criteria": {"true": "outage", "false": "chit-chat"}}}

    @classmethod
    def setUpClass(cls):
        _, cls.QUESTIONS = sysone.validate_request({"state": "x", "questions": cls.RAW})

    def test_the_prompt_letters_the_options(self):
        prompt = sl.render_answer_prompt("STATE", "tier", self.QUESTIONS[0][1])
        self.assertIn('A) basic ("The free tier")', prompt)
        self.assertIn('B) pro ("The paid pro plan")', prompt)
        self.assertIn("answer with exactly one of these letters: A, B", prompt)
        self.assertIn('start your answer line with exactly "tier:"', prompt)

    def test_the_noul_and_the_score_are_lettered_too(self):
        noul = sl.render_answer_prompt("STATE", "urgent", self.QUESTIONS[2][1])
        self.assertIn('A) yes means "outage"', noul)
        self.assertIn('B) no means "chit-chat"', noul)
        score = sl.render_answer_prompt("STATE", "mood", self.QUESTIONS[1][1])
        self.assertIn("A) unhappy", score)
        self.assertIn("B) neutral", score)
        self.assertIn("C) happy", score)
        self.assertIn("answer with exactly one of these letters: A, B, C", score)

    def test_the_letter_maps_back_to_the_option(self):
        tier = [("tier", self.QUESTIONS[0][1])]
        self.assertEqual(sl.parse_answer_text("A", tier), {"tier": "basic"})
        self.assertEqual(sl.parse_answer_text("B", tier), {"tier": "pro"})
        self.assertEqual(sl.parse_answer_text("b", tier), {"tier": "pro"})
        self.assertEqual(sl.parse_answer_text("B)", tier), {"tier": "pro"})
        self.assertEqual(sl.parse_answers("tier: B\n", tier), {"tier": "pro"})

    def test_the_score_letter_is_the_level_index(self):
        mood = [("mood", self.QUESTIONS[1][1])]
        self.assertEqual(sl.parse_answer_text("A", mood), {"mood": 0})
        self.assertEqual(sl.parse_answer_text("C", mood), {"mood": 2})

    def test_the_noul_letter_is_yes_or_no(self):
        urgent = [("urgent", self.QUESTIONS[2][1])]
        self.assertEqual(sl.parse_answer_text("A", urgent), {"urgent": True})
        self.assertEqual(sl.parse_answer_text("B", urgent), {"urgent": False})

    def test_a_rate_is_still_a_rate(self):
        # the calibration items are answered with the rate the state gives, and the
        # letter path does not lose that: the value is not a letter, so the direct
        # path's own rule applies
        urgent = [("urgent", self.QUESTIONS[2][1])]
        self.assertEqual(sl.parse_answer_text("0.7", urgent), {"urgent": 0.7})

    def test_a_word_is_not_a_letter(self):
        # "no" is not the letter N, and "Apple" is not the letter A: a value that is
        # not a letter is read by the direct path's own rules
        tier = [("tier", self.QUESTIONS[0][1])]
        with self.assertRaises(StructuredOutputError):
            sl.parse_answer_text("Apple", tier)

    def test_past_26_options_the_question_is_answered_by_name(self):
        # there are 26 letters, so a question with more options than that has no
        # letter for every option: it is shown and answered by name, the way the
        # direct path shows it
        criteria = {f"opt_{i}": None for i in range(27)}
        _, questions = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": criteria}}})
        prompt = sl.render_answer_prompt("STATE", "q", questions[0][1])
        self.assertIn("- opt_0", prompt)
        self.assertNotIn("A) opt_0", prompt)
        self.assertEqual(sl.parse_answer_text("opt_5", questions), {"q": "opt_5"})
        # and the letter path declines the question: there is no letter for every
        # option, so the direct path answers it
        self.assertIsNone(sl.readings(questions, {65: 2.0}, ByteTokenizer(), "q:", 3.0))

    def test_the_letters_are_one_token_each(self):
        # the byte tokenizer of the tests gives every letter its own id, so the
        # letters cannot collide: that is why the trie walk is not needed
        ids = sl.letter_token_ids(["A", "B", "C"], ByteTokenizer(), "q:")
        flat = [i for v in ids.values() for i in v]
        self.assertTrue(all(ids.values()))
        self.assertEqual(len(set(flat)), len(flat))

    def test_the_letter_path_takes_one_engine_request(self):
        # the options share their first token ("rule_1" and "rule_2" both start with
        # " rule"), so the direct path needs the trie walk; the letters cannot
        # collide, so one engine request answers the question
        svc = svc_with("q: A\n")
        svc.systemone_readout = "letters"
        svc.engine.logits_topk = {65: 2.0, 66: 1.0}
        calls = count_logits(svc)
        code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                         "criteria": {"rule_1": None, "rule_2": None}}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(len(calls), 1)                     # one engine request, no walk
        self.assertEqual(body["answers"]["q"]["choice"], "rule_1")
        self.assertTrue(body["calibrated"])

    def test_the_direct_path_needs_the_walk_for_the_same_options(self):
        # the same options, read as their own tokens: the first token cannot tell
        # them apart, so the walk asks the engine again
        svc = svc_with()
        tok = svc.tok
        ids = lft.option_token_ids(["rule_1", "rule_2"], tok, "q:")
        a, b = ids["rule_1"], ids["rule_2"]
        shared = 0
        while shared < len(a) and shared < len(b) and a[shared] == b[shared]:
            shared += 1
        self.assertGreater(shared, 0)
        rows = [({a[i]: 3.0}, 3.0) for i in range(shared)]
        rows.append(({a[shared]: 2.0, b[shared]: 1.0}, 3.0))
        svc.engine.logits_rows = rows
        calls = count_logits(svc)
        code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                         "criteria": {"rule_1": None, "rule_2": None}}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        self.assertGreater(len(calls), 1)                    # the walk's extra requests

    def test_a_garbled_answer_falls_back_the_same_way(self):
        # the model wrote something that is not an answer: the readings answer it,
        # the forced choice with the confidence carrying the doubt - the same as the
        # direct path
        svc = svc_with("q: hmm\n")
        svc.systemone_readout = "letters"
        svc.engine.logits_topk = {65: 2.0, 66: 1.0, 90: 9.0}     # 90 is not a letter
        svc.engine.logits_tokens = svc.tok.encode("q: hmm")
        code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                         "criteria": {"rule_1": None, "rule_2": None}}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q"]["choice"], "rule_1")
        self.assertLess(body["answers"]["q"]["confidence"], 0.5)

    def test_the_answer_is_in_the_scorer_s_format(self):
        # the harness's own mapping (tools/decision_policy/opendecisions.py): the
        # choice's probabilities are keyed by the option names, so the official
        # scorer reads the letter arm exactly like the direct one
        from tools.decision_policy.opendecisions import to_probs
        svc = svc_with("q: A\n")
        svc.systemone_readout = "letters"
        svc.engine.logits_topk = {65: 3.0, 66: 1.0}
        code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                         "criteria": {"rule_1": None, "rule_2": None}}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        item = {"question": {"type": "choice"}, "labels": ["rule_1", "rule_2"]}
        probs = to_probs(item, body["answers"]["q"])
        self.assertEqual(sorted(probs), ["rule_1", "rule_2"])
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_the_temperature_scales_the_letter_arm_too(self):
        # the letter arm and the token arm have to report the same thing: the
        # temperature scales the closed distribution on both, or the comparison
        # between the arms would be a comparison of the temperatures
        def run(temperature):
            svc = svc_with("q: A\n")
            svc.systemone_readout = "letters"
            svc.systemone_temperature = {"choice": temperature, "noul": 1.0, "score": 1.0}
            svc.engine.logits_topk = {65: 3.0, 66: 0.0}
            code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                             "criteria": {"rule_1": None, "rule_2": None}}}),
                                        threading.Event())
            self.assertEqual(code, 200)
            return body["answers"]["q"]["probabilities"]
        plain, sharp = run(1.0), run(0.5)
        self.assertGreater(sharp["rule_1"], plain["rule_1"])      # a temperature below 1 sharpens
        self.assertAlmostEqual(sum(plain.values()), 1.0)
        self.assertAlmostEqual(sum(sharp.values()), 1.0)

    def test_the_flag_off_keeps_the_direct_path(self):
        svc = svc_with()
        self.assertEqual(svc.systemone_readout, "tokens")
        self.assertIs(svc._systemone_readout(), sysone)
        svc.systemone_readout = "letters"
        self.assertIs(svc._systemone_readout(), sl)


if __name__ == "__main__":
    unittest.main()
