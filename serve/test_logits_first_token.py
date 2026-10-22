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
        # the confidence is the model's OWN probability of the pick (the open
        # reading), not the distribution's maximum (the closed reading)
        resp = lft.reading_response(lft.first_token_probs(
            self.qs, self._logits({"a": 2.0, "b": 1.0}), self.tok, PREFIX, logz=3.0)["q"])
        self.assertAlmostEqual(resp["confidence"], math.exp(2.0 - 3.0))
        self.assertLessEqual(resp["confidence"], resp["probabilities"]["a"])

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

    def test_the_model_wanting_something_else_is_a_flag_not_a_refusal(self):
        # the model's top token is not one of the options (it wants to write a digit,
        # a phrase, the end of the turn): the readings are still computed, and the flag
        # says the model was not choosing among the options - the caller's answer then
        # comes from the model's own text if that text is a valid answer, and from
        # these readings if it is not
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        logits = self._logits({"a": 2.0, "b": 1.0})
        logits[stop_id] = 5.0
        readings = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)
        self.assertIsNotNone(readings)
        r = readings["q"]
        self.assertFalse(r["model_picked"])
        # the doubt is in the confidence: the open reading is the model's own
        # probability of the pick, and the mass it put elsewhere is the difference
        self.assertLess(r["open"]["a"], r["closed"]["a"])
        self.assertAlmostEqual(r["outside"], 1.0 - sum(r["open"].values()))

    def test_the_model_choosing_an_option_is_flagged(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        readings = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)
        self.assertTrue(readings["q"]["model_picked"])

    def test_collision_falls_back(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": {"rule_1": None, "rule_2": None}}}})
        tok = self.tok
        shared = lft.first_token_ids(["rule_1"], tok, PREFIX)["rule_1"]
        self.assertIsNone(lft.first_token_probs(qs, {shared: 2.0}, tok, PREFIX))

    def test_missing_falls_back(self):
        # every option outside the window: there is nothing to compare
        self.assertIsNone(lft.first_token_probs(self.qs, {999: 2.0}, self.tok, PREFIX))

    def test_an_option_outside_the_window_is_scored_as_zero(self):
        # the engine sends the top k logits, so an option can be below the window.  It
        # is then below every option inside the window: it cannot be the pick, and the
        # question is still answered.
        logits = self._logits({"a": 2.0})                # "b" is outside the window
        readings = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)
        self.assertIsNotNone(readings)
        r = readings["q"]
        self.assertEqual(max(r["closed"], key=r["closed"].get), "a")
        self.assertEqual(r["closed"]["b"], 0.0)
        self.assertAlmostEqual(sum(r["closed"].values()), 1.0)
        self.assertEqual(r["open"]["b"], 0.0)

    def test_every_option_outside_the_window_falls_back(self):
        # nothing to score: there is no option to compare
        self.assertIsNone(lft.first_token_probs(self.qs, {999: 2.0}, self.tok, PREFIX))

    def test_the_fallback_reason_says_which_way_it_fell_back(self):
        # the failures mean different things: a collision is what the trie walk
        # answers, an option outside the window is not a failure at all, and the model
        # wanting something else is
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": {"rule_1": None, "rule_2": None}}}})
        self.assertIn("collision", lft.fallback_reason(qs, {1: 2.0}, self.tok, PREFIX))
        self.assertIn("outside the window",
                      lft.fallback_reason(self.qs, {999: 2.0}, self.tok, PREFIX))
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        logits = self._logits({"a": 2.0, "b": 1.0})
        logits[stop_id] = 5.0
        self.assertIn("top token", lft.fallback_reason(self.qs, logits, self.tok, PREFIX))

    def test_the_fallback_reason_is_none_when_the_readings_exist(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        self.assertIsNotNone(lft.first_token_probs(self.qs, logits, self.tok, PREFIX))
        self.assertIn("no reason", lft.fallback_reason(self.qs, logits, self.tok, PREFIX))

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


class TrieWalk(unittest.TestCase):
    """The trie walk: the options' token sequences form a trie, and the row at a node
    of the trie tells the options that pass through it apart.  It is what answers a
    question whose options share a first token ("rule_1" and "rule_2" both start with
    " rule"), and it stops where the options are told apart, so it is short."""

    def setUp(self):
        self.tok = Tok()
        _, self.collide = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": {"rule_1": None, "rule_2": None}}}})

    def _plan(self, options):
        plan = lft.trie_plan(options, self.tok, PREFIX)
        self.assertIsNotNone(plan)
        return plan

    def test_the_options_are_told_apart_at_their_first_differing_token(self):
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        # the mock tokenizer splits "rule_1" into "rule", "_", "1" (the model's BPE
        # gives " rule" and "_1"), so the options are told apart at the third token
        self.assertEqual(plan["resolution"]["rule_1"], 2)
        self.assertEqual(plan["resolution"]["rule_2"], 2)
        self.assertEqual(plan["walk"], [tuple(ids["rule_1"][:1]), tuple(ids["rule_1"][:2])])

    def test_the_walk_reads_every_node_on_the_way(self):
        # the weight of an option is a factor per node, so the row after "rule" is
        # needed as well as the row after "rule", "_"
        plan = self._plan(["rule_1", "rule_2"])
        self.assertEqual(len(plan["walk"]), 2)

    def test_options_of_different_lengths_are_told_apart(self):
        # "bill" and "bill now": the shorter one ENDS where the longer one continues,
        # so the turn has to end there for the shorter one to be the answer
        plan = self._plan(["bill", "bill now"])
        self.assertEqual(plan["resolution"]["bill"], 1)
        self.assertEqual(plan["resolution"]["bill now"], 1)
        self.assertEqual(len(plan["walk"]), 1)

    def test_a_partial_collision_is_told_apart_at_the_root(self):
        # "No change" and "Cross-service" are alone at the first token; "One file" and
        # "One service" are not, so they are told apart one node deeper
        plan = self._plan(["No change", "One file", "One service", "Cross-service"])
        self.assertEqual(plan["resolution"]["No change"], 0)
        self.assertEqual(plan["resolution"]["Cross-service"], 0)
        self.assertEqual(plan["resolution"]["One file"], 1)
        self.assertEqual(plan["resolution"]["One service"], 1)
        self.assertEqual(len(plan["walk"]), 1)          # only the "One" node

    def test_the_weight_is_the_product_of_the_token_probabilities(self):
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        rows = {(): ({ids["rule_1"][0]: 3.0}, 3.0),
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0, ids["rule_2"][2]: 1.0}, 3.0)}
        probs = lft.trie_probs(plan, rows)
        self.assertAlmostEqual(probs["rule_1"],
                               math.exp(3.0 - 3.0) * math.exp(2.0 - 3.0) * math.exp(2.0 - 3.0))
        self.assertAlmostEqual(probs["rule_2"],
                               math.exp(3.0 - 3.0) * math.exp(2.0 - 3.0) * math.exp(1.0 - 3.0))
        self.assertGreater(probs["rule_1"], probs["rule_2"])

    def test_a_token_outside_the_window_is_no_answer(self):
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        rows = {(): ({ids["rule_1"][0]: 3.0}, 3.0),
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0}, 3.0)}   # rule_2 missing
        probs = lft.trie_probs(plan, rows)
        # rule_2's token is below the window's tail: it carries no probability, and
        # rule_1 - which is in the window - is the answer
        self.assertEqual(probs["rule_2"], 0.0)
        self.assertGreater(probs["rule_1"], 0.0)
        r = lft.trie_readings(self.collide[0][1], plan, rows)
        self.assertIsNotNone(r)
        self.assertEqual(max(r["closed"], key=r["closed"].get), "rule_1")

    def test_the_shorter_option_needs_the_turn_to_end(self):
        # "bill" and "bill now": the end-of-turn token is what makes "bill" the answer
        plan = self._plan(["bill", "bill now"])
        ids = plan["tokens"]
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        node = tuple(ids["bill"][:1])
        rows = {(): ({ids["bill"][0]: 3.0}, 3.0),
                node: ({ids["bill now"][1]: 2.0, stop_id: 1.0}, 3.0)}
        probs = lft.trie_probs(plan, rows)
        # "bill": the prefix times the probability that the turn ends there
        self.assertAlmostEqual(probs["bill"], math.exp(3.0 - 3.0) * math.exp(1.0 - 3.0))
        # "bill now": the prefix times the probability of the token it continues with
        self.assertAlmostEqual(probs["bill now"], math.exp(3.0 - 3.0) * math.exp(2.0 - 3.0))
        self.assertGreater(probs["bill now"], probs["bill"])

    def test_the_reading_separates_the_options(self):
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        rows = {(): ({ids["rule_1"][0]: 3.0}, 3.0),
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0, ids["rule_2"][2]: 1.0}, 3.0)}
        r = lft.trie_readings(self.collide[0][1], plan, rows)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(sum(r["closed"].values()), 1.0)
        self.assertEqual(max(r["closed"], key=r["closed"].get), "rule_1")
        # the open reading: the model's own probability of each option
        self.assertAlmostEqual(r["open"]["rule_1"],
                               math.exp(3.0 - 3.0) * math.exp(2.0 - 3.0) * math.exp(2.0 - 3.0))
        self.assertLessEqual(r["outside"], 1.0)

    def test_the_model_wanting_something_else_is_a_flag_not_a_refusal(self):
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        rows = {(): ({ids["rule_1"][0]: 3.0}, 3.0),
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0, ids["rule_2"][2]: 1.0,
                                           stop_id: 9.0}, 3.0)}
        r = lft.trie_readings(self.collide[0][1], plan, rows)
        # the reading is computed; the model wanted to end the turn at the separating
        # node, and the doubt is in the confidence
        self.assertIsNotNone(r)
        self.assertLess(r["open"]["rule_1"], r["closed"]["rule_1"])

    def test_the_answer_position_is_checked_too(self):
        # the options share their first token, so the answer position is not a node
        # where they are told apart - but the reading is still about the context the
        # model is in, so the row at the answer position is held to the same rule the
        # first-token path holds it to
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        rows = {(): ({ids["rule_1"][0]: 3.0, 999: 9.0}, 3.0),      # the model wants 999
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0, ids["rule_2"][2]: 1.0}, 3.0)}
        r = lft.trie_readings(self.collide[0][1], plan, rows)
        self.assertIsNotNone(r)
        self.assertFalse(r["model_picked"])

    def test_the_failure_reason_says_which_way_it_failed(self):
        plan = self._plan(["rule_1", "rule_2"])
        ids = plan["tokens"]
        stop_id = self.tok.encode("<|im_end|>", parse_special=True)[0]
        rows = {(): ({ids["rule_1"][0]: 3.0}, 3.0),
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0, ids["rule_2"][2]: 1.0,
                                           stop_id: 9.0}, 3.0)}
        self.assertIn("wanted something else", lft.trie_failure_reason(plan, rows))


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
        # the fast path cannot score the options - the model's top token is not one of
        # them, it was going to say something else - but THIS request generated the
        # answer
        svc = svc_with("q: b\n")
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0, 1002: 9.0}    # 1002 is not an option
        svc.engine.logits_tokens = svc.tok.encode("q: b")
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "b")
        self.assertFalse(body["calibrated"])

    def test_an_option_outside_the_window_is_still_answered(self):
        # the engine sends the top k logits: an option below the window cannot be the
        # pick (it is below every option inside it), so the question is still answered
        svc = svc_with("q: b\n")
        svc.engine.logits_topk = {1000: 2.0}                       # 1001 missing
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        self.assertEqual(body["answers"]["q"]["probabilities"]["b"], 0.0)

    def test_the_readings_answer_when_the_model_wrote_something_else(self):
        # the model's top token is not an option and the text it wrote is not an
        # answer: the forced choice, with the confidence carrying the doubt
        svc = svc_with("q: hmm\n")
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0, 1002: 9.0}    # 1002 is not an option
        svc.engine.logits_tokens = svc.tok.encode("q: hmm")
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        # the doubt is in the confidence: the model's own probability of the pick
        self.assertLess(body["answers"]["q"]["confidence"], 0.5)
        self.assertAlmostEqual(body["answers"]["q"]["probabilities"]["a"]
                               + body["answers"]["q"]["probabilities"]["b"], 1.0)

    def test_the_noul_carries_a_confidence(self):
        # the value is the closed reading (the probability of yes among the two
        # answers); the confidence is the model's own probability of the side it
        # picked - so a confident "no" reports a small value and a large confidence
        svc = svc_with()
        svc.engine.logits_topk = {1000: 3.0, 1001: 0.0}
        svc.engine.logits_logz = 4.0
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"yes": 1000, "no": 1001}):
            code, body = svc.systemone(req(questions={"q": {"type": "noul",
                                                             "instructions": "Is it?"}}),
                                       threading.Event())
        self.assertEqual(code, 200)
        ans = body["answers"]["q"]
        self.assertEqual(ans["type"], "noul")
        self.assertIn("confidence", ans)
        # the value is the closed reading of "yes", the confidence the open one
        self.assertAlmostEqual(ans["noul"], math.exp(3.0 - 4.0) / (math.exp(3.0 - 4.0)
                                                           + math.exp(0.0 - 4.0)))
        self.assertAlmostEqual(ans["confidence"], math.exp(3.0 - 4.0))
        self.assertLessEqual(ans["confidence"], ans["noul"])

    def test_the_end_of_turn_token_is_not_part_of_the_answer(self):
        # the model wrote the answer and then ended its turn.  The end token must not
        # be decoded into the answer: "rule_2<|im_end|>" is not one of the options, so
        # the request failed with a 502 (measured, on the collision case).
        svc = svc_with("q: b\n")
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0, 1002: 9.0}    # 1002 is not an option
        svc.engine.logits_tokens = svc.tok.encode("q: b") + [next(iter(svc.stop_ids))]
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "b")

    def test_colliding_options_use_the_trie_walk(self):
        # "x1" and "x2" share their first tokens, so the first-token row cannot
        # separate them: the trie walk reads the rows that do, and the answer comes
        # from the probabilities
        svc = svc_with()
        tok = svc.tok
        ids = lft.option_token_ids(["x1", "x2"], tok, PREFIX)
        a, b = ids["x1"], ids["x2"]
        shared = 0
        while shared < len(a) and shared < len(b) and a[shared] == b[shared]:
            shared += 1
        self.assertGreater(shared, 0)
        # row 0 is the answer position, rows 1..shared follow the shared tokens, and
        # the last row holds the two options' own tokens
        rows = [({a[i]: 3.0}, 3.0) for i in range(shared)]
        rows.append(({a[shared]: 2.0, b[shared]: 1.0}, 3.0))
        svc.engine.logits_rows = rows
        code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                         "criteria": {"x1": None, "x2": None}}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q"]["choice"], "x1")
        self.assertGreater(body["answers"]["q"]["confidence"], 0.0)
        self.assertLessEqual(body["answers"]["q"]["confidence"],
                             max(body["answers"]["q"]["probabilities"].values()))

    def test_multi_question_is_answered_per_question(self):
        # several questions: each one runs the single-question path, so each answer
        # carries the model's own probability (the rows are the two answer positions)
        svc = svc_with()
        svc.engine.logits_rows = [({1000: 2.0, 1001: 1.0}, 3.0), ({1000: 1.0, 1001: 2.0}, 3.0)]
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(questions={
                "q1": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}},
                "q2": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}}),
                threading.Event())
        self.assertEqual(code, 200)
        self.assertTrue(body["calibrated"])
        self.assertEqual(body["answers"]["q1"]["choice"], "a")      # the first answer position
        self.assertEqual(body["answers"]["q2"]["choice"], "b")      # the second one
        self.assertLess(body["answers"]["q1"]["confidence"], 1.0)

    def test_the_shared_prompt_is_still_there(self):
        # the old path, behind the config flag: one prompt answers every question, so
        # the prefill is shared and the answers carry no probability
        svc = svc_with("q1: a\nq2: b\n")
        svc.systemone_multi = "shared"
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
        # the response's confidence is the OPEN reading; the diagnostics carry both
        self.assertAlmostEqual(diag["open_confidence"], body["answers"]["q"]["confidence"])
        self.assertNotAlmostEqual(diag["closed_confidence"], body["answers"]["q"]["confidence"])
        self.assertGreaterEqual(diag["outside"], 0.0)

    def test_empty_reply_falls_back_to_the_classic_prompt(self):
        # the model wanted to end the turn at the answer position: the classic prompt,
        # where it writes the whole line, answers it instead of erroring
        svc = svc_with("q: a\n")
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0, 1002: 9.0}    # 1002 is not an option
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        self.assertFalse(body["calibrated"])


class Temperature(unittest.TestCase):
    """The per-type temperature: the distribution over the options is scaled, the pick
    is not.

    The temperature divides the logits before the softmax, so the closed reading (the
    options renormalized over themselves) becomes normalize(closed ** (1/T)).  That
    identity is exact, which is why a temperature fitted offline on a stored run log
    predicts what the endpoint reports.
    """

    def setUp(self):
        self.tok = Tok()
        _, self.qs = sysone.validate_request(req())
        _, self.collide = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": {"rule_1": None, "rule_2": None}}}})

    def _logits(self, values, prefix=PREFIX):
        out = {}
        for opt, v in values.items():
            out[lft.first_token_ids([opt], self.tok, prefix)[opt]] = v
        return out

    def _trie_rows(self):
        plan = lft.trie_plan(["rule_1", "rule_2"], self.tok, PREFIX)
        ids = plan["tokens"]
        rows = {(): ({ids["rule_1"][0]: 3.0}, 3.0),
                tuple(ids["rule_1"][:1]): ({ids["rule_1"][1]: 2.0}, 3.0),
                tuple(ids["rule_1"][:2]): ({ids["rule_1"][2]: 2.0, ids["rule_2"][2]: 1.0}, 3.0)}
        return plan, rows

    # (a) the default is the path before the feature
    def test_the_default_changes_nothing(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        plain = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)
        scaled = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0, temperature=1.0)
        self.assertEqual(plain["q"], scaled["q"])
        self.assertNotIn("temperature", plain["q"])
        # the same for the trie walk
        plan, rows = self._trie_rows()
        self.assertEqual(lft.trie_readings(self.collide[0][1], plan, rows),
                         lft.trie_readings(self.collide[0][1], plan, rows, temperature=1.0))
        # and the response is the same
        self.assertEqual(lft.reading_response(plain["q"]),
                         lft.reading_response(scaled["q"]))

    # (b) the closed reading is exactly normalize(closed ** (1/T))
    def test_the_closed_reading_is_the_scaled_one(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        plain = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)["q"]
        for t in (0.5, 2.0, 3.7):
            scaled = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0,
                                           temperature=t)["q"]
            weights = {o: plain["closed"][o] ** (1.0 / t) for o in plain["closed"]}
            total = sum(weights.values())
            for o in plain["closed"]:
                self.assertAlmostEqual(scaled["closed"][o], weights[o] / total, places=12)
            self.assertAlmostEqual(sum(scaled["closed"].values()), 1.0, places=12)

    def test_the_trie_path_scales_the_closed_reading_too(self):
        plan, rows = self._trie_rows()
        plain = lft.trie_readings(self.collide[0][1], plan, rows)
        scaled = lft.trie_readings(self.collide[0][1], plan, rows, temperature=2.0)
        weights = {o: plain["closed"][o] ** 0.5 for o in plain["closed"]}
        total = sum(weights.values())
        for o in plain["closed"]:
            self.assertAlmostEqual(scaled["closed"][o], weights[o] / total, places=12)
        self.assertAlmostEqual(sum(scaled["closed"].values()), 1.0, places=12)

    # (g) the identity the offline fit relies on
    def test_the_logits_divided_by_T_are_the_closed_reading_at_T(self):
        # dividing the raw logits by T at the softmax site gives exactly
        # normalize(closed_1 ** (1/T)) - that is why the fit can be made offline
        logits = self._logits({"a": 2.0, "b": 1.0})
        ids = lft.first_token_ids(["a", "b"], self.tok, PREFIX)
        for t in (0.5, 2.0, 3.7):
            readings = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, temperature=t)
            scaled = [logits[ids[o]] / t for o in ("a", "b")]
            m = max(scaled)
            exps = [math.exp(v - m) for v in scaled]
            s = sum(exps)
            for i, o in enumerate(("a", "b")):
                self.assertAlmostEqual(readings["q"]["closed"][o], exps[i] / s, places=12)

    # (c) the pick does not change
    def test_the_pick_does_not_change(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        plain = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0)["q"]
        pick = max(plain["closed"], key=plain["closed"].get)
        for t in (0.1, 0.5, 1.0, 2.0, 10.0):
            scaled = lft.first_token_probs(self.qs, logits, self.tok, PREFIX, logz=3.0,
                                           temperature=t)["q"]
            self.assertEqual(max(scaled["closed"], key=scaled["closed"].get), pick)
            self.assertEqual(lft.reading_response(scaled)["choice"],
                             lft.reading_response(plain)["choice"])
        plan, rows = self._trie_rows()
        plain = lft.trie_readings(self.collide[0][1], plan, rows)
        pick = max(plain["closed"], key=plain["closed"].get)
        for t in (0.1, 0.5, 2.0, 10.0):
            scaled = lft.trie_readings(self.collide[0][1], plan, rows, temperature=t)
            self.assertEqual(max(scaled["closed"], key=scaled["closed"].get), pick)

    # (d) the confidence stays inside the distribution
    def test_the_confidence_stays_inside_the_distribution(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        for t in (0.1, 0.5, 1.0, 2.0, 10.0):
            resp = lft.reading_response(lft.first_token_probs(
                self.qs, logits, self.tok, PREFIX, logz=3.0, temperature=t)["q"])
            self.assertGreaterEqual(resp["confidence"], 0.0)
            self.assertLessEqual(resp["confidence"], 1.0)
            self.assertLessEqual(resp["confidence"], max(resp["probabilities"].values()))

    def test_a_larger_temperature_lowers_the_confidence(self):
        logits = self._logits({"a": 2.0, "b": 1.0})
        last = None
        for t in (1.0, 1.5, 2.0, 4.0, 10.0):
            resp = lft.reading_response(lft.first_token_probs(
                self.qs, logits, self.tok, PREFIX, logz=3.0, temperature=t)["q"])
            if last is not None:
                self.assertLess(resp["confidence"], last)
            last = resp["confidence"]

    # (e) the noul's value is the scaled one
    def test_the_noul_value_is_the_scaled_one(self):
        _, qs = sysone.validate_request({"state": "x", "questions": {
            "q": {"type": "noul", "instructions": "Is it?"}}})
        logits = self._logits({"yes": 3.0, "no": 0.0})
        plain = lft.first_token_probs(qs, logits, self.tok, PREFIX)["q"]
        scaled = lft.first_token_probs(qs, logits, self.tok, PREFIX, temperature=2.0)["q"]
        weights = {o: plain["closed"][o] ** 0.5 for o in plain["closed"]}
        total = sum(weights.values())
        self.assertAlmostEqual(scaled["closed"]["yes"], weights["yes"] / total, places=12)
        # the response's value is the closed reading, so it is the scaled one
        resp = lft.reading_response(scaled)
        self.assertAlmostEqual(resp["noul"], weights["yes"] / total, places=12)
        # a temperature below 1 sharpens: the yes grows
        sharp = lft.first_token_probs(qs, logits, self.tok, PREFIX, temperature=0.5)["q"]
        self.assertGreater(sharp["closed"]["yes"], plain["closed"]["yes"])


class ServiceTemperature(unittest.TestCase):
    """The temperature at the endpoint: the config's key, and every path that answers
    a decision."""

    def test_the_single_question_path_scales_the_distribution(self):
        svc = svc_with()
        svc.engine.logits_topk = {1000: 2.0, 1001: 1.0}
        svc.engine.logits_logz = 3.0
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(), threading.Event())
            plain = body["answers"]["q"]["probabilities"]
            svc.systemone_temperature = {"choice": 2.0, "noul": 2.0, "score": 2.0}
            code, body = svc.systemone(req(), threading.Event())
        self.assertEqual(code, 200)
        scaled = body["answers"]["q"]["probabilities"]
        weights = {o: plain[o] ** 0.5 for o in plain}
        total = sum(weights.values())
        for o in plain:
            self.assertAlmostEqual(scaled[o], weights[o] / total, places=12)
        self.assertEqual(body["answers"]["q"]["choice"], "a")      # the pick did not move

    def test_the_merged_path_scales_the_distribution_too(self):
        # several questions: every one of them runs the single-question path, so every
        # answer carries the scaled distribution
        svc = svc_with()
        svc.engine.logits_rows = [({1000: 2.0, 1001: 1.0}, 3.0), ({1000: 1.0, 1001: 2.0}, 3.0)]
        questions = {"q1": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}},
                     "q2": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}}
        with mock.patch.object(lft, "first_token_ids", lambda opts, tok, prefix="": {"a": 1000, "b": 1001}):
            code, body = svc.systemone(req(questions=questions), threading.Event())
            plain = {name: body["answers"][name]["probabilities"] for name in ("q1", "q2")}
            svc.systemone_temperature = {"choice": 2.0, "noul": 2.0, "score": 2.0}
            svc.engine.logits_turns = 0            # the same rows again
            code, body = svc.systemone(req(questions=questions), threading.Event())
        self.assertEqual(code, 200)
        for name in ("q1", "q2"):
            probs = body["answers"][name]["probabilities"]
            weights = {o: plain[name][o] ** 0.5 for o in plain[name]}
            total = sum(weights.values())
            for o in plain[name]:
                self.assertAlmostEqual(probs[o], weights[o] / total, places=12)
            # the closed reading at T = 2 is flatter than the one at T = 1
            self.assertLessEqual(max(probs.values()), max(plain[name].values()))

    def test_the_trie_path_scales_the_distribution(self):
        svc = svc_with()
        svc.systemone_temperature = {"choice": 2.0, "noul": 2.0, "score": 2.0}
        tok = svc.tok
        ids = lft.option_token_ids(["x1", "x2"], tok, PREFIX)
        a, b = ids["x1"], ids["x2"]
        shared = 0
        while shared < len(a) and shared < len(b) and a[shared] == b[shared]:
            shared += 1
        rows = [({a[i]: 3.0}, 3.0) for i in range(shared)]
        rows.append(({a[shared]: 2.0, b[shared]: 1.0}, 3.0))
        svc.engine.logits_rows = rows
        code, body = svc.systemone(req(questions={"q": {"type": "choice", "instructions": "Pick",
                                                         "criteria": {"x1": None, "x2": None}}}),
                                    threading.Event())
        self.assertEqual(code, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "x1")
        probs = body["answers"]["q"]["probabilities"]
        # the distribution is flatter than the raw one: the pick's share fell
        self.assertLess(probs["x1"], 1.0)
        self.assertGreater(probs["x2"], 0.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)

    def test_the_default_temperature_is_one(self):
        svc = svc_with()
        self.assertEqual(svc._decision_temperature("choice"), 1.0)
        svc.systemone_temperature = {"choice": 2.0}
        self.assertEqual(svc._decision_temperature("choice"), 2.0)
        self.assertEqual(svc._decision_temperature("noul"), 1.0)      # the rest is untouched


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
        # the DONE line carries the prompt's own token count: that is how the server
        # knows the answer it read is the answer to THIS request (serve/server.py's
        # Engine.logits), so the fixture carries the three ids that were sent.
        eng = self._engine(["Z 3.0", "L 1000 2.0", "T 7", "DONE 1 3 3 4 stop"])
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
