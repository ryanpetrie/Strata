"""serve/test_logits_prob.py - unit tests for logits_prob.

    python -m unittest serve.test_logits_prob -v
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import logits_prob as lp  # noqa: E402


class SoftmaxTests(unittest.TestCase):
    def test_known_values(self):
        logits = [2.0, 1.0, 0.0]
        probs = lp.softmax(logits)
        e = math.exp(1.0)
        expected = [e * e / (e * e + e + 1.0),
                    e / (e * e + e + 1.0),
                    1.0 / (e * e + e + 1.0)]
        for p, q in zip(probs, expected):
            self.assertAlmostEqual(p, q, places=12)
        self.assertAlmostEqual(sum(probs), 1.0, places=12)

    def test_uniform(self):
        probs = lp.softmax([0.0, 0.0, 0.0, 0.0])
        for p in probs:
            self.assertAlmostEqual(p, 0.25, places=12)

    def test_one_dominant(self):
        probs = lp.softmax([10.0, 0.0, 0.0])
        self.assertGreater(probs[0], 0.99)
        self.assertLess(probs[1], 0.01)
        self.assertLess(probs[2], 0.01)
        self.assertAlmostEqual(sum(probs), 1.0, places=12)

    def test_large_logits_no_overflow(self):
        probs = lp.softmax([1000.0, 1001.0, 999.0])
        self.assertTrue(all(math.isfinite(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertEqual(probs.index(max(probs)), 1)   # max logit wins

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            lp.softmax([])


class LogsumexpTests(unittest.TestCase):
    def test_known(self):
        # log(e^1 + e^2 + e^3) = 3 + log(e^-2 + e^-1 + 1)
        self.assertAlmostEqual(lp.logsumexp([1.0, 2.0, 3.0]),
                               3.0 + math.log(math.exp(-2) + math.exp(-1) + 1.0),
                               places=12)

    def test_large_no_overflow(self):
        v = lp.logsumexp([1000.0, 1001.0, 999.0])
        self.assertTrue(math.isfinite(v))
        self.assertAlmostEqual(v, 1001.0 + math.log(math.exp(-1) + 1.0 + math.exp(-2)),
                               places=12)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            lp.logsumexp([])


class SequenceLogprobTests(unittest.TestCase):
    def test_single_token(self):
        logits = [[2.0, 1.0, 0.0]]
        p = lp.softmax([2.0, 1.0, 0.0])[0]
        self.assertAlmostEqual(lp.sequence_logprob([0], logits), math.log(p), places=12)

    def test_two_token_chain_rule(self):
        # Position 0 favors token 0; position 1 (conditioned on token 0) favors token 1.
        logits = [[3.0, 1.0], [0.0, 2.0]]
        p0 = math.exp(3.0) / (math.exp(3.0) + math.exp(1.0))   # P(token 0 at pos 0)
        p1 = math.exp(2.0) / (math.exp(0.0) + math.exp(2.0))   # P(token 1 at pos 1 | token 0)
        expected_log = math.log(p0) + math.log(p1)
        self.assertAlmostEqual(lp.sequence_logprob([0, 1], logits), expected_log, places=12)

    def test_too_few_positions_raises(self):
        with self.assertRaises(ValueError):
            lp.sequence_logprob([0, 1], [[1.0, 2.0]])

    def test_extra_positions_ok(self):
        # A 1-token sequence uses position 0 only; extra positions are ignored.
        logits = [[2.0, 1.0], [9.0, 9.0]]
        p = lp.softmax([2.0, 1.0])[0]
        self.assertAlmostEqual(lp.sequence_logprob([0], logits), math.log(p), places=12)

    def test_token_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            lp.sequence_logprob([5], [[1.0, 2.0]])

    def test_empty_sequence_is_log_one(self):
        self.assertAlmostEqual(lp.sequence_logprob([], []), 0.0, places=12)


class AnswerLogprobTests(unittest.TestCase):
    def test_multiple_surface_forms(self):
        # "yes" = token 0, "YES" = token 1; both single-token, same logits vector.
        logits = [[2.0, 1.0, 0.0]]
        p_yes = lp.softmax([2.0, 1.0, 0.0])[0]
        p_yes_up = lp.softmax([2.0, 1.0, 0.0])[1]
        expected = math.log(p_yes + p_yes_up)
        self.assertAlmostEqual(lp.answer_logprob([[0], [1]], logits), expected, places=12)

    def test_empty_sequences_raises(self):
        with self.assertRaises(ValueError):
            lp.answer_logprob([], [[1.0, 2.0]])


class DistributionTests(unittest.TestCase):
    def test_sums_to_one(self):
        answers = {"a": [[0]], "b": [[1]], "c": [[2]]}
        logits = [[2.0, 1.0, 0.0]]
        dist = lp.distribution(answers, logits)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)
        self.assertEqual(set(dist), {"a", "b", "c"})

    def test_matches_softmax(self):
        answers = {"a": [[0]], "b": [[1]], "c": [[2]]}
        logits = [[2.0, 1.0, 0.0]]
        dist = lp.distribution(answers, logits)
        sm = lp.softmax(logits[0])
        self.assertAlmostEqual(dist["a"], sm[0], places=12)
        self.assertAlmostEqual(dist["b"], sm[1], places=12)
        self.assertAlmostEqual(dist["c"], sm[2], places=12)

    def test_empty_answers_raises(self):
        with self.assertRaises(ValueError):
            lp.distribution({}, [[1.0, 2.0]])


class DecideTests(unittest.TestCase):
    def test_argmax_and_confidence(self):
        answers = {"a": [[0]], "b": [[1]], "c": [[2]]}
        logits = [[2.0, 1.0, 0.0]]
        pick, conf, dist = lp.decide(answers, logits)
        self.assertEqual(pick, "a")
        self.assertAlmostEqual(conf, lp.softmax(logits[0])[0], places=12)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)

    def test_uniform_picks_first(self):
        answers = {"a": [[0]], "b": [[1]], "c": [[2]]}
        logits = [[0.0, 0.0, 0.0]]
        pick, conf, dist = lp.decide(answers, logits)
        self.assertEqual(pick, "a")
        self.assertAlmostEqual(conf, 1.0 / 3.0, places=12)


class NoulGroupingTests(unittest.TestCase):
    def test_positive_vs_negative_groups(self):
        # tokens: yes=0, YES=1, no=2, NO=3
        answers = {"yes": [[0], [1]], "no": [[2], [3]]}
        logits = [[2.0, 1.0, 0.5, 0.0]]
        dist = lp.distribution(answers, logits)
        sm = lp.softmax(logits[0])
        self.assertAlmostEqual(dist["yes"], sm[0] + sm[1], places=12)
        self.assertAlmostEqual(dist["no"], sm[2] + sm[3], places=12)
        self.assertAlmostEqual(dist["yes"] + dist["no"], 1.0, places=12)


class MultiTokenAnswerTests(unittest.TestCase):
    def test_multi_token_option(self):
        # "ny" = tokens [0, 1]; "la" = tokens [2, 3].
        # Position 0 favors token 0; position 1 (conditioned on token 0) favors token 1.
        answers = {"ny": [[0, 1]], "la": [[2, 3]]}
        logits = [[3.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]
        dist = lp.distribution(answers, logits)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)
        # raw_ny = [e^3/(e^3+e^1+2)] * [e^2/(3+e^2)]; raw_la = [1/(e^3+e^1+2)] * [1/(3+e^2)]
        # dist["ny"] = raw_ny/(raw_ny+raw_la) = e^5/(e^5+1)
        e5 = math.exp(5.0)
        self.assertAlmostEqual(dist["ny"], e5 / (e5 + 1.0), places=12)
        self.assertAlmostEqual(dist["la"], 1.0 / (e5 + 1.0), places=12)

    def test_mixed_single_and_multi_token(self):
        # "yes" = token 0 (single); "no way" = tokens [1, 2] (multi).
        answers = {"yes": [[0]], "no_way": [[1, 2]]}
        logits = [[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
        dist = lp.distribution(answers, logits)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)
        p_yes = lp.softmax([2.0, 1.0, 0.0])[0]
        p_no = lp.softmax([2.0, 1.0, 0.0])[1]
        p_way_given_no = lp.softmax([0.0, 0.0, 3.0])[2]
        expected_no = p_no * p_way_given_no
        self.assertAlmostEqual(dist["yes"], p_yes / (p_yes + expected_no), places=12)
        self.assertAlmostEqual(dist["no_way"], expected_no / (p_yes + expected_no), places=12)


class BuildResponseTests(unittest.TestCase):
    def test_noul(self):
        answers = {"yes": [[0]], "no": [[1]]}
        logits = [[2.0, 1.0]]
        dist = lp.distribution(answers, logits)
        resp = lp.build_response("noul", answers, dist)
        self.assertEqual(resp["type"], "noul")
        self.assertAlmostEqual(resp["noul"], dist["yes"], places=12)
        self.assertTrue(0.0 <= resp["noul"] <= 1.0)

    def test_choice(self):
        answers = {"a": [[0]], "b": [[1]], "c": [[2]]}
        logits = [[2.0, 1.0, 0.0]]
        dist = lp.distribution(answers, logits)
        resp = lp.build_response("choice", answers, dist)
        self.assertEqual(resp["type"], "choice")
        self.assertEqual(resp["choice"], "a")
        self.assertAlmostEqual(resp["confidence"], dist["a"], places=12)
        self.assertEqual(set(resp["probabilities"]), {"a", "b", "c"})
        self.assertAlmostEqual(sum(resp["probabilities"].values()), 1.0, places=12)

    def test_score(self):
        answers = {"low": [[0]], "mid": [[1]], "high": [[2]]}
        logits = [[0.0, 1.0, 2.0]]
        dist = lp.distribution(answers, logits)
        resp = lp.build_response("score", answers, dist)
        self.assertEqual(resp["type"], "score")
        self.assertEqual(resp["score"], 2.0)   # "high" is index 2
        self.assertEqual(resp["legend"], {"0": "low", "1": "mid", "2": "high"})
        self.assertAlmostEqual(resp["probabilities"]["2"], dist["high"], places=12)
        self.assertAlmostEqual(sum(resp["probabilities"].values()), 1.0, places=12)
        self.assertAlmostEqual(resp["confidence"], dist["high"], places=12)

    def test_unknown_qtype_raises(self):
        with self.assertRaises(ValueError):
            lp.build_response("maybe", {"a": [[0]]}, {"a": 1.0})


if __name__ == "__main__":
    unittest.main()
