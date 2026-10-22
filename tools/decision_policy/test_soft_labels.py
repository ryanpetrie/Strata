"""tools/decision_policy/test_soft_labels.py - unit tests for soft_labels.

    python -m unittest tools.decision_policy.test_soft_labels -v

The tests do not touch the network and do not need the downloaded data: they
build the rows themselves, in the shape jev-bench ships them.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import soft_labels as sl  # noqa: E402


def choice_row(soft: str | None = None) -> dict:
    return {"id": "c/1", "source": "c", "primitive": "choice", "split": "validation",
            "state": json.dumps({"text": "hello"}),
            "question": json.dumps({"type": "choice", "instructions": "which one?",
                                    "criteria": {"a": "first", "b": "second"}}),
            "label": "a", "soft_label": soft}


def score_row(soft: str | None = None) -> dict:
    return {"id": "s/1", "source": "s", "primitive": "score", "split": "validation",
            "state": "plain text state",
            "question": json.dumps({"type": "score", "instructions": "how much?",
                                    "criteria": ["low", "mid", "high"]}),
            "label": "1", "soft_label": soft}


def noul_row(soft: str | None = None) -> dict:
    return {"id": "n/1", "source": "n", "primitive": "noul", "split": "validation",
            "state": json.dumps({"text": "hello"}),
            "question": json.dumps({"type": "noul", "instructions": "is it?",
                                    "criteria": {"true": "yes", "false": "no"}}),
            "label": "1", "soft_label": soft}


class ToItemTests(unittest.TestCase):
    def test_choice(self):
        item = sl.to_item(choice_row('{"a": 0.7, "b": 0.3}'), "hard")
        self.assertEqual(item["labels"], ["a", "b"])
        self.assertEqual(item["expected"], "a")
        self.assertEqual(item["tier"], "hard")
        self.assertEqual(item["target_probs"], {"a": 0.7, "b": 0.3})
        self.assertEqual(item["state"], {"text": "hello"})
        self.assertEqual(sl.check_item(item), [])

    def test_score(self):
        item = sl.to_item(score_row("[0.2, 0.5, 0.3]"), "easy")
        self.assertEqual(item["labels"], ["0", "1", "2"])
        self.assertEqual(item["expected"], 1)
        self.assertIsInstance(item["expected"], int)
        self.assertEqual(item["target_probs"], {"0": 0.2, "1": 0.5, "2": 0.3})
        self.assertEqual(sl.check_item(item), [])

    def test_noul(self):
        item = sl.to_item(noul_row("0.25"), "judge")
        self.assertEqual(item["labels"], ["no", "yes"])
        self.assertEqual(item["expected"], "yes")
        self.assertEqual(item["target_probs"], {"yes": 0.25, "no": 0.75})
        self.assertEqual(sl.check_item(item), [])

    def test_no_soft_label(self):
        item = sl.to_item(choice_row(None), "hard")
        self.assertIsNone(item["target_probs"])
        self.assertEqual(sl.check_item(item), [])

    def test_state_may_be_plain_text(self):
        item = sl.to_item(score_row(None), "hard")
        self.assertEqual(item["state"], "plain text state")


class CheckItemTests(unittest.TestCase):
    def test_bad_tier(self):
        item = sl.to_item(choice_row(None), "hard")
        item["tier"] = "medium"
        self.assertTrue(any("tier" in p for p in sl.check_item(item)))

    def test_target_probs_keys_must_match_the_labels(self):
        item = sl.to_item(choice_row('{"a": 0.7, "b": 0.3}'), "hard")
        item["target_probs"] = {"a": 1.0}
        self.assertTrue(any("do not match" in p for p in sl.check_item(item)))

    def test_target_probs_must_sum_to_one(self):
        item = sl.to_item(choice_row('{"a": 0.7, "b": 0.3}'), "hard")
        item["target_probs"] = {"a": 0.7, "b": 0.7}
        self.assertTrue(any("sum to" in p for p in sl.check_item(item)))

    def test_noul_expected_must_be_yes_or_no(self):
        item = sl.to_item(noul_row(None), "hard")
        item["expected"] = 1
        self.assertTrue(any("noul expected" in p for p in sl.check_item(item)))

    def test_score_expected_must_be_an_int(self):
        item = sl.to_item(score_row(None), "hard")
        item["expected"] = "1"
        self.assertTrue(any("not an int" in p for p in sl.check_item(item)))

    def test_noul_criteria_may_be_missing(self):
        row = noul_row(None)
        row["question"] = json.dumps({"type": "noul", "instructions": "is it?"})
        self.assertEqual(sl.check_item(sl.to_item(row, "hard")), [])

    def test_choice_criteria_may_not_be_missing(self):
        row = choice_row(None)
        row["question"] = json.dumps({"type": "choice", "instructions": "which one?"})
        with self.assertRaises(ValueError):
            sl.to_item(row, "hard")


class SampleTests(unittest.TestCase):
    def _pool(self):
        pool = []
        for i in range(50):
            for row, qtype in ((choice_row, "choice"), (score_row, "score"), (noul_row, "noul")):
                item = sl.to_item(row(), "hard")
                item["id"] = "%s/%d" % (qtype, i)
                pool.append(item)
        return pool

    def test_stratified_and_deterministic(self):
        pool = self._pool()
        first = sl.stratified_sample(pool, 10, 20261020)
        second = sl.stratified_sample(pool, 10, 20261020)
        self.assertEqual([it["id"] for it in first], [it["id"] for it in second])
        counts = {}
        for item in first:
            counts[item["question"]["type"]] = counts.get(item["question"]["type"], 0) + 1
        self.assertEqual(counts, {"choice": 10, "noul": 10, "score": 10})

    def test_a_different_seed_gives_a_different_sample(self):
        pool = self._pool()
        first = [it["id"] for it in sl.stratified_sample(pool, 10, 1)]
        second = [it["id"] for it in sl.stratified_sample(pool, 10, 2)]
        self.assertNotEqual(first, second)

    def test_the_order_of_the_pool_does_not_matter(self):
        pool = self._pool()
        first = [it["id"] for it in sl.stratified_sample(pool, 10, 7)]
        second = [it["id"] for it in sl.stratified_sample(list(reversed(pool)), 10, 7)]
        self.assertEqual(first, second)

    def test_a_quota_larger_than_the_pool_takes_everything(self):
        pool = self._pool()
        picked = sl.stratified_sample(pool, 1000, 3)
        self.assertEqual(len(picked), len(pool))


class SummaryTests(unittest.TestCase):
    def test_type_mix_and_soft_label_count(self):
        rows = [choice_row('{"a": 1.0, "b": 0.0}'), score_row(None), noul_row("0.5")]
        mix, with_soft = sl.summarize(rows)
        self.assertEqual(mix, {"choice": 1, "score": 1, "noul": 1})
        self.assertEqual(with_soft, 2)


class NullPredictionTests(unittest.TestCase):
    def test_uniform_over_the_labels(self):
        item = sl.to_item(score_row(None), "hard")
        probs = sl.null_prediction(item)
        self.assertEqual(set(probs), {"0", "1", "2"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
