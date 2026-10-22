"""tools/decision_policy/test_temperature.py - unit tests for the temperature fit.

    python -m unittest tools.decision_policy.test_temperature -v

The tests that need the official scorer use it where the dataset keeps it (the same
path the fitting script defaults to); they skip themselves when it is not there, so
the suite still runs on a machine that has not fetched the dataset.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import temperature as T  # noqa: E402

SCORER = T.DEFAULT_SCORER


def scorer_or_skip():
    if not SCORER.exists():
        raise unittest.SkipTest(f"the official scorer is not at {SCORER}")
    return T.load_scorer(SCORER)


class ScaleTests(unittest.TestCase):
    """The identity the server applies: normalize(closed ** (1/T))."""

    def test_one_is_the_identity(self):
        closed = {"a": 0.6, "b": 0.3, "c": 0.1}
        self.assertEqual(T.scale(closed, 1.0), closed)

    def test_the_weights_are_raised_to_one_over_T(self):
        closed = {"a": 0.6, "b": 0.3, "c": 0.1}
        scaled = T.scale(closed, 2.0)
        weights = {k: v ** 0.5 for k, v in closed.items()}
        total = sum(weights.values())
        for k, v in closed.items():
            self.assertAlmostEqual(scaled[k], weights[k] / total, places=12)
        self.assertAlmostEqual(sum(scaled.values()), 1.0, places=12)

    def test_a_zero_stays_zero(self):
        scaled = T.scale({"a": 1.0, "b": 0.0}, 3.0)
        self.assertEqual(scaled["b"], 0.0)
        self.assertEqual(scaled["a"], 1.0)

    def test_the_pick_cannot_change(self):
        closed = {"a": 0.6, "b": 0.4}
        for t in (0.1, 0.5, 1.0, 2.0, 10.0, 1000.0):
            scaled = T.scale(closed, t)
            self.assertEqual(max(scaled, key=scaled.get), "a")

    def test_a_larger_temperature_flattens(self):
        closed = {"a": 0.9, "b": 0.1}
        self.assertLess(T.scale(closed, 4.0)["a"], T.scale(closed, 1.0)["a"])
        self.assertGreater(T.scale(closed, 0.5)["a"], T.scale(closed, 1.0)["a"])


class UndoTemperatureTests(unittest.TestCase):
    def test_it_inverts_the_scale(self):
        closed = {"a": 0.6, "b": 0.25, "c": 0.15}
        for t in (0.5, 0.62, 0.77, 2.0, 10.0):
            back = T.undo_temperature(T.scale(closed, t), t)
            for k, v in closed.items():
                self.assertAlmostEqual(back[k], v, places=12)

    def test_one_is_the_identity(self):
        closed = {"a": 0.6, "b": 0.4}
        self.assertEqual(T.undo_temperature(closed, 1.0), closed)


class FitTests(unittest.TestCase):
    """The fit: a grid search over the temperature, deterministic."""

    def _records(self):
        recs = []
        for i, (closed, gold) in enumerate([({"a": 0.7, "b": 0.3}, "a"),
                                            ({"a": 0.2, "b": 0.8}, "b")]):
            recs.append({"id": f"i{i}", "type": "choice", "tier": "easy",
                         "labels": ["a", "b"], "closed": closed, "gold": gold,
                         "fast_path": True, "task": {}})
        return recs

    def test_the_brier_fit_prefers_a_sharper_distribution_when_the_model_is(self):
        # the model is under-confident: the right answer carries 0.7 and it is right
        # both times, so a temperature below 1 lowers the Brier score
        t, value = T.fit_temperature(None, self._records(), "choice", "brier")
        self.assertLess(t, 1.0)
        self.assertLess(value, T.mean_brier(self._records(), 1.0))

    def test_the_fit_is_deterministic(self):
        recs = self._records()
        first = T.fit_temperature(None, recs, "choice", "brier")
        second = T.fit_temperature(None, recs, "choice", "brier")
        self.assertEqual(first, second)

    def test_the_grid_is_respected(self):
        recs = self._records()
        t, _ = T.fit_temperature(None, recs, "choice", "brier", 0.9, 1.5)
        self.assertGreaterEqual(t, 0.9)
        self.assertLessEqual(t, 1.5)


class BootstrapTests(unittest.TestCase):
    """The two uncertainty statements, on a synthetic set where the answer is known."""

    def _scorer_and_items(self):
        scorer = scorer_or_skip()
        items = []
        for i in range(40):
            qtype = ("choice", "noul", "score")[i % 3]
            if qtype == "choice":
                task = {"id": f"c{i}", "question": {"type": "choice"},
                        "labels": ["a", "b"], "expected": "a" if i % 2 else "b", "tier": "hard"}
                closed = {"a": 0.8 if i % 2 else 0.2, "b": 0.2 if i % 2 else 0.8}
            elif qtype == "noul":
                task = {"id": f"n{i}", "question": {"type": "noul"},
                        "labels": ["no", "yes"], "expected": "yes" if i % 2 else "no", "tier": "easy"}
                closed = {"yes": 0.8 if i % 2 else 0.2, "no": 0.2 if i % 2 else 0.8}
            else:
                task = {"id": f"s{i}", "question": {"type": "score"},
                        "labels": ["0", "1"], "expected": 0 if i % 2 else 1, "tier": "easy"}
                closed = {"0": 0.8 if i % 2 else 0.2, "1": 0.2 if i % 2 else 0.8}
            items.append({"id": task["id"], "type": qtype, "tier": task["tier"],
                          "labels": task["labels"], "closed": closed, "gold": task["expected"],
                          "fast_path": True, "task": task})
        return scorer, items

    def test_the_paired_bootstrap_sees_a_better_arm(self):
        scorer, items = self._scorer_and_items()
        sharp = {it["id"]: T.scale(it["closed"], 0.5) for it in items}
        flat = {it["id"]: it["closed"] for it in items}
        out = T.paired_bootstrap(scorer, items, flat, sharp, resamples=50, seed=42)
        self.assertIn("capability", out)
        self.assertEqual(out["capability"]["share_better"], 1.0)
        self.assertGreater(out["capability"]["mean"], 0.0)

    def test_the_paired_bootstrap_sees_no_difference(self):
        scorer, items = self._scorer_and_items()
        same = {it["id"]: it["closed"] for it in items}
        out = T.paired_bootstrap(scorer, items, same, same, resamples=20, seed=42)
        self.assertAlmostEqual(out["capability"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(out["capability"]["p025"], 0.0, places=9)
        self.assertAlmostEqual(out["capability"]["p975"], 0.0, places=9)

    def test_the_out_of_bag_bootstrap_is_deterministic(self):
        scorer, items = self._scorer_and_items()
        choice = [it for it in items if it["type"] == "choice"]
        first = T.bootstrap_ece(scorer, choice, "choice", "brier", resamples=20, seed=42)
        second = T.bootstrap_ece(scorer, choice, "choice", "brier", resamples=20, seed=42)
        self.assertEqual(first, second)
        self.assertIn("method", first)


if __name__ == "__main__":
    unittest.main()
