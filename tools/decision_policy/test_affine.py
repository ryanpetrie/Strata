"""tools/decision_policy/test_affine.py - unit tests for the affine calibration's map.

    python -m unittest tools.decision_policy.test_affine -v

No network, no GPU, no engine, no data on this machine: every test builds what it needs.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import affine as af  # noqa: E402


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class TheMapTests(unittest.TestCase):
    def test_the_noul_is_the_logit_scale_and_shift(self):
        for p in (0.001, 0.05, 0.446, 0.9, 0.999):
            for t, b in ((1.0, -2.0), (2.5, -1.3), (0.7, 0.4), (5.0, 0.0)):
                out = af.apply_map({"yes": p, "no": 1.0 - p}, "noul", ["no", "yes"], t, b)
                self.assertAlmostEqual(logit(out["yes"]), logit(p) / t + b, places=9)

    def test_the_map_is_invertible_on_every_type(self):
        cases = [
            ("noul", ["no", "yes"], {"yes": 0.446, "no": 0.554}),
            ("noul", ["no", "yes"], {"yes": 1.0, "no": 0.0}),
            ("score", ["0", "1", "2"], {"0": 0.6, "1": 0.3, "2": 0.1}),
            ("score", ["0", "1", "2", "3"], {"0": 0.05, "1": 0.25, "2": 0.4, "3": 0.3}),
            ("choice", ["a", "b", "c"], {"a": 0.7, "b": 0.2, "c": 0.1}),
        ]
        for qtype, labels, closed in cases:
            for t, b in ((1.0, 0.0), (0.6, -1.2), (3.7, 0.9), (0.5, 0.0)):
                self.assertLess(af.round_trip(closed, qtype, labels, t, b), 1e-14,
                                "%s at T %r b %r" % (qtype, t, b))

    def test_b_zero_is_the_server_temperature(self):
        from serve.logits_first_token import apply_temperature
        for qtype, labels in (("choice", ["a", "b", "c"]), ("score", ["0", "1", "2"]),
                              ("noul", ["no", "yes"])):
            closed = {lab: (1.0 + i) for i, lab in enumerate(labels)}
            total = sum(closed.values())
            closed = {lab: v / total for lab, v in closed.items()}
            reading = {"closed": closed, "open": closed}
            for t in (0.62, 1.0, 2.5):
                self.assertEqual(af.apply_map(closed, qtype, labels, t, 0.0),
                                 apply_temperature(reading, t)["closed"])

    def test_the_choice_has_no_shift(self):
        closed = {"a": 0.7, "b": 0.2, "c": 0.1}
        none = af.apply_map(closed, "choice", ["a", "b", "c"], 1.7, 0.0)
        some = af.apply_map(closed, "choice", ["a", "b", "c"], 1.7, 2.0)
        for lab in closed:
            self.assertAlmostEqual(none[lab], some[lab], places=15,
                                   msg="a shift over the options is cancelled by the normalizer")

    def test_the_score_tilt_moves_the_expected_level_the_right_way(self):
        closed = {"0": 0.6, "1": 0.3, "2": 0.1}
        down = af.apply_map(closed, "score", ["0", "1", "2"], 1.0, -0.5)
        up = af.apply_map(closed, "score", ["0", "1", "2"], 1.0, 0.5)
        self.assertLess(af.expected_level(down, ["0", "1", "2"]), 0.5)
        self.assertGreater(af.expected_level(up, ["0", "1", "2"]), 0.5)
        self.assertAlmostEqual(af.expected_level(closed, ["0", "1", "2"]), 0.5)

    def test_a_zero_stays_zero_and_the_item_still_sums_to_one(self):
        out = af.apply_map({"yes": 0.0, "no": 1.0}, "noul", ["no", "yes"], 2.0, -3.0)
        self.assertEqual(out["yes"], 0.0)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=15)


class TheFitTests(unittest.TestCase):
    def _noul_items(self, t_star: float, b_star: float, n: int = 200) -> list[dict]:
        items = []
        for i in range(n):
            # the closed P(yes) spread over the whole range, logit-spaced
            p = sigmoid(-5.0 + 10.0 * i / (n - 1))
            q = sigmoid(logit(p) / t_star + b_star)
            items.append({"labels": ["no", "yes"],
                          "logp": [math.log(1.0 - p) if p < 1.0 else None,
                                   math.log(p) if p > 0.0 else None],
                          "coords": [0.0, 1.0],
                          "target": {"no": 1.0 - q, "yes": q}})
        return items

    def test_the_fit_recovers_the_parameters_it_was_given(self):
        for t_star, b_star in ((2.0, -1.5), (3.5, 0.4), (1.0, -2.5)):
            fit = af.fit_map(self._noul_items(t_star, b_star), "noul", "soft",
                             b_min=-4.0, b_max=4.0)
            self.assertAlmostEqual(fit["temperature"], t_star, delta=0.03)
            self.assertAlmostEqual(fit["shift"], b_star, delta=0.03)

    def test_the_fit_beats_t_one_on_the_objective(self):
        items = self._noul_items(2.5, -1.0)
        fit = af.fit_map(items, "noul", "soft", b_min=-4.0, b_max=4.0)
        self.assertLess(fit["objective_value"], fit["objective_at_t1"])

    def test_the_choice_fit_returns_no_shift(self):
        items = []
        for i in range(50):
            p = 0.02 + 0.96 * i / 49.0
            items.append({"labels": ["a", "b"], "logp": [math.log(p), math.log(1.0 - p)],
                          "coords": [0.0, 0.0], "target": {"a": p, "b": 1.0 - p}})
        fit = af.fit_map(items, "choice", "soft")
        self.assertEqual(fit["shift"], 0.0)
        self.assertFalse(fit["shift_identified"])

    def test_the_objective_is_the_cross_entropy(self):
        items = [{"labels": ["a", "b"], "logp": [math.log(0.5), math.log(0.5)],
                  "coords": [0.0, 0.0], "target": {"a": 0.5, "b": 0.5}}]
        # a uniform target against a uniform distribution is log 2
        self.assertAlmostEqual(af.objective_value(items, 1.0, 0.0, "soft"), math.log(2.0),
                               places=12)


class TheMetricsTests(unittest.TestCase):
    def test_the_auc_with_ties_is_half(self):
        self.assertEqual(af.auc_binary([0.5, 0.5], [1.0, 0.0]), 0.5)
        self.assertEqual(af.auc_binary([0.9, 0.1], [1.0, 0.0]), 1.0)
        self.assertEqual(af.auc_binary([0.1, 0.9], [1.0, 0.0]), 0.0)

    def test_the_binary_auc_is_invariant_under_the_noul_map(self):
        # a monotone map cannot move the ranking, so the noul's discrimination cannot move
        records = [{"id": str(i), "type": "noul", "labels": ["no", "yes"],
                    "closed": {"yes": p, "no": 1.0 - p}, "gold": ("yes" if i % 2 else "no"),
                    "target": {"yes": 0.5, "no": 0.5}, "task": {}}
                   for i, p in enumerate((0.1, 0.35, 0.45, 0.55, 0.65, 0.9))]
        before = af.type_metrics(records, {r["id"]: r["closed"] for r in records}, "noul")
        after = af.type_metrics(records, {r["id"]: af.apply_map(
            r["closed"], "noul", r["labels"], 2.5, -1.2) for r in records}, "noul")
        self.assertAlmostEqual(before["auc"], after["auc"], places=15)

    def test_the_subset_table_reproduces_the_all_three_arm(self):
        # the decomposition the rule's (c) rests on: a toy scorer summary of the shape
        # combine_capability reads
        def arm(cal, comp):
            return {"scorer": {"calibration_per_type": {"choice": cal[0], "noul": cal[1],
                                                        "score": cal[2]},
                               "competence_per_type": {"choice": comp[0], "noul": comp[1],
                                                       "score": comp[2]},
                               "capability": (sum(cal) / 3 + sum(comp) / 3) / 2}}
        arms = {"T = 1": arm((60, 70, 80), (50, 60, 70)),
                "only choice": arm((40, 70, 80), (50, 60, 70)),
                "only noul": arm((60, 20, 80), (50, 30, 70)),
                "only score": arm((60, 70, 90), (50, 60, 10)),
                "all": arm((40, 20, 90), (50, 30, 10))}
        combined = af.combine_capability(arms, ["choice", "noul", "score"])
        self.assertAlmostEqual(combined["calibration"], (40 + 20 + 90) / 3)
        self.assertAlmostEqual(combined["intelligence"], (50 + 30 + 10) / 3)
        self.assertAlmostEqual(combined["capability"],
                               ((40 + 20 + 90) / 3 + (50 + 30 + 10) / 3) / 2)
        af.check_decomposition(arms, "all")
        # a subset uses the baseline for the types it does not map
        one = af.combine_capability(arms, ["noul"])
        self.assertAlmostEqual(one["calibration"], (60 + 20 + 80) / 3)
        self.assertAlmostEqual(one["intelligence"], (50 + 30 + 70) / 3)


if __name__ == "__main__":
    unittest.main()
