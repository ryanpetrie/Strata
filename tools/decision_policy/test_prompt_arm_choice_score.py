#!/usr/bin/env python3
"""The choice's and the score's arm analysis, without a server or a GPU.

    python -m unittest tools.decision_policy.test_prompt_arm_choice_score -v

The arithmetic is the thing to pin: the TVD, the score's expected level on the item's own
lowest-first order, the top-label mass, the mean-level prior, and the four configs' own
numbers.  The pre-registration is companion doc §27.
"""
from __future__ import annotations

import unittest

from tools.decision_policy import prompt_arm_choice_score as A


def item(iid: str, qtype: str, config: str, *, soft: dict | None = None,
         expected=None, k: int = 3) -> dict:
    it = {"id": iid, "config": config, "question": {"type": qtype,
                                                   "criteria": [f"level {i}" for i in range(k)]}}
    if soft is not None:
        it["target_probs"] = soft
    if expected is not None:
        it["expected"] = expected
    return it


class Arith(unittest.TestCase):
    def test_tvd_is_half_the_l1_distance(self):
        self.assertAlmostEqual(A.tvd({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}), 1.0)
        self.assertAlmostEqual(A.tvd({"a": 0.75, "b": 0.25}, {"a": 0.25, "b": 0.75}), 0.5)
        # a key only one side has counts as 0 on the other
        self.assertAlmostEqual(A.tvd({"a": 1.0}, {"b": 1.0}), 1.0)

    def test_expected_level_uses_the_item_s_own_order(self):
        self.assertAlmostEqual(A.expected_level({"0": 1.0, "1": 0.0, "2": 0.0}, 3), 0.0)
        self.assertAlmostEqual(A.expected_level({"0": 0.0, "1": 0.5, "2": 0.5}, 3), 1.5)
        self.assertAlmostEqual(A.expected_level({"0": 0.2, "1": 0.3, "2": 0.5}, 3), 1.3)

    def test_the_paired_interval_holds_a_constant(self):
        lo, hi = A.paired_ci([0.05] * 30)
        self.assertAlmostEqual(lo, 0.05)
        self.assertAlmostEqual(hi, 0.05)
        lo, hi = A.paired_ci([-0.1] * 20 + [0.3] * 20)
        self.assertLess(lo, 0.1)
        self.assertGreater(hi, 0.1)


class ChoiceBlock(unittest.TestCase):
    def items(self):
        return [item("s1", "choice", "go_emotions", soft={"a": 0.9, "b": 0.1}),
                item("s2", "choice", "go_emotions", soft={"a": 0.5, "b": 0.5}),
                item("h1", "choice", "banking77", expected="a", k=2),
                item("h2", "choice", "banking77", expected="a", k=2)]

    def test_identical_arms_move_nothing(self):
        base = {"s1": {"a": 0.9, "b": 0.1}, "s2": {"a": 0.5, "b": 0.5},
                "h1": {"a": 0.8, "b": 0.2}, "h2": {"a": 0.8, "b": 0.2}}
        r = A.choice_block(self.items(), base, dict(base))
        self.assertEqual(r["soft"]["d_tvd"], 0.0)
        self.assertEqual(r["soft"]["picks_moved"], 0)
        self.assertEqual(r["soft"]["acc_items_lost"], 0)
        self.assertEqual(r["hard"]["items_lost"], 0)
        self.assertEqual(r["hard"]["d_ece_max"], 0.0)

    def test_a_flattened_arm_is_a_worse_tvd_and_reports_the_mass(self):
        base = {"s1": {"a": 0.9, "b": 0.1}, "s2": {"a": 0.5, "b": 0.5},
                "h1": {"a": 0.8, "b": 0.2}, "h2": {"a": 0.8, "b": 0.2}}
        arm = {"s1": {"a": 0.5, "b": 0.5}, "s2": {"a": 0.5, "b": 0.5},
               "h1": {"a": 0.8, "b": 0.2}, "h2": {"a": 0.8, "b": 0.2}}
        r = A.choice_block(self.items(), base, arm)
        # s1: the humans are 0.9/0.1; the base is exact, the arm is 0.4 away
        self.assertAlmostEqual(r["soft"]["tvd_base"], 0.0)
        self.assertAlmostEqual(r["soft"]["tvd_arm"], 0.2)
        self.assertLess(r["soft"]["d_tvd"], 0.0)
        self.assertAlmostEqual(r["soft"]["top_mass_base"], 0.7)
        self.assertAlmostEqual(r["soft"]["top_mass_arm"], 0.5)
        self.assertAlmostEqual(r["soft"]["top_mass_humans"], 0.7)
        self.assertEqual(r["soft"]["acc_items_lost"], 0)

    def test_the_hard_half_is_per_config(self):
        base = {"s1": {"a": 0.9, "b": 0.1}, "s2": {"a": 0.5, "b": 0.5},
                "h1": {"a": 0.8, "b": 0.2}, "h2": {"a": 0.8, "b": 0.2}}
        arm = dict(base)
        arm["h2"] = {"a": 0.2, "b": 0.8}                    # one item lost
        r = A.choice_block(self.items(), base, arm)
        self.assertEqual(r["hard_per_config"]["banking77"]["items_lost"], 1)
        self.assertEqual(r["hard"]["items_lost"], 1)


class ScoreBlock(unittest.TestCase):
    def items(self):
        return [item("s1", "score", "measuring_hate_speech", soft={"0": 0.5, "1": 0.5, "2": 0.0}),
                item("s2", "score", "measuring_hate_speech", soft={"0": 1.0, "1": 0.0, "2": 0.0}),
                item("h1", "score", "helpsteer2_verbosity", expected=0, k=5),
                item("h2", "score", "helpsteer2_verbosity", expected=4, k=5)]

    def test_identical_arms_move_nothing(self):
        base = {"s1": {"0": 0.2, "1": 0.8, "2": 0.0}, "s2": {"0": 0.9, "1": 0.1, "2": 0.0},
                "h1": {"0": 0.6, "1": 0.2, "2": 0.1, "3": 0.05, "4": 0.05},
                "h2": {"0": 0.05, "1": 0.05, "2": 0.1, "3": 0.2, "4": 0.6}}
        r = A.score_block(self.items(), base, dict(base))
        self.assertAlmostEqual(r["soft"]["d_signed_toward_zero"], 0.0)
        self.assertAlmostEqual(r["soft"]["d_tvd"], 0.0)
        self.assertEqual(r["soft"]["top_items_lost"], 0)
        self.assertAlmostEqual(r["hard_per_config"]["helpsteer2_verbosity"]["d_mae"], 0.0)

    def test_a_higher_arm_moves_the_signed_error_away(self):
        base = {"s1": {"0": 0.5, "1": 0.5, "2": 0.0}, "s2": {"0": 1.0, "1": 0.0, "2": 0.0},
                "h1": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
                "h2": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}}
        arm = {"s1": {"0": 0.0, "1": 0.5, "2": 0.5}, "s2": {"0": 0.0, "1": 1.0, "2": 0.0},
               "h1": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0},
               "h2": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}}
        r = A.score_block(self.items(), base, arm)
        # the humans' mean level is (0.5 + 0)/2 = 0.25 and the base's is the same; the
        # arm's is (1.5 + 1.0)/2 = 1.25, so the signed error grows
        self.assertAlmostEqual(r["soft"]["level_humans"], 0.25)
        self.assertAlmostEqual(r["soft"]["signed_base"], 0.0)
        self.assertAlmostEqual(r["soft"]["signed_arm"], 1.0)
        self.assertLess(r["soft"]["d_signed_toward_zero"], 0.0)
        # verbosity: h1 goes from exact (0) to 4 levels off; h2 stays exact
        self.assertAlmostEqual(r["hard_per_config"]["helpsteer2_verbosity"]["d_mae"], 2.0)

    def test_the_mean_level_prior_is_reported(self):
        base = {"s1": {"0": 0.5, "1": 0.5, "2": 0.0}, "s2": {"0": 1.0, "1": 0.0, "2": 0.0},
                "h1": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
                "h2": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}}
        r = A.score_block(self.items(), base, dict(base))
        self.assertAlmostEqual(r["soft"]["mean_level_prior"], 0.25)
        self.assertAlmostEqual(r["soft"]["mean_level_prior_mae"], 0.25)


class Rules(unittest.TestCase):
    def test_choice_rule_checks_both_margins(self):
        good = {"soft": {"d_tvd": 0.02, "tvd_base": 0.3, "tvd_arm": 0.28,
                         "top_mass_base": 0.8, "top_mass_arm": 0.7, "top_mass_humans": 0.6,
                         "picks_moved": 0, "n_both": 250, "pick_matches_base": 200,
                         "pick_matches_arm": 200, "acc_items_lost": 0,
                         "tvd_ci": [0.01, 0.03]},
                "hard": {"items_lost": 0, "d_ece_max": 0.001, "acc_base": 0.8,
                         "acc_arm": 0.8, "n_both": 250},
                "hard_per_config": {}}
        text = "\n".join(A.rule_choice(good))
        self.assertIn("PASS", text)

    def test_score_rule_fails_when_the_location_moves_the_wrong_way(self):
        bad = {"soft": {"level_base": 1.0, "level_arm": 1.2, "level_humans": 0.7,
                        "signed_base": 0.3, "signed_arm": 0.5, "d_signed_toward_zero": -0.2,
                        "tvd_base": 0.4, "tvd_arm": 0.45, "d_tvd": -0.05,
                        "top_matches_base": 200, "top_matches_arm": 200,
                        "top_items_lost": 0, "n_both": 250,
                        "signed_ci": [-0.3, -0.1], "tvd_ci": [-0.06, -0.04],
                        "mean_level_prior": 0.7, "mean_level_prior_mae": 0.4,
                        "mae_base": 0.3, "mae_arm": 0.5},
                "hard_per_config": {}}
        text = "\n".join(A.rule_score(bad))
        self.assertIn("FAIL", text)


if __name__ == "__main__":
    unittest.main()
