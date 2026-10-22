"""tools/decision_policy/test_prompt_probe.py - unit tests for the prompt probe.

    python -m unittest tools.decision_policy.test_prompt_probe -v

The probe's own arithmetic: the design set's half A, the three readouts (the flip
rate, the swap gap and the AUC) and the pre-registered reading of them.  No server,
no engine, no GPU: the arms are two synthetic prediction files.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import prompt_probe as P  # noqa: E402
from serve import systemone as sysone  # noqa: E402


def item(iid: str, human_yes: float, state: str = "STATE", true: str = "outage",
         false: str = "chit-chat") -> dict:
    return {"id": iid, "state": state, "config": "civil_comments", "tier": "hard",
            "expected": "yes" if human_yes >= 0.5 else "no",
            "target_probs": {"yes": human_yes, "no": 1.0 - human_yes},
            "question": {"type": "noul", "instructions": "Is it x?",
                         "criteria": {"true": true, "false": false}}}


def preds(items: list[dict], ps: list[float]) -> dict[str, dict]:
    return {it["id"]: {"id": it["id"], "probs": {"yes": p, "no": 1.0 - p}}
            for it, p in zip(items, ps)}


class DesignSet(unittest.TestCase):
    def test_half_a_is_the_noul_items_with_a_human_distribution(self):
        rows = [item("a", 0.0), item("b", 0.5),
                {"id": "c", "state": "x", "question": {"type": "noul", "instructions": "?",
                                                       "criteria": {"true": "t", "false": "f"}}},
                {"id": "d", "state": "x", "question": {"type": "choice", "instructions": "?",
                                                       "criteria": {"x": None}},
                 "target_probs": {"x": 1.0}}]
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            path = Path(f.name)
        try:
            half_a = P.design_set(path)
            self.assertEqual([r["id"] for r in half_a], ["a", "b"])
        finally:
            path.unlink()

    def test_a_file_without_a_noul_distribution_is_refused(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"id": "c", "state": "x",
                                "question": {"type": "choice", "instructions": "?",
                                             "criteria": {"x": None}}}) + "\n")
            path = Path(f.name)
        try:
            with self.assertRaises(SystemExit):
                P.design_set(path)
        finally:
            path.unlink()


class Readouts(unittest.TestCase):
    """The three readouts, on two synthetic arms."""

    def test_the_pick_rule(self):
        self.assertEqual(P.pick_yes(0.5), "yes")       # the scorer's own rule
        self.assertEqual(P.pick_yes(0.4999), "no")
        self.assertEqual(P.pick_yes(1.0), "yes")

    def test_the_auc(self):
        self.assertEqual(P.auc([0.9, 0.1], [1, 0]), 1.0)
        self.assertEqual(P.auc([0.1, 0.9], [1, 0]), 0.0)
        self.assertEqual(P.auc([0.5, 0.5], [1, 0]), 0.5)

    def test_nothing_changes(self):
        items = [item("a", 0.0), item("b", 0.5), item("c", 0.25)]
        base = preds(items, [0.2, 0.7, 0.4])
        m = P.compare(items, base, base)
        self.assertEqual(m["n_both"], 3)
        self.assertEqual(m["flips"], 0)
        self.assertEqual(m["flip_rate"], 0.0)
        self.assertAlmostEqual(m["swap_gap"], 0.0, places=12)
        self.assertAlmostEqual(m["mean_p_yes_base"], m["mean_p_yes_swap"], places=12)
        self.assertAlmostEqual(m["auc_base"], m["auc_swap"], places=12)
        self.assertIn("DEAD", P.verdict(m))

    def test_the_swap_gap_is_the_mean_of_the_differences(self):
        items = [item("a", 0.0), item("b", 0.5), item("c", 0.25)]
        base = preds(items, [0.60, 0.70, 0.40])
        swap = preds(items, [0.40, 0.60, 0.20])
        m = P.compare(items, base, swap)
        self.assertAlmostEqual(m["swap_gap"], (0.20 + 0.10 + 0.20) / 3, places=12)
        self.assertAlmostEqual(m["mean_p_yes_base"], (0.60 + 0.70 + 0.40) / 3, places=12)
        self.assertAlmostEqual(m["mean_p_yes_swap"], (0.40 + 0.60 + 0.20) / 3, places=12)
        self.assertAlmostEqual(m["mean_human_yes"], (0.0 + 0.5 + 0.25) / 3, places=12)
        self.assertAlmostEqual(m["bias_base"], m["mean_p_yes_base"] - m["mean_human_yes"], places=12)
        self.assertAlmostEqual(m["bias_swap"], m["mean_p_yes_swap"] - m["mean_human_yes"], places=12)

    def test_the_flip_rate_counts_the_moved_picks(self):
        items = [item("a", 0.0), item("b", 0.5), item("c", 0.25), item("d", 0.0)]
        base = preds(items, [0.60, 0.40, 0.40, 0.10])
        swap = preds(items, [0.40, 0.60, 0.45, 0.10])   # two picks move
        m = P.compare(items, base, swap)
        self.assertEqual(m["flips"], 2)
        self.assertAlmostEqual(m["flip_rate"], 0.5, places=12)

    def test_only_the_items_in_both_arms_are_compared(self):
        items = [item("a", 0.0), item("b", 0.5)]
        base = preds(items[:1], [0.2])
        swap = preds(items[1:], [0.7])
        m = P.compare(items, base, swap)
        self.assertEqual(m["n_items"], 2)
        self.assertEqual(m["n_base"], 1)
        self.assertEqual(m["n_swap"], 1)
        self.assertEqual(m["n_both"], 0)
        self.assertIn("NOTHING TO COMPARE", P.verdict(m))

    def test_a_real_gap_is_read_as_a_live_axis(self):
        items = [item(f"i{i}", 0.0) for i in range(10)]
        base = preds(items, [0.80] * 10)
        swap = preds(items, [0.20] * 10)
        m = P.compare(items, base, swap)
        self.assertAlmostEqual(m["swap_gap"], 0.6, places=12)
        self.assertEqual(m["flips"], 10)
        self.assertIn("ALIVE", P.verdict(m))


class ThePrompts(unittest.TestCase):
    """The probe renders the prompt with the shipped code, and the swap arm is the
    full mirror of the base one."""

    def test_the_swap_mirrors_the_base(self):
        it = item("a", 0.0)
        base = P.render_prompt(it, "base").split("\n")
        swap = P.render_prompt(it, "swap").split("\n")
        self.assertIn('  yes means "outage" ; no means "chit-chat"', base)
        self.assertIn("  answer yes or no", base)
        self.assertIn('  no means "chit-chat" ; yes means "outage"', swap)
        self.assertIn("  answer no or yes", swap)
        # the answer tokens do not change: the model still writes " yes" or " no"
        self.assertIn("  start your answer line with exactly \"q:\"", swap)
        self.assertEqual(len(base), len(swap))

    def test_the_rendered_prompt_is_the_endpoint_s_prompt(self):
        it = item("a", 0.0)
        _, questions = sysone.validate_request({"state": it["state"],
                                                 "questions": {"q": it["question"]}})
        name, q = questions[0]
        want = sysone.render_answer_prompt(sysone.serialize_state(it["state"]), name, q, "swap")
        self.assertEqual(P.render_prompt(it, "swap"), want)


if __name__ == "__main__":
    unittest.main()
