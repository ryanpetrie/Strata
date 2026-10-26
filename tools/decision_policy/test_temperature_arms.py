"""tools/decision_policy/test_temperature_arms.py - unit tests for the paired arms tool.

    python -m unittest tools.decision_policy.test_temperature_arms -v

The tests do not touch the network, do not need a GPU and do not need the engine.
The one test that needs the official scorer says so when it is not there.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import temperature_arms as ta  # noqa: E402

SCORER = Path(tempfile.gettempdir()) / "opencode" / "opendecisions" / "score.py"


class ParseTests(unittest.TestCase):
    def test_the_named_form(self):
        self.assertEqual(ta.parse_temperatures("choice=0.62,noul=0.67,score=0.77"),
                         {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_the_positional_form_follows_the_type_order(self):
        self.assertEqual(ta.parse_temperatures("0.62,0.67,0.77"),
                         {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_an_arm_needs_a_name(self):
        with self.assertRaises(SystemExit):
            ta.parse_arms(["soft"])

    def test_an_arm_may_be_positional(self):
        arms = ta.parse_arms(["soft=3.65,3.75,3.73"])
        self.assertEqual(arms, [("soft", {"choice": 3.65, "noul": 3.75, "score": 3.73})])

    def test_a_temperature_must_be_above_zero(self):
        with self.assertRaises(SystemExit):
            ta.parse_temperatures("choice=0")


class TvdTests(unittest.TestCase):
    def test_it_is_the_half_sum_of_the_differences(self):
        self.assertAlmostEqual(ta.tvd({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}, ["a", "b"]), 0.5)

    def test_an_exact_match_is_zero(self):
        self.assertAlmostEqual(ta.tvd({"a": 0.25, "b": 0.75}, {"a": 0.25, "b": 0.75}, ["a", "b"]), 0.0)


class ClosedTests(unittest.TestCase):
    def test_a_noul_gets_its_no_filled_in(self):
        item = {"labels": ["no", "yes"], "question": {"type": "noul"}}
        self.assertEqual(ta.closed_of(item, {"yes": 0.25}), {"no": 0.75, "yes": 0.25})


@unittest.skipUnless(SCORER.is_file(), "the official score.py is not here")
class CliTests(unittest.TestCase):
    def _pool(self, directory: Path):
        """A small run: three items, one of every type, served at a temperature."""
        served = {"choice": 2.0, "noul": 2.0, "score": 2.0}
        items, runlog = [], []
        for iid, qtype, labels, target in (
                ("c/1", "choice", ["a", "b"], {"a": 0.5, "b": 0.5}),
                ("n/1", "noul", ["no", "yes"], {"yes": 0.4, "no": 0.6}),
                ("s/1", "score", ["0", "1"], {"0": 0.7, "1": 0.3})):
            items.append({"id": iid, "question": {"type": qtype}, "labels": labels,
                          "expected": labels[0], "tier": "hard", "target_probs": target,
                          "config": "cfg", "state": "a state"})
            # the run reported the T = 1 reading scaled by the served temperature
            from tools.decision_policy import temperature as tp
            at_one = {"a": 0.8, "b": 0.2} if qtype == "choice" else (
                {"yes": 0.8, "no": 0.2} if qtype == "noul" else {"0": 0.8, "1": 0.2})
            closed = tp.scale(at_one, served[qtype])
            record = {"id": iid, "type": qtype, "config": "cfg", "predicted": True,
                      "fast_path": True, "pick": max(closed, key=closed.get),
                      "closed_confidence": closed[max(closed, key=closed.get)],
                      "probabilities": closed if qtype != "noul" else None,
                      "open_probs": {k: v * 0.9 for k, v in closed.items()},
                      "target_probs": target, "tvd": 0.0}
            runlog.append(record)
        paths = []
        for name, rows in (("items.jsonl", items), ("runlog.jsonl", runlog)):
            path = directory / name
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                            encoding="utf-8")
            paths.append(path)
        return paths[0], paths[1]

    def test_the_arms_are_recovered_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            items, runlog = self._pool(tmpdir)
            out = tmpdir / "report.md"
            argv = sys.argv
            sys.argv = ["temperature_arms.py", "--items", str(items), "--runlog", str(runlog),
                        "--served", "2.0,2.0,2.0", "--arm", "soft=4.0,4.0,4.0",
                        "--scorer", str(SCORER), "--out-report", str(out),
                        "--out-predictions", str(tmpdir / "t1.jsonl")]
            try:
                self.assertEqual(ta.main(), 0)
            finally:
                sys.argv = argv
            report = out.read_text(encoding="utf-8")
            self.assertIn("T = 1 (recovered)", report)
            self.assertIn("soft", report)
            self.assertTrue((tmpdir / "t1.jsonl").is_file())
            # the recovered T = 1 arm: the run's reading put back through 1/T
            rows = {json.loads(l)["id"]: json.loads(l)["probs"]
                    for l in (tmpdir / "t1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
            self.assertAlmostEqual(rows["c/1"]["a"], 0.8, places=9)
            self.assertAlmostEqual(rows["n/1"]["yes"], 0.8, places=9)


if __name__ == "__main__":
    unittest.main()
