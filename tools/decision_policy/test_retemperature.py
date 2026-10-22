"""tools/decision_policy/test_retemperature.py - unit tests for the re-scaling tool.

    python -m unittest tools.decision_policy.test_retemperature -v

The tests do not touch the network, do not need a GPU and do not need the engine.
One test needs the 235-item OpenDecisions run of 2026-10-21 (the run with the
temperature and the same run with it undone) and says so when it is not here.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import retemperature as rt  # noqa: E402

DATA_DIR = Path(os.environ.get("STRATA_OPENDECISIONS_DIR",
                               r"C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions"))
ITEMS = DATA_DIR / "sample10_items.jsonl"
WITH_TEMPERATURE = DATA_DIR / "fix_serial_temp.jsonl"
TEMPERATURE_UNDONE = DATA_DIR / "fix_serial_temp_t1.jsonl"


class ParseTests(unittest.TestCase):
    def test_the_named_form(self):
        self.assertEqual(rt.parse_temperatures("choice=0.62,noul=0.67,score=0.77"),
                         {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_the_bare_form_follows_the_type_order(self):
        self.assertEqual(rt.parse_temperatures("0.62,0.67,0.77"),
                         {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_a_bad_type_is_refused(self):
        with self.assertRaises(SystemExit):
            rt.parse_temperatures("nope=0.62")

    def test_a_bad_value_is_refused(self):
        with self.assertRaises(SystemExit):
            rt.parse_temperatures("choice=warm")

    def test_a_temperature_must_be_above_zero(self):
        with self.assertRaises(SystemExit):
            rt.parse_temperatures("choice=0")


class TransformTests(unittest.TestCase):
    def test_undoing_the_served_reading_gives_the_unscaled_one(self):
        closed = {"a": 0.7, "b": 0.3}
        served = rt.scale_closed(closed, 0.62)
        at_one = rt.undo(served, 0.62)
        for key in closed:
            self.assertAlmostEqual(at_one[key], closed[key], places=12)

    def test_the_round_trip_is_the_identity(self):
        closed = {"a": 0.7, "b": 0.3}
        served = rt.scale_closed(closed, 0.62)
        back = rt.scale_closed(rt.undo(served, 0.62), 0.62)
        for key in closed:
            self.assertAlmostEqual(back[key], served[key], places=12)

    def test_undo_gives_the_unscaled_reading(self):
        closed = {"a": 0.6, "b": 0.4}
        served = rt.scale_closed(closed, 0.5)
        at_one = rt.undo(served, 0.5)
        self.assertAlmostEqual(at_one["a"], 0.6, places=12)
        self.assertAlmostEqual(at_one["b"], 0.4, places=12)

    def test_it_is_the_server_expression(self):
        from serve.logits_first_token import apply_temperature
        closed = {"a": 0.7, "b": 0.3}
        reading = {"closed": closed, "open": closed}
        self.assertEqual(rt.scale_closed(closed, 0.77),
                         apply_temperature(reading, 0.77)["closed"])


class CliTests(unittest.TestCase):
    def test_it_writes_the_rescaled_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            items = [{"id": "c/1", "question": {"type": "choice"}, "labels": ["a", "b"],
                      "expected": "a", "tier": "hard"}]
            (tmpdir / "items.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")
            (tmpdir / "preds.jsonl").write_text(
                json.dumps({"id": "c/1", "probs": {"a": 0.6, "b": 0.4}}) + "\n", encoding="utf-8")
            argv = sys.argv
            sys.argv = ["retemperature.py", "--items", str(tmpdir / "items.jsonl"),
                        "--predictions", str(tmpdir / "preds.jsonl"),
                        "--served", "choice=0.62", "--to", "choice=0.5",
                        "--out", str(tmpdir / "out.jsonl")]
            try:
                self.assertEqual(rt.main(), 0)
            finally:
                sys.argv = argv
            row = json.loads((tmpdir / "out.jsonl").read_text(encoding="utf-8").strip())
            # the predictions file holds the closed reading AS SERVED (at 0.62), so
            # the tool undoes that and applies 0.5 to the unscaled reading
            served = {"a": 0.6, "b": 0.4}
            expected = rt.scale_closed(rt.undo(served, 0.62), 0.5)
            self.assertAlmostEqual(row["probs"]["a"], expected["a"], places=12)
            self.assertAlmostEqual(row["probs"]["b"], expected["b"], places=12)

    def test_an_item_the_items_file_does_not_have_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "items.jsonl").write_text("", encoding="utf-8")
            (tmpdir / "preds.jsonl").write_text(
                json.dumps({"id": "c/1", "probs": {"a": 1.0}}) + "\n", encoding="utf-8")
            argv = sys.argv
            sys.argv = ["retemperature.py", "--items", str(tmpdir / "items.jsonl"),
                        "--predictions", str(tmpdir / "preds.jsonl"),
                        "--served", "0.62", "--to", "0.5", "--out", str(tmpdir / "out.jsonl")]
            try:
                with self.assertRaises(SystemExit):
                    rt.main()
            finally:
                sys.argv = argv


@unittest.skipUnless(ITEMS.is_file() and WITH_TEMPERATURE.is_file() and TEMPERATURE_UNDONE.is_file(),
                     "the 235-item OpenDecisions run of 2026-10-21 is not on this machine")
class RegressionTests(unittest.TestCase):
    """The re-scaling must reproduce what the 1.1 measurement computed by hand.

    The 1.1 run has two files: the run as the server served it (with the
    temperature) and the same run with the temperature undone.  The undone arm was
    made offline, by putting the closed reading back through
    normalize(closed_T ** T).  This script must land on the same numbers.
    """

    def test_undoing_the_temperature_gives_the_1_1_undone_arm(self):
        served = {r["id"]: r for r in rt.read_jsonl(WITH_TEMPERATURE)}
        undone = {r["id"]: r for r in rt.read_jsonl(TEMPERATURE_UNDONE)}
        items = {it["id"]: it for it in rt.read_jsonl(ITEMS)}
        worst = 0.0
        for iid, row in served.items():
            task = items[iid]
            probs = row["probs"]
            if task["question"]["type"] == "noul":
                closed = {"yes": float(probs["yes"]), "no": 1.0 - float(probs["yes"])}
            else:
                closed = {lab: float(probs[lab]) for lab in task["labels"]}
            at_one = rt.undo(closed, {"choice": 0.62, "noul": 0.67, "score": 0.77}[task["question"]["type"]])
            for key, value in at_one.items():
                if task["question"]["type"] == "noul":
                    expected = undone[iid]["probs"]["yes"] if key == "yes" else 1.0 - undone[iid]["probs"]["yes"]
                else:
                    expected = float(undone[iid]["probs"][key])
                worst = max(worst, abs(expected - value))
        self.assertLess(worst, 1e-15)


if __name__ == "__main__":
    unittest.main()
