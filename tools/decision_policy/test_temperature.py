"""tools/decision_policy/test_temperature.py - unit tests for the temperature fit.

    python -m unittest tools.decision_policy.test_temperature -v

The tests do not touch the network, do not need a GPU and do not need the engine.
Most of them are hermetic (no data files).  Two need the artifacts that live
outside the repository, and they say so when the artifacts are not there:

  - the CLI test needs the official scorer (score.py);
  - the regression test needs the 235-item OpenDecisions run of 2026-10-21
    (the items, the run log and the predictions), under
    $STRATA_OPENDECISIONS_DIR (the default is the machine's own copy).

The tests that need the official scorer use it where the dataset keeps it (the
same path the fitting script defaults to); they skip themselves when it is not
there, so the suite still runs on a machine that has not fetched the dataset.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.decision_policy import temperature as tp  # noqa: E402
from serve.logits_first_token import apply_temperature as server_apply_temperature  # noqa: E402

SCORER = tp.DEFAULT_SCORER
DATA_DIR = Path(os.environ.get("STRATA_OPENDECISIONS_DIR",
                               str(Path(tempfile.gettempdir()) / "opencode" / "opendecisions")))
ITEMS = DATA_DIR / "sample10_items.jsonl"
RUNLOG = DATA_DIR / "fix_serial.runlog.jsonl"
PREDICTIONS = DATA_DIR / "fix_serial.jsonl"


def scorer_or_skip():
    if not SCORER.exists():
        raise unittest.SkipTest(f"the official scorer is not at {SCORER}")
    return tp.load_scorer(SCORER)


# ------------------------------------------------------------------ the fixtures

def make_item(iid: str, qtype: str, labels: list[str], expected, tier: str = "hard",
              target: dict | None = None) -> dict:
    item = {"id": iid, "question": {"type": qtype, "instructions": "which one?"},
            "labels": labels, "expected": expected, "tier": tier, "state": "a state"}
    if target is not None:
        item["target_probs"] = target
    return item


def make_record(iid: str, qtype: str, closed: dict[str, float], gold: str,
                tier: str = "hard", target: dict | None = None,
                labels: list[str] | None = None) -> dict:
    """A record, the shape build_records() produces."""
    return {"id": iid, "type": qtype, "tier": tier,
            "labels": list(labels if labels is not None else closed),
            "closed": dict(closed), "gold": gold, "target": target,
            "fast_path": True, "task": {"id": iid, "question": {"type": qtype},
                                        "labels": list(closed), "expected": gold, "tier": tier}}


def synthetic_pool() -> list[dict]:
    """A small pool: three types, ten items each, one of them with a human
    distribution."""
    records = []
    for i in range(10):
        p = 0.2 + 0.06 * i
        records.append(make_record("c/%02d" % i, "choice", {"a": p, "b": 1 - p}, "a",
                                   tier="hard" if i % 2 else "easy"))
        records.append(make_record("n/%02d" % i, "noul", {"yes": p, "no": 1 - p},
                                   "yes" if p > 0.5 else "no",
                                   tier="hard" if i % 2 else "easy"))
        records.append(make_record("s/%02d" % i, "score", {"0": p, "1": 1 - p}, "0",
                                   tier="hard" if i % 2 else "easy",
                                   labels=["0", "1"]))
    records[0]["target"] = {"a": 0.5, "b": 0.5}
    return records


# ------------------------------------------------------------------ the transform

class ScaleTests(unittest.TestCase):
    """The identity the server applies: normalize(closed ** (1/T))."""

    def test_it_is_the_server_expression(self):
        closed = {"a": 0.6, "b": 0.4}
        reading = {"closed": closed, "open": closed}
        for t in (0.5, 0.77, 1.0, 2.5, 10.0):
            self.assertEqual(tp.scale(closed, t), server_apply_temperature(reading, t)["closed"])

    def test_the_known_value(self):
        # normalize(0.6 ** 2, 0.4 ** 2)
        scaled = tp.scale({"a": 0.6, "b": 0.4}, 0.5)
        self.assertAlmostEqual(scaled["a"], 0.6923076923076923, places=12)
        self.assertAlmostEqual(scaled["b"], 0.30769230769230776, places=12)
        self.assertAlmostEqual(sum(scaled.values()), 1.0, places=12)

    def test_one_is_the_identity(self):
        closed = {"a": 0.6, "b": 0.3, "c": 0.1}
        self.assertEqual(tp.scale(closed, 1.0), closed)

    def test_the_weights_are_raised_to_one_over_T(self):
        closed = {"a": 0.6, "b": 0.3, "c": 0.1}
        scaled = tp.scale(closed, 2.0)
        weights = {k: v ** 0.5 for k, v in closed.items()}
        total = sum(weights.values())
        for k, v in closed.items():
            self.assertAlmostEqual(scaled[k], weights[k] / total, places=12)
        self.assertAlmostEqual(sum(scaled.values()), 1.0, places=12)

    def test_a_zero_stays_zero(self):
        scaled = tp.scale({"a": 1.0, "b": 0.0}, 0.5)
        self.assertEqual(scaled["b"], 0.0)
        self.assertAlmostEqual(scaled["a"], 1.0, places=12)

    def test_the_pick_cannot_change(self):
        closed = {"a": 0.6, "b": 0.4}
        for t in (0.1, 0.5, 1.0, 2.0, 10.0, 1000.0):
            scaled = tp.scale(closed, t)
            self.assertEqual(max(scaled, key=scaled.get), "a")

    def test_a_smaller_temperature_sharpens(self):
        closed = {"a": 0.6, "b": 0.4}
        self.assertGreater(tp.scale(closed, 0.5)["a"], closed["a"])
        self.assertLess(tp.scale(closed, 2.0)["a"], closed["a"])

    def test_a_larger_temperature_flattens(self):
        closed = {"a": 0.9, "b": 0.1}
        self.assertLess(tp.scale(closed, 4.0)["a"], tp.scale(closed, 1.0)["a"])
        self.assertGreater(tp.scale(closed, 0.5)["a"], tp.scale(closed, 1.0)["a"])


class UndoTemperatureTests(unittest.TestCase):
    """The paired control: the same run with the temperature's arithmetic removed."""

    def test_it_inverts_the_scale(self):
        closed = {"a": 0.6, "b": 0.25, "c": 0.15}
        for t in (0.5, 0.62, 0.77, 2.0, 10.0):
            back = tp.undo_temperature(tp.scale(closed, t), t)
            for k, v in closed.items():
                self.assertAlmostEqual(back[k], v, places=12)

    def test_one_is_the_identity(self):
        closed = {"a": 0.6, "b": 0.4}
        self.assertEqual(tp.undo_temperature(closed, 1.0), closed)

    def test_it_is_the_server_expression(self):
        # the inverse of "apply T" is "apply 1/T", so the server's own function
        # does the work: there is one place in the tools that knows the transform
        closed = {"a": 0.6, "b": 0.4}
        reading = {"closed": closed, "open": closed}
        self.assertEqual(tp.undo_temperature(closed, 0.62),
                         server_apply_temperature(reading, 1.0 / 0.62)["closed"])


# ------------------------------------------------------------------ the fit

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
        t, value = tp.fit_temperature(None, self._records(), "choice", "brier")
        self.assertLess(t, 1.0)
        self.assertLess(value, tp.mean_brier(self._records(), 1.0))

    def test_the_fit_is_deterministic(self):
        recs = self._records()
        first = tp.fit_temperature(None, recs, "choice", "brier")
        second = tp.fit_temperature(None, recs, "choice", "brier")
        self.assertEqual(first, second)

    def test_the_grid_is_respected(self):
        recs = self._records()
        t, _ = tp.fit_temperature(None, recs, "choice", "brier", 0.9, 1.5)
        self.assertGreaterEqual(t, 0.9)
        self.assertLessEqual(t, 1.5)


# ------------------------------------------------------------------ the human distribution

class HumanTargetTests(unittest.TestCase):
    def test_a_choice_item(self):
        task = make_item("c/1", "choice", ["a", "b"], "a", target={"a": 0.7, "b": 0.3})
        target, problem = tp.human_target(task)
        self.assertIsNone(problem)
        self.assertEqual(target, {"a": 0.7, "b": 0.3})

    def test_a_noul_that_gives_only_the_yes_share(self):
        task = make_item("n/1", "noul", ["no", "yes"], "yes", target={"yes": 0.25})
        target, problem = tp.human_target(task)
        self.assertIsNone(problem)
        self.assertEqual(target, {"yes": 0.25, "no": 0.75})

    def test_a_distribution_over_some_of_the_labels_only(self):
        # a level the people never chose has a target mass of 0
        task = make_item("s/1", "score", ["0", "1", "2", "3"], 0,
                         target={"0": 0.78, "1": 0.21, "2": 0.01})
        target, problem = tp.human_target(task)
        self.assertIsNone(problem)
        self.assertEqual(target, {"0": 0.78, "1": 0.21, "2": 0.01})

    def test_a_key_that_is_not_a_label_is_a_problem(self):
        task = make_item("c/1", "choice", ["a", "b"], "a", target={"a": 0.7, "b": 0.3, "z": 0.0})
        target, problem = tp.human_target(task)
        self.assertIsNone(target)
        self.assertIn("not labels", problem)

    def test_no_distribution(self):
        target, problem = tp.human_target(make_item("c/1", "choice", ["a", "b"], "a"))
        self.assertIsNone(target)
        self.assertIsNone(problem)


# ------------------------------------------------------------------ the soft objective

class SoftObjectiveTests(unittest.TestCase):
    def test_at_one_it_is_the_cross_entropy_of_the_human_distribution(self):
        closed = {"a": 0.6, "b": 0.4}
        target = {"a": 0.5, "b": 0.5}
        expected = -(0.5 * math.log(0.6) + 0.5 * math.log(0.4))
        self.assertAlmostEqual(tp.soft_ce(closed, target, 1.0), expected, places=12)

    def test_it_is_minimized_where_the_model_agrees_with_the_people(self):
        closed = {"a": 0.6, "b": 0.4}
        record = make_record("c/1", "choice", closed, "a", target=dict(closed))
        best, value = tp.fit_temperature(None, [record], "choice", "soft")
        self.assertEqual(best, 1.0)
        # the cross-entropy at the minimum is the entropy of the human distribution
        entropy = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        self.assertAlmostEqual(value, entropy, places=9)
        # and it is lower than anywhere else on the grid
        for t in (0.5, 0.8, 1.2, 2.0, 5.0):
            self.assertGreater(tp.soft_ce(closed, closed, t), value)

    def test_a_flatter_human_distribution_wants_a_bigger_temperature(self):
        # the model says 0.9, the people were split 50/50: flattening pays
        closed = {"a": 0.9, "b": 0.1}
        record = make_record("c/1", "choice", closed, "a", target={"a": 0.5, "b": 0.5})
        best, _ = tp.fit_temperature(None, [record], "choice", "soft")
        self.assertGreater(best, 1.0)

    def test_the_items_without_a_distribution_do_not_count(self):
        with_target = make_record("c/1", "choice", {"a": 0.6, "b": 0.4}, "a", target={"a": 0.5, "b": 0.5})
        without = make_record("c/2", "choice", {"a": 0.9, "b": 0.1}, "a")
        value = tp.mean_soft_ce([with_target, without], 1.0)
        self.assertAlmostEqual(value, tp.soft_ce(with_target["closed"], with_target["target"], 1.0),
                               places=12)

    def test_no_item_with_a_distribution_is_infinite(self):
        self.assertEqual(tp.mean_soft_ce([make_record("c/1", "choice", {"a": 1.0}, "a")], 1.0),
                         float("inf"))


# ------------------------------------------------------------------ no leakage

class NoLeakTests(unittest.TestCase):
    """The temperature that scores an item must not have seen that item."""

    def test_the_fold_temperature_is_the_one_fitted_without_the_fold(self):
        records = synthetic_pool()
        per_item, _table = tp.cross_validate(None, records, "brier")
        folds = tp.stratified_folds(records)
        for f in range(tp.FOLDS):
            train = [r for g in range(tp.FOLDS) if g != f for r in folds[g]]
            expected = tp.fold_temperatures(None, train, "brier")
            for rec in folds[f]:
                self.assertEqual(per_item[rec["id"]], expected[rec["type"]])

    def test_the_held_items_cannot_move_their_own_temperature(self):
        records = synthetic_pool()
        before, _ = tp.cross_validate(None, records, "brier")
        folds = tp.stratified_folds(records)
        for f in range(tp.FOLDS):
            held = {r["id"] for r in folds[f]}
            # the fold's own items get a flat distribution: a temperature fitted
            # WITH them would move, and the one fitted WITHOUT them must not
            changed = []
            for rec in records:
                copy = dict(rec)
                if rec["id"] in held:
                    copy["closed"] = {k: 1.0 / len(rec["closed"]) for k in rec["closed"]}
                changed.append(copy)
            after, _ = tp.cross_validate(None, changed, "brier")
            for rec in folds[f]:
                self.assertEqual(after[rec["id"]], before[rec["id"]])

    def test_the_folds_do_not_overlap(self):
        records = synthetic_pool()
        folds = tp.stratified_folds(records)
        seen = set()
        for f in range(tp.FOLDS):
            ids = {r["id"] for r in folds[f]}
            self.assertFalse(seen & ids)
            seen |= ids
        self.assertEqual(len(seen), len(records))

    def test_every_item_is_in_exactly_one_fold(self):
        records = synthetic_pool()
        per_item, table = tp.cross_validate(None, records, "brier")
        self.assertEqual(len(per_item), len(records))
        self.assertEqual(sum(row["n"] for row in table), len(records))


# ------------------------------------------------------------------ the single scalar

class SingleScalarTests(unittest.TestCase):
    """The arm the roadmap asks for: if per-type does not beat ONE scalar, say so."""

    def test_it_is_one_number_for_every_type(self):
        records = synthetic_pool()
        t, _ = tp.fit_single_scalar(None, records, "brier")
        self.assertGreater(t, 0.0)
        # the same records, per type, give three numbers
        per_type = {q: tp.fit_temperature(None, [r for r in records if r["type"] == q], q, "brier")[0]
                    for q in tp.TYPES}
        self.assertEqual(len(per_type), 3)

    def test_the_out_of_fold_scalar_is_leak_free(self):
        records = synthetic_pool()
        before, table = tp.cross_validate_single(None, records, "brier")
        folds = tp.stratified_folds(records)
        for f in range(tp.FOLDS):
            train = [r for g in range(tp.FOLDS) if g != f for r in folds[g]]
            expected, _ = tp.fit_single_scalar(None, train, "brier")
            for rec in folds[f]:
                self.assertEqual(before[rec["id"]], expected)
            # the fold's own items get a flat distribution: the scalar fitted
            # WITHOUT them must not move
            held = {r["id"] for r in folds[f]}
            changed = []
            for rec in records:
                copy = dict(rec)
                if rec["id"] in held:
                    copy["closed"] = {k: 1.0 / len(rec["closed"]) for k in rec["closed"]}
                changed.append(copy)
            after, _ = tp.cross_validate_single(None, changed, "brier")
            for rec in folds[f]:
                self.assertEqual(after[rec["id"]], before[rec["id"]])

    def test_every_item_is_in_exactly_one_fold(self):
        records = synthetic_pool()
        per_item, table = tp.cross_validate_single(None, records, "brier")
        self.assertEqual(len(per_item), len(records))
        self.assertEqual(sum(row["n"] for row in table), len(records))


# ------------------------------------------------------------------ the uncertainty

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
        sharp = {it["id"]: tp.scale(it["closed"], 0.5) for it in items}
        flat = {it["id"]: it["closed"] for it in items}
        out = tp.paired_bootstrap(scorer, items, flat, sharp, resamples=50, seed=42)
        self.assertIn("capability", out)
        self.assertEqual(out["capability"]["share_better"], 1.0)
        self.assertGreater(out["capability"]["mean"], 0.0)

    def test_the_paired_bootstrap_sees_no_difference(self):
        scorer, items = self._scorer_and_items()
        same = {it["id"]: it["closed"] for it in items}
        out = tp.paired_bootstrap(scorer, items, same, same, resamples=20, seed=42)
        self.assertAlmostEqual(out["capability"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(out["capability"]["p025"], 0.0, places=9)
        self.assertAlmostEqual(out["capability"]["p975"], 0.0, places=9)

    def test_the_out_of_bag_bootstrap_is_deterministic(self):
        scorer, items = self._scorer_and_items()
        choice = [it for it in items if it["type"] == "choice"]
        first = tp.bootstrap_ece(scorer, choice, "choice", "brier", resamples=20, seed=42)
        second = tp.bootstrap_ece(scorer, choice, "choice", "brier", resamples=20, seed=42)
        self.assertEqual(first, second)
        self.assertIn("method", first)


# ------------------------------------------------------------------ the CLI

def write_pool(directory: Path) -> tuple[Path, Path, Path]:
    """A small items/runlog/predictions trio, in the harness's own shapes."""
    items, predictions, runlog = [], [], []
    for i in range(12):
        p = 0.25 + 0.04 * i
        iid = "c/%02d" % i
        items.append(make_item(iid, "choice", ["a", "b"], "a", tier="hard" if i % 2 else "easy",
                               target={"a": 0.5, "b": 0.5} if i == 0 else None))
        predictions.append({"id": iid, "probs": {"a": p, "b": 1 - p}})
        runlog.append({"id": iid, "type": "choice", "tier": "hard" if i % 2 else "easy",
                       "fast_path": True, "probabilities": {"a": p, "b": 1 - p},
                       "open_probs": {"a": 0.9 * p, "b": 0.9 * (1 - p)}})
        iid = "n/%02d" % i
        items.append(make_item(iid, "noul", ["no", "yes"], "yes" if p > 0.5 else "no",
                               tier="hard" if i % 2 else "easy"))
        predictions.append({"id": iid, "probs": {"yes": p}})
        runlog.append({"id": iid, "type": "noul", "tier": "hard" if i % 2 else "easy",
                       "fast_path": True, "probabilities": {"yes": p},
                       "open_probs": {"yes": 0.9 * p, "no": 0.9 * (1 - p)}})
        iid = "s/%02d" % i
        items.append(make_item(iid, "score", ["0", "1"], 0, tier="hard" if i % 2 else "easy"))
        predictions.append({"id": iid, "probs": {"0": p, "1": 1 - p}})
        runlog.append({"id": iid, "type": "score", "tier": "hard" if i % 2 else "easy",
                       "fast_path": True, "probabilities": {"0": p, "1": 1 - p},
                       "open_probs": {"0": 0.9 * p, "1": 0.9 * (1 - p)}})
    paths = []
    for name, rows in (("items.jsonl", items), ("predictions.jsonl", predictions),
                       ("runlog.jsonl", runlog)):
        path = directory / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        paths.append(path)
    return paths[0], paths[2], paths[1]


class CliTests(unittest.TestCase):
    @unittest.skipUnless(SCORER.is_file(), "the official score.py is not here")
    def test_the_fit_writes_the_constants_and_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            items, runlog, predictions = write_pool(tmpdir)
            out_json = tmpdir / "calibration.json"
            out_report = tmpdir / "report.md"
            argv = sys.argv
            sys.argv = ["temperature.py", "--items", str(items), "--runlog", str(runlog),
                        "--predictions", str(predictions), "--scorer", str(SCORER),
                        "--out-json", str(out_json), "--out-report", str(out_report),
                        "--bootstrap", "0", "--date", "2026-01-01"]
            try:
                code = tp.main()
            finally:
                sys.argv = argv
            self.assertEqual(code, 0)
            result = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(set(result["temperature"]), {"choice", "noul", "score"})
            self.assertEqual(set(result["temperature_by_objective"]),
                             {"nll", "brier", "ece", "soft"})
            self.assertEqual(result["objective"], "brier")
            self.assertEqual(result["seed"], 42)
            self.assertEqual(result["folds"], 5)
            # one choice item of the twelve carries a human distribution
            self.assertEqual(result["items_with_human_distribution"]["choice"], 1)
            self.assertEqual(result["items_with_human_distribution"]["noul"], 0)
            self.assertEqual(result["items_with_human_distribution"]["score"], 0)
            # the provenance block names the files it read
            self.assertEqual(result["source"]["files"]["items"]["path"], str(items))
            self.assertEqual(len(result["source"]["files"]["items"]["sha256"]), 64)
            # the out-of-fold numbers are in there
            self.assertIn("brier", result["validation"]["out_of_fold"])
            self.assertIn("calibration_after", result["validation"]["out_of_fold"]["brier"])
            self.assertTrue(out_report.is_file())
            report = out_report.read_text(encoding="utf-8")
            self.assertIn("# The per-type temperature: the fit", report)
            self.assertIn("| brier | the multiclass Brier score |", report)

    def test_the_objective_choices(self):
        # the CLI rejects an objective that does not exist
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            items, runlog, predictions = write_pool(tmpdir)
            argv = sys.argv
            sys.argv = ["temperature.py", "--items", str(items), "--runlog", str(runlog),
                        "--predictions", str(predictions), "--objective", "nonsense"]
            try:
                with self.assertRaises(SystemExit):
                    tp.main()
            finally:
                sys.argv = argv


# ------------------------------------------------------------------ the regression

@unittest.skipUnless(ITEMS.is_file() and RUNLOG.is_file() and PREDICTIONS.is_file(),
                     "the 235-item OpenDecisions run is not on this machine")
class RegressionTests(unittest.TestCase):
    """The refit must reproduce the constants of 2026-10-21 exactly.

    They are the ones that were served with the 235-item sample: choice 0.62,
    noul 0.67, score 0.77 (the Brier objective).  If a change to this script
    moves them, the served numbers are no longer the ones that were measured.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(cls.tmp.name)
        cls.out_json = tmpdir / "calibration.json"
        argv = sys.argv
        sys.argv = ["temperature.py", "--items", str(ITEMS), "--runlog", str(RUNLOG),
                    "--predictions", str(PREDICTIONS), "--scorer", str(SCORER),
                    "--out-json", str(cls.out_json),
                    "--out-report", str(tmpdir / "report.md"), "--bootstrap", "0"]
        try:
            tp.main()
        finally:
            sys.argv = argv
        cls.result = json.loads(cls.out_json.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_shipped_constants(self):
        self.assertEqual(self.result["temperature"], {"choice": 0.62, "noul": 0.67, "score": 0.77})

    def test_the_other_objectives(self):
        by_objective = self.result["temperature_by_objective"]
        self.assertEqual(by_objective["nll"], {"choice": 1.16, "noul": 1.57, "score": 0.94})
        self.assertEqual(by_objective["ece"], {"choice": 0.58, "noul": 0.67, "score": 0.75})

    def test_the_out_of_fold_ece_table(self):
        brier = self.result["validation"]["out_of_fold"]["brier"]
        self.assertAlmostEqual(brier["ece_before"]["choice"], 0.13127217543588357, places=9)
        self.assertAlmostEqual(brier["ece_before"]["noul"], 0.07561924931885602, places=9)
        self.assertAlmostEqual(brier["ece_before"]["score"], 0.10818720700054921, places=9)
        self.assertAlmostEqual(brier["ece_after"]["choice"], 0.1224, places=4)
        self.assertAlmostEqual(brier["ece_after"]["noul"], 0.0786, places=4)
        self.assertAlmostEqual(brier["ece_after"]["score"], 0.0649, places=4)

    def test_the_official_scorer_numbers(self):
        brier = self.result["validation"]["out_of_fold"]["brier"]["scorer"]
        self.assertAlmostEqual(brier["capability"], 81.47, places=1)
        self.assertAlmostEqual(brier["intelligence"], 79.43, places=1)
        self.assertAlmostEqual(brier["calibration"], 83.52, places=1)
        base = self.result["validation"]["official_scorer_baseline"]
        self.assertAlmostEqual(base["capability"], 79.83, places=1)

    def test_the_human_distribution_counts(self):
        counts = self.result["items_with_human_distribution"]
        self.assertEqual(counts, {"choice": 3, "noul": 6, "score": 4})


if __name__ == "__main__":
    unittest.main()
