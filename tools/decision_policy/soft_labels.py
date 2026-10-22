#!/usr/bin/env python3
"""tools/decision_policy/soft_labels.py - the soft-label data in the harness's shape.

Praveenrajus/jev-bench (HuggingFace) is 22 configs of System One questions: a
state, a typed question (choice / noul / score) and a gold answer. Four configs
carry a HUMAN LABEL DISTRIBUTION - the "calibration gold" ones: go_emotions,
measuring_hate_speech, civil_comments and chaosnli. Three of them publish a
train split and a validation split. That human distribution is the scarce
resource for the decision policy: self-distillation can only copy the model's
own thinking-mode distribution, so an over-confident mode is inherited.

This script turns a downloaded jev-bench split into items the harness can run
and the official scorer can read:

    {"id", "state", "question": {"type", "instructions", "criteria"},
     "labels", "expected", "tier", "target_probs"}

The rules the official scorer enforces (see score.py):
  - the probabilities over `labels` must sum to 1 within 0.01;
  - a noul item's `expected` is the string "yes" or "no" and its `target_probs`
    is {"yes": p, "no": 1 - p};
  - a score item's `expected` is the INT level index and its `target_probs` is
    keyed "0".."k-1";
  - `tier` is one of easy | standard | judge | hard;
  - `target_probs` keys match `labels`.

The row -> item conversion is NOT reinvented here: jevbench_sample.py already
does it (to_item, and the soft_label handling - a dict over the option names
for a choice, a list over the level indices for a score, a single number, the
share of the annotators who said yes, for a noul). This script reuses it and
adds the checks, the stratified sample and the scorer proof.

    python tools/decision_policy/soft_labels.py --data-dir <D> --out items.jsonl
    python tools/decision_policy/soft_labels.py --data-dir <D> --sample --out sample.jsonl
    python tools/decision_policy/soft_labels.py --data-dir <D> --validate --out items.jsonl
    python tools/decision_policy/soft_labels.py --data-dir <D> \
        --plan tools/decision_policy/soft_label_pool.json --out pool.jsonl

THE POOL PLAN.  A plan file (soft_label_pool.json, tracked in the repository) names
the per-config counts and the seed, so the fitting pool is reproducible: the same
plan, the same data and the same converter always give the same items file.  The
plan also carries the licence verdict of every config, taken from the licence
record (docs/research/decision_policy_training_data.md).  The data itself stays
outside the repository.

Pure standard library. Nothing is written inside the repository.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jevbench_sample                                       # noqa: E402  (same folder)

TIERS = ("easy", "standard", "judge", "hard")
TYPES = ("choice", "noul", "score")
SPLITS = ("train", "validation", "test")

# The official scorer, if it is not given with --scorer.
SCORER_CANDIDATES = (
    os.environ.get("OPENDECISIONS_SCORER", ""),
    r"C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\score.py",
    str(Path(__file__).resolve().parent / "score.py"),
)

# The tier jev-bench items are filed under. The dataset carries no difficulty
# label, so there is no honest mapping onto the four JevBench tiers; "hard" for
# every item collapses the tier weights to one (the same choice jevbench_sample.py
# makes) and puts every choice item inside the tier the scorer measures the
# choice calibration on.
DEFAULT_TIER = "hard"


# ------------------------------------------------------------------ reading

def read_rows(path: Path) -> list[dict]:
    """The rows of a jev-bench split file."""
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def summarize(rows: list[dict]) -> tuple[dict[str, int], int]:
    """(the type mix, how many rows carry a soft label)."""
    per_type: dict[str, int] = {}
    with_soft = 0
    for row in rows:
        try:
            qtype = json.loads(row["question"])["type"]
        except (KeyError, TypeError, ValueError):
            qtype = "unknown"
        per_type[qtype] = per_type.get(qtype, 0) + 1
        if row.get("soft_label") is not None:
            with_soft += 1
    return per_type, with_soft


def find_splits(data_dir: Path, configs: list[str], splits: list[str]) -> list[tuple[str, str, Path]]:
    """The (config, split, path) triples that exist on disk."""
    out = []
    for cfg in configs:
        for split in splits:
            path = data_dir / cfg / ("%s.jsonl" % split)
            if path.exists():
                out.append((cfg, split, path))
    return out


# ------------------------------------------------------------------ the item

def to_item(row: dict, tier: str = DEFAULT_TIER) -> dict:
    """One jev-bench row -> one item of the harness's shape.

    The conversion itself is jevbench_sample.to_item (the row's `question` is
    already the typed question object, `labels` follows the scorer, and the
    `soft_label` is turned into `target_probs` in the scorer's label space).

    A row that cannot be converted (a choice row without criteria, say) raises
    a ValueError naming the row, so the caller can report it instead of
    losing the run to a traceback.
    """
    try:
        return jevbench_sample.to_item(row, tier)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("row %r: %s" % (row.get("id"), exc)) from exc


def check_item(item: dict) -> list[str]:
    """The invariants the official scorer enforces, checked here.

    Returns the list of problems; an empty list means the item is in shape.
    """
    problems: list[str] = []
    for key in ("id", "state", "question", "labels", "expected", "tier"):
        if key not in item:
            problems.append("missing %r" % key)
    if problems:
        return problems
    q = item["question"]
    if not isinstance(q, dict):
        return ["question is not an object"]
    qtype = q.get("type")
    if qtype not in TYPES:
        problems.append("question type %r is not one of %s" % (qtype, ", ".join(TYPES)))
    if "instructions" not in q:
        problems.append("question has no instructions")
    # a noul question may leave the criteria out (the endpoint then asks the
    # plain yes/no); a choice or score question needs them.
    if "criteria" not in q and qtype != "noul":
        problems.append("question has no criteria")
    labels = item["labels"]
    if not isinstance(labels, list) or not labels:
        problems.append("labels is not a non-empty list")
        return problems
    if len(set(labels)) != len(labels):
        problems.append("labels are not unique")
    if item["tier"] not in TIERS:
        problems.append("tier %r is not one of %s" % (item["tier"], ", ".join(TIERS)))
    expected = item["expected"]
    if qtype == "choice":
        if expected not in labels:
            problems.append("expected %r is not one of the labels" % (expected,))
    elif qtype == "noul":
        if expected not in ("yes", "no"):
            problems.append("noul expected %r is not 'yes' or 'no'" % (expected,))
    elif qtype == "score":
        if not isinstance(expected, int) or isinstance(expected, bool):
            problems.append("score expected %r is not an int" % (expected,))
        elif not 0 <= expected < len(labels):
            problems.append("score expected %r is outside 0..%d" % (expected, len(labels) - 1))
    target = item.get("target_probs")
    if target is not None:
        if not isinstance(target, dict):
            problems.append("target_probs is not an object")
        else:
            if set(target) != set(labels):
                problems.append("target_probs keys %s do not match the labels %s"
                                % (sorted(target), sorted(labels)))
            total = 0.0
            for key, value in target.items():
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    problems.append("target_probs[%r] is not a number" % (key,))
                    continue
                if f < 0.0 or f != f or f in (float("inf"), float("-inf")):
                    problems.append("target_probs[%r] = %r is not a probability" % (key, value))
                total += f
            if abs(total - 1.0) > 0.01:
                problems.append("target_probs sum to %.4f, not 1" % total)
    return problems


def null_prediction(item: dict) -> dict:
    """The uniform distribution over the item's labels.

    This is the "null prediction" the --validate mode feeds the scorer: it
    exercises the scorer's own normalization (which raises when the labels and
    the probabilities do not fit each other) on every item.
    """
    labels = item["labels"]
    p = 1.0 / len(labels)
    return {lab: p for lab in labels}


# ------------------------------------------------------------------ sampling

def stratified_sample(items: list[dict], per_type: int, seed: int) -> list[dict]:
    """A deterministic, stratified sample of the pool: `per_type` items of every
    question type.

    THE RULE. The pool is sorted by (type, id), so its order does not depend on
    the order the files were read in. One seeded generator (random.Random(seed))
    draws the sample of every type, the types in the fixed order choice, noul,
    score. The same pool and the same seed always give the same sample.
    """
    pool: dict[str, list[dict]] = {t: [] for t in TYPES}
    for item in items:
        pool.setdefault(item["question"]["type"], []).append(item)
    rng = random.Random(seed)
    picked: list[dict] = []
    for qtype in TYPES:
        group = sorted(pool.get(qtype, []), key=lambda it: str(it["id"]))
        if not group:
            continue
        take = min(per_type, len(group))
        chosen = rng.sample(group, take)
        picked.extend(sorted(chosen, key=lambda it: str(it["id"])))
    return picked


# ------------------------------------------------------------------ the pool plan

DEFAULT_PLAN = Path(__file__).resolve().parent / "soft_label_pool.json"


def load_plan(path: Path) -> dict:
    """The pool plan: the per-config counts, the seed and the licence verdicts."""
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit("cannot read the pool plan %s: %s" % (path, exc))
    for key in ("pool", "seed"):
        if key not in plan:
            raise SystemExit("the pool plan %s has no %r" % (path, key))
    return plan


def plan_sample(items: list[dict], plan: dict) -> list[dict]:
    """The pool the plan names: per question type, per config, the plan's count.

    THE RULE. The types are taken in the fixed order choice, noul, score; inside a
    type the configs are taken in sorted order, so the plan's own key order does
    not change the draw. One seeded generator (random.Random(plan["seed"])) draws
    every config's share, in that order, from the config's items sorted by id.
    The same plan, the same data and the same seed always give the same pool.

    A config the plan names but the data does not have, or a count the config's
    split cannot fill, is an error: the pool would silently be smaller than the
    plan says, and a temperature fitted on it would be fitted on the wrong data.
    """
    rng = random.Random(int(plan["seed"]))
    pool = plan.get("pool") or {}
    unknown = set(pool) - set(TYPES)
    if unknown:
        raise SystemExit("the plan names the question types %s, which do not exist"
                         % ", ".join(sorted(unknown)))
    picked: list[dict] = []
    for qtype in TYPES:
        counts = pool.get(qtype) or {}
        for config in sorted(counts):
            want = int(counts[config])
            group = sorted((it for it in items
                            if it["question"]["type"] == qtype and it.get("config") == config),
                           key=lambda it: str(it["id"]))
            if not group:
                raise SystemExit("the plan wants %d %s items of %s, but the data has none"
                                 % (want, qtype, config))
            if want > len(group):
                raise SystemExit("the plan wants %d %s items of %s, but the split has %d"
                                 % (want, qtype, config, len(group)))
            chosen = rng.sample(group, want)
            picked.extend(sorted(chosen, key=lambda it: str(it["id"])))
    return picked


def plan_counts(items: list[dict]) -> dict[str, dict[str, int]]:
    """-> {type: {config: how many items}}, for the report."""
    out: dict[str, dict[str, int]] = {}
    for item in items:
        qtype = item["question"]["type"]
        config = item.get("config") or "unknown"
        out.setdefault(qtype, {}).setdefault(config, 0)
        out[qtype][config] += 1
    return {t: dict(sorted(v.items())) for t, v in sorted(out.items())}


# ------------------------------------------------------------------ the scorer

def find_scorer(given: str | None) -> Path | None:
    """The official score.py, if it can be found."""
    for candidate in ([given] if given else []) + list(SCORER_CANDIDATES):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def run_scorer(scorer: Path, items_path: Path, predictions_path: Path) -> tuple[int, str]:
    """The official scorer over the items file, as a subprocess.

    The scorer reads the items with its own read_jsonl and writes the result;
    a non-zero exit means it raised.
    """
    cmd = [sys.executable, str(scorer), str(predictions_path), "--data", str(items_path), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ------------------------------------------------------------------ the CLI

def split_list(values: list[str] | None) -> list[str]:
    """A repeatable flag that also takes a comma-separated list."""
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="the data root: <config>/<split>.jsonl")
    ap.add_argument("--config", action="append", default=None,
                    help="a config to read; repeatable or comma-separated. Default: every config "
                         "found under the data dir")
    ap.add_argument("--split", action="append", default=None,
                    help="a split to read: train | validation | test. Repeatable. Default: the plan's "
                         "split, or validation")
    ap.add_argument("--type", action="append", default=None,
                    help="keep only this question type: choice | noul | score. Repeatable")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of items emitted (0 = no cap)")
    ap.add_argument("--per-type", type=int, default=200,
                    help="the stratified sample quota per question type (--sample)")
    ap.add_argument("--seed", type=int, default=None,
                    help="the sampling seed. Default: the plan's seed, or 20261020")
    ap.add_argument("--plan", type=Path, default=None,
                    help="the pool plan: the per-config counts, the seed and the licence verdicts. "
                         "With it, the pool is the plan's, per config (it implies --sample)")
    ap.add_argument("--tier", default=DEFAULT_TIER, choices=TIERS,
                    help="the tier every item is filed under (jev-bench has no difficulty label; "
                         "the default puts every item in the tier the scorer measures choice "
                         "calibration on)")
    ap.add_argument("--out", type=Path, default=None,
                    help="the items file to write. Default: standard output")
    ap.add_argument("--sample", action="store_true",
                    help="emit the stratified, seeded sample of the pool instead of the whole pool")
    ap.add_argument("--validate", action="store_true",
                    help="check every item and prove the official scorer parses them")
    ap.add_argument("--scorer", default=None, help="the official score.py to use in --validate")
    args = ap.parse_args()

    plan = load_plan(args.plan) if args.plan else None
    seed = args.seed if args.seed is not None else (int(plan["seed"]) if plan else 20261020)
    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit("no such data dir: %s" % data_dir)
    splits = split_list(args.split) or ([plan["split"]] if plan and plan.get("split") else ["validation"])
    for split in splits:
        if split not in SPLITS:
            raise SystemExit("split %r is not one of %s" % (split, ", ".join(SPLITS)))
    types = split_list(args.type)
    for qtype in types:
        if qtype not in TYPES:
            raise SystemExit("type %r is not one of %s" % (qtype, ", ".join(TYPES)))

    configs = split_list(args.config)
    if not configs:
        configs = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    found = find_splits(data_dir, configs, splits)
    if not found:
        raise SystemExit("no %s split found under %s for %s" % ("/".join(splits), data_dir, ", ".join(configs)))

    # ---- the pool
    pool: list[dict] = []
    broken: list[str] = []
    per_file = []
    for cfg, split, path in found:
        rows = read_rows(path)
        items = []
        for row in rows:
            try:
                items.append(to_item(row, args.tier))
            except ValueError as exc:
                broken.append("%s: %s" % (path.name, exc))
        if types:
            items = [it for it in items if it["question"]["type"] in types]
        pool.extend(items)
        per_type, with_soft = summarize(rows)
        per_file.append({"config": cfg, "split": split, "path": str(path), "rows": len(rows),
                         "type_mix": per_type, "with_soft_label": with_soft})
    if broken:
        print("FAILED: %d rows could not be converted" % len(broken))
        for problem in broken[:20]:
            print("  " + problem)
        return 1
    # a stable order, so the sample does not depend on the read order
    pool.sort(key=lambda it: str(it["id"]))

    if plan:
        # the plan's pool: per type, per config, the plan's counts
        items = plan_sample(pool, plan)
        if args.limit:
            items = items[:args.limit]
    elif args.sample:
        per_type_quota = args.per_type
        if args.limit:
            per_type_quota = min(per_type_quota, max(1, args.limit // len(TYPES)))
        items = stratified_sample(pool, per_type_quota, seed)
    else:
        items = pool
        if args.limit:
            items = items[:args.limit]

    # ---- the checks
    problems = []
    for item in items:
        for problem in check_item(item):
            problems.append("%s: %s" % (item.get("id"), problem))
    if problems:
        print("FAILED: %d problems in %d items" % (len(problems), len(items)))
        for problem in problems[:20]:
            print("  " + problem)
        return 1

    # ---- the output
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("wrote %d items to %s" % (len(items), args.out))
    else:
        sys.stdout.write(text)

    mix, with_soft = summarize([{"question": json.dumps(it["question"]), "soft_label":
                                 json.dumps(it["target_probs"]) if it.get("target_probs") is not None else None}
                                for it in items])
    print("per type: %s" % mix)
    print("items with a human distribution: %d" % with_soft)
    counts = plan_counts(items)
    for qtype in TYPES:
        if qtype in counts:
            print("  %-6s %s" % (qtype, ", ".join("%s %d" % (c, n) for c, n in counts[qtype].items())))

    # ---- the scorer proof
    if args.validate:
        scorer = find_scorer(args.scorer)
        if scorer is None:
            raise SystemExit("no official scorer found; give --scorer PATH")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            items_path = tmpdir / "items.jsonl"
            items_path.write_text(text, encoding="utf-8")
            empty = tmpdir / "predictions_empty.jsonl"
            empty.write_text("", encoding="utf-8")
            code, output = run_scorer(scorer, items_path, empty)
            print("scorer %s over %d items with an empty predictions file: exit %d"
                  % (scorer.name, len(items), code))
            print(output.strip()[:1200])
            if code != 0:
                return 1
            null = tmpdir / "predictions_null.jsonl"
            null.write_text("".join(json.dumps({"id": it["id"], "probs": null_prediction(it)}) + "\n"
                                    for it in items), encoding="utf-8")
            code, output = run_scorer(scorer, items_path, null)
            print("scorer %s over %d items with a uniform prediction on every item: exit %d"
                  % (scorer.name, len(items), code))
            print(output.strip()[:1200])
            if code != 0:
                return 1
        meta = {"data_dir": str(data_dir), "configs": [c for c, _, _ in found],
                "splits": splits, "types": types or list(TYPES), "seed": seed,
                "sample": bool(args.sample), "per_type_quota": args.per_type if args.sample else None,
                "limit": args.limit, "tier": args.tier, "items": len(items),
                "type_mix": mix, "with_soft_label": with_soft, "per_config": counts,
                "plan": {"path": str(args.plan), "seed": plan.get("seed"), "pool": plan.get("pool"),
                         "licence": plan.get("licence")} if plan else None,
                "files": per_file}
        if args.out:
            args.out.with_suffix(args.out.suffix + ".meta.json").write_text(
                json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
