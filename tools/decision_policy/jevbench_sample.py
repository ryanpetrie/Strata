#!/usr/bin/env python3
"""tools/decision_policy/jevbench_sample.py - build the jev-bench sample and its items file.

Praveenrajus/jev-bench (HuggingFace) is 22 configs of System One questions: a state, a
typed question (choice / score / noul) and a gold answer, with a human label
DISTRIBUTION on four of the configs - the "calibration-gold" ones (go_emotions,
measuring_hate_speech, civil_comments, chaosnli). 22,773 test records in all.

This script takes a stratified sample of the test records and writes three files:

  jevbench_sample3.jsonl        the dataset's own rows, as selected
  jevbench_sample3_items.jsonl  the same rows in the official scorer's shape
  jevbench_sample3_ids.txt      the ids, one per line (the harness's --ids-file)

The sample is stratified over the 22 configs, so every config is in it. The four
calibration-gold configs get the bigger share (--calibration-fraction, 5% of their own
records by default): their human distribution is what the TVD is measured on, and 3%
of a 1,000-record config would leave 30 items for it. The rest of the budget is spread
over the other configs in proportion to their size, so the sample still stands for the
whole set. The whole sample is --fraction of the test set (3% = 683 records).

Everything generated here stays OUTSIDE the repo: the dataset says "validation only,
do not train", so no dataset text is committed.

    python tools/decision_policy/jevbench_sample.py --data-dir <D>\\jevbench_data --out-dir <D>

THE TIER. The official scorer (score.py) weights the four JevBench tiers (easy 0.1,
standard 0.2, judge 0.3, hard 0.4) and computes the choice calibration on the hard
tier alone. jev-bench carries no difficulty label, so there is no honest mapping from
its 22 configs onto those four tiers. The default writes "hard" for every item: the
tier weights then collapse to one (the competence of a type is its chance-corrected
competence over every item of that type, with no invented weighting), and the choice
calibration covers every choice item instead of none. The config each item came from
is kept in the item's "config" field and in the run log, so the per-config numbers are
still available offline. --tier-mode source writes the config name into "tier" instead:
that is the dataset's own grouping, but the official scorer then reports Intelligence
as null, because it only knows the four JevBench tier names.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

REPO = "Praveenrajus/jev-bench"
BASE = "https://huggingface.co/datasets/%s/resolve/main/" % REPO
TIER_NAMES = ("easy", "standard", "judge", "hard")


def download(url: str, dest: Path) -> None:
    """The file to its place, over a temporary name, so an interrupted run leaves
    no half-written file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print("  fetching %s" % url, flush=True)
    with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def load_manifest(data_dir: Path) -> dict:
    """The dataset's own manifest: the configs, their sizes and their licences."""
    path = data_dir / "manifest.json"
    if not path.exists():
        download(BASE + "manifest.json", path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_config_rows(data_dir: Path, cfg: str) -> list[dict]:
    """The test rows of one config."""
    path = data_dir / cfg / "test.jsonl"
    if not path.exists():
        download(BASE + "data/%s/test.jsonl" % cfg, path)
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def quotas(sizes: dict[str, int], fraction: float, calibration_fraction: float,
           calibration: set[str]) -> dict[str, int]:
    """The per-config sample sizes.

    The calibration-gold configs take `calibration_fraction` of their own records (the
    human distribution is what the TVD is measured on). The rest of the budget is
    spread over the other configs in proportion to their size, with the largest
    remainder taking the last units, so the total is exactly the budget.
    """
    total = round(fraction * sum(sizes.values()))
    out = {c: round(calibration_fraction * sizes[c]) for c in calibration}
    rest_total = total - sum(out.values())
    others = [c for c in sizes if c not in calibration]
    rest_size = sum(sizes[c] for c in others)
    if rest_total < len(others):                      # too small to give every config one
        raise SystemExit("the budget is too small: %d items for %d configs" % (rest_total, len(others)))
    exact = {c: rest_total * sizes[c] / rest_size for c in others}
    for c in others:
        out[c] = int(exact[c])
    left = rest_total - sum(out[c] for c in others)
    for c in sorted(others, key=lambda c: (-(exact[c] - int(exact[c])), c))[:left]:
        out[c] += 1
    for c in sizes:
        out[c] = min(out[c], sizes[c])
    return out


def parse_state(raw: str):
    """The state's value: the JSON object it holds when it holds one, else the text.

    Most configs put an object in the state ({"passage": ..., "question": ...}); the
    rest put plain text. The endpoint takes both, and an object is sent as an object
    so the server serializes it canonically (sorted keys) - the same prompt for the
    same state whatever the key order.
    """
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    if isinstance(value, (dict, list, str)):
        return value
    return raw


def target_probs(row: dict, qtype: str, labels: list[str]) -> dict | None:
    """The human label distribution, in the scorer's label space, or None.

    The dataset's soft_label has three shapes: a dict over the option names (the
    choice configs), a list over the level indices (the score config) and a single
    number, the share of the annotators who said yes (the noul config).
    """
    soft = row.get("soft_label")
    if soft is None:
        return None
    value = json.loads(soft)
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items()}
    if isinstance(value, list):
        return {str(i): float(v) for i, v in enumerate(value)}
    if qtype == "noul":
        p = float(value)
        return {"yes": p, "no": 1.0 - p}
    return None


def to_item(row: dict, tier: str) -> dict:
    """One dataset row -> one item of the official scorer's shape.

    The scorer reads: id, question (the typed question, an object), labels (the
    answer names), expected (the gold answer), tier, state and target_probs (the
    human distribution, which drives the TVD).

    The labels and the expected answer follow the scorer, not the dataset:
      choice  the option names; the gold answer is the option key
      score   the level indices "0".."k-1"; the gold answer is the index
      noul    ["no", "yes"]; the gold answer is "yes" when the dataset's label is "1"
    """
    q = json.loads(row["question"])
    qtype = q["type"]
    if qtype == "choice":
        labels = list(q["criteria"].keys())
        expected = row["label"]
    elif qtype == "score":
        labels = [str(i) for i in range(len(q["criteria"]))]
        expected = int(row["label"])
    else:
        labels = ["no", "yes"]
        expected = "yes" if str(row["label"]) == "1" else "no"
    return {"id": row["id"], "question": q, "labels": labels, "expected": expected,
            "tier": tier, "state": parse_state(row["state"]),
            "target_probs": target_probs(row, qtype, labels),
            "config": row.get("source"), "primitive": row.get("primitive")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="the dataset cache: <config>/test.jsonl and manifest.json (downloaded if missing)")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the sample files go")
    ap.add_argument("--name", default="jevbench_sample3", help="the sample's name (the file prefix)")
    ap.add_argument("--fraction", type=float, default=0.03, help="the share of the test set")
    ap.add_argument("--calibration-fraction", type=float, default=0.05,
                    help="the share of the calibration-gold configs' own records to take")
    ap.add_argument("--seed", type=int, default=20261020, help="the sampling seed")
    ap.add_argument("--tier-mode", choices=("scorer", "source", "primitive"), default="scorer",
                    help="scorer: every item is filed under the 'hard' tier (the four JevBench tiers "
                         "are the only ones the official scorer knows, and jev-bench has no difficulty "
                         "label); source/primitive: the dataset's own grouping, which the official "
                         "scorer cannot weight (Intelligence comes out null)")
    args = ap.parse_args()

    manifest = load_manifest(args.data_dir)
    sources = manifest["sources"]
    calibration = {c for c, info in sources.items() if info.get("has_soft_labels")}
    rows_by_cfg, sizes = {}, {}
    for cfg in sources:
        rows = load_config_rows(args.data_dir, cfg)
        rows_by_cfg[cfg] = rows
        sizes[cfg] = len(rows)
    total = sum(sizes.values())
    take = quotas(sizes, args.fraction, args.calibration_fraction, calibration)

    rng = random.Random(args.seed)
    picked = []
    for cfg in sorted(sizes):                      # sorted: the same seed gives the same sample
        rows = rows_by_cfg[cfg]
        idx = sorted(rng.sample(range(len(rows)), take[cfg]))
        picked.extend(rows[i] for i in idx)
    rng.shuffle(picked)                            # the run is not config-ordered: the configs
                                                   # differ in prompt size, and an ordered run
                                                   # would make the latency curve meaningless

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / (args.name + ".jsonl")
    items_path = out_dir / (args.name + "_items.jsonl")
    ids_path = out_dir / (args.name + "_ids.txt")
    with sample_path.open("w", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with items_path.open("w", encoding="utf-8") as f:
        for row in picked:
            tier = "hard"
            if args.tier_mode == "source":
                tier = row.get("source") or "unknown"
            elif args.tier_mode == "primitive":
                tier = row.get("primitive") or "unknown"
            f.write(json.dumps(to_item(row, tier), ensure_ascii=False) + "\n")
    ids_path.write_text("\n".join(row["id"] for row in picked) + "\n", encoding="utf-8")

    # what the sample holds, for the report
    per_cfg, per_type, with_soft = {}, {}, 0
    for row in picked:
        per_cfg[row["source"]] = per_cfg.get(row["source"], 0) + 1
        t = json.loads(row["question"])["type"]
        per_type[t] = per_type.get(t, 0) + 1
        if row.get("soft_label") is not None:
            with_soft += 1
    meta = {"dataset": REPO, "dataset_version": manifest.get("version"), "seed": args.seed,
            "fraction": args.fraction, "calibration_fraction": args.calibration_fraction,
            "tier_mode": args.tier_mode, "test_records": total, "sample_records": len(picked),
            "per_config": per_cfg, "per_type": per_type, "with_soft_label": with_soft,
            "calibration_configs": sorted(calibration), "files": {
                "sample": str(sample_path), "items": str(items_path), "ids": str(ids_path)},
            "licenses": {c: sources[c].get("license") for c in sorted(sources)}}
    (out_dir / (args.name + "_meta.json")).write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    print("test records %d, sample %d (%.2f%%)" % (total, len(picked), 100.0 * len(picked) / total))
    print("per config:")
    for cfg in sorted(per_cfg):
        mark = " (calibration gold)" if cfg in calibration else ""
        print("  %-26s %4d%s" % (cfg, per_cfg[cfg], mark))
    print("per type: %s" % per_type)
    print("records with a human distribution: %d" % with_soft)
    print("wrote %s" % sample_path)
    print("wrote %s" % items_path)
    print("wrote %s" % ids_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
