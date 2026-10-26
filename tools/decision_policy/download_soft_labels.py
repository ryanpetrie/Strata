#!/usr/bin/env python3
"""tools/decision_policy/download_soft_labels.py - fetch the licence-clean soft-label data.

Praveenrajus/jev-bench (HuggingFace) repacks 22 public datasets as System One
questions: a state, a typed question (choice / noul / score) and a gold answer.
Four configs carry a HUMAN LABEL DISTRIBUTION (the "calibration gold" ones):
go_emotions, measuring_hate_speech, civil_comments and chaosnli. The first
three have a train split and a validation split on the Hub; chaosnli has test
only (and it is CC BY-NC, see docs/research/decision_policy_training_data.md).

This script fetches the train + validation splits of the three soft-label
configs and the validation split of a set of type-covering configs, then writes
a manifest next to the data: the URL, the byte size, the sha256, the line
count, the type mix and how many records carry a soft label.

Everything it writes stays OUTSIDE the repo. The dataset's own card says the
set is for evaluation, and some of the upstream sources carry conditions, so no
dataset text is committed to this repository.

    python tools/decision_policy/download_soft_labels.py --out-dir <D>

Pure standard library: urllib for the fetch, hashlib for the digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import soft_labels

REPO = "Praveenrajus/jev-bench"
BASE = "https://huggingface.co/datasets/%s/resolve/main/" % REPO

# The three soft-label configs with a train split: the human distribution is
# the scarce resource, so these are fetched whole (train and validation).
SOFT_LABEL_CONFIGS = ("civil_comments", "measuring_hate_speech", "go_emotions")

# Validation only, one per question type and spread over the domains, so the
# validation pool covers choice / noul / score.
TYPE_COVER_CONFIGS = ("boolq", "strategyqa_closed", "mmlu", "banking77", "stsb",
                      "ledgar", "sst5", "arc_challenge", "massive", "clinc150",
                      "paws", "fever_evidence", "sms_spam", "helpsteer2_helpfulness",
                      "helpsteer2_verbosity", "strategyqa_grounded")


def wanted(configs: list[str]) -> list[tuple[str, str]]:
    """The (config, split) pairs to fetch."""
    out: list[tuple[str, str]] = []
    for cfg in configs:
        splits = ("train", "validation") if cfg in SOFT_LABEL_CONFIGS else ("validation",)
        for split in splits:
            out.append((cfg, split))
    return out


def download(url: str, dest: Path) -> None:
    """The file to its place, over a temporary name, so an interrupted run
    leaves no half-written file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print("  fetching %s" % url, flush=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code == 404:
                raise SystemExit("404: %s" % url)
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise SystemExit("could not fetch %s" % url)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="the data root: <config>/<split>.jsonl and manifest.json go here")
    ap.add_argument("--config", action="append", default=None,
                    help="a config to fetch; repeatable. Default: the three soft-label configs "
                         "plus the type-covering set")
    ap.add_argument("--manifest-only", action="store_true",
                    help="do not fetch; only measure what is already on disk")
    args = ap.parse_args()

    configs = args.config or (list(SOFT_LABEL_CONFIGS) + list(TYPE_COVER_CONFIGS))
    pairs = wanted(configs)

    files = []
    for cfg, split in pairs:
        url = BASE + "data/%s/%s.jsonl" % (cfg, split)
        dest = args.out_dir / cfg / ("%s.jsonl" % split)
        if not args.manifest_only:
            download(url, dest)
        if not dest.exists():
            print("  MISSING %s" % dest)
            continue
        files.append((cfg, split, url, dest))

    manifest = {"dataset": REPO, "manifest_of": "downloaded files", "files": []}
    total = 0
    for cfg, split, url, path in files:
        rows = soft_labels.read_rows(path)
        per_type, with_soft = soft_labels.summarize(rows)
        entry = {"config": cfg, "split": split, "url": url, "path": str(path),
                 "bytes": path.stat().st_size, "sha256": sha256_of(path),
                 "lines": len(rows), "type_mix": per_type, "with_soft_label": with_soft,
                 "soft_label_configs": sorted({r.get("source", cfg) for r in rows if r.get("soft_label") is not None})}
        manifest["files"].append(entry)
        total += path.stat().st_size
        print("  %-24s %-11s %9d bytes  %5d rows  %s  soft=%d" %
              (cfg, split, entry["bytes"], entry["lines"], per_type, with_soft))
    manifest["total_bytes"] = total
    manifest["files_count"] = len(manifest["files"])
    out = args.out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("total %d bytes in %d files" % (total, len(manifest["files"])))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
