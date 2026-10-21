#!/usr/bin/env python3
"""tools/decision_policy/jevbench_rewrites_sample.py - build the JevBench rewrite sample.

JevBench (https://huggingface.co/datasets/JevBench/jevbench, Apache-2.0) ships a frozen
test plan: 12,000 tests over 240 cases, one per relation per case.  Each test is a
REWRITE of the case's questions - a paraphrase, a negation, a conjunction - and a law
of probability that the two answer distributions must obey.  This script takes a slice
of that plan and writes it in the shape the harness runs:

    jevbench_rewrites_sample.jsonl   one row per test: the case, the relation, the
                                     base questions and the rewritten ones, each with
                                     the id of the request that answers it.

The slice is chosen for the endpoint's budget, not for a score: one question per
request (the endpoint's answer position is per question), and about 3.9 s per
question on this machine.  The default is one case per domain over 41 relations -
every relation except the batch ones (BAT: they need several questions in one
request) and the choice-set ones (CHO: they change the menu, not the question) and
threshold_sweep (its probe set is eight or nine questions per case).

Every request in the sample is checked against the endpoint's OWN validation
(serve/systemone.py), so a sample that the endpoint would reject never leaves here.

Everything generated stays OUTSIDE the repo: the benchmark's data are downloaded
into the data directory, and the sample file is written where the caller asks.

    python tools/decision_policy/jevbench_rewrites_sample.py \
        --data-dir <D>\\jevbench_suite --out <D>\\jevbench_rewrites_sample.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import systemone as sysone                                    # noqa: E402
from tools.decision_policy.jevbench_rewrites import CLASSES, request_id  # noqa: E402

REPO = "JevBench/jevbench"
SUITE = "jevbench-240"
BASE = "https://huggingface.co/datasets/%s/resolve/main/suites/%s/" % (REPO, SUITE)
SECONDS_PER_QUESTION = 3.9        # measured on this machine (the long state costs a prefill)


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


def ensure_suite(data_dir: Path) -> dict:
    """The suite's cases, its frozen plan and its manifest (downloaded if missing).

    The manifest records the SHA-256 of every file of the suite, and it is checked:
    the plan is frozen data, and a file that does not match the manifest is not the
    benchmark.
    """
    files = {}
    for name in ("manifest.json", "cases.jsonl", "tests.jsonl.gz"):
        path = data_dir / name
        if not path.exists():
            download(BASE + name, path)
        files[name] = path
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    for name, path in files.items():
        if name == "manifest.json":
            continue
        want = manifest.get("files", {}).get(name)
        if not want:
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit("%s: SHA-256 %s, the manifest says %s - the file is not the benchmark's"
                             % (path, got, want))
    return {"manifest": manifest, "files": files}


def read_jsonl(path: Path) -> list[dict]:
    op = gzip.open if path.name.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_cases(cases: list[dict], per_domain: int) -> list[dict]:
    """The first cases of each domain in the suite's own order.

    A convenience slice, not a random sample: it is one case per domain by default,
    so every domain is in the run and the rules are the same for every model that
    runs this sample.
    """
    picked, count = [], Counter()
    for case in cases:
        domain = case.get("domain", "")
        if count[domain] < per_domain:
            count[domain] += 1
            picked.append(case)
    return picked


def decode(case: dict, items: list) -> list[tuple[str, dict]]:
    """The plan's encoded questions -> the questions themselves.

    jevbench/plan.py, decode_requests: a question identical to one of the case's is
    stored as {"@": its key}.
    """
    out = []
    for key, q in items:
        out.append((key, case["questions"][q["@"]] if set(q) == {"@"} else q))
    return out


def variant_keys(relation: str, params: dict, targets: list[str], keys: list[str],
                 case: dict) -> dict:
    """Which variant key stands for which base question.

    Most relations keep the question's key; two rename it (the official KeyRename
    and KeyInstructionConflict), and the sample says which is which so the laws can
    read the right pair.
    """
    if relation == "key_rename":
        return {params["new_key"]: targets[0]}
    if relation == "key_instruction_conflict":
        return {sent_as: key for key, sent_as in params["keys"].items()}
    return {k: k for k in keys if k in case["questions"]}


def build_row(case: dict, plan_row: dict, relation: str, group: str, cls: str) -> dict:
    """One test of the plan -> one row of the sample."""
    reduced = plan_row.get("reduced") or {"requests": plan_row["requests"], "targets": plan_row["targets"]}
    variant = decode(case, list(reduced["requests"].values())[0])
    base_keys = list(plan_row["targets"])
    base = [(k, case["questions"][k]) for k in base_keys]
    vkeys = variant_keys(relation, plan_row.get("params") or {}, base_keys, [k for k, _ in variant], case)
    return {
        "id": "%s|%s|%s|v%s" % (case["id"], relation, plan_row["key"], plan_row["v"]),
        "case_id": case["id"],
        "domain": case.get("domain"),
        "relation": relation,
        "class": cls,
        "group": group,
        "targets": base_keys,
        "state": case["state"],
        "base": [{"key": k, "question": q, "id": request_id(case["id"], k, q)} for k, q in base],
        "variant": [{"key": k, "question": q, "id": request_id(case["id"], k, q)} for k, q in variant],
        "variant_keys": vkeys,
        "params": plan_row.get("params") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="the suite cache: cases.jsonl, tests.jsonl.gz, manifest.json "
                         "(downloaded if missing)")
    ap.add_argument("--out", type=Path, required=True, help="the sample file to write")
    ap.add_argument("--meta", type=Path, default=None, help="the summary file (default: <out>.meta.json)")
    ap.add_argument("--per-domain", type=int, default=1, help="cases per domain")
    ap.add_argument("--relations", default=None,
                    help="the relations to include (default: every relation the harness implements)")
    ap.add_argument("--seconds-per-question", type=float, default=SECONDS_PER_QUESTION,
                    help="the endpoint's measured cost of one question, for the time estimate")
    args = ap.parse_args()

    wanted = None
    if args.relations:
        wanted = {r.strip() for r in args.relations.split(",") if r.strip()}
    relation_class = {r: (cls, group) for cls, rows in CLASSES.items() for r, group in rows}
    if wanted:
        unknown = wanted - set(relation_class)
        if unknown:
            raise SystemExit("unknown relations: %s" % sorted(unknown))
        relations = sorted(wanted)
    else:
        relations = sorted(relation_class)

    suite = ensure_suite(args.data_dir)
    cases = read_jsonl(suite["files"]["cases.jsonl"])
    plan = read_jsonl(suite["files"]["tests.jsonl.gz"])
    head, rows = plan[0], plan[1:]
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_case[r["case"]][r["relation"]] = r

    picked = pick_cases(cases, args.per_domain)
    sample, missing = [], Counter()
    for case in picked:
        for relation in relations:
            row = by_case[case["id"]].get(relation)
            if row is None:
                missing[relation] += 1
                continue
            cls, group = relation_class[relation]
            sample.append(build_row(case, row, relation, group, cls))

    # every request must be one the endpoint accepts
    bad = []
    for row in sample:
        for role in ("base", "variant"):
            for e in row[role]:
                try:
                    sysone.validate_request({"state": row["state"],
                                             "questions": {e["key"]: e["question"]}})
                except ValueError as exc:
                    bad.append("%s %s/%s: %s" % (row["id"], role, e["key"], exc))
    if bad:
        for line in bad[:20]:
            print("  INVALID REQUEST: %s" % line, file=sys.stderr)
        raise SystemExit("%d requests the endpoint would reject" % len(bad))

    requests = {}
    for row in sample:
        for role in ("base", "variant"):
            for e in row[role]:
                requests[e["id"]] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_class = Counter(r["class"] for r in sample)
    per_rel = Counter(r["relation"] for r in sample)
    per_type = Counter(e["question"]["type"] for r in sample for e in r["base"] + r["variant"])
    seconds = len(requests) * args.seconds_per_question
    meta = {
        "dataset": "%s (%s %s)" % (REPO, SUITE, suite["manifest"].get("version")),
        "license": "apache-2.0",
        "plan": {"file": "tests.jsonl.gz", "seed": head.get("seed"), "selection": head.get("selection"),
                 "sha256": suite["manifest"].get("files", {}).get("tests.jsonl.gz")},
        "cases": [c["id"] for c in picked],
        "domains": sorted({c.get("domain") for c in picked}),
        "per_domain": args.per_domain,
        "relations": relations,
        "tests": len(sample),
        "requests": len(requests),
        "per_class": dict(per_class),
        "per_relation": dict(per_rel),
        "questions_by_type": dict(per_type),
        "seconds_per_question": args.seconds_per_question,
        "expected_seconds": round(seconds, 1),
        "expected_minutes": round(seconds / 60.0, 1),
        "files": {"sample": str(args.out), "suite": str(args.data_dir)},
    }
    meta_path = args.meta or args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    print("%d cases, %d relations, %d tests, %d requests"
          % (len(picked), len(relations), len(sample), len(requests)))
    print("  domains: %s" % ", ".join(meta["domains"]))
    print("  tests per class: %s" % ", ".join("%s %d" % (c, per_class[c]) for c in sorted(per_class)))
    print("  questions by type: %s" % dict(per_type))
    print("  expected time: %.1f min at %.1f s per question"
          % (seconds / 60.0, args.seconds_per_question))
    if missing:
        print("  relations with no test on a picked case: %s" % dict(missing))
    print("  wrote %s" % args.out)
    print("  wrote %s" % meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
