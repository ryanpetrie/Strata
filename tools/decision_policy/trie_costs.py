"""The trie walk's cost over the 235-item sample, using the SHIPPED code.

tools/decision_policy/trie_costs.py - how many extra engine requests the trie walk
would need for every item of the OpenDecisions sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import strata_tokenizer as ST  # noqa: E402
from serve import logits_first_token as lft  # noqa: E402
from tools.decision_policy.opendecisions import shape_question  # noqa: E402

PACK = Path(r"C:\tools\Strata-data\packs\unsloth-ud-iq4_xs\tokenizer")


def load_tok():
    """The tokenizer the way serve/server.py builds it: the piece list ordered by id
    and the token types, so the option token sequences are the ones the model sees."""
    vocab = json.loads((PACK / "vocab.json").read_text(encoding="utf-8"))
    pieces = [None] * len(vocab)
    for t, i in vocab.items():
        pieces[i] = t
    merges = (PACK / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((PACK / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(pieces, merges, types)


def options_of(q):
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    return [str(x) for x in q["criteria"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--prefix", default="q1:", help="the answer line's prefix")
    ap.add_argument("--show", type=int, default=0, help="print the N costliest items")
    args = ap.parse_args()
    tok = load_tok()

    items = []
    with args.items.open(encoding="utf-8") as f:
        for line in f:                       # split on \n only: the file holds a raw U+2028
            line = line.strip()
            if line:
                items.append(json.loads(line))

    collide = 0
    cost = Counter()
    total = 0
    detail = []
    for it in items:
        shaped, _ = shape_question(it["question"])
        options = options_of(shaped)
        if len(options) < 2:
            continue
        firsts = lft.first_token_ids(options, tok, args.prefix)
        if len(set(firsts.values())) == len(firsts):
            continue                      # the first-token row separates them
        collide += 1
        plan = lft.trie_plan(options, tok, args.prefix)
        if plan is None:
            cost["no plan"] += 1
            continue
        n = len(plan["walk"])
        cost[n] += 1
        total += n
        detail.append((n, it["id"], shaped["type"], len(options)))

    print(f"items {len(items)}   with a first-token collision {collide}")
    print("\nextra engine requests per colliding item:")
    for n in sorted(cost, key=str):
        print(f"  {n}: {cost[n]:3d} items")
    print(f"\ntotal extra requests over the colliding items: {total}")
    if args.show:
        print("\nthe costliest items:")
        for n, iid, qtype, nopts in sorted(detail, reverse=True)[:args.show]:
            print(f"  {n:3d} requests  {iid:36s} {qtype:6s} {nopts} options")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
