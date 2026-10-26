"""tools/decision_policy/letter_costs.py - the letter readout's cost over the sample.

How many engine requests does the letter readout need for an item?  One.  The options
are shown as A, B, C ... and the letters are one token each, and no two of them are
the same token, so the row at the answer position tells them apart: the trie walk -
the extra engine requests the direct path pays for the items whose options share
their first token - is never needed.

This computes that with the SHIPPED code (serve/systemone_letters.py) and the
server's own tokenizer, over the items of a run, so the number is the one the
endpoint would pay.  No engine and no server: the letters' token ids are a property
of the tokenizer, not of the model.

    python tools/decision_policy/letter_costs.py --items <items.jsonl>
    python tools/decision_policy/letter_costs.py --items <items.jsonl> --show 5
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
from serve import systemone_letters as sl  # noqa: E402
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
    ap.add_argument("--prefix", default="q:", help="the answer line's prefix")
    ap.add_argument("--show", type=int, default=0, help="print the N items with the most options")
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
    not_one_token = 0
    too_many = 0
    detail = []
    for it in items:
        shaped, _ = shape_question(it["question"])
        options = options_of(shaped)
        if len(options) < 2:
            continue
        if len(options) > sl.MAX_LETTERED_OPTIONS:
            too_many += 1
            continue
        letters = sl.LETTERS[:len(options)]
        ids = sl.letter_token_ids(letters, tok, args.prefix)
        if not all(ids.values()):
            not_one_token += 1
            continue
        # the trie plan of the letters: the walk is the extra engine requests
        plan = lft.trie_plan(letters, tok, args.prefix)
        if plan is None:
            cost["no plan"] += 1
            continue
        firsts = [plan["tokens"][letter][0] for letter in letters]
        if len(set(firsts)) != len(firsts):
            collide += 1             # two letters share their first token: they would collide
        n = len(plan["walk"])
        cost[n] += 1
        total += n
        detail.append((n, it["id"], shaped["type"], len(options)))

    print(f"items {len(items)}   with a first-token collision {collide}")
    print("\nextra engine requests per item (the letter readout):")
    for n in sorted(cost, key=str):
        print(f"  {n}: {cost[n]:3d} items")
    print(f"\ntotal extra requests over the items: {total}")
    print(f"items whose letters are not one token: {not_one_token}")
    print(f"items with more options than letters: {too_many}")
    if args.show:
        print("\nthe items with the most options:")
        for n, iid, qtype, nopts in sorted(detail, key=lambda d: -d[3])[:args.show]:
            print(f"  {iid:36s} {qtype:6s} {nopts:3d} options, {n} extra requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
