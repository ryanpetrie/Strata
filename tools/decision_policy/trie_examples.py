"""tools/decision_policy/trie_examples.py - the colliding questions, as the tokens see them.

Prints the questions whose options share a first token, with each option's token
sequence and the depth at which the trie walk tells the options apart - so the shape
of the problem is visible: the options that are alone at the first token, and the ones
that need a deeper row.
"""
from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument("--prefix", default="q1:")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    tok = load_tok()

    items = []
    with args.items.open(encoding="utf-8") as f:
        for line in f:                       # split on \n only: the file holds a raw U+2028
            line = line.strip()
            if line:
                items.append(json.loads(line))

    shown = 0
    for it in items:
        shaped, _ = shape_question(it["question"])
        options = options_of(shaped)
        if len(options) < 2:
            continue
        firsts = lft.first_token_ids(options, tok, args.prefix)
        if len(set(firsts.values())) == len(firsts):
            continue                      # the first-token row separates them
        plan = lft.trie_plan(options, tok, args.prefix)
        shown += 1
        if shown > args.limit:
            break
        print(f"\n--- {it['id']}  ({shaped['type']})  "
              f"{len(plan['walk'])} extra engine request(s)")
        for o in options:
            toks = [tok.decode([t]) for t in plan["tokens"][o]]
            r = plan["resolution"][o]
            ends = " (ends here: the turn has to end for it to be the answer)" \
                if r == len(plan["tokens"][o]) else ""
            print(f"   told apart at {r:2d}  {o!r}")
            print(f"                 {toks}{ends}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
