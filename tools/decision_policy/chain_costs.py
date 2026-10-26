"""tools/decision_policy/chain_costs.py - how the chain rule would apply to the sample.

The chain rule separates the options that share a first token by walking the TRIE of
their token sequences: the row at a node gives the distribution over the token that
comes next, so a node's row tells the options that pass through it apart.

Two ways to read the walk:

  full     every node on every option's path: the probability of an option is the
           product of ALL its token probabilities (the model's own probability of
           writing that option's text).  The cost is the size of the trie.
  to-split the nodes up to where an option separates from the others: the options
           are told apart, and the probability is the product up to that point (an
           upper bound on the option's own probability, as the first-token reading
           is).

The root (the answer position) is given, so it is not counted.  This measures both
costs on the sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.strata_tokenizer import Tokenizer  # noqa: E402

PACK = Path(r"C:\tools\Strata-data\packs\unsloth-ud-iq4_xs\tokenizer")


def load_tok() -> Tokenizer:
    cfg = json.loads((PACK / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = json.loads((PACK / "vocab.json").read_text(encoding="utf-8"))
    merges = [m for m in
              (PACK / "merges.txt").read_text(encoding="utf-8").split("\n") if m]
    return Tokenizer(list(vocab), merges, pre=cfg.get("pre", "qwen35"),
                     special_ids=cfg.get("special_ids"))


def option_tokens(tok: Tokenizer, prefix: str, option: str) -> list[int]:
    """The option's token ids after the prefix (the ids the model would write)."""
    pre = list(tok.encode(prefix, parse_special=True))
    ids = list(tok.encode(prefix + " " + option, parse_special=True))
    n = 0
    while n < len(pre) and n < len(ids) and pre[n] == ids[n]:
        n += 1
    return ids[n:]


def costs(seqs: dict[str, list[int]]) -> tuple[int, int, int]:
    """-> (the full trie size, the rows up to the split, the branching rows)"""
    children: dict[tuple, set] = {}
    ends: dict[tuple, bool] = {}
    for ids in seqs.values():
        path = ()
        for tokid in ids:
            children.setdefault(path, set()).add(tokid)
            path = path + (tokid,)
        ends[path] = True

    # the depth at which each option separates from the others
    depth: dict[str, int] = {}
    for name, ids in seqs.items():
        d = 0
        for i in range(len(ids)):
            path = tuple(ids[:i])
            # the option is unique here when no other option shares this path
            if sum(1 for o2, ids2 in seqs.items() if o2 != name and ids2[:i] == list(path)) == 0:
                d = i
                break
        else:
            d = len(ids)
        depth[name] = d

    up_to_split = set()
    for name, ids in seqs.items():
        for i in range(min(depth[name] + 1, len(ids))):
            up_to_split.add(tuple(ids[:i]))
    branching = {p for p, c in children.items() if len(c) > 1 or ends.get(p, False)}
    return len(children), len(up_to_split) - 1, len(branching) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    args = ap.parse_args()
    tok = load_tok()

    items = []
    with args.items.open(encoding="utf-8") as f:
        for line in f:                       # split on \n only: the file holds a raw U+2028
            line = line.strip()
            if line:
                items.append(json.loads(line))

    collide = 0
    full, split, branch = Counter(), Counter(), Counter()
    for it in items:
        q = it["question"]
        crit = q.get("criteria")
        if q["type"] == "noul":
            options = ["yes", "no"]
        elif isinstance(crit, dict):
            options = list(crit.keys())
        else:
            options = list(crit or [])
        if len(options) < 2:
            continue
        seqs = {o: option_tokens(tok, "q:", o) for o in options}
        firsts = Counter(seqs[o][0] for o in options if seqs[o])
        if not any(c > 1 for c in firsts.values()):
            continue                      # the first-token lookup separates them
        collide += 1
        f_, s_, b_ = costs(seqs)
        full[f_] += 1
        split[s_] += 1
        branch[b_] += 1

    print(f"items {len(items)}   with a first-token collision {collide}")
    for name, c in (("the full trie", full), ("the rows up to the split", split),
                    ("the branching rows", branch)):
        print(f"\n{name} (engine requests each):")
        for n, k in sorted(c.items()):
            print(f"  {n:3d}: {k:4d} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
