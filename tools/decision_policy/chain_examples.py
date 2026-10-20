"""tools/decision_policy/chain_examples.py - the colliding questions, as the tokens see them.

Prints the questions whose options share their first token, with each option's token
sequence, so the shape of the problem is visible: the options that are the same
length, and the ones that are not.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    tok = load_tok()

    items = []
    with args.items.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    shown = 0
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
            continue
        shown += 1
        if shown > args.limit:
            break
        print(f"\n--- {it['id']}  ({q['type']})")
        for o in options:
            toks = [tok.decode([t]) for t in seqs[o]]
            print(f"   {len(seqs[o]):2d} tokens  {o!r}")
            print(f"                {toks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
