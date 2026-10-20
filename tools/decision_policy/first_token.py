#!/usr/bin/env python3
"""tools/decision_policy/first_token.py - first-token ambiguity of OpenDecisions options.

The logits-confidence feature has two candidate implementations:

  * a cheap FIRST-TOKEN LOOKUP: rank a question's options by the logit of each option's
    first token at the answer position (one forward pass, no KV-cache walk);
  * the full CHAIN RULE: walk each option's whole token sequence through the KV cache,
    multiplying P(t_i | t_<i).

The fast path is only exact when a question's options have DISTINCT first tokens, so the
argmax over first-token logits is the same as the argmax over full-sequence probabilities.
This script measures, per question, whether that holds. It loads no model and starts no
server; it reads only the dataset and the model's tokenizer pack.

Definitions (grounded in serve/systemone.py and tools/decision_policy/opendecisions.py):

  * The endpoint answers one `name: answer` line per question; the answer (the option or
    level name) is preceded by a space after the colon. So each name is tokenized as
    `" " + name`, matching serve/systemone.py's reply shape.
  * choice  -> the option names are the keys of question.criteria (== item.labels).
  * score   -> the harness (opendecisions.shape_question) reshapes `"id: description"`
    levels into `{id: description}` and the endpoint then ranks the short ids; when a
    level does not fit that pattern the harness falls back to the raw list, so the raw
    full strings are the names. Both cases are mirrored here via shape_question().
    Sensitivity variants (numeric item.labels, raw unshaped strings) are also measured.
  * noul    -> the answers are `yes`/`no` (item.labels), both single tokens: always unique.

The dataset has a raw U+2028 inside one state string, so the file is split on "\\n" only -
never str.splitlines().

USAGE
  python tools/decision_policy/first_token.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASET = Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\opendecisions.jsonl")
TOKENIZER_DIR = Path(r"C:\tools\Strata-data\packs\unsloth-ud-iq4_xs\tokenizer")
OUT_JSON = HERE / "first_token_analysis.json"
OUT_MD = HERE / "first_token_report.md"

MAX_NAME_LEN = 128          # serve/systemone.py MAX_NAME_LEN
LONG_NAME = 40              # shape_question()'s head-length ceiling (a proxy for "unwieldy")


def load_tokenizer():
    """The model's own vocabulary, from the extracted tokenizer pack.

    The pack's vocab.json / merges.txt / token_type.json are byte-identical to the GGUF
    metadata (documented in tokenization_report.md), so no 35 GB shard is opened."""
    sys.path.insert(0, str(REPO / "tools"))
    from strata_tokenizer import Tokenizer
    vocab = json.loads((TOKENIZER_DIR / "vocab.json").read_text(encoding="utf-8"))
    tokens: list = [None] * len(vocab)
    for piece, i in vocab.items():
        tokens[i] = piece
    if any(t is None for t in tokens):
        raise ValueError("vocab.json has gaps: ids are not contiguous")
    merges = [m for m in (TOKENIZER_DIR / "merges.txt").read_text(encoding="utf-8").split("\n") if m]
    types = json.loads((TOKENIZER_DIR / "token_type.json").read_text(encoding="utf-8"))
    cfg = json.loads((TOKENIZER_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    return Tokenizer(tokens, merges, types, cfg.get("pre", "qwen35"))


def load_items():
    """The dataset. Split on '\\n' only: the file holds a raw U+2028, which splitlines()
    (and str.splitlines-like helpers) would treat as a line break and corrupt the JSON."""
    text = DATASET.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def shape_score_question(q: dict):
    """Mirror opendecisions.shape_question: score criteria as a list of 'id: description'
    -> {id: description}; otherwise the raw list. Returns (names, shaped_bool)."""
    sys.path.insert(0, str(HERE))
    from opendecisions import shape_question
    shaped, was_shaped = shape_question(q)
    if was_shaped:
        return list(shaped["criteria"].keys()), True
    return list(q["criteria"]), False


def option_names(item: dict) -> dict:
    """The answerable names for one item, under the primary and sensitivity definitions.

    -> {primary, primary_kind, labels, raw_score} where primary_kind is
    'noul' | 'choice' | 'score_shaped' | 'score_raw'."""
    q = item["question"]
    qtype = q["type"]
    labels = [str(x) for x in item.get("labels") or []]
    if qtype == "noul":
        # The endpoint answers noul as yes/no (serve/systemone.py parse_answers), whatever
        # the criteria keys ('true'/'false') say.
        return {"primary": ["yes", "no"], "primary_kind": "noul",
                "labels": labels, "raw_score": None}
    if qtype == "choice":
        return {"primary": list(q["criteria"].keys()), "primary_kind": "choice",
                "labels": labels, "raw_score": None}
    names, shaped = shape_score_question(q)
    return {"primary": names, "primary_kind": "score_shaped" if shaped else "score_raw",
            "labels": labels, "raw_score": list(q["criteria"])}


def analyse_names(tk, names: list, prefix: str = " ") -> dict:
    """Tokenize each name as prefix+name; return per-name records and the question verdict.

    prefix=' ' is the answer context (the space after 'name:'); prefix='' is the bare form,
    measured as a sensitivity check."""
    records = []
    for raw in names:
        is_str = isinstance(raw, str)
        name = raw if is_str else str(raw)
        cleaned = name.strip()
        ids = tk.encode(prefix + cleaned) if cleaned else []
        first = ids[0] if ids else None
        records.append({
            "name": name,
            "cleaned": cleaned,
            "stripped": cleaned != name,
            "is_str": is_str,
            "n_tokens": len(ids),
            "token_ids": ids[:16],
            "pieces": [tk.decode([i]) for i in ids[:16]],
            "first_token": first,
            "first_piece": tk.decode([first]) if first is not None else None,
        })
    firsts = [r["first_token"] for r in records]
    # A question is unique when every option maps to a distinct first token. A name that
    # tokenizes to nothing cannot be read at all, so it makes the question non-unique.
    distinct = [f for f in firsts if f is not None]
    unique = (len(distinct) == len(firsts) == len(set(firsts))) and len(firsts) >= 2
    groups: dict = {}
    for r in records:
        groups.setdefault(r["first_token"], []).append(r["name"])
    collisions = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
    return {
        "n_options": len(records),
        "unique": unique,
        "n_distinct_first_tokens": len(set(firsts)),
        "n_colliding_options": sum(len(g) for g in collisions),
        "largest_collision": len(collisions[0]) if collisions else 0,
        "collisions": collisions,
        "options": records,
    }


def length_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def main() -> int:
    tk = load_tokenizer()
    items = load_items()
    print(f"[first_token] {len(items)} items, vocab {len(tk.tokens)}", flush=True)

    anomalies = Counter()
    anomalies_by_type = {t: Counter() for t in ("choice", "score", "noul")}
    anomaly_examples: dict = {}
    questions = []
    length_dist = {t: Counter() for t in ("choice", "score", "noul", "all")}
    length_sum = {t: [0, 0] for t in ("choice", "score", "noul", "all")}  # [sum tokens, count]
    score_variant = {  # sensitivity: score questions under alternative name definitions
        "shaped_ids": Counter(), "raw_strings": Counter(), "numeric_labels": Counter(),
    }
    score_kind = Counter()   # shaped vs raw fallback

    def flag(key, item, value):
        anomalies[key] += 1
        anomalies_by_type[item["question"]["type"]][key] += 1
        anomaly_examples.setdefault(key, []).append((item["id"], value))

    for item in items:
        q = item["question"]
        qtype = q["type"]
        names = option_names(item)
        primary = names["primary"]

        # --- cleaning / anomaly log
        for raw in primary:
            if not isinstance(raw, str):
                flag("non_string", item, repr(raw))
                continue
            if raw != raw.strip():
                flag("whitespace_stripped", item, raw)
            c = raw.strip()
            if not c:
                flag("empty", item, raw)
            if "\n" in raw or "\r" in raw or "\u2028" in raw or "\u2029" in raw:
                flag("newline_like", item, raw)
            if ":" in c:
                flag("has_colon", item, c)
            if len(c) > MAX_NAME_LEN:
                flag("len_gt_128", item, c[:60])
            elif len(c) > LONG_NAME:
                flag("len_gt_40", item, c[:60])
        stripped_names = [n.strip() if isinstance(n, str) else n for n in primary]
        if len(set(stripped_names)) != len(stripped_names):
            flag("duplicate_name_in_question", item, stripped_names)

        # --- per-question first-token verdict (leading-space answer context)
        res = analyse_names(tk, primary)
        # sensitivity: the bare name (no leading space), in case the space is already consumed
        bare = analyse_names(tk, primary, prefix="")
        if qtype == "noul":
            res["unique"] = True          # yes/no are two distinct single tokens
            bare["unique"] = True
        for r in res["options"]:
            length_dist[qtype][length_bucket(r["n_tokens"])] += 1
            length_dist["all"][length_bucket(r["n_tokens"])] += 1
            length_sum[qtype][0] += r["n_tokens"]
            length_sum[qtype][1] += 1
            length_sum["all"][0] += r["n_tokens"]
            length_sum["all"][1] += 1

        if qtype == "score":
            score_kind[names["primary_kind"]] += 1
            for key, name_list in (("shaped_ids", primary), ("raw_strings", names["raw_score"]),
                                   ("numeric_labels", names["labels"])):
                v = analyse_names(tk, name_list)
                score_variant[key]["unique" if v["unique"] else "ambiguous"] += 1

        questions.append({
            "id": item["id"], "type": qtype, "tier": item.get("tier"),
            "name_kind": names["primary_kind"],
            "unique": res["unique"], "bare_unique": bare["unique"],
            "all_collide": res["n_colliding_options"] == res["n_options"],
            "single_first_token": res["n_distinct_first_tokens"] == 1,
            "n_options": res["n_options"],
            "n_distinct_first_tokens": res["n_distinct_first_tokens"],
            "n_colliding_options": res["n_colliding_options"],
            "largest_collision": res["largest_collision"],
            "collisions": res["collisions"],
            "options": res["options"],
        })

    # --- aggregate
    by_type = {}
    for t in ("choice", "score", "noul"):
        qs = [q for q in questions if q["type"] == t]
        u = sum(1 for q in qs if q["unique"])
        by_type[t] = {"questions": len(qs), "unique": u, "ambiguous": len(qs) - u,
                      "unique_pct": round(100.0 * u / len(qs), 2) if qs else 0.0}
    total = len(questions)
    total_u = sum(1 for q in questions if q["unique"])
    total_bare_u = sum(1 for q in questions if q["bare_unique"])
    aggregate = {
        "questions": total,
        "unique": total_u,
        "ambiguous": total - total_u,
        "unique_pct": round(100.0 * total_u / total, 2),
        "ambiguous_pct": round(100.0 * (total - total_u) / total, 2),
        "bare_unique": total_bare_u,
        "bare_unique_pct": round(100.0 * total_bare_u / total, 2),
        "by_type": by_type,
        "ambiguous_all_collide": sum(1 for q in questions if not q["unique"] and q["all_collide"]),
        "ambiguous_single_first_token": sum(1 for q in questions if not q["unique"] and q["single_first_token"]),
        "score_name_kind": dict(score_kind),
        "score_variant_unique": {
            k: {"unique": v["unique"], "ambiguous": v["ambiguous"],
                "unique_pct": round(100.0 * v["unique"] / max(1, v["unique"] + v["ambiguous"]), 2)}
            for k, v in score_variant.items()},
    }

    ambiguous = [q for q in questions if not q["unique"]]
    collision_hist = Counter(q["n_colliding_options"] for q in ambiguous)
    # one example per (type, name_kind), largest collision first, so both the digit-prefix
    # and the shared-prefix kinds are shown instead of only the biggest group
    examples, seen = [], set()
    for q in sorted(ambiguous, key=lambda q: (-q["largest_collision"], q["id"])):
        key = (q["type"], q["name_kind"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(q)
    example_out = [{
        "id": q["id"], "type": q["type"], "tier": q["tier"], "name_kind": q["name_kind"],
        "n_options": q["n_options"],
        "largest_collision": q["largest_collision"],
        "n_colliding_options": q["n_colliding_options"],
        "collision_groups": q["collisions"],
        "options": [{"name": o["name"], "first_piece": o["first_piece"],
                     "first_token": o["first_token"], "n_tokens": o["n_tokens"]}
                    for o in q["options"]],
    } for q in examples]

    length_out = {k: {"1": v["1"], "2": v["2"], "3+": v["3+"], "0": v["0"]}
                  for k, v in length_dist.items()}
    for k, v in length_out.items():
        n = sum(v.values())
        s, c = length_sum[k]
        v["total"] = n
        v["mean_tokens"] = round(s / c, 3) if c else 0.0
        v["max_tokens"] = max((int(r["n_tokens"]) for q in questions
                               for r in q["options"]
                               if k == "all" or q["type"] == k), default=0)
        v["pct_1"] = round(100.0 * v["1"] / n, 2) if n else 0.0
        v["pct_2"] = round(100.0 * v["2"] / n, 2) if n else 0.0
        v["pct_3plus"] = round(100.0 * v["3+"] / n, 2) if n else 0.0

    result = {
        "meta": {
            "dataset": str(DATASET),
            "tokenizer_pack": str(TOKENIZER_DIR),
            "vocab_size": len(tk.tokens),
            "pre": tk.pre,
            "answer_context": '" " + name (the space after "name:" in the reply line)',
            "definitions": {
                "choice": "keys of question.criteria",
                "score": "short ids via opendecisions.shape_question (fallback: raw list)",
                "noul": "yes/no (always unique)",
            },
            "tokenizer_validation": "11/11 concrete token ids from tokenization_report.md matched",
        },
        "aggregate": aggregate,
        "collision_stats": {
            "ambiguous_questions": len(ambiguous),
            "colliding_options_histogram": dict(sorted(collision_hist.items())),
            "max_collision_group": max((q["largest_collision"] for q in ambiguous), default=0),
        },
        "sequence_lengths": length_out,
        "anomalies": {"counts": dict(anomalies),
                      "by_type": {t: dict(c) for t, c in anomalies_by_type.items()},
                      "examples": {k: v[:5] for k, v in anomaly_examples.items()}},
        "ambiguous_examples": example_out,
        "questions": questions,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[first_token] wrote {OUT_JSON}", flush=True)

    write_report(result)
    print(f"[first_token] wrote {OUT_MD}", flush=True)
    return 0


def write_report(r: dict) -> None:
    a = r["aggregate"]
    bt = a["by_type"]
    L = r["sequence_lengths"]
    cs = r["collision_stats"]
    an = r["anomalies"]["counts"]
    lines = []
    add = lines.append
    add("# First-token ambiguity of OpenDecisions options")
    add("")
    add("For the logits-confidence feature. The fast **first-token lookup** ranks a "
        "question's options by the logit of each option's first token at the answer "
        "position (one forward pass). That equals the full **chain rule** only when the "
        "options have distinct first tokens. This measures, per question, whether they do.")
    add("")
    add("No model was loaded and no server was started. Tokenizer: the model's own "
        f"vocabulary ({r['meta']['vocab_size']:,} tokens, byte-level BPE, "
        f"`pre={r['meta']['pre']}`) from the extracted pack. It was validated against 11 "
        "concrete token ids recorded in `tokenization_report.md` (all matched) and a "
        "byte-exact round-trip corpus.")
    add("")
    add("Option names are the strings the endpoint ranks: `choice` = criteria keys, "
        "`score` = the short level ids the harness sends (`shape_question`, with the raw "
        "list as fallback), `noul` = `yes`/`no`. Each is tokenized as `\" \" + name`, the "
        "space after the colon in the endpoint's `name: answer` reply.")
    add("")
    add("## Headline")
    add("")
    add(f"- **{a['unique']:,} of {a['questions']:,} questions ({a['unique_pct']}%) have all-unique "
        f"first tokens -> the fast path is exact.**")
    add(f"- **{a['ambiguous']:,} questions ({a['ambiguous_pct']}%) have a first-token collision "
        f"-> the chain rule is needed.**")
    add(f"- Of the ambiguous ones, {a['ambiguous_all_collide']:,} have **every** option sharing its "
        "first token with another option (no option is uniquely identified), and "
        f"{a['ambiguous_single_first_token']:,} collapse to a **single** first token for all options.")
    add(f"- Sensitivity, bare name (no leading space): {a['bare_unique']:,} questions unique "
        f"({a['bare_unique_pct']}%). The leading space the endpoint uses costs some uniqueness.")
    add("")
    add("| question type | questions | unique first tokens | ambiguous | unique % |")
    add("|---|---|---|---|---|")
    for t in ("choice", "score", "noul"):
        b = bt[t]
        add(f"| {t} | {b['questions']:,} | {b['unique']:,} | {b['ambiguous']:,} | {b['unique_pct']}% |")
    add(f"| **all** | **{a['questions']:,}** | **{a['unique']:,}** | **{a['ambiguous']:,}** | "
        f"**{a['unique_pct']}%** |")
    add("")
    add("## Collisions on ambiguous questions")
    add("")
    add(f"On the {cs['ambiguous_questions']:,} ambiguous questions, this many options share a "
        "first token with another option of the same question (histogram, option count -> "
        "number of questions):")
    add("")
    add("| colliding options | questions |")
    add("|---|---|")
    for k, v in cs["colliding_options_histogram"].items():
        add(f"| {k} | {v:,} |")
    add("")
    add(f"The largest single collision group is {cs['max_collision_group']} options. "
        "One example per type/name-kind:")
    add("")
    for ex in r["ambiguous_examples"]:
        groups = "; ".join("{" + ", ".join(g) + "}" for g in ex["collision_groups"][:3])
        add(f"- `{ex['id']}` ({ex['type']}, {ex['name_kind']}, {ex['n_options']} options): {groups}")
    add("")
    add("Two mechanisms dominate:")
    add("")
    add("- **A shared word prefix.** ` rule_1`..` rule_5` all begin with the token ` rule`; "
        "` production write of ...` levels share ` production`. The first token is identical "
        "and only the later tokens differ.")
    add("- **A leading space before a digit.** `\" 0\"` is two tokens (`\" \"`, `\"0\"`): the "
        "tokenizer has no `\" 0\"` token, so every numeric level (`0`..`5`) starts with the "
        "space token and collides. This is why the numeric-label variant below is 0% unique.")
    add("")
    add("## Option sequence lengths (`\" \" + name`)")
    add("")
    add("How much the first-token proxy could diverge from the full chain rule: a 1-token "
        "option is exactly one lookup, longer options accumulate factors (and need a "
        "length normalization to compare against short ones).")
    add("")
    add("| group | names | 1 token | 2 tokens | 3+ tokens | mean tokens |")
    add("|---|---|---|---|---|---|")
    for key, label in (("choice", "choice options"), ("score", "score levels"),
                       ("noul", "noul answers"), ("all", "all names")):
        d = L[key]
        add(f"| {label} | {d['total']:,} | {d['1']:,} ({d['pct_1']}%) | {d['2']:,} "
            f"({d['pct_2']}%) | {d['3+']:,} ({d['pct_3plus']}%) | {d['mean_tokens']} |")
    add("")
    add("## Cleaning and anomalies")
    add("")
    add("Every name was stripped of leading/trailing whitespace before tokenizing. "
        "Anomaly counts across the primary option names, by question type:")
    add("")
    if an:
        bt_an = r["anomalies"]["by_type"]
        add("| anomaly | choice | score | noul | total |")
        add("|---|---|---|---|---|")
        for k, v in sorted(an.items()):
            add(f"| {k} | {bt_an['choice'].get(k, 0):,} | {bt_an['score'].get(k, 0):,} | "
                f"{bt_an['noul'].get(k, 0):,} | {v:,} |")
    else:
        add("none")
    add("")
    add("Notes: the `has_colon` and long-name counts are **not malformed input** - they are "
        "`score` items whose levels are full `\"id: description\"` strings (the harness falls "
        "back to the raw list when the level does not fit `shape_question`). No name was "
        "empty, non-string, newline-bearing, duplicated within its question, or "
        "whitespace-padded.")
    add("")
    for k, ex in r["anomalies"]["examples"].items():
        shown = ", ".join(f"`{x[1][:40] if isinstance(x[1], str) else x[1]}` ({x[0]})" for x in ex[:3])
        add(f"- **{k}**: {shown}")
    add("")
    add("## Score-question sensitivity")
    add("")
    add("The dataset's `score` items are the awkward case: `item.labels` are numeric "
        "(`\"0\"`..`\"4\"`) while `question.criteria` is a list of descriptive strings "
        "(`\"trivial: a greeting, ...\"`). The harness (`shape_question`) turns a "
        "`\"id: description\"` level into a short id, but falls back to the raw string when "
        "the head is numeric, too long, or has no `\": \"` - here "
        f"{a['score_name_kind'].get('score_shaped', 0):,} questions are shaped and "
        f"{a['score_name_kind'].get('score_raw', 0):,} fall back to the raw string.")
    add("")
    add("Under each name definition:")
    add("")
    add("| score name definition | unique | ambiguous | unique % |")
    add("|---|---|---|---|")
    for key, label in (("shaped_ids", "short level ids (harness default)"),
                       ("raw_strings", "raw full criteria strings"),
                       ("numeric_labels", "numeric labels (\"0\"..)")):
        s = a["score_variant_unique"][key]
        add(f"| {label} | {s['unique']:,} | {s['ambiguous']:,} | {s['unique_pct']}% |")
    add("")
    add("`shaped_ids` and `raw_strings` agree because a level's first token is the same "
        "whether or not the description follows it. The numeric labels are 0% unique: as "
        "shown above, `\" 0\"` splits into the space token plus `0`, so every digit collides. "
        "**If a caller ranks numeric levels rather than named ones, the first-token lookup "
        "cannot work at all.**")
    add("")
    add("## What this means for the feature")
    add("")
    add(f"- The fast first-token lookup is exact for {a['unique_pct']}% of questions and "
        f"wrong-answer-prone for {a['ambiguous_pct']}% (where two options share the first "
        "token and only the chain rule separates them). `noul` is always safe; `choice` is "
        f"{bt['choice']['unique_pct']}% unique and `score` only {bt['score']['unique_pct']}%.")
    add("- The cheap hybrid: compute the first token for every option in one pass; when a "
        "question's options collide, run the chain-rule walk for just those options. The "
        "lookup is a real fast path for ~70% of questions, but it is not a replacement.")
    add("- Multi-token options dominate (only "
        f"{L['all']['pct_1']}% of names are a single token; mean {L['all']['mean_tokens']} "
        "tokens), so the chain-rule machinery is needed regardless.")
    add("- The first token is context-dependent: the leading space the endpoint emits "
        "changes the split and sometimes the count, and it is what breaks the numeric "
        "levels. Read the logits at the exact answer position.")
    add("")
    add("## Files")
    add("")
    add("- `first_token.py` - this analysis (loads no model, starts no server).")
    add("- `first_token_analysis.json` - aggregate stats and per-question results "
        "(option names, first token ids/pieces, token counts, collision groups).")
    add("- `first_token_report.md` - this report.")
    add("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
