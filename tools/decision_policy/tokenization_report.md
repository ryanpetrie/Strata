# Tokenization check: are answer options single tokens?

For the logits-confidence feature. Question: for each answer-relevant string, is it a
single token in the Qwen3.8-Flash-Next tokenizer, or a multi-token sequence? A single
token needs one logit lookup; a multi-token sequence needs a chain-rule walk through the
KV cache.

## Method

- **Tokenizer:** the model's own vocabulary, read from the GGUF metadata
  (`tokenizer.ggml.tokens`, `.token_type`, `.merges`) at
  `C:\tools\Strata-data\models\unsloth-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`.
  It is byte-level BPE, `model = gpt2`, `pre = qwen35`, **248,320 tokens**, **247,587 merges**.
- The GGUF vocabulary, token types and merges are **byte-identical** to the tokenizer pack
  at `C:\tools\Strata-data\packs\unsloth-ud-iq4_xs\tokenizer` (`vocab.json`, `merges.txt`,
  `token_type.json`), so either source gives the same answer.
- Tokenization was done with the project's reference implementation of record,
  `tools/strata_tokenizer.py` (`Tokenizer.from_gguf`). Every string was independently
  re-tokenized with a separate byte-level BPE written for this check; the two agree on all
  2,698 strings. The `qwen35` pre-tokenizer regex used is the one from
  `third_party/llama.cpp/src/llama-vocab.cpp` L396.
- Two forms are reported per string: the **bare** string, and the string with one
  **leading space** (`" STOP"`). The leading space matters: in a rendered prompt an option
  is almost always preceded by a space or newline, and the space changes the first token.
- No model was loaded and no server was started. The API key in the config was never read
  or printed.

## Action vocabulary

| string | single token (bare)? | bare ids / pieces | single token (`" "+s`)? | with-space ids / pieces |
|---|---|---|---|---|
| `CONTINUE`  | no  | `[22587, 47139]` = `CONT`+`INUE` | no  | `[15645, 47139]` = `ĠCONT`+`INUE` |
| `CALL_TOOL` | no  | `[24622, 55274]` = `CALL`+`_TOOL` | no  | `[24932, 55274]` = `ĠCALL`+`_TOOL` |
| `ANSWER`    | no  | `[11355, 38050]` = `ANS`+`WER` | no  | `[93365, 622]` = `ĠANSW`+`ER` |
| `ASK_USER`  | no  | `[7150, 8834]` = `ASK`+`_USER` | no  | `[5574, 42, 8834]` = `ĠAS`+`K`+`_USER` |
| `STOP`      | **yes** | `[48964]` = `STOP` | **yes** | `[44020]` = `ĠSTOP` |

Only **`STOP`** is a single token. The other four are two tokens, and `ASK_USER` becomes
**three** tokens once a leading space is present.

## OpenDecisions names

Extracted from `opendecisions.jsonl` (2,352 lines, JevBench format). Each line's
`question` has a `type` of `choice` (1,178), `noul` (586) or `score` (588).

- `choice`/`noul` options are the keys of `question.criteria` (a dict).
- `score` levels are the part before `": "` in each entry of `question.criteria` (a list) -
  the same short ids `tools/decision_policy/opendecisions.py` `shape_question()` produces.

| group | strings | single token (bare) | multi token (bare) | single (`" "+s`) | single in either form |
|---|---|---|---|---|---|
| choice options | 2,268 | 122 (5.4%) | 2,146 | 175 | 178 |
| noul options (`true`,`false`) | 2 | 2 | 0 | 2 | 2 |
| score levels | 423 | 67 (15.8%) | 356 | 88 | 96 |
| **OpenDecisions total** | **2,693** | **191 (7.1%)** | **2,502 (92.9%)** | **265** | **276 (10.2%)** |

Token-count distribution (bare form):

- choice options: 1 token 122, 2: 659, 3: 792, 4: 408, 5: 161, 6: 70, 7: 40, 8: 7, 9: 5, 10: 4
- score levels: 1 token 67, 2: 136, 3: 94, 4: 62, 5: 21, 6: 23, 7: 6, 8: 1, 9: 2, 10: 5, 11: 4, 12: 1, 14: 1
- Mean length: ~3.1 tokens per name.

The model's actual `noul` answers are `yes`/`no` (see `to_probs`), not `true`/`false`.
Both are single tokens: `yes` = `[9405]`, `no` = `[2083]`, `Yes` = `[9175]`, `No` = `[2665]`.

### Examples of multi-token names

- `trivial` -> `tr`+`ivial`
- `involved` -> `inv`+`olved`
- `extraction_json` -> `extract`+`ion`+`_json` (bare)
- `Low stakes` -> `Low`+`Ġstakes`
- `Read-only` -> `Read`+`-only`
- `On that topic, and meets exactly one of` -> 14 tokens
- non-Latin names split into byte-level pieces: `Низька` -> `Н`+`из`+`ька`, `計算不要` -> 2 tokens

## Surprises

1. **Multi-token is the rule, not the exception.** 92.9% of OpenDecisions names are
   multi-token, average ~3.1 tokens. The chain-rule walk is the common path. A plain logit
   lookup only covers ~7% of names (and, at best, ~10% counting the leading-space form).
2. **Only one of the five action words is a single token.** `STOP` is one token; `CONTINUE`,
   `CALL_TOOL`, `ANSWER` and `ASK_USER` are two. Any action-probability read must handle
   multi-token actions for four of the five.
3. **A leading space changes the split, sometimes the count.** `ASK_USER` is 2 tokens bare
   but 3 tokens as `" ASK_USER"` (`ĠAS`+`K`+`_USER`). `ANSWER` is `ANS`+`WER` bare but
   `ĠANSW`+`ER` with a space. Even single-token names can break under a space: `faq`,
   `structuring` and `dpa` are single tokens bare but multi-token with a leading space,
   while 56 choice options and 29 score levels are the reverse. The first-token logit to
   read depends on the exact context (space, newline, or none).
4. **Some "option names" are long phrases or non-Latin.** Score levels include full
   sentences up to 14 tokens (`On that topic, and meets exactly one of`) and scripts
   (Cyrillic, CJK, Devanagari, Arabic) that split into many byte-level pieces. These are
   poor candidates for a single-step probability read.
5. **No name collides with a special token.** None of the 2,698 strings contains a
   `<|...|>`, `<think>`, or `<tool_call>` literal, so no special-token handling is needed
   for these names.

## What this means for the feature

- The design must treat a **multi-token sequence as the normal case**. Compute the joint
  probability of an option with the chain rule: probability of the first token given the
  prefill context, times the probability of each later token given the earlier ones, each
  read from the model's logits at the corresponding position.
- For a **single-token option** the chain rule degenerates to one lookup, so one code path
  can cover both cases; no separate "simple" branch is required.
- The **first token depends on the surrounding text**. Read the first-token logits at the
  exact position where the option would be emitted, with the same leading whitespace the
  prompt uses (space, newline, or none). Comparing options of different lengths also needs
  a length normalization, since a 3-token option accumulates more factors than a 1-token one.
- The five action words should be treated as sequences, not single tokens, except `STOP`.

## Files

- Raw results (per string: name, group, `is_single_token`, token ids and pieces for the bare
  and leading-space forms, token counts, token-type): `tools/decision_policy/tokenization_results.json`
- This report: `tools/decision_policy/tokenization_report.md`
