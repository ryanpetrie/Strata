# First-token ambiguity of OpenDecisions options

For the logits-confidence feature. The fast **first-token lookup** ranks a question's options by the logit of each option's first token at the answer position (one forward pass). That equals the full **chain rule** only when the options have distinct first tokens. This measures, per question, whether they do.

No model was loaded and no server was started. Tokenizer: the model's own vocabulary (248,320 tokens, byte-level BPE, `pre=qwen35`) from the extracted pack. It was validated against 11 concrete token ids recorded in `tokenization_report.md` (all matched) and a byte-exact round-trip corpus.

Option names are the strings the endpoint ranks: `choice` = criteria keys, `score` = the short level ids the harness sends (`shape_question`, with the raw list as fallback), `noul` = `yes`/`no`. Each is tokenized as `" " + name`, the space after the colon in the endpoint's `name: answer` reply.

## Headline

- **1,649 of 2,352 questions (70.11%) have all-unique first tokens -> the fast path is exact.**
- **703 questions (29.89%) have a first-token collision -> the chain rule is needed.**
- Of the ambiguous ones, 278 have **every** option sharing its first token with another option (no option is uniquely identified), and 268 collapse to a **single** first token for all options.
- Sensitivity, bare name (no leading space): 1,737 questions unique (73.85%). The leading space the endpoint uses costs some uniqueness.

| question type | questions | unique first tokens | ambiguous | unique % |
|---|---|---|---|---|
| choice | 1,178 | 813 | 365 | 69.02% |
| score | 588 | 250 | 338 | 42.52% |
| noul | 586 | 586 | 0 | 100.0% |
| **all** | **2,352** | **1,649** | **703** | **70.11%** |

## Collisions on ambiguous questions

On the 703 ambiguous questions, this many options share a first token with another option of the same question (histogram, option count -> number of questions):

| colliding options | questions |
|---|---|
| 2 | 290 |
| 3 | 100 |
| 4 | 243 |
| 5 | 67 |
| 6 | 3 |

The largest single collision group is 6 options. One example per type/name-kind:

- `jb16-llm_router-057d3467e2` (score, score_raw, 6 options): {0: read-only, auto-run, 1: dev/staging writes only, auto-run and logged, 2: production write under 1,000 rows, requester confirms, 3: production write of 1,000-49,999 rows, one approver, 4: production write of 50,000+ rows or production schema change, two approvers and change ticket, 5: destructive production operation, blocked}
- `jb16-composite_gen-6f641b54b1` (choice, choice, 6 options): {rule_1, rule_2, rule_3, rule_4, rule_5}
- `jb16-fairness-0edc52b9eb` (score, score_shaped, 5 options): {Band 0, Band 1, Band 2, Band 3, Band 4}

Two mechanisms dominate:

- **A shared word prefix.** ` rule_1`..` rule_5` all begin with the token ` rule`; ` production write of ...` levels share ` production`. The first token is identical and only the later tokens differ.
- **A leading space before a digit.** `" 0"` is two tokens (`" "`, `"0"`): the tokenizer has no `" 0"` token, so every numeric level (`0`..`5`) starts with the space token and collides. This is why the numeric-label variant below is 0% unique.

## Option sequence lengths (`" " + name`)

How much the first-token proxy could diverge from the full chain rule: a 1-token option is exactly one lookup, longer options accumulate factors (and need a length normalization to compare against short ones).

| group | names | 1 token | 2 tokens | 3+ tokens | mean tokens |
|---|---|---|---|---|---|
| choice options | 5,135 | 907 (17.66%) | 1,989 (38.73%) | 2,239 (43.6%) | 2.537 |
| score levels | 2,376 | 464 (19.53%) | 316 (13.3%) | 1,596 (67.17%) | 6.58 |
| noul answers | 1,172 | 1,172 (100.0%) | 0 (0.0%) | 0 (0.0%) | 1.0 |
| all names | 8,683 | 2,543 (29.29%) | 2,305 (26.55%) | 3,835 (44.17%) | 3.436 |

## Cleaning and anomalies

Every name was stripped of leading/trailing whitespace before tokenizing. Anomaly counts across the primary option names, by question type:

| anomaly | choice | score | noul | total |
|---|---|---|---|---|
| has_colon | 0 | 215 | 0 | 215 |
| len_gt_128 | 0 | 14 | 0 | 14 |
| len_gt_40 | 0 | 426 | 0 | 426 |

Notes: the `has_colon` and long-name counts are **not malformed input** - they are `score` items whose levels are full `"id: description"` strings (the harness falls back to the raw list when the level does not fit `shape_question`). No name was empty, non-string, newline-bearing, duplicated within its question, or whitespace-padded.

- **has_colon**: `None: copy, format or restate.` (jb16-llm_router-8c4dde74db), `Low stakes: a minor, easily reversible d` (jb16-llm_router-74eda20fbc), `計算不要:答えは文中にそのまま書かれている。` (jb16-llm_router-7a24a8c84a)
- **len_gt_40**: `A multi-step calculation (three or more ` (jb16-llm_router-8c4dde74db), `General information with no decision at ` (jb16-llm_router-74eda20fbc), `Low stakes: a minor, easily reversible d` (jb16-llm_router-74eda20fbc)
- **len_gt_128**: `Affects a person's employment or legal s` (jb16-llm_router-7dc3580c3b), `1 : dysfonctionnement d'une fonction sec` (jb16-multilingual-ffad820e9a), `Kontoen sperres og saken går til sikkerh` (jb16-multilingual-e2cd1530a3)

## Score-question sensitivity

The dataset's `score` items are the awkward case: `item.labels` are numeric (`"0"`..`"4"`) while `question.criteria` is a list of descriptive strings (`"trivial: a greeting, ..."`). The harness (`shape_question`) turns a `"id: description"` level into a short id, but falls back to the raw string when the head is numeric, too long, or has no `": "` - here 238 questions are shaped and 350 fall back to the raw string.

Under each name definition:

| score name definition | unique | ambiguous | unique % |
|---|---|---|---|
| short level ids (harness default) | 250 | 338 | 42.52% |
| raw full criteria strings | 250 | 338 | 42.52% |
| numeric labels ("0"..) | 0 | 588 | 0.0% |

`shaped_ids` and `raw_strings` agree because a level's first token is the same whether or not the description follows it. The numeric labels are 0% unique: as shown above, `" 0"` splits into the space token plus `0`, so every digit collides. **If a caller ranks numeric levels rather than named ones, the first-token lookup cannot work at all.**

## What this means for the feature

- The fast first-token lookup is exact for 70.11% of questions and wrong-answer-prone for 29.89% (where two options share the first token and only the chain rule separates them). `noul` is always safe; `choice` is 69.02% unique and `score` only 42.52%.
- The cheap hybrid: compute the first token for every option in one pass; when a question's options collide, run the chain-rule walk for just those options. The lookup is a real fast path for ~70% of questions, but it is not a replacement.
- Multi-token options dominate (only 29.29% of names are a single token; mean 3.436 tokens), so the chain-rule machinery is needed regardless.
- The first token is context-dependent: the leading space the endpoint emits changes the split and sometimes the count, and it is what breaks the numeric levels. Read the logits at the exact answer position.

## Files

- `first_token.py` - this analysis (loads no model, starts no server).
- `first_token_analysis.json` - aggregate stats and per-question results (option names, first token ids/pieces, token counts, collision groups).
- `first_token_report.md` - this report.
