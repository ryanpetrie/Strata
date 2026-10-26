# The letter readout: the options as A, B, C ...

**Status:** built and unit-tested, not yet measured. The 235-item OpenDecisions
sample has not been run with the letter readout (the engine was busy with another
measurement), so this document says what the readout is and what it has to be
measured against; the numbers go here when the run is done.

**Date:** 2026-10-21

This is item 1.3 of the decision-policy roadmap: the letter-readout benchmark. The
question is whether Strata should read the option's own tokens (what it does today)
or the letter of a lettered choice (what most of the literature does).

## What the letter readout is

The options are shown with a letter each and the model answers with the letter.

```
[q | choice] Classify the task type of this request for routing.
  A) extraction_json ("Pull given details into structured fields.")
  B) translation ("Render given text into another language.")
  C) coding ("Write or fix program code.")
  D) creative_writing ("Compose original prose or verse.")
  answer with exactly one of these letters: A, B, C, D
  start your answer line with exactly "q:"
```

The prompt is Strata's own prompt with the options lettered, so the comparison
against the direct path is about the readout and not about two different prompts.
(The OpenDecisions set's own reference implementation, `predict_letters.py`, uses
another shape entirely: a JSON message with "evidence", "criterion" and "options".
Using that would have changed the prompt and the readout at the same time.)

The answer line's prefix is unchanged: the assistant's turn is still prefilled with
`q:`, so the next token the model writes is the answer itself - here the letter.

**The letters never leave the server.** The response is the shape the endpoint
already has: a choice's probabilities are keyed by the option names, a score's by
the level indices, a yes/no's value is the probability of yes. The same scorer reads
both arms.

## How it is turned on

The config's `"systemone"` block, one key:

```json
"systemone": {"enabled": true, "diagnostics": true, "readout": "letters"}
```

Without the key the readout is `"tokens"`: today's path, byte for byte. A value that
is neither `"tokens"` nor `"letters"` refuses the start, like the other config
values do.

The per-type temperature (the config's `"systemone": {"temperature": ...}`) scales
the letter reading exactly as it scales the direct one - the same
`normalize(closed ** (1/T))`, through the same `apply_temperature`. Without that the
two arms would report different things, and the comparison between them would be a
comparison of the temperatures.

## The tokenization, measured

The letter has to be ONE token at the answer position, or the reading is the trie
walk's problem again. Measured with the model's own tokenizer
(`tools/strata_tokenizer.py`, the pack the server loads):

| letter | bare | space-merged (after `q:`) |
|---|---|---|
| A | `[32]` | `[357]` |
| B | `[33]` | `[417]` |
| C | `[34]` | `[351]` |
| ... | ... | ... |
| Z | `[57]` | `[1799]` |

All 26 letters are one token in both forms. The reading uses both: the answer
position is a prefix that ends at a colon, so the model writes ` A` (the space
belongs to the letter's first token), but the bare `A` is the same answer, and the
probabilities of the two forms are added. The reading does not depend on which form
the model happens to write.

**Past 26 options:** there are 26 letters, so a question with more options than
that has no letter for every option. Such a question is shown and answered by name,
the way the direct path shows it, and no letter is read for it. The letter set is not
extended with `AA` or `AB`: those are two tokens, and a two-token answer is the trie
walk's problem again. The 235-item sample has no such question (its widest question
has 7 options).

## The cost, measured without the engine

The letters are one token each and no two of them are the same token, so the options
cannot collide, so the trie walk is never needed: the row at the answer position
answers the question by itself. That is a structural fact, and it holds on the real
data (`tools/decision_policy/letter_costs.py`, the server's own tokenizer, the
235-item sample):

| readout | items with a first-token collision | extra engine requests |
|---|---|---|
| the options' own tokens | 72 of 235 | 113 (one to seven per colliding item) |
| the letters | 0 of 235 | 0 |

(The direct path's numbers are `tools/decision_policy/trie_costs.py` over the same
235 items with the same prefix.)

## The baseline to beat, re-scored

The baseline is the serial run of the 235 items (`fix_serial.jsonl` and its run log).
Re-scored from those files with the official scorer, it reads:

| | |
|---|---|
| Capability | 79.8 (Intelligence 78.0, Calibration 81.7) |
| predicted | 235 of 235 |
| fast path | 235 of 235 |
| requests that failed | 0 (no 502) |
| latency per question | median 3121 ms, p95 5682 ms |

Those are the numbers the letter arm has to beat or match.  They come from the
baseline's own files, not from a new run, and the engine that run used is not
recorded here - so the direct arm is run again, next to the letter arm, before the
two are compared (numbers from a long-running engine are invalid, companion doc
§12.5).

## What is not measured yet

Everything that needs the model:

- the Capability the official scorer reports for the letter arm;
- the fast-path rate and the rate of answers the endpoint could not parse (a 502);
- the latency per question (median and p95);
- how often the model's top token at the answer position is a letter at all, and
  whether the model prefers the first letter (the position bias every lettered
  readout has).

Until those are measured, the letter readout is an option, not a result. The
roadmap's rule applies: ship it only if it wins on at least two of {Capability,
latency, simplicity} without losing the other.

## The measurement (Phase B)

Both arms run over the same 235 ids, from the same build, on a freshly restarted
engine (numbers from a long-running engine are invalid - the companion doc §12.5).

```sh
# the engine: stop the running one, start it again with the arm's config, wait for
# GET http://127.0.0.1:8080/v1/status to answer
C:\tools\strata\.venv\Scripts\python.exe strata.py serve --config <arm>.json

# the run (the arm's config decides the readout)
C:\tools\strata\.venv\Scripts\python.exe tools\decision_policy\opendecisions.py \
  --items C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\opendecisions.jsonl \
  --out <arm>.jsonl \
  --ids-file C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\sample10.txt \
  --batch-k 1

# the score (the official scorer)
C:\tools\strata\.venv\Scripts\python.exe C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\score.py \
  <arm>.jsonl --data C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\sample10_items.jsonl

# the comparison (the picks that moved, the fast-path rate, the latency)
C:\tools\strata\.venv\Scripts\python.exe C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\compare_arms.py \
  --items C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions\sample10_items.jsonl \
  --before <direct>.jsonl --after <letters>.jsonl \
  --before-runlog <direct>.runlog.jsonl --after-runlog <letters>.runlog.jsonl
```

The two run configs are copies of the live run config that differ in one key
(`"systemone": {"readout": ...}`); `make_letter_configs.py` (next to the dataset,
outside the repository) writes them.

## Files

| File | What it is |
|---|---|
| `serve/systemone_letters.py` | the readout: the lettered prompt, the letters' token ids, the reading, the parsers |
| `serve/systemone.py` | the direct path; the two readouts have the same shape, so the server does not change |
| `serve/server.py` | the config's `"readout"` key and the switch |
| `serve/test_systemone.py`, `serve/test_logits_first_token.py` | the tests |
| `tools/decision_policy/letter_costs.py` | the letter readout's engine-request cost over a set |
