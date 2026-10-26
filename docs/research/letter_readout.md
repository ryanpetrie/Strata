# The letter readout: the options as A, B, C ...

**Status:** built, unit-tested and measured on two populations. On the 235-item
OpenDecisions sample (2026-10-21) the letter readout scored **82.7** Capability
against the direct readout's **79.3** in the same session, was faster, and needs one
engine request per question where the direct path needs 113 extra over the same
items. **On jev-bench - the sample that carries the human label distributions - it
loses: 48.3 against 55.1, and the mean distance to the humans goes from 0.4439 to
0.4688.** The OpenDecisions win does not generalize; the letter readout is not
ready to be the default. The comparisons are in the companion doc's sections 20 and
21.

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
comparison of the temperatures. The measurement below was made with **no
temperature in either arm**: the fitted constants are not being shipped (companion
doc §19), so the comparison is at T = 1, against the readout that ships.

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

## The measurement (2026-10-21)

Both arms ran over the same 235 ids, in one session, on a freshly restarted engine
each time (numbers from a long-running engine are invalid - the companion doc
§12.5), with **no temperature in either arm**: the fitted temperatures are not being
shipped (companion doc §19), so T = 1 compares the letter readout against the
readout that ships.  The arms differ in one key of the launch config:
`"systemone": {"readout": "letters"}`.

| | the direct readout | the letter readout | the stored baseline |
|---|---:|---:|---:|
| Capability | 79.3 | **82.7** | 79.8 |
| Intelligence | 77.4 | **82.4** | 78.0 |
| Calibration | 81.1 | **83.0** | 81.7 |
| predicted | 235 of 235 | 235 of 235 | 235 of 235 |
| fast path | 235 of 235 | 235 of 235 | 235 of 235 |
| requests that failed | 0 | 0 | 0 |
| latency median | 3951 ms | **3606 ms** | 3121 ms |
| latency p95 | 7288 ms | **5427 ms** | 5682 ms |
| extra engine requests per item | 113 over 235 items | **0** | - |

**The letter readout wins the official Capability by 3.4 points** against the arm
measured in the same session, and it is faster and simpler.  **But it does not pick
more items right**: 204 of the 235 against the direct readout's 206.  The win is in
the shape of the distribution - the score questions' expected level sits closer to
the gold level (the mean normalized distance 0.0908 against 0.1013) - and in the
calibration.  The companion doc §20 has the whole comparison, the two numbers that
go the other way (the distance to the true probabilities on the 13 calibration items,
0.256 against 0.216, and the score questions' raw accuracy, 72.9% against 79.7%),
and the limits.

**No position bias**: the letter readout picked the first letter on 25.3% of the 166
choice and score items; an unbiased readout would pick it 24.5% of the time.

The exit criterion (roadmap item 1.3) - ship it only if it wins on at least two of
{Capability, latency, simplicity} without losing the other - **is met on this
sample**: it wins Capability and simplicity, and the latency does not lose.

## The second population: jev-bench, against the humans

The check that decides the default question, on the sample that carries the **human
label distributions** (683 items, 280 of them with a human distribution over the
options).  The direct arm's numbers there were measured by roadmap item 1.2; the
letter arm was run on 2026-10-21, on a freshly restarted engine, with the same
harness and the same ids.

| | the direct readout | the letter readout |
|---|---:|---:|
| Capability | **55.1** | 48.3 |
| Intelligence | **46.6** | 44.4 |
| Calibration | **63.6** | 52.2 |
| choice accuracy / calibration | 71.1% / 69.3 | 69.7% / 44.9 |
| yes/no accuracy / calibration | 74.3% / 57.8 | 72.5% / 56.5 |
| score accuracy / calibration | 48.5% / 63.6 | 47.3% / 55.2 |
| the mean TVD to the humans | **0.4439** | 0.4688 |
| the TVD, choice / yes/no / score | **0.4699** / 0.3844 / **0.4955** | 0.5268 / **0.3559** / 0.5437 |
| predicted / failures | 683 / 0 | 683 / 0 |
| the fast path | 683 | 539 (144 questions have more options than there are letters) |

**The letter readout loses by 6.9 Capability points, and the loss is the
calibration.**  Two reasons, both measured:

* **144 of the 683 questions have more than 26 options** (go_emotions with 28,
  ledgar with 100, massive with 60, banking77 with 77, clinc150 with 151), and there
  are only 26 letters.  Those questions are answered from the decode path, which
  reports a degenerate 1.0/0.0 distribution: their calibration collapses from 56.5 to
  18.1.  That is a fix, not a verdict - those questions should be read the direct way.
* **Even on the 539 questions the letter readout did serve, it loses**: Capability
  57.8 against 53.8, the calibration 67.0 against 60.8.  The choice questions' accuracy
  is the same (80.0% against 80.0%).

**The hedging hypothesis is refuted.**  The letter readout's distributions are
**sharper**, not flatter: the choice questions' entropy is 0.188 bits against the
direct readout's 0.531, the mass on the mode 0.950 against 0.866.  On the items with
a human distribution the humans carry 1.144 bits, the direct readout answers with
0.803 and the letter readout with 0.310 - twice as certain as the direct readout and
far more certain than the humans, which is what the calibration component measures.
The score questions' gain on OpenDecisions (the expected level closer to the gold)
does not transfer: on jev-bench the same sharpening moves the expected level further
from the gold.

**The recommendation: do not ship it as the default.**  The exit criterion was met on
the OpenDecisions sample; it does not survive the population with the humans' labels.
The flag stays (it is faster and simpler where it works, and on the yes/no questions
it is closer to the humans), the default stays the direct readout, and the next steps
are the >26-options fix and a paired re-measurement on both populations.

## How to measure it again

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
