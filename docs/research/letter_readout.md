# The letter readout: the options as A, B, C ...

**Status:** built, unit-tested and measured on two populations. On the 235-item
OpenDecisions sample (2026-10-21) the letter readout scored **82.7** Capability
against the direct readout's **79.3** in the same session, was faster, and needs one
engine request per question where the direct path needs 113 extra over the same
items. **On jev-bench - the sample that carries the human label distributions - it
loses: 48.3 against 55.1, and the mean distance to the humans goes from 0.4439 to
0.4688.** The OpenDecisions win does not generalize; the letter readout is not
ready to be the default. The comparisons are in the companion doc's sections 20 and
21. **2026-10-23: the >26-option fallback was fixed and measured** - it costs 3.8 of
the 6.9 Capability points, and the readout itself costs the other 3.1; the flag stays
opt-in and no default changes (the result is below).

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
calibration.  The companion doc §21 has the whole OpenDecisions comparison, and §22
the jev-bench check: the two numbers that go the other way (the distance to the true
probabilities on the 13 calibration items,
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
it is closer to the humans), and the default stays the direct readout.  The
>26-options fix was done on 2026-10-23 (below): it recovers 3.8 of the 6.9 points and
the readout's own 3.1-point deficit remains, so the flag stays opt-in and nothing
becomes the default.

## The >26-option fallback: the rule (2026-10-23, written before the analysis)

**The question.** The letter readout's jev-bench loss (-6.9 Capability) has two candidate
causes: the 144 of 683 questions with more options than letters (answered by the decode
path, whose 1.0/0.0 distribution calibrates at 18.1), or the letter reading itself (the
539 questions it serves already lose 4 Capability points, measured in the section above).
This settles how much of the loss is the fallback. It is a decision aid, not a new
readout: no default changes on the result.

**The fix under test.** A request with a question that has more options than letters is
rendered and answered on the **direct** path from the start - the same prompt, the same
first-token reading, the same trie walk - instead of being lettered and falling back to
decode. A request that mixes a letterable question with an unletterable one goes direct as
a whole (the endpoint answers one question per prompt by default, so this is the rare
case). This is a small change to `serve/server.py`'s choice of readout; the letter module
is not touched.

**Why this needs no new engine run.** Under the fix, an unletterable question's request is
byte-identical to the direct arm's request for the same item, so its reading is the direct
arm's reading. The fixed arm's predictions are therefore exactly: the **stored direct
run** for the 144 unletterable items, and the **stored letter run** for the other 539.
That recombination is the measurement, and it is paired item by item.

**The arms.** (1) the direct readout, stored (`jevbench_sample3_predictions.jsonl`);
(2) the letter readout, stored (`jevbench_sample3_letter_predictions.jsonl`); (3)
letters + direct fallback, recombined as above. Same items, same order, same scorer.

**The rule.** "The fallback was the loss" only if the recombined arm's **official
Capability** is within **1.8 points** of the direct arm's (the programme's measured
run-to-run spread) **and** its mean **TVD to the humans** on the 280 items with a human
distribution is no worse by more than **0.005**. Anything larger on either means the loss
is the letter reading, and the 539 served items' own loss stands. The per-type numbers and
the two subsets are reported either way.

**What the result decides.** If the fallback is the loss, the letter readout's flag gets
the fix and a re-measurement before any default question. If it is not, the flag stays
opt-in and unused (or is deleted) as `letter_readout.md`'s recommendation already says -
and the fix is still worth having, because a decode-path answer with a 1.0/0.0
distribution is wrong for any caller that reads probabilities, whatever the arm is called.

## The >26-option fallback: the result (2026-10-23)

**The fix is built** (`serve/server.py`, `_systemone_single` and `_systemone_classic`): a
question with more options than letters is rendered and read on the **direct** path, and a
mixed request goes direct as a whole.  The test is
`serve/test_systemone.py::LetterReadout::test_past_26_options_the_server_answers_it_on_the_direct_path`
(27 options, one trie step, the fast path's probabilities keyed by the option names).  The
default path is untouched: the branch only exists in letter mode.

**The analysis** (`tools/decision_policy/letter_fallback.py`, on the two stored runs):

| arm | Capability | Intelligence | Calibration | mean TVD to the humans |
|---|---:|---:|---:|---:|
| the direct readout | **55.2** | **46.7** | **63.7** | **0.443** |
| the letter readout (as measured) | 48.3 | 44.4 | 52.2 | 0.469 |
| letters + the direct fallback | 52.1 | 44.8 | 59.5 | 0.449 |

| subset | arm | accuracy | ECE | human TVD | mode mass |
|---|---|---:|---:|---:|---:|
| >26 options (144, 50 with humans) | direct | 61.1% | 0.2177 | 0.7219 | 0.6757 |
| | letters | 59.0% | 0.4097 | 0.8327 | **1.0000** |
| | fixed | 61.1% | 0.2177 | 0.7219 | 0.6757 |
| the 539 the letters serve (230 with humans) | direct | 67.7% | 0.1877 | 0.3826 | 0.8576 |
| | letters | 66.8% | 0.2225 | 0.3897 | 0.8761 |
| | fixed | 66.8% | 0.2225 | 0.3897 | 0.8761 |

**The rule: the fallback is NOT the whole loss.** The fixed arm's Capability is **3.1
points** below the direct arm's (the margin is 1.8) and its human TVD is **0.0060** worse
(the margin is 0.005).  The decomposition is clean:

* **The fallback costs 3.8 of the 6.9 Capability points** - 55% of the loss - and 7.3 of
  the 11.5 calibration points.  The mode mass of **1.0000** on those 144 items is the
  decode path's degenerate distribution, seen directly.
* **The readout costs the other 3.1**, on the 539 items it does serve: accuracy 0.9 points
  worse, ECE 0.035 higher, the human TVD 0.007 higher, and the mode mass 0.019 *higher*
  than the direct arm's - the sharpening §22 measured, still there.  The letter readout is
  a worse reading even where it works.

**The decision.** The flag stays **opt-in**; nothing becomes the default.  The fix ships
anyway, for two reasons: a decode-path answer that reports 1.0 on one option and 0.0 on the
rest is wrong for any caller that reads probabilities, and it recovers more than half of
the flag's loss.  The flag's remaining 3.1-point deficit is the readout's own and is not a
bug to fix; a user who turns the flag on gets a faster, simpler readout that is slightly
worse against people.  If the owner would rather remove the flag than carry that, this
result says removing it costs nothing measurable: no default uses it.

**The limits.** The fixed arm's numbers for the 144 items are the **stored direct run's**
readings, not a fresh run's.  The code path is identical (the same prompt, the same
functions), but the engine's logits wobble run to run (§5.2 of `prompt_variants.md`), so
this is a paired estimate inside the 1.8-point noise floor, not a byte-exact transplant.
The 235 cannot check it: its widest question has 7 options.  One model, one quantization,
the jev-bench test sample only.


## How to measure it again

Both arms run over the same 235 ids, from the same build, on a freshly restarted
engine (numbers from a long-running engine are invalid - the companion doc §12.5).

In the commands below, `<opencode temp>` is the temporary run directory (on this
machine, `%LOCALAPPDATA%\Temp\opencode`).

```sh
# the engine: stop the running one, start it again with the arm's config, wait for
# GET http://127.0.0.1:8080/v1/status to answer
C:\tools\strata\.venv\Scripts\python.exe strata.py serve --config <arm>.json

# the run (the arm's config decides the readout)
C:\tools\strata\.venv\Scripts\python.exe tools\decision_policy\opendecisions.py \
  --items <opencode temp>\opendecisions\opendecisions.jsonl \
  --out <arm>.jsonl \
  --ids-file <opencode temp>\opendecisions\sample10.txt \
  --batch-k 1

# the score (the official scorer)
C:\tools\strata\.venv\Scripts\python.exe <opencode temp>\opendecisions\score.py \
  <arm>.jsonl --data <opencode temp>\opendecisions\sample10_items.jsonl

# the comparison (the picks that moved, the fast-path rate, the latency)
C:\tools\strata\.venv\Scripts\python.exe <opencode temp>\opendecisions\compare_arms.py \
  --items <opencode temp>\opendecisions\sample10_items.jsonl \
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
