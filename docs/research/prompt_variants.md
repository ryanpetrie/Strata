# Per-type prompt variants: a pre-registered ablation

**Status:** the design is written and **the mechanism is built** (2026-10-21).
Nothing has been **run**: no engine was started for this document, no request went
to a live server, and no variant has been measured. Every number below is either
**measured** (and says where) or **static analysis** (and says so). The
implementation is in §4.4, and what it changed about the design is in §4.6.

This is the plan for roadmap item 1.5
([decision_policy_roadmap.md](decision_policy_roadmap.md) §1.5). It is a plan, not
a result. It says what the prompt looks like today, what could be changed in it,
how a change would be selected at serve time, and how a change would be measured so
that the measurement cannot be a hill-climb.

**Where this sits.** §1.5 of the roadmap is a **working-copy** item: it is in the
orchestrator's uncommitted roadmap edits, not yet in a commit. The companion doc's
§§12.5, 14, 18 and 19 that this document cites are in the committed
`shared_backbone_decision_policy.md`. The licence record
(`decision_policy_training_data.md` §7) is on the data branch
`feat/data-softlabels`; it is cited here, not copied.

**The question the user asked:** "do we have any plans to test different
instructions/preamble based on question type?" This document is the plan. The
short answer is in §7: the plan as the roadmap writes it would not be able to
ship the fix it is looking for, and this document changes the rule in three
places to fix that.

**How to read this**

- **Measured** means a number came out of a file on this machine in this session, or
  out of the companion doc by section. The source is named with every number.
- **Static analysis** means a reading of the code, or a hypothesis about the model.
  It is not a finding. A static-analysis claim is never used to justify a decision;
  it is used to decide what to *measure*.
- Files are named with line numbers so a reviewer can check the map.

---

## 0. The short version

1. The noul's template is **four lines**: the header with the question, the
   criteria line, `answer yes or no`, and the answer-line instruction. §1 has all
   nine prompts rendered, literally.
2. The noul's failure is a **bias**, and a bias has two possible homes: the
   **template** (the order the two sides are named, the frame around the criteria)
   or the **task** (the question's own meaning). The template is a cheap thing to
   change; the task is not. §2 says which parts of the template could plausibly
   produce a yes-bias. All of it is static analysis.
3. §3 lists **five** noul variants, **four** choice variants and **four** score
   variants, one axis each, and marks the ones that are **base-rate shifts** - the
   kind that flatter the TVD without making the model better at ranking. It also
   gives one **diagnostic probe** that is not a shipping candidate.
4. §4 is the mechanism: a per-type variant in the launch config, absent = today's
   bytes exactly, an unknown value refused at start-up, the variant reported in
   `/v1/status`. With the file:line map. **It is built** (§4.4); it is not
   measured.
5. §5 is the measurement design: the design set, the measured noise floor, the
   power arithmetic, the GPU budget, and the yardsticks.
6. §6 is the **pre-registered decision rule**, written before the run.
7. §4.5 is the probe's usage, and §4.6 is what the implementation changed.

---

## 1. The template, as rendered today

Everything in this section was rendered by the worktree's own
`serve/systemone.py` and `serve/frontend.py` on 2026-10-21, from the three real
item files named in each heading. The question name is `q` and the score levels are
shaped, because that is what the two harnesses
(`tools/decision_policy/opendecisions.py`, `tools/decision_policy/jevbench.py`) send.
The text is the text; nothing is paraphrased.

### 1.1 The parts that never change

**The system turn** (`serve/systemone.py:86-90`), verbatim, and the same for every
question type:

```
You are a classification module. Read the state, then answer every question on its own line as `name: answer`, with no other text.
Copy each question name exactly. Do not repeat the question, the type, or the instructions.
- A yes/no question takes exactly `yes` or `no`.
- A choice question takes exactly one option name, copied exactly.
- A score question takes exactly one level name, copied exactly.
```

**The user turn's preamble** (`serve/systemone.py:373-376`), the same for every
question type:

```
State:
<the state>

Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.

Questions:
```

(The shared prompt path, `render_prompt` at `serve/systemone.py:333-358`, has the
same preamble with "every question" and a closing `Valid line starts:` line. It is
not the default path; the default answers one question per prompt, so every answer
carries the model's own probability.)

### 1.2 The noul

**OpenDecisions**, `sample10_items.jsonl`, item `jb16-llm_router-7fd07f8b0f`
(gold `no`, and the item carries `trap_answer: "yes"`):

```
State:
Rule: attach the expert advisor to production outages and security incidents.
Message: "how do i change my profile picture in the app?"


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | noul] Should the router attach the expert advisor to this message?
  yes means "The message is about a production outage or a security incident." ; no means "The message is about neither a production outage nor a security incident."
  answer yes or no
  start your answer line with exactly "q:"
```

**jev-bench**, `jevbench_sample3_items.jsonl`, item `civil_comments/test/32284`
(gold `no`; the humans' distribution is `{"yes": 0.0, "no": 1.0}`):

```
State:
sour grapes?


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | noul] Is this comment toxic?
  yes means "Rude, disrespectful, or likely to make someone leave the discussion." ; no means "Civil, even if critical or blunt."
  answer yes or no
  start your answer line with exactly "q:"
```

**The licence-clean pool**, `decision_policy_training/items/p1b_pool.jsonl`, item
`civil_comments/validation/8822` (gold `no`; the humans' distribution is
`{"yes": 0.0, "no": 1.0}`):

```
State:
Naturally there's the other side of the story i.e  in addition to the supposedly leftist NA media, read what any number of European newspapers and news magazines report on the subject.
One will very quickly realize that what some call "leftist media" here in NA is pretty well on the same wavelength as those sources.
If PM May thought she would get "her way or the highway" privileges she is sorely mistaken, if for no other reason than the EU learned from past obstructionist tactics of the UK.
The  "have your cake AND eat it" times are past.


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | noul] Is this comment toxic?
  yes means "Rude, disrespectful, or likely to make someone leave the discussion." ; no means "Civil, even if critical or blunt."
  answer yes or no
  start your answer line with exactly "q:"
```

### 1.3 The choice

**OpenDecisions**, item `jb16-llm_router-9bb828d310` (gold `creative_writing`):

```
State:
Write a short, warm four-line poem I can read at my grandma's 80th birthday.


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | choice] Classify the task type of this request for routing.
  answer with exactly one of these option names
  - extraction_json ("Pull given details into structured fields.")
  - translation ("Render given text into another language.")
  - coding ("Write or fix program code.")
  - creative_writing ("Compose original prose or verse.")
  start your answer line with exactly "q:"
```

**jev-bench**, item `arc_challenge/test/689` (gold `D`):

```
State:
{"question":"Which pair of instruments can be used to measure wind chill?"}


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | choice] Which option is the correct answer to `question`?
  answer with exactly one of these option names
  - A ("barometer and wind vane")
  - B ("wind vane and anemometer")
  - C ("thermometer and barometer")
  - D ("anemometer and thermometer")
  start your answer line with exactly "q:"
```

**The pool**, item `go_emotions/validation/189` (gold `annoyance`; the humans'
distribution is `anger 0.333, annoyance 0.667`):

```
State:
Grow up.


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | choice] Which emotion does the comment primarily express?
  answer with exactly one of these option names
  - admiration
  - amusement
  - anger
  - annoyance
  - approval
  - caring
  - confusion
  - curiosity
  - desire
  - disappointment
  - disapproval
  - disgust
  - embarrassment
  - excitement
  - fear
  - gratitude
  - grief
  - joy
  - love
  - nervousness
  - optimism
  - pride
  - realization
  - relief
  - remorse
  - sadness
  - surprise
  - neutral
  start your answer line with exactly "q:"
```

### 1.4 The score

**OpenDecisions**, item `jb16-llm_router-d6982f3d76` (gold level 3 of 4). The
harness shaped the levels into `id ("description")` pairs
(`tools/decision_policy/opendecisions.py:98`):

```
State:
{"router_trace":{"pre_classifier":"arithmetic / percentages","pre_classifier_suggested_effort":"low"},"user_message":"Quick one, should be simple percentages. Our AOI station checks boards for bad solder joints. If a board really has a bad joint, the station flags it 98% of the time. If the board is fine, it still flags it 3% of the time. Historically 1 board in 400 has a bad joint. Board A-5521 got flagged, so we sent it through the same station again (independent re-test) and it got flagged a second time. What's the chance it actually has a bad joint? Just the percentage please, it goes into the ticket."}


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | score] Which reasoning effort should the router set for the reply?
  answer with exactly one of these level names, lowest first
  - none ("no computation: a fact, a greeting, or a confirmation")
  - low ("a single direct computation (one formula, one percentage, one unit conversion)")
  - medium ("a chain of standard steps in a fixed order, with no conditioning on observed events and no case analysis")
  - high ("needs conditioning on observed events, case analysis, or a proof, where a careless setup gives a wrong number")
  start your answer line with exactly "q:"
```

**jev-bench**, item `helpsteer2_helpfulness/test/883` (gold level 4 of 5). The
levels were **not** shaped here: the harness gives up when a level cannot be given
a short id (`tools/decision_policy/jevbench.py:144-193`).

```
State:
{"prompt":"Where is the quote \"frankly my dear, I don't give a damn\" from?\n<extra_id_1>Assistant\nIt's a famous quote from the movie Gone with the Wind.\n<extra_id_1>User\nWhat is this movie about?\n<extra_id_1>Assistant\nThe movie tells the story of the American Civil War and the Reconstruction era, through the life of a southern woman named Scarlett O'Hara. It's the highest-grossing film of all time, and is widely considered one of the greatest movies ever made.\n<extra_id_1>User\nMake a poem about the story of gone with the wind","response":"In the heart of the South, in a time of strife,\n Lived Scarlett O'Hara, a girl full of life.\n Belle of Tara, her plantation home,\n Dreamed of wealth and power, her fate to roam.\n \n\n But war came to Georgia, flames encroaching,\n Leaving ash and ruin, the land dew-drenched.\n Scarlett fought and persevered, love ever near,\n Through loss and heartache, her fortitude clear.\n \n\n After the war, desperation deep,\n Scarlett vowed to save her ancestral keep.\n Rhett Butler, the rogue she'd met,\n A partner in her plans, their fates set.\n \n\n With strength and determination, she reclaimed her land,\n Bankrupting Yankees, ruthless and grand.\n But Scarlett's love for Rhett, never secure,\n Left their passionate bond, cracked and unsure.\n \n\n Her heart belonged to one, who'd gone away,\n To Ashley Wilkes, a lost cause of the gray.\n Yet Rhett loved Scarlett, despite her flaws,\n His final words to her, he left no pause:\n \n\n \"Frankly, my dear, I don't give a damn,\"\n He left her love, on that sorrowful span.\n Scarlett was left with the fire inside,\n To build her empire, never to be denied."}


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | score] How helpful is `response` as an answer to `prompt`?
  answer with exactly one of these level names, lowest first
  - Not useful or helpful at all; completely missed the essence of what the user wanted.
  - Borderline unhelpful: mostly does not capture what the user was looking for, but still usable and helpful in a small way.
  - Partially helpful: misses the overall goal of the user's query in some way; did not fully satisfy what the user was looking for.
  - Mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
  - Extremely helpful and completely aligned with the spirit of what the prompt was asking for.
  start your answer line with exactly "q:"
```

**The pool**, item `measuring_hate_speech/validation/852` (gold level 0 of 3; the
humans' distribution is `0: 0.5, 1: 0.25, 2: 0.25`):

```
State:
FUCK ISRAEL


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | score] Does this comment contain hate speech?
  answer with exactly one of these level names, lowest first
  - No hate speech ("the comment does not attack or demean a group.")
  - Unclear or borderline ("offensive or hostile, but not clearly hate speech.")
  - Hate speech ("attacks, dehumanizes or incites against a group.")
  start your answer line with exactly "q:"
```

### 1.5 The whole prompt, once

The user turn above is wrapped in the chat template and the assistant's turn is
prefilled with the answer line's prefix (`serve/server.py:4650`, the prefix is
`f"{name}:"`). The pool's noul, whole:

```
<|im_start|>system
You are a classification module. Read the state, then answer every question on its own line as `name: answer`, with no other text.
Copy each question name exactly. Do not repeat the question, the type, or the instructions.
- A yes/no question takes exactly `yes` or `no`.
- A choice question takes exactly one option name, copied exactly.
- A score question takes exactly one level name, copied exactly.<|im_end|>
<|im_start|>user
State:
sour grapes?


Answer the question on its own line as `name: answer`, and nothing else. Copy each question name exactly. Do not repeat the question, the type, or the instructions.


Questions:
[q | noul] Is this comment toxic?
  yes means "Rude, disrespectful, or likely to make someone leave the discussion." ; no means "Civil, even if critical or blunt."
  answer yes or no
  start your answer line with exactly "q:"<|im_end|>
<|im_start|>assistant
<think>

</think>

q:
```

The last two lines matter for the readout: the model's next token is the answer's
first token, and the answer tokens are the merged `" yes"` and `" no"`
(`serve/logits_first_token.py:55-65`).

---

## 2. Where the noul's yes-bias could come from

### 2.1 The measured failure, restated

Measured on the jev-bench test sample's 100 noul items that carry a human
distribution (all `civil_comments`), from the stored T = 1 run
(`jevbench_sample3_predictions.jsonl`, the run the companion doc §14.1 records):

| | |
|---|---:|
| the model's mean P(yes), the closed reading | **0.4439** |
| the model's mean P(yes), the open reading | 0.4410 |
| the humans' mean P(yes) | **0.0892** |
| the mean signed difference | **+0.3547** |
| the mean absolute difference (= the mean TVD) | 0.3842 |
| the bias's share of the TVD | 92.3% |
| the model picks "yes" on | **45 of 100** items |
| the humans' majority is "yes" on | 8 of 100 items |
| the accuracy against the humans' majority | **0.55** |
| a constant "no" would score | **0.92** |
| the AUC of P(yes) against the humans' hard label | 0.6399 |

The companion doc §19.5 records the same slice as a mean signed difference of
+0.357 on a mean absolute 0.384, with 93% of the TVD in the bias. The numbers
here differ in the third decimal because the companion doc quotes the *recovered*
T = 1 arm (the temperature's arithmetic undone) and this table quotes the
*stored* run; the recovered arm gives +0.3570 and 0.3844, so the two agree.

**The AUC is the important number in that table.** The model is not random: 0.64
is well above 0.5. But 0.55 accuracy against a 0.92 constant is a *bad* trade.

### 2.2 The model's base rate does not move with the population's

This is the single most important measurement for the plan, and it is not in any
doc yet. Measured from the three stored runs:

| population | n | the gold "yes" rate | the model's mean P(yes) |
|---|---:|---:|---:|
| the pool's `civil_comments` (validation) | 250 | 9.2% (the humans' mean P(yes) 0.1096) | not measured yet (the design set) |
| the pool's `paws` + `strategyqa` (validation) | 250 | 45.2% | not measured yet (the design set) |
| the jev-bench test's `civil_comments` | 100 | 8.9% | **0.4439** |
| the jev-bench test's other nouls | 122 | - | 0.4952 (it picks "yes" on 50.8%) |
| the OpenDecisions 235 | 69 | 55.1% (38 of 69) | **0.4903** (the accuracy 91.3%) |

The model's mean P(yes) is 0.44 to 0.50 on **every** population, while the gold
rate is 0.09, 0.45 and 0.55. The endpoint's yes/no prior is a **constant**, not a
reading of the population.

**What that means for this item (static analysis, built on the table above).** A
template change can move that constant. It cannot make it population-aware. So:

- A variant that lowers the model's mean P(yes) from 0.44 to 0.11 will look
  superb on the toxicity population and will be **wrong** on the other two. It is
  a base-rate shift, and the plan has to be able to *see* that.
- The over-yes on the toxicity population is at least as likely to be a **task**
  mismatch as a template artifact: the criteria's "true" side ("Rude,
  disrespectful, or likely to make someone leave the discussion") is broader than
  the annotators' notion of toxicity, and on the rule-shaped OpenDecisions
  questions the same template does fine. The template is identical for both, so
  the difference is in the question, not in the frame. That is an inference from
  the measurements, and the exact share of it that is the template is **not
  measured**. §3.4's probe is what would measure it.

### 2.3 The six parts of the noul template, one at a time

All of this section is **static analysis**. None of it has been measured, and
none of it can be measured without the engine. Each part names the variant that
would test it, so nothing here is load-bearing on its own.

1. **The order the two sides are presented in.** The affirmative side is named
   first in **three** places: the system turn's `- A yes/no question takes
   exactly `yes` or `no`.`, the criteria line `yes means "X" ; no means "Y"`, and
   the line `answer yes or no`. A first-mentioned side is a plausible prior. The
   two sides carry the same information, so a swap can only move the **centre**,
   never the ranking. **This is the base-rate axis.** Tested by N1 and N2 (§3.1)
   and measured by the probe P1 (§3.4).

2. **The criteria's wording.** The *criteria text* is the dataset's, not the
   server's: it arrives in the request and the server may not change it without
   changing the question's meaning. So the axis is the **frame** around the
   criteria - the connective, the order, the position - not the criteria
   themselves. One measured observation: the `civil_comments` criteria are
   **asymmetric**. The "true" side lists three triggers; the "false" side is one
   permissive clause ("Civil, even if critical or blunt"). A reader that counts
   triggers will find more support for "yes". That asymmetry is in the dataset's
   text and a template change cannot remove it - only a change of the dataset
   could, and that is not a prompt experiment. (Static analysis of the measured
   criteria strings: the 100 `civil_comments` items in the jev-bench test and the
   250 in the pool all carry the same pair, and the 24 `boolq` items in the
   jev-bench test carry a symmetric pair, "The evidence establishes the claim is
   true." / "The evidence shows the claim is false.")

3. **The bare `answer yes or no` line.** It states the answer space and nothing
   else. It does not say the two answers are equally acceptable, and it lists
   "yes" first. A line that names both sides as available might remove an
   implied default. Tested by N2 and N5 (§3.1).

4. **The instruction line, and its position.** The question's own instruction
   comes from the dataset and is the first thing after the header. For
   `civil_comments` it is *"Is this comment toxic?"* - a question whose
   affirmative answer is the salient one. Reading the question before the
   criteria may prime the answer ("is this toxic?" → "yes"); reading the
   criteria before the question may make the model apply them literally (and so
   find more "yes" cases). The two orders are opposite predictions, and only one
   of them can be right. Tested by N4 (§3.1). **This is the strongest "weighing"
   lever in the list**, because it changes what the model weighs, not what it
   defaults to.

5. **The `start your answer line with exactly "q:"` line.** It is a format
   instruction about the output, and it is the last thing the model reads. There
   is no plausible path by which it prefers "yes": the two answer tokens are
   read straight off the answer position's logits (§2.4), and the prefix is the
   same for both. **No variant targets it.** It matters to this document for a
   different reason: it is why the readout cannot be the cause of the bias.

6. **The preamble.** The system turn and the user turn's preamble describe the
   **output format** and say nothing about how to weigh the evidence. A model
   given a format and no decision rule falls back on its own reading of the
   question - which, on §2.2's table, is a constant prior. Adding one line of
   decision guidance is the other "weighing" lever. Tested by N5 (§3.1). Note
   the preamble is **shared by all three types**, so a per-type variant cannot
   touch it without a per-type system turn, which is a bigger change and is out
   of scope for item 1.5. The variants in §3 therefore live in the **per-type
   lines** only.

### 2.4 The readout cannot be the cause (static analysis)

The plan's probe has to be able to say "the bias is not in the parse". Reading
the code says it cannot be:

- `serve/logits_first_token.py:110-114`: for a noul the options are exactly
  `["yes", "no"]`, in that order. The order is a dict order and does not touch
  any value.
- `serve/logits_first_token.py:525-529`: the closed reading is a softmax over
  the two option token ids. A logit is attached to the token id the tokenizer
  gave the string "yes" or "no", so there is no path by which the "yes" slot
  can carry the "no" logits.
- `tools/decision_policy/logits_prob.py:152`: the reported noul value is
  `distribution["yes"]`, and the distribution's keys came from
  `option_names_for_question`. "yes" is "yes".
- `serve/systemone.py:426-432` and `484-487`: the decode path maps the text
  "yes" to yes and "no" to no, the same way.

And the stored run agrees with itself: over the 100 human-labelled noul items the
closed reading's `P(yes) >= 0.5` disagrees with the run log's recorded pick on
**0** items, and the open and the closed readings differ by at most 0.0088 on any
item. The open-versus-closed choice moves the mean P(yes) by 0.0029, so it is not
where the bias lives either.

### 2.5 The score's error, and why its lever is the direction

The score's error on the jev-bench slice is a **location** error, not a shape
one. Measured on the 50 `measuring_hate_speech` items with a human distribution:
the model's mean expected level is **1.0002** against the humans' **0.6717** - the
model reads these comments as more hateful than the annotators do, by 0.33 of a
three-level scale. The model's top level agrees with the humans' top level on 21
of 50 items. The companion doc §19.5 measured the score's *shape* as close to the
humans' (the model's top-label mass 0.7977 against the humans' 0.8233).

So the score's lever is the **ordinal direction** - "lowest first" - and not the
scale. That is why §3.3 leads with the direction axis.

---

## 3. The variant space

One axis per variant. Each row says what changes, what it tests, what it risks,
and what it costs. The cost is in prompt tokens, estimated at the measured
**1.5 to 1.8 characters per token** (the median of the stored run logs: 1.51 for
the noul, 1.57 for the score, 1.80 for the choice).

**The mark that matters:** a variant marked **BASE RATE** shifts the model's
prior. It can move the mean P(yes) or the mean P(option 1) and it cannot make
the model better at ranking the answers. It will look good on a population whose
base rate it matches and bad on a population whose base rate it does not. A
variant not marked that way changes **how the model weighs the evidence**, and
is the only kind that can be expected to transfer.

### 3.1 The noul (five variants)

The base for every row is the block in §1.2. `{true}` and `{false}` are the
dataset's own criteria.

| id | axis | what changes | hypothesis | risk | cost |
|---|---|---|---|---|---:|
| **N1** `criteria-order` | the order the two sides are presented in | the criteria line becomes `no means "{false}" ; yes means "{true}"` and nothing else changes | the affirmative side is named first in the template and carries a prior; taking it out lowers the mean P(yes) | **BASE RATE.** The two sides carry the same information, so the ranking cannot change. On the OpenDecisions 235 (55.1% "yes") it will lose accuracy: a uniform log-odds shift of -0.25 already costs 2 of the 69 items, -1.0 costs 4 and -1.5 costs 8 (measured from the stored run) | 0 tokens (a reorder) |
| **N2** `answer-order` | the order the two sides are presented in, in the answer line | `answer yes or no` becomes `answer no or yes` | the same prior, in the one place the model reads when it decides | **BASE RATE**, and it is confounded with N1: the two move the same prior. Running both is what measures how big the prior is | 0 tokens (a reorder) |
| **N3** `rule-form` | the criteria's connective | `yes means "{true}" ; no means "{false}"` becomes `answer yes if the state is "{true}"; answer no if the state is "{false}"` | the criteria read as a **definition** ("this is what yes means") invite a category judgement; read as a **rule** they invite a test ("does the state meet it?"). The noul's failure is a judgement that is too harsh, and a test reads more strictly than a category | it can also make the model *more* literal and therefore more likely to say yes. The direction is not predictable, which is the point | +~9 tokens (a 15-character edit) |
| **N4** `criteria-first` | the instruction line's position relative to the criteria | the criteria line is emitted **before** the `[q | noul] ...` header line; everything else is unchanged | the model reads the question *through* the criteria instead of the criteria *through* the question. On `civil_comments` the question is "Is this comment toxic?" - a question whose affirmative is salient. Reading the criteria first removes that priming | it can go the other way: knowing what "rude" means before reading the comment may make the model find more rude comments. The two orders make **opposite** predictions, which is why this is the best single variant to run | 0 tokens (a reorder) |
| **N5** `decision-rule` | whether the template states a decision rule | one line is added: `answer yes only when the state satisfies the "yes" criterion; otherwise answer no` | the template says nothing about what to **do** with the criteria, so the model falls back on its own prior. A rule that points at the criterion changes the weighing | **confounded.** A rule of this shape also pushes the base rate down, so a win here cannot be told apart from N1/N2. It is the most useful variant and the least clean | +~41 tokens (a 65-character line) |

**Why these five and not more.** The roadmap names four axes (the order, the
criteria wording, the "either side" statement, the instruction's position). The
"either side" statement is folded into N2 and N5, because "either answer may be
correct" and "answer yes only when the state satisfies the criterion" are the
same instruction in a softer and a harder form, and running both would spend a
run on a near-duplicate. The five above cover the four axes and add the
connective, which is the one place the *dataset's* criteria can be turned into a
*procedure* without changing their meaning.

### 3.2 The choice (four variants)

The base is the block in §1.3. The choice's measured failure is a **shape** one,
not a bias: the model's top-label mass is 0.7862 against the humans' 0.6215
(companion doc §19.5), so the model is already more peaked than the humans, which
is why the temperature made it worse. The levers are therefore the ones that
change the *evidence*, and one that removes the option order's prior.

| id | axis | what changes | hypothesis | risk | cost |
|---|---|---|---|---|---:|
| **C1** `options-order` | the order the options are listed in | the option lines are emitted in the reverse order | the first-listed option carries a prior, as "yes" does on the noul | **BASE RATE.** The options carry the same information in either order. The 235 holds 12 items from the dataset's own `option_order` family, so the effect is visible there | 0 tokens (a reorder) |
| **C2** `evidence` | the evidence rule | a line is added: `pick the option the state best supports; if none fits, pick the closest` | the template never says what to do when no option fits, and the model then falls back on a prior | it invites the model to guess, and on a 77-option choice (banking77) "the closest" is a weak instruction | +~46 tokens (a 73-character line) |
| **C3** `one-best` | the wording of the existing line | `answer with exactly one of these option names` becomes `answer with exactly one of these option names, the one the state best supports` | the same rule as C2 in the smallest possible edit, so the cost of the *extra line* is not confounded with the *content* of the rule | it is the same hypothesis as C2 in weaker form; if C3 and C2 agree, the rule is what mattered | +~21 tokens (a 33-character edit) |
| **C4** `options-first` | the instruction line's position relative to the options | the option lines are emitted **before** the `[q | choice] ...` header line | the same lever as the noul's N4, on the choice: the model reads the options before it reads the question | it can also narrow the model onto the first option that looks plausible, which is the opposite of what is wanted | 0 tokens (a reorder) |

### 3.3 The score (four variants)

The base is the block in §1.4. The score's measured failure is a **location**
error (§2.5): the model's mean expected level is 0.33 of a three-level scale
above the humans'. The lever is the ordinal direction.

| id | axis | what changes | hypothesis | risk | cost |
|---|---|---|---|---|---:|
| **S1** `direction` | the direction the levels are presented in | the levels are listed highest-first and the line becomes `answer with exactly one of these level names, highest first` | the template tells the model the scale runs bottom-up, and the model has a prior about which end to pick | **BASE RATE** along the scale. It can move the mean expected level by 0.33 and nothing else | 0 tokens (a reorder) |
| **S2** `wording` | the direction's **words** | the line loses `, lowest first` and the levels stay listed lowest-first | the same lever with the words removed, so S1 and S2 together say whether the *words* or the *list order* carries the prior | **BASE RATE**; and it is inert on the items whose levels the harness could not shape (the long ones), because there the levels are a bare list of sentences | -~9 tokens (the words come off) |
| **S3** `levels-first` | the instruction line's position relative to the levels | the level lines are emitted **before** the `[q | score] ...` header line | the same lever as N4 and C4: the model reads the scale before it reads the question, so the question is read *through* the scale | as with C4, it can narrow the model onto the ends | 0 tokens (a reorder) |
| **S4** `full-scale` | whether the template says the whole scale may be used | one line is added: `answer with exactly one of these level names; the state may support any of them, not only the ends` | the model may be using only the scale's ends, and the middle is where the humans' uncertainty lives | it is the weakest of the four: there is no measurement of an end-bias to motivate it, so it is a shot in the dark | +~63 tokens (a 100-character line) |

### 3.4 The diagnostic probe (one, and it is not a shipping candidate)

**P1 `swap`.** The full mirror of the noul's template: the criteria line's two
sides are swapped **and** the answer line reads `answer no or yes`. The answer
**tokens** do not change - the model still writes " yes" or " no", and "yes" still
means the same thing - so the two arms are two **presentations of the same
question**, and the readout is untouched.

**The three readouts.**

1. **The flip rate.** The share of items whose *pick* changes between the two
   arms. The engine's own run-to-run wobble flips **1 of 100** noul items
   (measured, §5.2), so that is the floor.
2. **The swap gap**, `d_i = P_yes(base) − P_yes(swap)`, and its mean. `d > 0`
   means the first-listed side carries a prior; `d ≈ 0` means the two
   presentations produce the same judgement.
3. **The AUC.** A presentation change should not change the ranking, so the AUC
   should not move.

**What the probe would show if the bias were elsewhere.** This is the point of
the probe.

| what the probe shows | what it means |
|---|---|
| the flip rate is near 1% and `mean(d)` is near 0, while the level stays high in **both** arms | the template has **no order component**. The over-yes is in the criteria's meaning, in the question, or in the task - and no order variant will fix it. **Stop looking at the template's order.** This is the most likely outcome, and the probe costs 17 minutes to find out |
| the flip rate is well above 1% and `mean(d)` is clearly positive | the **position** of the two sides is a real, measurable component of the bias. N1 and N2 are worth their runs, and `mean(d)` prices them |
| the level stays high in both arms and `mean(d) ≈ 0`, but the *labels* are what the model prefers | the component is a "yes"-**label** prior, not a position prior. A variant that renames the answer space (not the order) would be the lever - and that is a bigger change than item 1.5 |
| the AUC moves | something other than the order moved - the prompt's shape, not its order. That is a bug, and the run should be thrown away |
| both arms report a systematically wrong sign | the bias would be in the **parse**. §2.4 says the code cannot do that, and the stored run agrees with itself (0 disagreements between the closed reading's pick and the run log's pick over the 100 items) |
| the two arms differ by about 0.0029 in the mean | the bias would be in the **readout's open/closed choice**. That is the measured worth of that choice, so it is not it either |

**The probe is cheap and it is the first thing to run.** It costs one arm on the
design set's 250 human-labelled noul items: 250 items at 3.5 s is **15 minutes**
of GPU time plus a restart. If it finds no order component, N1 and N2 are dead
and the budget goes to N3, N4 and N5 instead. **That is the whole point of
running the probe first.** P1 is N1 and N2 **together**, so it measures the
order as a whole; it cannot say which of the two carries it, and it is not
meant to.

### 3.5 The rule that catches a base-rate shift

The rule is in §6. In one line: **the design set's 500 noul items are two
populations with different gold "yes" rates (9.2% and 45.2%, measured), and a
variant must not improve one of them by wrecking the other.** A base-rate shift
moves the level in the same direction on both halves and moves the accuracy in
*opposite* directions, so the two halves together catch it. §5.1 and §6 spell it
out.

### 3.6 The exact text of every variant

So that the implementation has nothing to decide. The noul's base block is in
§1.2; `{true}` and `{false}` are the dataset's criteria, `{instructions}` is the
dataset's instruction, and the answer-line instruction
(`  start your answer line with exactly "q:"`) is the same in every arm.

**The noul**, in the order the lines appear:

| id | the lines |
|---|---|
| base | `  yes means "{true}" ; no means "{false}"` <br> `  answer yes or no` |
| N1 `criteria-order` | `  no means "{false}" ; yes means "{true}"` <br> `  answer yes or no` |
| N2 `answer-order` | `  yes means "{true}" ; no means "{false}"` <br> `  answer no or yes` |
| N3 `rule-form` | `  answer yes if the state is "{true}"; answer no if the state is "{false}"` <br> `  answer yes or no` |
| N4 `criteria-first` | the criteria line `  yes means "{true}" ; no means "{false}"` is emitted **before** the header line `[q | noul] {instructions}`; the rest is as in the base |
| N5 `decision-rule` | `  yes means "{true}" ; no means "{false}"` <br> `  answer yes only when the state satisfies the "yes" criterion; otherwise answer no` <br> `  answer yes or no` |
| P1 `swap` (the probe) | `  no means "{false}" ; yes means "{true}"` <br> `  answer no or yes` |

**The choice:**

| id | the lines |
|---|---|
| base | `  answer with exactly one of these option names` <br> `  - {option} ("{description}")` ... |
| C1 `options-order` | the option lines in the reverse order; the rest as in the base |
| C2 `evidence` | `  answer with exactly one of these option names` <br> `  pick the option the state best supports; if none fits, pick the closest` <br> `  - {option} ...` |
| C3 `one-best` | `  answer with exactly one of these option names, the one the state best supports` <br> `  - {option} ...` |
| C4 `options-first` | the option lines are emitted **before** the header line `[q | choice] {instructions}` |

**The score:**

| id | the lines |
|---|---|
| base | `  answer with exactly one of these level names, lowest first` <br> `  - {level} ("{description}")` ... |
| S1 `direction` | `  answer with exactly one of these level names, highest first` <br> the level lines in the reverse order |
| S2 `wording` | `  answer with exactly one of these level names` <br> the level lines in the base order |
| S3 `levels-first` | the level lines are emitted **before** the header line `[q | score] {instructions}` |
| S4 `full-scale` | `  answer with exactly one of these level names, lowest first` <br> `  answer with exactly one of these level names; the state may support any of them, not only the ends` <br> `  - {level} ...` |

**Two implementation notes that the text above makes plain.**

- A noul item with **no** criteria prints only `  answer yes or no` today
  (`serve/systemone.py:252`). So **N1, N3 and N4 are inert on half B** (the 250
  `paws` and `strategyqa` items): they rewrite or move the criteria line, and
  there is none. **N2 and N5 are not inert** - the answer line and the added
  rule are there for every noul item. That matters for the base-rate check:
  half B's accuracy can catch a base-rate shift made by **N2** and **N5** in the
  design set, but **not** one made by N1, N3 or N4. Those three are caught by
  the 235 instead, where the noul's gold "yes" rate is 55.1% and every item
  carries criteria (69 of 69, measured). The rule in §6 says this: (c) is the
  design-set half of the base-rate check and (d) is the yardstick half, and
  **every** variant has to clear both.
- The score's level list is **not** reversed by S2: the point of S2 is to remove
  the *words* while leaving the *order* alone, so S1 and S2 together say which
  of the two carries the ordinal prior.

---

## 4. The mechanism (a design; not implemented)

### 4.1 The config

```json
"systemone": {
  "enabled": true,
  "prompt": {"noul": "criteria-order", "choice": "base", "score": "base"}
}
```

The `prompt` block sits beside the `temperature` block that item 1.1 put in the
same place, and it is read the same way.

- **Absent** - every type is on `base`, and `base` is today's text, **byte for
  byte**. The default path is unchanged.
- **A type absent from the `prompt` block** - that type is on `base`.
- **An unknown type, or an unknown variant name** - refused at start-up, with the
  same shape as the other config values: the resolver raises `ValueError` and
  `serve/server.py` turns it into `SystemExit` with the message
  `[strata] config ...`, so the server does not start on a typo.
- **The two knobs do not interact.** The temperature scales the *reading*
  (`serve/logits_first_token.py:534-536`, after the closed and the open readings
  are computed); the prompt changes the *reading's input*. A run of this
  ablation sets the temperature to 1.0, and the two are never varied together -
  that would be two axes in one arm, and item 1.2 already measured that the
  fitted constants lose against the humans on every type.
- **`/v1/status`** reports the **resolved** variants, so the status always says
  what is in force, not what was asked for:

```json
"systemone": {"enabled": true, "diagnostics": false, "multi": "separate",
              "temperature": {"choice": 1.0, "noul": 1.0, "score": 1.0},
              "prompt": {"noul": "criteria-order", "choice": "base", "score": "base"}}
```

### 4.2 The file:line map

**As built** (2026-10-21, this branch). The line numbers are against
`feat/p1a-prompt-design`; the base is `4bb9bcd8` plus the implementation commit.

| file:line | what it is now |
|---|---|
| `serve/systemone.py:79-88` | `PROMPT_VARIANTS`: the registry. One list per question type, the name of every variant that type accepts. It is the only list, and the config reader and the renderer both read it, so they cannot drift |
| `serve/systemone.py:232-271` | `_noul_lines(crit, variant)` -> (the lines in front of the header, the lines after it). The noul's six variants, each one axis |
| `serve/systemone.py:273-288` | `_choice_lines(crit, variant)`: the choice's four variants |
| `serve/systemone.py:290-309` | `_score_lines(q, variant)`: the score's four variants |
| `serve/systemone.py:311-331` | `_question_lines(name, q, variant="base")`: assembles the question's lines, and **refuses an unknown variant** (a typo is an error, not a silent default) |
| `serve/systemone.py:333-358` | `render_prompt(state_text, questions, variants=None)`: the shared path. `variants` maps a question **type** to a variant name; a type that is not in it is rendered on "base" |
| `serve/systemone.py:360-377` | `render_answer_prompt(state_text, name, q, variant="base")`: the default path, one question per prompt |
| `serve/systemone.py:86-90` | `SYSTEM`: **unchanged**. A per-type system turn is a bigger change and is out of scope for item 1.5 (§2.3, point 6) |
| `serve/server.py:79` | `from serve.systemone import PROMPT_VARIANTS` |
| `serve/server.py:8377-8405` | `systemone_prompt_variants(cfg)`: the config's `"systemone": {"prompt": {...}}`, in the shape of `systemone_temperatures` (8328-8359): the default when the block is absent, and a `ValueError` for an unknown type, an unknown variant name, a value that is not a string, or a block that is not an object |
| `serve/server.py:3535` | `Service.systemone_prompt`: the resolved variants, next to `systemone_enabled`, `systemone_diagnostics`, `systemone_multi` and `systemone_temperature` |
| `serve/server.py:4368-4379` | `Service._decision_prompt_variant(qtype)`: the call site's own rule - the question's **type** picks the variant, and a type that is not in the config is on "base". It sits beside `_decision_temperature`, the temperature's own rule |
| `serve/server.py:8790-8798` | the config plumbing: the resolver inside the same `try` / `except ValueError` / `SystemExit` that `systemone_multi` and `systemone_temperature` use, plus the startup line that names the variants in force |
| `serve/server.py:4966-4970` | the `/v1/status` block gains `"prompt": {...}`, beside the `"temperature"` block |
| `serve/server.py:4651-4654` | the single-question call site: the variant comes from `_decision_prompt_variant` |
| `serve/server.py:4607-4610` | the shared-path call site: the whole map goes in |
| `serve/test_systemone.py:511-800` | `PromptVariants`: the byte-identity tests and the per-variant tests |
| `serve/test_systemone.py:801-870` | `PromptVariantConfig`: the config reader, the status, and the call site's rule |

**The variant names**, exactly as they are in the registry:

```
noul:   base, criteria-order, answer-order, rule-form, criteria-first, decision-rule, swap
choice: base, options-order, evidence, one-best, options-first
score:  base, direction, wording, levels-first, full-scale
```

**`swap` IS in the noul's list, and that is a change from the design.** The design
said the probe "must not be reachable from a launch config" and would be run from a
branch that is not merged. The implementation needs the opposite: the probe is a
measurement somebody else runs later, against a **live server**, and the variant is
a launch-config value, so the probe's arm has to be startable. `swap` is therefore
a known name, it is in the registry beside the others, and it is marked in the
code and in this document as **the probe, not a shipping candidate**. Nothing
else changed: the six shipping candidates are the same five plus the base.

### 4.3 The tests

Four things have to be pinned, and the first one was a gap in the tests that were
there. All four are now written (§4.4).

1. **The base arm is byte-identical.** The render test that was there only checks
   `assertIn` for a list of needles (`serve/test_systemone.py:141-150`), so a
   change to the base template's wording would not fail anything. The new test
   holds the **whole** expected prompt for one question of each type as a literal
   string and compares it exactly, with the variant absent and with
   `variant="base"`.
2. **Each variant renders its own lines and not the base's.** The new tests pin
   the **whole line list** of the question for every variant, so a variant that
   changed one line too many would fail.
3. **The config resolver.** Absent -> all `base`; the three types; an unknown
   type refused; an unknown variant name refused; a `prompt` block that is not a
   dict refused; a value that is not a name refused.
4. **The status.** `/v1/status` carries the resolved variants, and carries `base`
   for every type when the block is absent.

### 4.4 The implementation (2026-10-21)

**What is built.** The mechanism of §4.1-§4.3, in full:

- `serve/systemone.py`: the registry `PROMPT_VARIANTS` (`:79`), the three per-type
  line builders (`:232`, `:273`, `:290`), the assembler `_question_lines`
  (`:311`), and the two renderers (`:333`, `:360`). All **fourteen** variants are
  implemented: the noul's N1-N5 and the probe P1, the choice's C1-C4 and the
  score's S1-S4. The mechanism generalised cleanly to the choice and the score -
  the same shape (a lead block, a body, a suffix) carries all three types - so all
  three are there, each with its own one-axis test.
- `serve/server.py`: the config reader `systemone_prompt_variants` (`:8377`), the
  `Service` attribute (`:3535`), the call site's rule `_decision_prompt_variant`
  (`:4368`), the start-up plumbing and the startup line (`:8790-8798`), the
  `/v1/status` field (`:4966-4970`) and the two call sites (`:4607-4610`,
  `:4651-4654`).
- `serve/test_systemone.py`: `PromptVariants` (`:511-800`) and
  `PromptVariantConfig` (`:801-870`).
- `tools/decision_policy/prompt_probe.py`: the probe (§4.5).
- `tools/decision_policy/prompt_design_set.py`: the design set's composition.
- `tools/decision_policy/test_prompt_probe.py`: the probe's own arithmetic.

**The byte-identity test, and what it proves.** The expected strings in
`PromptVariants.BASE_*` were taken from `serve/systemone.py` **as it was before
this feature** (the commit `f331cb8a`), not from the new code: so they are the
bytes the endpoint has always sent. The test then asserts that the renderer
produces exactly those bytes with the variant absent, with `variant="base"`, and
with the config reader's resolved variants, for the noul, the choice, the score,
the noul without criteria, and the shared prompt.

```
python -m unittest serve.test_systemone -v
...
Ran 62 tests in 0.404s
OK
```

and the new tests alone:

```
python -m unittest serve.test_systemone.PromptVariants serve.test_systemone.PromptVariantConfig -v
Ran 18 tests in 0.047s
OK
```

**The per-variant tests.** Each variant's **whole line list** is pinned, so a
variant that changed one line too many fails. `test_the_noul_variants`,
`test_the_choice_variants` and `test_the_score_variants` cover the fourteen
variants; `test_a_noul_without_criteria` covers the ones that are inert when the
question has no criteria (N1, N3, N4) and the ones that are not (N2, N5, P1);
`test_an_unknown_variant_is_refused` covers the refusal, including a variant that
belongs to another type; `test_the_service_picks_the_variant_by_type` covers the
call site's own rule.

**The whole server suite.** `python -m unittest discover -s serve -p "test_*.py"`
on the final tree: **1,053 tests, OK, 20 skipped, no failures.**

One test in that suite is flaky: `test_responses.NoLeakedSampler.test_server_close_ends_the_sampler`
counts the asyncio event-loop threads, so it depends on what ran before it. It
failed on two earlier runs of this tree and **it also fails on the base commit**
(1,035 tests, the same one failure), and it passes in isolation both before and
after this change. It is not caused by the variants; it is a pre-existing
order-dependent test, and the final run of the whole suite is green.

### 4.5 The probe's usage

`tools/decision_policy/prompt_probe.py` is the P1 measurement: the noul's
template mirrored, on the design set's half A.

**The design set, first** (no server, no engine, no GPU):

```sh
python tools/decision_policy/prompt_design_set.py --items <the pool's p1b_pool.jsonl>
```

It prints the composition: the noul's 500 items in two halves - **half A**, the
250 `civil_comments` items with a human distribution (the human's mean P(yes)
0.1096, 174 of them at exactly 0) and a 9.2% gold "yes" rate, and **half B**, the
250 `paws` and `strategyqa` items at a 45.2% gold "yes" rate and no criteria.

**The dry run** (prints the prompt the endpoint would send, and the item count;
sends nothing):

```sh
python tools/decision_policy/prompt_probe.py --items <p1b_pool.jsonl> --out <swap.jsonl> --dry-run
```

**The run.** The variant is a launch-config value, so the arm is a server start.
The server must be started with

```json
"systemone": {"enabled": true, "diagnostics": true, "prompt": {"noul": "swap"}}
```

and then:

```sh
python tools/decision_policy/prompt_probe.py \
  --items <p1b_pool.jsonl> --out <swap.jsonl> \
  --base <the pool run's predictions.jsonl> --report <report.md>
```

**The launch config is per machine and is not in the repository** (it is
`.gitignore`d, `/strata-*.json`), so on a fresh checkout the script says so and
uses the default port with `STRATA_API_KEY`; pass `--config <this machine's
config>` for the port and the key. The script is resume-safe, like the other
harnesses: the ids already in `--out` are skipped, so an interrupted run
continues.

**The base arm is free, if the pool run left its artifacts.** Roadmap item 1.1b's
pool run is the base arm: the pool at T = 1, the direct readout, today's template.
`--base` points at that run's predictions file and the probe runs the **swap arm
only** and computes the gap against the stored base. **On this machine those
artifacts are not there yet** - the data root holds the pool's *items* file and no
predictions from a pool run - so the fallback applies: run the base arm first
(with the server on the default template, no `"prompt"` key), and then point
`--base` at it:

```sh
# 1. the server on the default template
python tools/decision_policy/prompt_probe.py --items <p1b_pool.jsonl> --out <base.jsonl> --arm base
# 2. restart the server with "prompt": {"noul": "swap"} and run the probe
python tools/decision_policy/prompt_probe.py --items <p1b_pool.jsonl> --out <swap.jsonl> \
  --base <base.jsonl> --report <report.md>
```

**What the report says.** The three readouts of §3.4 - the **flip rate** (the
share of items whose pick changes between the arms; the engine's own wobble flips
about 1 in 100), the **swap gap** `d = mean(P_yes(base) - P_yes(swap))`, and the
**AUC** of P(yes) against the humans' hard label in each arm - and the
pre-registered reading of them (§6): a swap gap below 0.05 with the flip rate at
the wobble means the order axis is dead.

The probe's arithmetic has its own tests
(`python -m unittest tools.decision_policy.test_prompt_probe -v`, 11 tests, all
passing), so the three readouts and the reading of them are checked without a
server. The whole CLI was exercised end to end without a server as well: with the
output file already holding every id (the resume-safe path sends nothing), the
script does the comparison, writes the report and exits 0.

### 4.6 What changed from the design

Three things, and one of them is a defect the implementation found.

1. **`swap` is now a launch-config name** (§4.2). The design said the probe must
   not be reachable from a config; the implementation needs it to be, because the
   probe is a run somebody else does later against a live server. The probe is
   marked as such in the code and here.
2. **The choice's and the score's arms are implemented** (C1-C4, S1-S4), not only
   the noul's. The mechanism generalised cleanly, and §6's last paragraph says the
   choice's and the score's arms need their own pre-registration before they are
   run; that is still true, and nothing has been run.
3. **N5's rule line is applied to a noul with no criteria too.** The design's
   §3.6 says N5 is not inert on half B, and the implementation follows it. **The
   wording is a defect the implementation found**: the line says "answer yes only
   when the state satisfies the *\"yes\" criterion*", and a noul without criteria
   has no such criterion - the words have no referent. The one-line fix, before a
   run, is to make N5 inert without criteria (which would also mean half B cannot
   catch N5's base-rate shift, so §6(c) would lean on the 235 for it) or to
   reword the line. **The design's text is kept as written, so the pre-registration
   is not silently changed; the decision is the operator's.**

---

## 5. The measurement design

### 5.1 The design set: the licence-clean pool, and why it splits in two

The design set is item 1.1b's pool, `decision_policy_training/items/p1b_pool.jsonl`
(seed 20261021, 1,500 items, 500 a type, 750 with a human distribution). It is
licence-clean, so using it for an ablation is allowed: the licence record
(`decision_policy_training_data.md` §7, on the data branch) puts `civil_comments`
(CC0), `paws` (free for any purpose), `strategyqa` (MIT) and the rest of the
pool's configs in the "usable" list, and the `test` splits of every source stay a
yardstick. **OpenDecisions stays validation-only** and is never used to choose
anything.

**The pool's noul slice is 500 items in two halves, and that is the most useful
thing about it** (measured from the pool file):

| half | items | configs | the human distribution | the gold "yes" rate |
|---|---:|---|---|---:|
| **A** | 250 | `civil_comments` 250 | yes: the humans' mean P(yes) 0.1096; 174 of the 250 are exactly 0, 76 are between 0 and 1, none is 1 | **9.2%** (23 of 250) |
| **B** | 250 | `paws` 170, `strategyqa_closed` 50, `strategyqa_grounded` 30 | no | **45.2%** (113 of 250) |

Half A is the same population as the jev-bench test's `civil_comments` slice, on
the **validation** split, so it is disjoint from the yardstick's **test** split.
Half A carries the true/false criteria; half B has none, so a variant that
changes the criteria's *frame* (N1, N3, N4) is inert on half B, and a variant
that changes the *answer line* (N2) or the *decision rule* (N5) is not.

**Why the two halves are the point.** A variant that only shifts the base rate
lowers the TVD on half A (the gold rate is 9.2%) and raises it on half B (the
gold rate is 45.2%). One arm, the same engine, the same items: the shift shows
up as a **gain on one half and a loss on the other**. That is the hill-climbing
trap caught inside the design set, and it costs nothing extra.

**How many items per variant.** All 500, for every arm. The reason is not
statistical - it is that half B only exists as the base-rate check if it is in
the same run.

### 5.2 The noise floor (measured)

The engine's run-to-run wobble is the floor every number sits on. It is measured
here from two stored runs of the **same code** on the same engine: the jev-bench
T = 1 run (§14.1) and the temperature run (§19.1), with the temperature's
arithmetic undone (`tools/decision_policy/jevbench_paired.py`'s inverse). The two
recovered T = 1 arms are two independent samples of the wobble.

| | noul | choice | score |
|---|---:|---:|---:|
| the mean per-item change in the model's P(answer) between the two runs | 0.0193 | 0.0307 | 0.0268 |
| the per-item **SD** of that change | **0.0337** | **0.0348** | **0.0255** |
| the largest single change | 0.1454 | - | - |
| the picks that moved | **1 of 100** | - | - |
| the SE of the mean paired change in the TVD to the human distribution | **0.00338** | 0.00305 | 0.00361 |
| the AUC's run-to-run change (noul, against the humans' hard label) | **-0.0095** | - | - |

The same floor on the OpenDecisions 235 is in the companion doc §10.8: the same
235 items scored **79.5** and **78.9** Capability on two runs, three picks
flipped, the confidence moved by a median of 0.004. So the 235's run-to-run
spread is **0.6 Capability points**.

### 5.3 The power (the arithmetic)

A variant's effect is a **paired** change on the same items, so the SE of the
mean change is the per-item SD divided by the square root of n. The MDE at 80%
power and a two-sided 5% is `2.802 x SE` (1.96 + 0.842).

**The calibration metric (the TVD to the human distribution), the MDE:**

| n | SE of the mean paired change | MDE (80% power, 5%) | MDE with the Bonferroni correction over 5 variants (α = 0.01, 3.418 x SE) |
|---:|---:|---:|---:|
| 100 | 0.00338 | 0.00945 | 0.01154 |
| 250 | 0.00213 | 0.00598 | **0.00728** |
| 500 | 0.00151 | 0.00423 | 0.00516 |

The design set's half A has 250 items, so the **MDE on the mean TVD change is
0.006**, and 0.007 after the correction for picking the best of five variants.
The measured bias is 0.3547, so the design resolves a change of about **2%** of
the bias. That is ample.

**The discrimination metric (the AUC), the MDE.** A bootstrap over the items
(2,000 resamples) gives:

| | the noul (against the humans' hard label) | the choice (the model's confidence against its own correctness) |
|---|---:|---:|
| the AUC's value | 0.6399 | 0.6779 |
| the bootstrap SD at the yardstick's n (100 / 130) | 0.0929 | 0.0478 |
| the MDE at n = 100 / 130 | 0.260 | 0.134 |
| the MDE at n = 250 | **0.165** | **0.097** |

**This is the number that changes the plan.** The discrimination metric on the
noul cannot be *improved* measurably on 250 items: the MDE is 0.165 AUC. So for
the noul, discrimination is a **guard** (a non-inferiority check, with a margin
below the MDE but well above the measured run-to-run change of 0.0095), never a
criterion. The roadmap's rule as written - "improves discrimination or
calibration without hurting the other" - cannot be applied to the noul as an
"improves discrimination" test. §6 says what the rule is instead.

### 5.4 The yardsticks

Used **only** for the variants already chosen on the design set, and only after
the choice is written down.

1. **The OpenDecisions 235** (`sample10_items.jsonl`). The official scorer
   (`score.py`, the one in the OpenDecisions release) gives Capability,
   Intelligence and Calibration. The baseline to beat is the stored
   `fix_serial.jsonl` run: **Capability 79.8, Intelligence 78.0, Calibration 81.7,
   235/235** (measured with the official scorer in this session). The noul's
   share: 69 items, 38 "yes" (55.1%), the accuracy 91.3% (63 of 69). This is the
   **anti-base-rate yardstick**: a variant that lowers the model's yes rate loses
   here.
2. **The jev-bench test sample** (`jevbench_sample3_items.jsonl`, 683 items, the
   `test` split of every source). The official scorer's Capability, the ECE of
   P(yes) on the noul (the scorer's own measure), and the mean TVD to the human
   distribution on the 280 items that carry one. The noul's 100 `civil_comments`
   items are the ones that carry the human distribution.

**What happens if the design set and a yardstick disagree** is in §6, and it is
not "pick the yardstick".

### 5.5 The GPU budget

The per-item costs, measured:

| set | items | the measured cost | per item |
|---|---:|---|---:|
| the jev-bench test sample | 683 | 73.5 min (companion doc §19.1) | 6.46 s |
| the OpenDecisions 235 | 235 | ~14 min (roadmap item 1.1b) | ~3.6 s |
| the licence-clean pool | 1,500 | ~87 min (roadmap item 1.1b) | ~3.5 s |
| an engine restart | - | 2 min 13 s (companion doc §19.1: 19:24:33 to 19:26:46) | - |

**The first batch: the noul only.**

| what | arms | items each | time |
|---|---:|---:|---:|
| the probe P1 (`swap`) on half A | 1 | 250 | 15 min + 2 min restart = **17 min** |
| the design set: the base arm + N1..N5 | 6 | 500 | 6 x 29 min = 175 min, + 6 restarts x 2.2 min = 13 min -> **3 h 8 min** |
| the yardsticks for the chosen variant: the 235 | 1 | 235 | 14 min + restart = **16 min** |
| the yardsticks for the chosen variant: the jev-bench noul slice | 1 | 222 | 13 min + restart = **15 min** |
| the yardsticks, if the **whole** jev-bench sample is run for the official Capability | 1 | 683 | 74 min + restart = **76 min** |

**The first batch is about 4 hours** with the noul-only jev-bench slice, or
**about 5 hours** with the whole jev-bench sample. If the probe kills the order
axis, the design set is four arms instead of six and the first batch drops by
about 1 hour. The choice and the score arms would add two more design sets
(2 x 3 h 8 min) and two more yardstick pairs (2 x 1 h 32 min): **about 9 h 20
min** more, so all three types is about **14 hours**.

**The recommendation: run the noul arm first and stop.** The choice's and the
score's templates are worth a batch only if the noul's arm shows that a
per-type variant can change something at all - and the noul is where the failure
was measured, so it is the cheapest place to find out.

**One restart per arm.** The variant is a launch-config value, so each arm is a
fresh engine. Numbers taken on a long-running engine are invalid (companion doc
§12.5), so each arm gets a fresh restart and the run order is recorded.

### 5.6 The paired measurement

- The same items, the same order, the same harness, one attempt per item, nothing
  retried.
- The official `score.py` for the Capability numbers; the TVD to the human
  distribution for the calibration numbers; the AUC for the ranking numbers.
- Every arm writes its own predictions file and its own run log, and the
  comparison is **item by item** against the base arm (`tools/decision_policy/compare_arms.py`
  already does the picks and the distributions).
- The engine's state (VRAM, the model file, the temperature block) is recorded
  with each run, and the temperature is **1.0** in every arm: item 1.2 measured
  the fitted constants to be worse on every type, so they are not in this run.

---

## 6. The pre-registered decision rule

Written before the run. Nothing below may be changed after the first arm is
scored; if it is changed, the change is written down with its date and the run
it invalidates.

> **Scope.** The noul arm only. One question type, one design set, one metric
> family, five variants against one base arm.
>
> **The design set.** The licence-clean pool's 500 noul items
> (`p1b_pool.jsonl`). Half A: the 250 `civil_comments` items, which carry the
> human distribution and the true/false criteria. Half B: the 250 `paws` and
> `strategyqa` items, which carry a hard label and a 45.2% "yes" gold rate.
> Every arm runs all 500 items, in the same order, on a freshly restarted engine.
>
> **The arms.** The base arm (today's template, byte-identical) plus N1
> `criteria-order`, N2 `answer-order`, N3 `rule-form`, N4 `criteria-first` and
> N5 `decision-rule`. The probe P1 `swap` is run first, on half A only, and is
> **not** a candidate.
>
> **The metrics.**
> - **Primary - the bias.** Half A's mean signed difference between the model's
>   P(yes) and the humans' P(yes). The base arm's value is measured in the same
>   run; the variant must move it toward zero.
> - **Primary - the TVD.** Half A's mean total variation distance to the human
>   distribution.
> - **Guard 1 - the ranking.** Half A's AUC of P(yes) against the humans' hard
>   label.
> - **Guard 2 - the base rate.** Half B's accuracy at P(yes) >= 0.5, and half
>   B's ECE of P(yes) against the hard label (the official scorer's own rule).
> - **Guard 3 - the cost.** The fast-path rate, the median latency per item, and
>   the prompt tokens.
>
> **The rule.** A variant ships only if **all** of the following hold.
>
> **(a) Calibration, on the design set.** Half A's mean TVD to the human
> distribution falls by more than **0.007** against the base arm (the MDE at
> n = 250, 80% power, with the Bonferroni correction for the best of five
> variants), **and** half A's mean signed difference moves toward zero by more
> than **0.007**.
>
> **(b) Ranking, on the design set.** Half A's AUC does not fall by more than
> **0.05** against the base arm. This is a **non-inferiority** check, not an
> improvement test: the MDE on the AUC at n = 250 is 0.165, so an improvement
> smaller than that is not measurable and is not asked for. The 0.05 margin is
> about five times the measured run-to-run change in the AUC (0.0095).
>
> **(c) The base rate, on the design set.** Half B's accuracy does not fall by
> more than **3 items** (of 250; 1.2 accuracy points) against the base arm, and
> half B's ECE does not rise by more than **0.02**. **This is the check that
> kills a base-rate shift.** Half B's gold "yes" rate is 45.2% (measured), and
> the base arm's mean P(yes) on half B is expected to sit near it: on the
> jev-bench test's comparable noul slice (`paws`, `strategyqa`, `boolq`,
> `fever_evidence`, `sms_spam` - the same first two configs as half B) the base
> arm's mean P(yes) is **0.4952** and it picks "yes" on **50.8%** of them
> (measured). So a variant that drops the model's mean P(yes) to about 0.11 will
> lose on half B. The base arm's own number on half B is measured in this run,
> not assumed.
>
> **(d) The 235.** The noul's accuracy on the 69 noul items does not fall by
> more than **2 items** against the base arm (the stored baseline run scores 63
> of 69, measured), and the 235's overall Capability does not fall by more than
> the measured run-to-run spread (**0.6 points**) against the base arm's own run.
> **The base arm's own run must itself land within 0.6 of the stored 79.8**
> (Capability 79.8, Intelligence 78.0, Calibration 81.7, 235/235, measured with
> the official scorer), or the run is invalid and is repeated.
>
> **The two margins are not redundant, and the Capability one binds first.** Two
> noul items are 2.9 accuracy points, which is 5.8 noul-competence points,
> 1.9 Intelligence points and about **0.97 Capability points** - more than the
> 0.6 margin. One noul item is about **0.48 Capability points**, so the 0.6
> margin allows a loss of **one** noul item. Measured on the stored run, a
> uniform log-odds shift of **-0.10 to -0.24** on the noul's P(yes) costs
> exactly one item, and -0.25 costs two. So the 235's Capability is the tighter
> of the two checks, and it catches a base-rate shift of about a quarter of a
> logit - which on the design set's half A moves the model's mean P(yes) by
> about 0.025, three times the 0.007 that §6(a) demands.
>
> **(e) The jev-bench test sample.** The noul's mean TVD to the human
> distribution on the 100 `civil_comments` test items does not rise by more than
> **0.005** against the base arm (the same fifth-of-a-fifth threshold item 1.1b
> used), and the noul's ECE of P(yes) does not rise by more than **0.02**.
>
> **(f) The cost.** The fast-path rate does not fall below the base arm's, and
> the median latency per item does not rise by more than **10%**.
>
> **If the design set and a yardstick disagree.** The variant does **not** ship.
> It is recorded as a **population-specific** effect, with the numbers, exactly
> as item 1.2 recorded the temperature's population-specificity. The write-up
> names the population it helps and the population it hurts. If the disagreement
> is a base-rate shift (guard 2 or (d) fails while (a) passes), the conclusion
> is that the template's bias is a **calibration** problem and not a prompt
> problem: the run then hands over to roadmap item 1.4 (an affine fit per type),
> which can move a distribution's centre, and it says so. **A variant that only
> moves the base rate is not a result; it is item 1.4 wearing a prompt's
> clothes.**
>
> **If no variant ships.** That is an acceptable and a likely answer. It would
> mean the noul's bias is a **task** mismatch - the criteria mean something
> different to the model than to the annotators (§2.2) - and the next step is
> 1.4's affine fit or a per-population calibration, not another prompt. The
> write-up records that, with the numbers, and the roadmap moves on.
>
> **The probe's result.** If P1's **swap gap** (`mean(P_yes(base) −
> P_yes(swap))`, §3.4) is below **0.05** in absolute value and the flip rate is
> near the engine's own 1% wobble, the order axis is dead: N1 and N2 are not run,
> and the remaining budget goes to N3, N4 and N5. The rule's numbers do not
> change; only the arms do, and the change is written down before the affected
> arms run.
>
> **The choice and the score.** This rule is written for the noul. The choice's
> and the score's arms are only run if the noul's arm produces a variant that
> ships or a result worth recording, and each gets its own pre-registration
> before its run - the same structure, with the metrics that fit its own
> measured failure (the choice's is a shape error, §3.2; the score's is a
> location error, §2.5).

---

## 7. What is wrong or missing in the roadmap's §1.5

Five things, in the order they matter.

1. **The rule as written cannot ship the fix it is looking for.** §1.5 says a
   variant ships only if it wins on the design set **and** clears the margin on
   **both** yardsticks. The design set's gold "yes" rate is 9.2% and the 235's
   is 55.1% (both measured). A variant that fixes the noul's bias necessarily
   lowers the model's yes rate, and that costs accuracy on the 235. So the rule
   would forbid shipping every variant that fixes the bias, and would allow
   shipping a variant that does nothing. The rule needs the third guard (§6(c))
   and a defined outcome for the conflict, and the outcome is in §6: a
   population-specific effect, recorded, not shipped.
2. **The obvious metric is gameable and the roadmap does not say so.** The TVD
   to the human distribution is the natural choice, and a **constant** P(yes) = 0
   scores a mean TVD of **0.0892** on the jev-bench noul slice against the
   model's **0.3842** (measured). A constant is four times "better" on the TVD
   and worthless: its AUC is 0.5 and its accuracy is 0.92 only because that
   population's gold rate is 8.9%. The rule has to name the ranking and the
   second-half accuracy as guards, or it measures nothing.
3. **"Improves discrimination or calibration" is not measurable as stated.** The
   noul's AUC's MDE at 250 items is **0.165** (§5.3). Discrimination cannot be an
   improvement criterion for the noul; it can only be a non-inferiority guard.
   The rule has to say which metric is primary and which is a guard, per type.
4. **The design set is already two populations, and the roadmap does not use
   that.** The pool's 500 noul items split into 250 at a 9.2% gold "yes" rate
   and 250 at 45.2%. That is a free base-rate check inside the design set, in
   the same run, on the same engine - the strongest thing this item has, and it
   costs nothing but the choice to run all 500 rather than 250.
5. **Smaller gaps.** §1.5 does not say how many items a variant needs, does not
   give the power, and does not correct for picking the best of five. The
   design resolves a mean TVD change of **0.006** at n = 250 (0.007 after the
   correction), and the discrimination metric of **0.165** AUC at the same n -
   two numbers that should be in the roadmap next to the item.

Two smaller notes.

- **The "criteria wording" axis is smaller than it looks.** The criteria's **text
  is the dataset's**. The server may change the frame around it (the connective,
  the order, the position) but not the criteria, without changing the question's
  meaning. §1.5's phrase "the criteria wording" should be read as the frame.
- **A dead default, worth knowing.** `serve/systemone.py:243-244` prints
  `crit.get("true", "yes")` and `crit.get("false", "no")`, so a request with only
  one side would print `no means "no"`. **No item in any of the three sources has
  a half-specified criteria** (measured: 0 of the 791 noul items across the
  three files - 69 in the 235, 222 in the jev-bench sample, 500 in the pool), so
  the default never fires. It is not a bug to fix; it is a sign that the
  criteria's *shape* is settled and only its *frame* is open.

**One thing §1.5 gets right, and it is the reason this item is worth doing
first.** It names the trap: the 235's run-to-run spread is 0.6-0.9 Capability
points, so the best of ten variants picked on it wins by selection alone. The
design set, the yardsticks, the single-axis rule and the pre-declared margin are
all the right shape. What is missing is the metric decomposition and the
base-rate guard, and without them the discipline is a formality.

---

## 8. The limits

- **Nothing here has been run.** The probe has not been run, no variant has been
  rendered into a live server, and the GPU numbers are budgets computed from
  other runs' measured rates, not measurements of this design.
- **Every mechanism claim in §2 and §3 is static analysis.** The hypotheses are
  what the arms would test. The one thing §2.4 establishes is a code-reading
  result: the readout cannot invert the sign.
- **The design set is a toxicity-heavy population.** Half A is 250
  `civil_comments` items and half B is 250 `paws` and `strategyqa` items. If a
  variant ships, it ships for populations that look like these. The 235 and the
  jev-bench test sample are the two other populations, and they are the yardsticks
  for exactly that reason.
- **The discrimination metric is weak on the noul.** The AUC's MDE is 0.165 at
  250 items, so a real but small improvement in the ranking would not be seen.
  The noul's arm can rule a ranking regression out; it cannot rule a small
  improvement in.
- **The model and the quantization are fixed.** The constants and the template
  belong to this model at this quantization. A variant that wins here is a
  candidate for this model, not a general claim about prompt templates.
- **The licence record governs the design set.** `p1b_pool.jsonl` holds only
  licences that oblige nothing beyond attribution, and the `test` split of every
  source stays a yardstick. OpenDecisions says in plain words that it is not for
  training, and it is used here only as a yardstick, never to choose a variant.
