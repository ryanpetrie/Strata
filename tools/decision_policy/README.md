# decision_policy - shared-backbone System 1 decision policy exploration

Research tooling for the task in
[docs/research/shared_backbone_decision_policy.md](../../../docs/research/shared_backbone_decision_policy.md).
It measures whether a small decision prompt, served by the already-running Strata server, is fast and accurate
enough to be worth a hidden-state classifier later. It changes nothing in the engine or the server.

## Run

```sh
python tools/decision_policy/make_examples.py                        # build/freeze the example set
python tools/decision_policy/baseline.py --warmup 2 --runs 3         # measure the prompt route
python tools/decision_policy/stage_a.py --runs 2 --with-compact-arm  # measure the fixed-preamble route
```

Each script reads the address, model id, and API key from the launch config, or from
`--host/--port/--model/--api-key` / `STRATA_API_KEY`. They start no server and load no model.

`stage_a.py` compares two preamble arms on the same examples: the verbose preamble (readable states) and the
compact preamble (`--with-compact-arm`: abbreviated `G:|LA:|NI:` states, matching compact worked examples).
Use `--with-short-arm` to re-run the Phase 1 prompt for contrast.

## The decision endpoint's own tools

These need the server running with `"systemone": {"enabled": true, "diagnostics": true}`.

```sh
python tools/decision_policy/logits_check.py      # the probe: seven cases, one run is decisive
python tools/decision_policy/opendecisions.py \
  --items <opendecisions.jsonl> --out <predictions.jsonl> --ids-file <ids.txt> --batch-k 1
python tools/decision_policy/readings_compare.py \
  --items <items.jsonl> --predictions <predictions.jsonl> --runlog <predictions.runlog.jsonl>
python tools/decision_policy/readings_split.py \
  --items <items.jsonl> --runlog <predictions.runlog.jsonl>
python tools/decision_policy/vram_headroom.py       # the free VRAM during a run
```

`logits_check.py` asks for a handful of small requests and reports, per case, whether the request took the
LOGITS fast path (a real probability) or fell back to the decode path (1.0/0.0). The cases cover the paths
that matter: a yes/no, a choice, a score, two options that share their first token, two options of different
lengths, two options where only some share a first token, and a request with two questions (each question is
answered on its own, so the fast path fires for each of them).

`opendecisions.py` runs a dataset through the endpoint and writes the scorer's input (the probabilities) and
a per-item run log (the fast path or not, and why not). It is resume-safe: it skips the ids already in the
output file, so an interrupted run restarts with the same `--out`.

`readings_compare.py` and `readings_split.py` compare the two readings of the confidence (the closed one, the
options renormalized over themselves, and the open one, the model's own probability) against the outcome, per
type: the AUC (which one ranks a correct answer above a wrong one) and the expected calibration error (which
one is calibrated).

`vram_headroom.py` watches the free VRAM while a run is going, because the conversation cache holds parked
conversations and the engine's warning threshold is what decides how many slots the server can keep.

## The per-type temperature

The endpoint can scale the reported distribution per question type: the temperature divides the logits
before the softmax, so the distribution over the options is flattened (T above 1) or sharpened (T below 1).
The **pick never changes** - the map is monotone - so the answer is the same under every temperature.

**The fit** (one scalar per type: choice, noul, score):

```sh
python tools/decision_policy/temperature.py \
  --items <items.jsonl> --runlog <runlog.jsonl> --predictions <predictions.jsonl> \
  --out-json tools/decision_policy/temperature_calibration.json \
  --out-report tools/decision_policy/temperature_report.md
```

The script reads a baseline run (the run log and the predictions the official scorer takes), rebuilds each
item's closed distribution, fits the temperature per type, cross-validates it (5 folds, stratified by type,
seed 42; the fit never sees the item it is scored on), scores the out-of-fold predictions with the official
`score.py`, and writes the constants and the report.  Three objectives are fitted and cross-validated
(`--objective nll|brier|ece`): they disagree on this data, and the report says why and which one is used.

It also says how sure the result is, in two ways, and neither of them is a point estimate on its own:

- an **out-of-bag** bootstrap per type: the items are resampled twice, the temperature is fitted on one
  sample and the ECE measured on the other, and the report says how often the temperature wins.  A share
  is not an effect size;
- a **paired item bootstrap** over the 235 items: both arms are scored on the same resample, so the
  difference is paired and the interval on it is honest.  Pass `--temp-runlog <the run that used the
  temperature>` to add the paired control - the same run with the temperature undone, which is the same
  engine and the same picks with the temperature's arithmetic removed.

**The application** - the launch config, per machine:

```json
"systemone": {
  "enabled": true,
  "diagnostics": true,
  "temperature": {"choice": 0.62, "noul": 0.67, "score": 0.77}
}
```

The key is **opt-in**: without it every type is 1.0, and 1.0 means "no temperature" - the reported numbers
are then exactly the ones the endpoint reported before the feature existed.  A temperature that is not a
number, is not above 0, or names a question type the endpoint does not have is a startup error, not a silent
default.  The server prints the temperatures it is using at startup.

The constants that were fitted for this model, with their provenance (the model, the date, the split rule,
the seed, the ECE before and after), are in `temperature_calibration.json`.  To use them, copy the three
numbers into the launch config's `"systemone": {"temperature": {...}}` and restart the server.

Two things to know about the math:

- The temperature is applied to the **option-level** effective logits (the closed distribution), not to the
  per-trie-row logits: an option's weight is a product of per-node token probabilities and the per-row
  normalizers differ per option.  For a single-row question the two forms are identical.
- The **open reading** (the reported confidence) is the closed reading at T scaled by the mass the model put
  on the options, measured at T = 1.  The model's own probability at temperature T would need the
  full-vocabulary normalizer at T, which the server does not have.

## The other datasets

`jevbench.py` and `jevbench_rewrites.py` run the two JevBench sets through the endpoint. They need the
server running with `"systemone": {"enabled": true, "diagnostics": true}`.

```sh
python tools/decision_policy/jevbench.py --items <items.jsonl> --out <predictions.jsonl>
python tools/decision_policy/jevbench_rewrites.py --items <sample.jsonl> --out <predictions.jsonl>
python tools/decision_policy/jevbench_rewrites.py --items <sample.jsonl> --out <predictions.jsonl> --report
```

`jevbench.py` runs the 3% sample of [Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench)
(683 records over 22 configs). **280 of them carry the human label distribution** (`soft_label`), so the
calibration can be scored against how uncertain the humans were and not only against one right answer. The
run log carries that distribution and the total variation distance to it per item. Caveats the sample's own
metadata records: jev-bench has no difficulty tier, so every item is filed under `"hard"` and the scorer's
tier weights collapse to one - say so in any write-up, it is not a difficulty claim; the licence is per-source
and mixed (several "research use only"); and the dataset says validation only.

`jevbench_paired.py` makes the **paired** comparison of a run made with a temperature against the same run
without it. The temperature's transform is invertible - the server scales the closed reading by
`closed_T(i) = normalize(closed_1(i) ** (1/T))`, so `closed_1(i) = normalize(closed_T(i) ** T)` - so a
tempered run carries the T = 1 distribution inside it:

```sh
python tools/decision_policy/jevbench_paired.py \
  --items <items.jsonl> --runlog <the tempered run's runlog.jsonl> \
  --baseline-runlog <a stored T = 1 run's runlog.jsonl> --scorer <path to score.py> \
  --out-predictions <the recovered T = 1 predictions.jsonl> --out-report <report.md>
```

Both arms then come from the **same run, the same items, the same engine**, so the difference between them is
the temperature and nothing else. The recovered arm is written as a predictions file, so the official scorer
scores it: the paired Capability on identical items. The script **validates the inversion** rather than
assuming it (the recomputed per-item TVD against the harness's stored one, the forward transform against the
arm the run reported, the noul's recovered P(yes) against the run log's own open reading) and reports the
distance from the recovered arm to a stored T = 1 run, which is the engine's run-to-run wobble.

`jevbench_rewrites.py` runs the JevBench coherence suite (the metamorphic rewrite cases). The official metric
is in the published package (`pip install jevbench`, [JevBench/jevbench](https://github.com/JevBench/jevbench),
Apache-2.0); the harness re-implements the laws and names the official file and class beside each one, so it
runs from a checkout with nothing installed. The metric needs no gold labels: it asks whether the distribution
stays coherent when a question is reworded, negated or combined. **One failed request drops that relation's
coverage, and a relation under 95% coverage is not scored at all** - so the engine's failures would silently
remove relations. The harness says so in the report rather than counting a missing answer as a law broken.

Both harnesses are resume-safe, both omit a failed request from the predictions file (the scorer counts a
missing answer as wrong), and neither retries a request that produced an answer.

## The options' shape

```sh
python tools/decision_policy/trie_costs.py --items <items.jsonl>    # the trie walk's cost per item
python tools/decision_policy/trie_examples.py --items <items.jsonl>  # the colliding questions, as tokens
```

`trie_costs.py` uses the shipped code (`serve/logits_first_token.py`) and the server's own tokenizer, so the
numbers are the ones the endpoint will pay: how many extra engine requests the trie walk needs for every item
whose options share a first token.

## Files

| File | Source or generated | Purpose |
| --- | --- | --- |
| `make_examples.py` | source | the state schema, the label rule (`decide`), and the generator. It asserts the rule matches every stated target. |
| `examples.jsonl` | generated | 60 synthetic, rule-labelled examples. Do not hand-edit; change `make_examples.py` and re-run. |
| `baseline.py` | source | the harness: prompts, timing, validation, metrics, report. |
| `baseline_report.json` / `.md` | generated | Phase 1 results. |
| `stage_a.py` | source | the fixed-preamble experiment: verbose vs compact arms, new-vs-repeat split, prompt-token counts. |
| `stage_a_report.json` / `.md` | generated | Stage A results. |
| `logits_check.py` | source | the probe: the fast path fires or it does not, per case. |
| `opendecisions.py` | source | the evaluation harness: the OpenDecisions set through the endpoint, with a per-item run log. |
| `readings_compare.py`, `readings_split.py` | source | the two readings of the confidence against the outcome. |
| `vram_headroom.py` | source | the free VRAM during a run. |
| `temperature.py` | source | the per-type temperature: the fit, the cross-validation, and the official scorer's numbers. |
| `temperature_calibration.json` | generated | the fitted constants and their provenance (the model, the date, the split rule, the seed, the ECE before and after). |
| `temperature_report.md` | generated | the fit's report: the three objectives, the cross-validation, the limits. |
| `trie_costs.py`, `trie_examples.py` | source | the options' token shape and the walk's cost. |
| `first_token.py` | source | the first-token ambiguity of the whole dataset (the Phase 1 study). |
| `logits_prob.py` | source | the probability helpers the response shape is built from. |
| `jevbench.py` | source | the jev-bench sample harness: one item per request, the human label distribution and the distance to it. |
| `jevbench_paired.py` | source | the paired temperature comparison: the T = 1 arm recovered from a tempered run, the official scorer's numbers for both arms, and the checks on the inversion. |
| `jevbench_sample.py` | source | builds the 3% sample and its scorer-shaped items file from the dataset. |
| `jevbench_rewrites.py` | source | the JevBench coherence metric over the endpoint's answers (the official laws, re-implemented). |
| `jevbench_rewrites_sample.py` | source | builds the rewrite sample from the official frozen plan (SHA-256 checked). |
| `tokenization_report.md`, `tokenization_results.json` | generated | the model's tokenizer against the reference one. |
| `first_token_report.md`, `first_token_analysis.json` | generated | the first-token study's results. |

## Labels are synthetic

Every target comes from `decide()` in `make_examples.py`, not from a real agent trajectory. Accuracy here is
agreement with that rule. It is not a learned-policy quality claim. Real labels are Stage D of the research doc,
and they do not exist yet.
