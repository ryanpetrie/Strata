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
| `trie_costs.py`, `trie_examples.py` | source | the options' token shape and the walk's cost. |
| `first_token.py` | source | the first-token ambiguity of the whole dataset (the Phase 1 study). |
| `logits_prob.py` | source | the probability helpers the response shape is built from. |
| `tokenization_report.md`, `tokenization_results.json` | generated | the model's tokenizer against the reference one. |
| `first_token_report.md`, `first_token_analysis.json` | generated | the first-token study's results. |

## Labels are synthetic

Every target comes from `decide()` in `make_examples.py`, not from a real agent trajectory. Accuracy here is
agreement with that rule. It is not a learned-policy quality claim. Real labels are Stage D of the research doc,
and they do not exist yet.
