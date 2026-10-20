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

## Files

| File | Source or generated | Purpose |
| --- | --- | --- |
| `make_examples.py` | source | the state schema, the label rule (`decide`), and the generator. It asserts the rule matches every stated target. |
| `examples.jsonl` | generated | 60 synthetic, rule-labelled examples. Do not hand-edit; change `make_examples.py` and re-run. |
| `baseline.py` | source | the harness: prompts, timing, validation, metrics, report. |
| `baseline_report.json` / `.md` | generated | Phase 1 results. |
| `stage_a.py` | source | the fixed-preamble experiment: verbose vs compact arms, new-vs-repeat split, prompt-token counts. |
| `stage_a_report.json` / `.md` | generated | Stage A results. |

## Labels are synthetic

Every target comes from `decide()` in `make_examples.py`, not from a real agent trajectory. Accuracy here is
agreement with that rule. It is not a learned-policy quality claim. Real labels are Stage D of the research doc,
and they do not exist yet.
