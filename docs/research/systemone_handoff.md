# SystemOne on Strata — session handoff

> **Historical (2026-10-19).** This is one session's handoff, kept as the record of
> that session. It is **not** the current state: the confidence is no longer
> degenerate (the endpoint reads the model's own next-token distribution, so a
> response says `"calibrated": true`), the fast path now carries 233 of the 235
> sample items (the options that share a first token are separated by a trie walk),
> and the conversation cache has 2 slots, not 4. The current state is
> [shared_backbone_decision_policy.md](shared_backbone_decision_policy.md) §10, and
> the tools are listed in
> [tools/decision_policy/README.md](../../tools/decision_policy/README.md).

**Date:** 2026-10-19. **Purpose:** let a fresh session continue exactly where this one stops.
**Rule zero:** the machine holds ONE Strata instance. Never start a second engine/server. No VRAM-heavy work without asking the user.

## 1. Current state of the world

- **Workspace = the instance.** Directories were renamed: old `C:\tools\strata-dev` → **`C:\tools\strata`** (the repo you are in); old install → `C:\tools\strata-old` (rollback only, read-only).
- **Server:** workspace `serve/server.py` on **port 8080**, config `C:\tools\strata\strata-unsloth-ud-iq4_xs.json` (gitignored via `/strata-*.json`; holds the API key — never print it, read it programmatically). Engine 0.1.45.3, model `qwen3.8-flash-next-unsloth-ud-iq4_xs`, loaded. Host is `0.0.0.0` with API key set (fine per AGENTS.md rule: never expose *without* a key).
- **Endpoint live:** `POST /v1/systemone` in `/v1/status` dialects. All code below is deployed (server process started after the last `serve/systemone.py` write; verified functionally, not just by timestamps).
- **Background jobs:** a same-question batch run may still be running (see §4). It is resume-safe: re-running the identical command skips ids already in the predictions file.

## 2. What was built (workspace = `C:\tools\strata`)

| File | What it is |
|---|---|
| `serve/systemone.py` | The endpoint's pure module: validate → render → parse → build. Types `noul`/`choice`/`score` (same names as the reference API). Score criteria accept a list OR `{id: description}`. Noul answers accept `yes`/`no`/decimal in [0,1] (reported as given; this is live and measured). Every response says `"calibrated": false`. Malformed output or truncation → `StructuredOutputError` (502), never a fabricated answer. |
| `serve/server.py` (+~100 lines) | `systemone_enabled` flag (default off), config `"systemone": {"enabled": true}`, admission + `begin_request` + dispatch, `Service.systemone` (validate → prepare → `openai_chunks` → `openai_collect` → parse → respond, temp 0, thinking off), `dialects` entry. |
| `serve/test_systemone.py` | 16 tests, MockEngine, no GPU. Run: `C:\tools\strata\.venv\Scripts\python.exe -m unittest serve.test_systemone -v` (system python lacks jinja2; use the workspace venv). Full `serve.test_server` suite: 341 pass, 3 skipped. |
| `tools/decision_policy/opendecisions.py` | OpenDecisions harness: serial (`--batch-k 1`) + batched (`--batch-k N`, `--batch-seed`, `--batch-order shuffle|file`) arms, resume-safe append, per-item runlog with latency/tokens, official `predictions.jsonl` output. One attempt per item; non-200 → omitted (counts wrong per dataset rules). |
| `tools/decision_policy/{baseline,stage_a,make_examples}.py` | Earlier phases: API-only baseline, fixed-preamble experiment, synthetic rule set. Superseded by the endpoint work but reproducible. |
| `docs/research/shared_backbone_decision_policy.md` | Findings doc §§6–9 (endpoint design, Stage A, OpenDecisions results). |
| `docs/research/shared_backbone_decision_policy.md` §9 | OpenDecisions validation results (update after full run). |

**Dropped, do not revive without user direction:** the policy-head track (linear head on `mixed`, LoRA, any engine changes). User explicitly killed Tier H: "what I want, in full, is a systemone-style API running with Strata, using Strata resources."

## 3. Measured results so far

- **Prompt-route baseline (synthetic, 60 rule-labelled examples):** new-state 1606 ms cold / 158 ms repeat; accuracy 0.75 vs rule. Compact preamble: 532 ms new-state, 1.0 rule-agreement. Engine log law: restore checkpoint + ~19 ms per new tail token; root checkpoint needs ≥2048-token system prompt.
- **Endpoint verification (22/22):** single + batched answers all 3 types, all eight 400s, both 401s, chat regression. Two prompt bugs found live and fixed (question-header echo, instruction echo) + regression tests.
- **OpenDecisions 10% sample (235 ids, seed 42, serial):** official `score.py` → Capability **65.6** (Intelligence 62.0, Calibration 69.2); accuracy on predicted 83.4% (choice 83.2 / noul 87.0 / score 62.7); format-failure 12/235 (5.1%, all 502s); trap picked 14/135 (10.4%); latency med 3.2 s, p95 5.4 s. Context: above no-training baseline SemIf 58.9, below trained 4B systems (67–77) on their board.
- **Batched arm, mixed questions (47×5, seed 43):** 175/235 predicted; Capability 54.0. Per-question latency 1.5 s vs 3.2 s serial (**2.1× amortization**). Serial-vs-batch agreement on co-predicted: **87.6%** (6 flips). Accuracy ON PREDICTED 86.3% vs 83.4% serial — batching doesn't corrupt judgment. The penalty is all format: 12/47 batches fully failed (zero partial — atomic per request by construction), 46 serial-correct items lost as collateral.
- **Batched arm, same question × 5 (47 batches, 47 distinct repeated signatures, seed 44):** 230/235 predicted, Capability **66.5** (Intelligence 60.6), correct 187/235 = **79.6%** ≈ serial 79.1%. Format failures nearly vanish: 5/235 (2.1%, all one composite_gen batch with bare-sentence levels). Per-question latency med 1.1 s (**2.9× vs serial**). No inertia: uniform model answers 3/46 batches vs uniform gold 4/46. Caveat: ran with decimal-noul live (serial/mixed ran strict), so TVD not comparable; accuracy/failure comparisons stand. Conclusion: same-question batching is the production shape (fixed vocabulary loop).
- **Calibration items:** model outputs probabilities unprompted on rate-stating states (`0.7` for P=0.70 exactly); endpoint now accepts them (verified live: `{"noul": 0.7}` HTTP 200).

## 4. In-flight / pending work (in order)

1. ~~Same-question batch run~~ DONE (see §3). Comparison complete; no further analysis owed.
2. **Full 2,352-item serial run** (~1.5–2 h background) → official Capability number. Command: same harness without `--limit`, fresh `--out` file. User deferred this; confirm before launching. NOTE: current server has decimal-noul live, so the full run will differ from the strict-parser 10% sample on calibration items — document as a protocol version (+decimals), or re-freeze strict. Decide with user.
3. **Hybrid fallback** (batch, then serial retry of failed batches): proposed, not implemented. Best effort/accuracy trade; needs no server change. Natural follow-up now that both batch modes are measured.
4. **Real-trajectory logging** (Stage D): not started. Needs user access to agent trajectories.

## 5. Key commands

```bat
:: status (key from config, never paste it)
:: GET /v1/status -> dialects must include /v1/systemone
:: smoke: 24 stratified items
cd C:\tools\strata && python tools/decision_policy/opendecisions.py --items <D>\opendecisions.jsonl --out <D>\smoke.jsonl --limit 24 --stratify
:: 10% serial sample (seed 42 ids file)
python tools/decision_policy/opendecisions.py --items <D>\opendecisions.jsonl --out <D>\sample10.jsonl --ids-file <D>\sample10.txt
:: mixed batch (seed 43) / same-question batch (seed 44, --batch-order file)
python tools/decision_policy/opendecisions.py --items <D>\opendecisions.jsonl --out <D>\batch5.jsonl --ids-file <D>\sample10.txt --batch-k 5 --batch-seed 43
python tools/decision_policy/opendecisions.py --items <D>\opendecisions.jsonl --out <D>\sameq5.jsonl --ids-file <D>\sample10_sameq.txt --batch-k 5 --batch-order file
:: score (official scorer, MIT)
python <D>\score.py <predictions.jsonl> --data <items.jsonl>
:: unit tests (workspace venv python, NOT system python)
C:\tools\strata\.venv\Scripts\python.exe -m unittest serve.test_systemone -v
```
`<D>` = `C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions` (data + predictions + runlogs live here, OUTSIDE the repo, by design: dataset says do-not-train; keep it out of git).

## 6. Gotchas learned the hard way

- **JSONL parsing:** the dataset file contains a raw U+2028 inside one state string. `str.splitlines()` splits on it and corrupts a row; iterate the file object or `split("\n")` instead. Harness already does this.
- **ids-file order:** the harness sorts items to the ids-file order (needed for cluster-adjacent batching). `--batch-order file` chunks consecutively; `shuffle` (default) re-randomizes.
- **Server caches imports:** `serve/systemone.py` is imported per-request but cached in `sys.modules` — code changes need a server restart. Only the user restarts it.
- **No second instance, ever.** Check `Get-NetTCPConnection -State Listen` before assuming anything about ports. Old install on 8095-era history is gone; current: workspace on 8080.
- **Decode is cheap, prefill is everything** (~19 ms/new tail token; cold full prefill dominates). Repeats are fast via conversation cache (4 slots) but unique states never hit it (0 cached tokens on all 235 sample items).
- **Endpoint strictness philosophy:** 502 on any unparseable output, never guess. The `Valid line starts:` closer + `start your line with exactly "name:"` per question are load-bearing prompt features — do not remove.
- **Noul decimals are intentional**, not leniency creep: reference `noul` is already a float; temp-0 transcription of a stated rate is reading, not sampling.
- **Score dict criteria `{id: desc}`**: answer with the short id; order preserved so scoring is unaffected. Falls back to raw list when any level lacks an `id: desc` split (pure-digit ids excluded — they collide with index semantics).

## 7. References

- Reference API: https://abacus.ai/help/route-llm/systemone (Typesafe Jev; state + questions, noul/choice/score, calibrated distributions, 0.2–0.5 s, input-only billing).
- Dataset: https://huggingface.co/datasets/jgeuter/OpenDecisions (v1.0.1, CC-BY-SA-4.0, validation only). Local copy + scorer in `<D>`. Paper: arXiv 2610.17205.
- Leaderboard context (Oct 2026): Jev 80.1, best open 85.1, trained 4B systems 67–77, SemIf-no-training 58.9. Ours (prompt-only, degenerate calibration): 65.6 on 10%.
- Repo docs that matter: `docs/DETAILS.md:1096-1106` (prompt-cache root rule), `serve/structured.py` docstring (failed structured output = error, never retry/forge).
- Code anchors: `serve/server.py` `Service.systemone` (:4060), dispatch (:6303), `_systemone` handler (:6851), admission/begin_request tuples (:6224/:6284), dialects (:4238), config application (:7893).

## 8. Suggested next session order

1. Check server on 8080 (`/v1/status` dialects). Nothing is running in the background (all three arms complete).
2. Ask user about the full 2,352 serial run (note the decimal-parser protocol difference vs the strict 10% sample), then the hybrid fallback experiment.
3. The remaining open question from the user: configurability of the endpoint's system prompt (see below) — no code exists for it yet.
