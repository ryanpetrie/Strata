# The decision policy's training data: what may be trained on

**Date:** 2026-10-21. **What this is:** the licence record for Phase 2 of the decision-policy work
(roadmap item 2.3, self-distillation and the human-agreement gap). **What it is not:** a training run.
Nothing was trained while this document was written.

## 1. Why this document exists

Strata's decision endpoint is well calibrated on paper and still disagrees with people. On the
OpenDecisions set it scores Calibration 81.7, but its total variation distance to the human
distribution is 0.4432 and it agrees with the human consensus on 43.9% of items.

The usual fix is self-distillation: let the model teach itself. That needs no labels. But
self-distillation copies the model's *own* thinking-mode distribution. If that mode is over-confident,
the adapter inherits the over-confidence and the human-agreement gap does not close.

The scarce resource is therefore **human uncertainty**, and a trainable copy of it exists. This
document records which human-uncertainty data may be trained on, which may not, and what a released
adapter would owe. The rules below come from the licences themselves. Every licence claim carries the
URL it was read from. Anything not read and confirmed is marked **not verified**.

The data itself is not in this repository. It lives outside the repo, under a data root, and
`tools/decision_policy/download_soft_labels.py` fetches it and writes a manifest of what it fetched.

## 2. How to read the tables

- **dataset** is the hub id (or the GitHub project) the data comes from.
- **split and size** is what exists on the hub. The sizes are jev-bench's own counts, from its
  `manifest.json` and its card. For the files this work downloaded, the sizes were confirmed by counting
  the lines (2026-10-21); the sizes of the splits that were **not** downloaded are the manifest's numbers,
  read from <https://huggingface.co/datasets/Praveenrajus/jev-bench/raw/main/manifest.json>.
- **the licence, verbatim** is the text as it stands in the card or README named in the link. Where the
  card only carries a licence tag, the tag is quoted as it stands.
- **training?** is the verdict: may a released adapter be trained on it.
- **soft labels?** says whether the data carries a human label distribution (the thing worth copying)
  and what the distribution is.

## 3. The jev-bench configs

[Praveenrajus/jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) repacks 22 public
datasets as System One questions: a state, a typed question (choice, noul or score) and a gold answer.
Its own card says:

> Licenses are those of the upstream datasets; this repackaging adds no restrictions. Per-source
> provenance is in `manifest.json`.
> — <https://huggingface.co/datasets/Praveenrajus/jev-bench/raw/main/README.md>

The card's `license` field is `other`, with `license_name: mixed-see-manifest` and
`license_link: https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md`. So the licence of every
config is the licence of the upstream dataset, read below.

**The hub tree carries `train.jsonl` and `validation.jsonl` for 21 of the 22 configs.** `chaosnli` has
`test.jsonl` only. That was read from the hub's own file tree on 2026-10-21:
<https://huggingface.co/api/datasets/Praveenrajus/jev-bench/tree/main?recursive=true>. The local copy
under `...\opendecisions\jevbench_data` is test-only, as the research pass said.

### 3.1 The three configs with a human distribution, and the fourth

| dataset | what it is | split | size | the licence, verbatim (read from) | training? | soft labels? |
|---|---|---|---|---|---|---|
| `google/civil_comments` | Reddit news-site comments, Jigsaw's toxicity labels | train, validation | 1,804,874 / 97,320 / 97,320 upstream; 8,000 / 500 in jev-bench | "This dataset is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)." — [card](https://huggingface.co/datasets/google/civil_comments/raw/main/README.md) | **Yes.** Nothing owed. | **Yes.** The noul question's `toxicity` share of the annotators. 263 distinct values over the 8,000 train rows, exact fractions such as 5/46, stored as float32. |
| `google-research-datasets/go_emotions` | Reddit comments with 27 emotions plus neutral | train, validation | 8,000 / 500 in jev-bench | "The GitHub repository which houses this dataset has an [Apache License 2.0](https://github.com/google-research/google-research/blob/master/LICENSE)." — [card](https://huggingface.co/datasets/google-research-datasets/go_emotions/raw/main/README.md) | **Yes.** Nothing owed. | **Yes.** The rater vote shares over the 28 emotions, one per rater. 3 raters on 5,153 of the 8,000 train rows, 5 on 2,545, 4 on 302. |
| `ucberkeley-dlab/measuring-hate-speech` | comments with a hate-speech level | train, validation | 8,000 / 500 in jev-bench | the card's front matter reads `license:` / `- cc-by-4.0` — [card](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/raw/main/README.md) | **Yes**, with attribution (see 5.3). | **Yes.** The annotator vote shares over the 3 levels. 3 annotators on 5,233 of the 8,000 train rows, 4 on 2,642; a few rows carry hundreds. |
| `metaeval/chaos-mnli-ambiguity` | MNLI pairs re-annotated by 100 crowdworkers each | test only | 1,599 | the hub card declares **no licence at all** (no `license:` field; the hub API returns no licence tag) — [card](https://huggingface.co/datasets/metaeval/chaos-mnli-ambiguity/raw/main/README.md). The upstream says: "**Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details." — [ChaosNLI README](https://github.com/easonnie/ChaosNLI) | **No.** Non-commercial (see 5.1). | **Yes**, but out of reach: test only, and non-commercial. |

### 3.2 The configs without a human distribution

| dataset | what it is | split | size | the licence, verbatim (read from) | training? | type mix |
|---|---|---|---|---|---|---|
| `mteb/banking77` | bank messages to 77 intents | train, validation | 8,000 / 500 | the card's front matter reads `license: mit` — [card](https://huggingface.co/datasets/mteb/banking77/raw/main/README.md). jev-bench records the upstream as "cc-by-4.0 (PolyAI; mirror: mit)". | **Yes** on the mirror's terms. | choice, 77 options |
| `clinc/clinc_oos` | assistant queries to 150 intents plus out-of-scope | train, validation | 8,000 / 500 | the card's front matter reads `license:` / `- cc-by-3.0` — [card](https://huggingface.co/datasets/clinc/clinc_oos/raw/main/README.md) | **Yes**, with attribution. | choice, 151 options |
| `mteb/amazon_massive_intent` | smart-assistant utterances to 60 intents | train, validation | 8,000 / 500 | the card's front matter reads `license: apache-2.0` — [card](https://huggingface.co/datasets/mteb/amazon_massive_intent/raw/main/README.md). jev-bench records the upstream as "cc-by-4.0 (Amazon; mirror: apache-2.0)". | **Yes** on the mirror's terms. | choice, 60 options |
| `coastalcph/lex_glue` (ledgar) | contract clauses to 100 provision types | train, validation | 8,000 / 500 | the card's front matter reads `license:` / `- cc-by-4.0` — [card](https://huggingface.co/datasets/coastalcph/lex_glue/raw/main/README.md) | **Yes**, with attribution. | choice, 100 options |
| `cais/mmlu` | multiple-choice exam questions | train, validation | 285 / 500 | the card's front matter reads `license:` / `- mit`, and its Licensing Information section says "[MIT License](https://github.com/hendrycks/test/blob/master/LICENSE)" — [card](https://huggingface.co/datasets/cais/mmlu/raw/main/README.md) | **Yes.** Nothing owed. | choice, 4 options |
| `allenai/ai2_arc` (arc_challenge) | science exam questions | train, validation | 1,119 / 299 | the card's front matter reads `license:` / `- cc-by-sa-4.0` — [card](https://huggingface.co/datasets/allenai/ai2_arc/raw/main/README.md) | **Yes**, with share-alike (see 5.2). | choice, variable options |
| `nyu-mll/glue` (mnli) | does the hypothesis follow from the premise? | train, validation | 8,000 / 500 | the card's front matter reads `license:` / `- other`, and its Licensing Information section says "The primary GLUE tasks are built on and derived from existing datasets. We refer users to the original licenses accompanying each dataset." — [card](https://huggingface.co/datasets/nyu-mll/glue/raw/main/README.md) | **No** (see 5.1). | choice, 3 options |
| `SetFit/sst5` | movie-review sentiment on 5 levels | train, validation | 8,000 / 500 | **no licence is declared**: the card has no `license:` field and no licence section; it only points at the Stanford Sentiment Treebank — [card](https://huggingface.co/datasets/SetFit/sst5/raw/main/README.md) | **No** (see 5.1). | score, 5 levels |
| `Yelp/yelp_review_full` (yelp5) | star ratings from review text | train, validation | 8,000 / 500 | the card's front matter reads `license:` / `- other` and `license_details: yelp-licence`; its Licensing Information section says "You can check the official [yelp-dataset-agreement](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf)." — [card](https://huggingface.co/datasets/Yelp/yelp_review_full/raw/main/README.md) | **No** (see 5.1). | score, 5 levels |
| `nvidia/HelpSteer2` (helpfulness, verbosity) | rate an assistant response's helpfulness and length | train, validation | 8,000 / 500 each | the card's front matter reads `license: cc-by-4.0` — [card](https://huggingface.co/datasets/nvidia/HelpSteer2/raw/main/README.md) | **Yes**, with attribution. | score, 5 levels each |
| `sentence-transformers/stsb` | sentence-pair similarity on the 0-5 scale | train, validation | 5,749 / 500 | **no licence is declared** on this hub mirror: the card has no `license:` field and no licence section — [card](https://huggingface.co/datasets/sentence-transformers/stsb/raw/main/README.md). jev-bench records "cc-by-sa-4.0 (STS Benchmark)". | **Not verified.** Treat as share-alike until the STS terms are read (see 5.2 and 5.6). | score, 6 levels |
| `google/boolq` | is the answer to the question yes, given a Wikipedia passage? | train, validation | 8,000 / 472 | the card's Licensing Information section says "BoolQ is released under the [Creative Commons Share-Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license." — [card](https://huggingface.co/datasets/google/boolq/raw/main/README.md) | **Yes**, with share-alike (see 5.2). | noul |
| `copenlu/fever_gold_evidence` (fever_evidence) | is the claim supported by gold Wikipedia evidence? | train, validation | 8,000 / 500 | the card's front matter reads `license:` / `- cc-by-sa-3.0` / `- gpl-3.0`, and its Licensing Information section says "[Needs More Information]" — [card](https://huggingface.co/datasets/copenlu/fever_gold_evidence/raw/main/README.md). The upstream FEVER card quotes its terms: "These annotations incorporate material from Wikipedia, which is licensed pursuant to the Wikipedia Copyright Policy... under the Creative Commons Attribution-ShareAlike License (version 3.0)..." — [FEVER card](https://huggingface.co/datasets/fever/fever/raw/main/README.md) | **Not verified.** The card carries GPL-3.0 as well (see 5.6). | noul |
| `google-research-datasets/paws` | do two sentences with high overlap mean the same thing? | train, validation | 8,000 / 500 | the card's front matter reads `license:` / `- other`; its Licensing Information section says "The dataset may be freely used for any purpose, although acknowledgement of Google LLC ("Google") as the data source would be appreciated. The dataset is provided "AS IS" without any warranty, express or implied. Google disclaims all liability for any damages, direct or indirect, resulting from the use of the dataset." — [card](https://huggingface.co/datasets/google-research-datasets/paws/raw/main/README.md) | **Yes.** Free for any purpose; attribution appreciated. | noul |
| `ucirvine/sms_spam` | is this SMS spam? | train, validation | 4,000 / 237 | the card's front matter reads `license:` / `- unknown` — [card](https://huggingface.co/datasets/ucirvine/sms_spam/raw/main/README.md) | **Not verified** (see 5.6). | noul |
| `ChilleD/StrategyQA` (closed, grounded) | implicit multi-hop yes/no, with and without the facts | train, validation | 1,400 / 160 each | the card's front matter reads `license: mit` — [card](https://huggingface.co/datasets/ChilleD/StrategyQA/raw/main/README.md) | **Yes.** Nothing owed. | noul |

## 4. The other sources the research pass named

| dataset | what it is | split and size | the licence, verbatim (read from) | training? | soft labels? |
|---|---|---|---|---|---|
| `ZefanCai/Open-Jev` | typed decision records: a yes/no probability, a distribution over choices, independent label probabilities, an ordinal decision | 12 frozen configs; train, calibration, validation, test and OOD published separately | the card's front matter reads `license: cc0-1.0`; its Licensing and provenance section says "Original generated records are marked **CC0-1.0** in their existing provenance. This dedication covers our generated content, not upstream wording, external assets, source data or model weights. Original source code is **MIT**." — [card](https://huggingface.co/datasets/ZefanCai/Open-Jev/raw/main/README.md) | **Yes** for the generated content. The upstream wording keeps its own terms. | Yes: distribution, binary, multilabel and ordinal targets. |
| `ZefanCai/Open-Jev-v1.1` | 326,619 typed decision records, of which 147,139 are training records, from the frozen `community-hard-mix-v2-final` mixture | train / calibration / validation / test / OOD | the card's front matter reads `license: other`, `license_name: source-dependent-cc0-1.0-and-cc-by-4.0`. Its LICENSE-DATA.md says "WANLI source text and its converted records remain under Creative Commons Attribution 4.0 International... Original generated Open-Jev content is dedicated under CC0 1.0 Universal... No Wikispeedia task payload is included." — [LICENSE-DATA.md](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1/blob/main/LICENSE-DATA.md) | **Yes**, with attribution for the WANLI part. | Yes. |
| `jylin001206/SII-Decision-4B-Training-Data` | 29,326 training and 7,506 validation questions with a teacher's distributions | train, validation | the card's front matter reads `license: other`, `license_name: mixed-upstream-licenses`; its Licensing section says "This collection combines multiple upstream datasets and generated decision questions. Source materials retain their upstream terms; no uniform license is asserted for the entire collection. The source inventory is provided in `SOURCES.json`." — [card](https://huggingface.co/datasets/jylin001206/SII-Decision-4B-Training-Data/raw/main/README.md) | **Not verified per source.** `SOURCES.json` names 23 sources and their row counts but **no licence per source** (read on 2026-10-21). The mixture includes `chaosnli` (1,777 rows), `mnli` (838) and `paws` (555), so it inherits terms this document excludes. | Yes: `train_target` is "Same-size teacher probability distribution", `p0` is "Original same-size non-thinking distribution". This is the self-distillation shape. |
| `jgeuter/OpenDecisions` | 2,352 Jev-style decisions, a validation set | test | the card's front matter reads `license: cc-by-sa-4.0`; its Licence section says "CC BY-SA 4.0. Items built from real requests keep the terms of their source (Chatbot Arena 55k: Apache-2.0; Dolly: CC BY-SA 3.0; GSM8K and MATH-500: MIT); each item names its source and licence in `provenance`. The code (`score.py`, `predict_letters.py`) is under the MIT License." — [card](https://huggingface.co/datasets/jgeuter/OpenDecisions/raw/main/README.md) | **No.** The card says, in the first paragraph: "**Please do not use for training.**" (read on 2026-10-21) | Yes on 129 items: `target_probs`, "the exact probability of each option implied by the state". |
| `Mapika/decider` (GitHub) | a decision model: code, weights and the recipe for its training mixture | repository | the README says "Apache 2.0. See [LICENSE](https://github.com/Mapika/decider/blob/main/LICENSE)." — [README](https://github.com/Mapika/decider/blob/main/README.md), [LICENSE](https://github.com/Mapika/decider/blob/main/LICENSE) | **Yes** for the code and the repository. The code licence does not settle the licences of the data in its mixture. | The mixture is "public datasets plus data labelled by a local Qwen3.5-27B teacher". That is teacher data, not human data. |

## 5. The rules that follow from the licences

### 5.1 Four configs are out

| config | why | what replaces it |
|---|---|---|
| `mnli` | GLUE's own card points to the original datasets' licences and asserts none of its own; jev-bench records it as "other (GLUE/MultiNLI, research use)". A released adapter must not carry research-use terms. | `paws` (free for any purpose) and `strategyqa_closed` (MIT) cover the noul and nli shapes. |
| `yelp5` | the Yelp dataset agreement is a research licence; jev-bench records "other (Yelp Dataset License, research use)". | `sst5` is also out (below), so the 5-level review shape has no replacement in jev-bench. `helpsteer2_helpfulness` (CC BY 4.0) is a 5-level scale and stands in. |
| `sst5` | the Stanford Sentiment Treebank terms are not stated anywhere on the hub card, and the card declares no licence. "Unspecified" is not permission. | same as above. |
| `chaosnli` | **CC BY-NC 4.0.** Non-commercial. A released adapter trained on it could not be used commercially, and the upstream dataset's terms do not allow it. | nothing in jev-bench. Its 100-annotator-per-item distributions are the best human uncertainty there is, and they are out of reach. This is the real cost. |

Two more are held back on weaker grounds, and the reason is written down so the next step does not
rediscover it:

- `sms_spam`: the card declares `license: unknown`. The UCI collection is public, but "unknown" is not
  permission. Keep it out of anything released; it may sit in an internal validation pool.
- `stsb`: the hub mirror declares no licence. The underlying STS Benchmark is recorded by jev-bench as
  CC BY-SA 4.0. Until the STS terms are read from their own source, treat it as share-alike.

### 5.2 The share-alike obligation

`arc_challenge` (CC BY-SA 4.0), `boolq` (CC BY-SA 3.0), `fever_evidence` (CC BY-SA 3.0) and `stsb`
(STS Benchmark, CC BY-SA 4.0 per jev-bench) are share-alike. The rule of ShareAlike is: if a work is
derived from the material and it is released, the release must carry the same licence.

A trained adapter is derived material. It contains no text anyone can read, but the weights carry what
the training taught them, and the conservative reading is that the adapter is adapted material.

**Conservative practice:**

1. Keep share-alike data **out** of any adapter that will be released. Train the released adapter on the
   free-licence and attribution-only sources (see the list in 7), so the released weights owe nothing
   beyond a citation.
2. If an adapter trained on share-alike data must be released, release it under the same licence
   (CC BY-SA 4.0 for `arc_challenge` and `stsb`, CC BY-SA 3.0 for `boolq` and `fever_evidence`) and say
   so plainly in the model card, naming the sources. Do not ship a permissive licence on weights that
   came from share-alike data.
3. Either way, name the sources in the model card and keep the training data's own terms attached to it.
   A share-alike obligation that nobody can trace is a broken obligation.
4. If the adapter is never released (an internal model, a research run), share-alike does not bite.
   Nothing is distributed, so nothing is licensed onward. This is the usual case for the Phase 2 work.

**Unconfirmed static analysis, read from the licence texts and not from a lawyer:** the reading in
point 1 (an adapter is adapted material) is the conservative one, not a settled question. The licences
were written for works that carry the material; a weight file carries none of it. The safe path costs
little here, because the share-alike configs carry no human distribution at all: they are useful for
the shape of the data, not for the uncertainty.

### 5.3 What must be attributed

Attribution is a requirement, not an obstacle. These sources may be trained on and the release must
name them:

- `ucberkeley-dlab/measuring-hate-speech` (CC BY 4.0)
- `nvidia/HelpSteer2` (CC BY 4.0), both the helpfulness and the verbosity scale
- `clinc/clinc_oos` (CC BY 3.0)
- `coastalcph/lex_glue` (CC BY 4.0)
- `google-research-datasets/paws` (other; attribution "would be appreciated", not required)
- `ZefanCai/Open-Jev-v1.1` for the WANLI part (CC BY 4.0)
- `mteb/banking77` and `mteb/amazon_massive_intent` (the mirrors are MIT and Apache-2.0; jev-bench
  records the upstreams as CC BY 4.0, so name both)

CC BY 4.0 also asks for a description of the changes when the material is modified. A rewritten state
template or a reworded instruction is a modification; say so.

### 5.4 What needs nothing

These may be trained on, and a release owes no licence and no share-alike:

- `google/civil_comments` (CC0 1.0)
- `google-research-datasets/go_emotions` (Apache-2.0)
- `cais/mmlu` (MIT)
- `ChilleD/StrategyQA` (MIT)
- `ZefanCai/Open-Jev` for its generated content (CC0 1.0)
- `Mapika/decider`'s code and repository (Apache-2.0)

`mteb/banking77` (MIT) and `mteb/amazon_massive_intent` (Apache-2.0) sit in between: the mirrors ask
nothing, but jev-bench records their upstreams as CC BY 4.0, so name them (5.3).

Apache-2.0 and MIT ask for the licence and the notice to be kept with the copy. That is the notice file
in the release, not a restriction on the weights.

### 5.5 Two records in the upstream manifest are wrong

**`chaosnli` is recorded wrongly, and the research pass was right about it.** jev-bench's `manifest.json`
records `"license": "cc-by-sa-4.0 (ChaosNLI; Hub mirror of the MNLI portion)"`, and the same wrong string
is in the card's table and in
[Jevify's DATASETS.md](https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md) (`chaosnli |
choice | 3 | nli | 100 annotators per item - calibration gold (eval only) | cc-by-sa-4.0`). The upstream
says otherwise, in two places I read myself:

> **Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.
> — <https://github.com/easonnie/ChaosNLI>

and the repository's `LICENSE` file is the Creative Commons Attribution-NonCommercial 4.0 International
licence (<https://github.com/easonnie/ChaosNLI/blob/master/LICENSE>). The hub mirror
`metaeval/chaos-mnli-ambiguity` declares **no licence at all**: no `license:` field in the card, and
the hub's own API returns no licence tag for it.

So the truth is: the upstream is CC BY-NC 4.0, the mirror says nothing, and the manifest's CC BY-SA 4.0
is wrong. The verdict does not change (the config is out either way, and it is out for a stronger
reason than the manifest gives). But a licence record that says "share-alike" where the truth is
"non-commercial" is the kind of error that gets a release into trouble.

**`fever_evidence` is recorded incompletely.** jev-bench's manifest records `"license": "cc-by-sa-3.0"`.
The hub card carries two licences, not one: `cc-by-sa-3.0` **and** `gpl-3.0`
(<https://huggingface.co/datasets/copenlu/fever_gold_evidence/raw/main/README.md>). GPL-3.0 is copyleft.
What it covers here is not stated anywhere on the card (its Licensing Information section reads
"[Needs More Information]"), and the upstream FEVER card quotes terms that mention only CC BY-SA 3.0 and
the Wikipedia terms. **Not verified** which of the two applies to the rows. Until that is settled, treat
`fever_evidence` as out, exactly like the share-alike and GPL configs.

### 5.6 What is still not verified

- The **STS Benchmark's own terms** (`stsb`). The hub mirror declares no licence; jev-bench records
  CC BY-SA 4.0. Not read from the STS Benchmark's own source.
- The **FEVER row-level terms** (`fever_evidence`), see 5.5.
- The **UCI SMS Spam Collection's terms** (`sms_spam`). The hub card says `unknown`.
- **Per-source terms for the SII-Decision mixture.** `SOURCES.json` names the sources and their row
  counts but no licences.
- **`Open-Jev` and `Open-Jev-v1.1` payload**: not downloaded here (see 6), so their contents were not
  checked against their cards.
- **The `Mapika/decider` mixture's per-source terms.** The repository's own README does not list the
  licences of the ~95 datasets in its mixture.
- **The research pass's claim that `Mapika/decider` is "about 72% of the 'Releasing the Inner Jev'
  training mixture".** Not confirmed by anything I read. The decider README says the mixture is "public
  datasets plus data labelled by a local Qwen3.5-27B teacher" and "Nothing was distilled from Jev"; the
  OpenDecisions README describes its own 2,352 items as 599 computed, 1,443 authored and 310 from real
  requests, and gives no mixture share. The two numbers in the decider README that look like 72 are a
  Decision Index score (72.1 for Jev 1.13.0) and a note that a model "agreed with only 72% of its own
  generic-option labels". **Not verified**; treat the claim as unsourced.

## 6. What was downloaded, and where

The data root is outside the repository:

```
C:\Users\ryanp\AppData\Local\Temp\opencode\decision_policy_training\
```

22 files, 44,966,196 bytes, 32,328 rows in all (24,000 train rows and 8,328 validation rows). Every
file came from `https://huggingface.co/datasets/Praveenrajus/jev-bench/resolve/main/data/<config>/<split>.jsonl`.
The manifest next to the data (`manifest.json`) holds the URL, the byte size, the sha256, the line count,
the type mix and the number of soft-label records for each file.

- train + validation for `civil_comments`, `measuring_hate_speech`, `go_emotions` (the three configs
  with a human distribution and a train split)
- validation for `boolq`, `strategyqa_closed`, `mmlu`, `banking77`, `stsb`, `ledgar`, `sst5`,
  `arc_challenge`, `massive`, `clinc150`, `paws`, `fever_evidence`, `sms_spam`,
  `helpsteer2_helpfulness`, `helpsteer2_verbosity`, `strategyqa_grounded` (the type-covering set)

`chaosnli` was not downloaded: it has no train or validation split, and its licence rules it out.

One file (`go_emotions/validation.jsonl`) was fetched a second time on 2026-10-21 and its sha256 matched
the manifest's, so the digests are the digests of the hub's files and not of a corrupted copy.

**The parquet sources are out of reach here, and that is a limitation of the machine, not of the plan.**
The workspace virtual environment has numpy, requests and tqdm, and no `pyarrow`, `pandas` or
`datasets`. Nothing was installed for this work. So:

- the upstream `google/civil_comments` train split (1,804,874 rows, "Size of downloaded dataset files:
  414.95 MB" per its card) is a parquet file that cannot be opened here;
- `ZefanCai/Open-Jev-v1.1` (326,619 records) ships as parquet and cannot be opened here either.

**The proposal, not done here:** add `pyarrow` to the workspace virtual environment
(`C:\tools\strata\.venv\Scripts\python.exe -m pip install pyarrow`). That is one wheel, about 40 MB, and
it opens every parquet source the hub serves. The install needs the user's go-ahead, because this
machine holds one engine and the work should not change the environment without being asked.

## 7. The rule for the next step

**What may be trained on, for an adapter that will be released:**

1. `google/civil_comments` (CC0 1.0) - the noul soft labels
2. `google-research-datasets/go_emotions` (Apache-2.0) - the choice soft labels
3. `ucberkeley-dlab/measuring-hate-speech` (CC BY 4.0) - the score soft labels, with attribution
4. `cais/mmlu`, `ChilleD/StrategyQA` (MIT), and `mteb/banking77`, `mteb/amazon_massive_intent` on
   their mirrors' terms (MIT / Apache-2.0; the upstreams are recorded as CC BY 4.0, so name them)
   - the shapes, no human distribution
5. `nvidia/HelpSteer2`, `clinc/clinc_oos`, `coastalcph/lex_glue`, `google-research-datasets/paws` -
   the shapes, with attribution
6. `ZefanCai/Open-Jev` for its generated content (CC0 1.0)

**What may not:** `mnli`, `yelp5`, `sst5`, `chaosnli` (section 5.1), `fever_evidence` until its terms
are settled (5.5), `sms_spam` and `stsb` until theirs are read (5.6), and `jgeuter/OpenDecisions`,
which says in plain words that it is not for training.

**What is validation-only, for every source:** the `test` split. The train and validation splits are
the only ones a training run may read, and the test split stays a yardstick.

**The share-alike question is settled conservatively** (5.2): a released adapter is trained on the
list above only, so it owes attribution but no share-alike and no licence on the weights.

**What to fit where.** The human distribution lives in three places: the noul share (civil_comments),
the choice distribution over 28 emotions (go_emotions) and the score distribution over 3 levels
(measuring_hate_speech). A copy of human uncertainty has to be fitted per question type, because the
three types have different label spaces and different numbers of levels.

**The fitting pool** is the jev-bench **validation** split, 500 items of every question type, with the
per-config counts and the seed fixed in `tools/decision_policy/soft_label_pool.json` (that file also
carries the licence verdict of every config, taken from the tables above, and the list of the configs
that are out and why). The pool is built by `tools/decision_policy/soft_labels.py --plan`; the data
itself stays outside the repository. The **test** split is the yardstick and is never fitted on.

The 200-item-per-type sample this work wrote first (seed 20261020) is still in the data directory; it is
the smaller of the two and the pool plan supersedes it for the temperature fit.

## 8. The limits of this document

- The licences were read on 2026-10-21 from the cards and READMEs named above. A card can change.
- This is a reading of licences, not legal advice. It says what the texts say, and it marks the
  readings that are conservative.
- No training happened, and no number in this document comes from a training run. The sizes, the
  sha256 digests, the type mixes and the soft-label counts were measured by reading the downloaded
  files on this machine.
- The share-alike reading in 5.2 is an unconfirmed static analysis, marked as such there.
