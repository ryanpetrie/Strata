# Decision policy: literature context and gaps

**Status:** research note, written 2026-10-21 after a web search across the 2026
"System One" decision-model literature.
**Companion:** [shared_backbone_decision_policy.md](shared_backbone_decision_policy.md)
is the measured record for this codebase; this document is the literature that
tells us what that record means and what is still missing.
**Scope:** Strata's `/v1/systemone` endpoint, which answers typed decisions
(choice / noul / score) by reading the backbone's own next-token distribution at
the answer position. No text generation, no second model, no heads.

---

## 1. What Strata is actually doing, in literature terms

Strata's path is a **frozen first-token decision readout**: one forward pass over
`state + question + options`, read the next-token distribution at the answer
position, restrict it to the option tokens (trie-walking where they collide).
That is a recognized, named, published method with its own family name:
"System One" / "typed decision model".

Two 2026 papers formalize it directly:

- **"LLM-as-Jev"** (arXiv 2610.02076): "extracts calibrated decisions directly
  from next-token probabilities over bracketed numeric identifiers… without
  altering architectures, tokenizers, or vocabularies." It shows an off-the-shelf
  Qwen3.5-4B *is already* an effective decision model zero-shot — matching
  community models built on the same backbone, strong probabilistic calibration,
  multimodal support, arbitrary option counts. The model stays a fully functional
  causal LLM.
- **"General Decision Models"** (arXiv 2610.03935): "First-Token Decision
  Readout: the next-token distribution at the answer position is restricted to the
  labels of the valid options. One forward pass gives the selected option for
  choice, the probability of true for noul, and the expected level for score."
  It argues an open-weight LLM *becomes* a general decision model through this
  readout, with the pretrained head intact.

A third paper, **"What do your logits know?"** (arXiv 2604.09885), supports the
premise at the mechanism level: the final logits contain primarily
*decision-relevant* information, while the residual stream leaks
decision-irrelevant detail. So reading the **logits** (not the hidden state) for a
decision is the right level of representation. Strata's choice to capture `mixed`
(the pre-projection hidden state) only as a *fallback* for a future head, and to
read logits for decisions, is consistent with this.

**Conclusion on validity:** Strata is not improvising. It is doing exactly what the
literature calls a "frozen readout", and its measured Capability (~79 on the
235-item OpenDecisions sample) sits next to the paper's zero-shot Gemma 4 31B
(79.6) and Jev (80.1). That proximity is the method working, not luck.

## 2. How Strata differs from Jev-like (trained) decision models

| | Strata (frozen readout) | "True" decision models (Jev, OpenJev, SII-Decision, OpenDecider) |
|---|---|---|
| Weights | Frozen base LLM | Fine-tuned (LoRA) on decision data, sometimes + decision head |
| Training | None | Supervised on Jev outputs / human labels / self-distillation |
| Readout | Option tokens directly, trie walk for collisions | Often letter readout (A/B/C) or pointer head; or option tokens |
| Calibration | Raw softmax (open vs closed readings) | Per-type fitted temperature scaling |
| Coherence | Not trained; weak on additivity/order | Trained; still weak on additivity (~7% pass) |
| Reasoning | Thinking forced off | Some distill reasoning into the readout (SII-Decision) |

The single most important published finding for Strata (from "Releasing the Inner
Jev", Geuter, Cheng, Alvarez-Melis, arXiv 2610.17205): **"training the decision
head alone provides little to no benefit"** — the improvement comes from *adapting
the language model* (LoRA), not from adding a classifier head.

This matters twice:

1. It **validates parking the policy-head idea** (companion doc §2.5, §2.6, §5).
   A linear head over `mixed` would not have closed the gap with Jev. Dropping it
   was correct.
2. It tells us where the real gap-closing work is if we ever want it: **LoRA on
   the backbone**, or **self-distillation** (SII-Decision's
   "Reasoning-to-Readout Self-Distillation"). Both require training; neither
   requires a second model or a second copy of the weights.

## 3. What is missing, ranked

### A. Temperature / calibration fitting per type — cheap, high value, not done

Fit one scalar temperature per question type (choice / noul / score) on a held-out
set, minimizing NLL or ECE. Apply `logits / T` at inference. No weights touched,
no training, no engine change — a post-processing constant in the probability
module.

Why it matters here: Strata's own numbers show the model is over-confident relative
to its own accuracy (real probabilities cost 2.8 Capability points vs. the
degenerate 1.0/0.0). The open-vs-closed reading analysis is essentially
discovering this post-hoc. A fitted temperature per type would directly shrink the
gap between "how sure the model says it is" and "how often it's right", which is
exactly what the Calibration component of Capability measures.

### B. Backbone adaptation (LoRA) — moderate cost, the real accuracy lever

Zero-shot readout ≈ 79.6 Capability; trained (LoRA) ≈ 84.4. That is +4.8 points,
the single biggest lever in the literature. Strata sits at ~79 (zero-shot). To
reach Jev-level (~80) Strata is already there; to *beat* Jev it needs this.

LoRA (Low-Rank Adaptation) freezes `W` and trains two small matrices `A` (r×d),
`B` (d×r) so the effective weight is `W + B·A`, rank `r` small (8–64). Adapter
file ~10–100 MB, ~1% or less of the base model. Base weights stay frozen and
resident; the adapter is a small delta applied on top.

**Coexistence with unmodified chat: yes, in principle — that is the point of LoRA.**
Two serving modes exist: load-time fusion (permanent, bad for us) and per-request
application (base weights untouched, adapter toggled per request — vLLM's
`LoRARequest`, MOLA's per-request dispatch, Ray Serve multi-LoRA). One copy of the
base model in memory, adapter applied on the decision path only.

Catches for Strata:
- Strata's C++/GGUF engine must support per-request LoRA application (engineering
  feature to build; the LOGITS path applies the adapter, the GEN path does not).
- MoE complicates it: Qwen3.8-Flash-Next is a mixture-of-experts model. LoRA on
  MoE needs MoE-aware adapters (attention + selected experts) and MoE-aware
  tooling (vLLM has `enable_mixed_moe_lora_format`). Training cost is higher than
  dense.
- Training infrastructure: no PyTorch in Strata's environment today. Needs user
  approval + install, plus a training pipeline.

### C. Reasoning-to-readout self-distillation — moderate cost, keeps single-pass speed

**Not** "turn on thinking at serve time" (that would be slow). At *training* time,
run the model with thinking enabled → it produces a better distribution at the
answer position → train the readout (thinking off) to reproduce that distribution.
At *serve* time it is still a single forward pass, thinking off, fast. The
reasoning is internalized into the weights, not performed at inference.

The SII-Decision page (sii-research.github.io/sii-decision): "Reasoning-to-Readout
Self-Distillation then teaches this readout to match the distribution the same
model reaches after reasoning, without gold labels or a stronger teacher." The
"General Decision Models" paper: "the model can build its intuition as humans do,
teaching its System One with its own System Two."

Do you pre-seed reasoning artificially? **Not required.** The standard form is
*self*-distillation: the model generates its own reasoning targets. You can also
distill from a stronger teacher, but SII's contribution is showing you don't need
one. The related technique (on-policy self-distillation, arXiv 2601.18734)
formalizes it: one LLM as both teacher and student, different contexts.

Practical shape for Strata: for each decision example, run Qwen with thinking on →
capture the answer-position distribution → train a LoRA adapter (thinking off) to
match it. No external labels needed. This is the most "Strata-shaped" improvement:
it preserves the single-pass serving contract while recovering accuracy that
thinking would give. Requires B's training infra, but not E's data.

### D. Coherence / consistency — hard, partly unsolved even for trained models

JevBench measures coherence (reordering options → probabilities follow; batch
independence; additivity; monotonicity; conditionality). First-token readouts are
known to be brittle: **additivity fails in ~93% of negation tests** on average
across models, and option ordering moves probabilities. Strata's doc notes
"order-robust 91.7%" and weak jaggedness/composite. Trained models improve
coherence but nobody has solved additivity. Strata's trie walk and `model_picked`
gate are engineering responses, not training. Real improvement needs training with
coherence losses or architectural changes (e.g., scoring options independently
rather than jointly — which Strata's merged-per-question path partially does, at a
cost).

### E. Real labels / human uncertainty — data, not code

Strata's synthetic rule labels measure agreement with a rule, not quality. The
jev-bench soft-label analysis (companion doc §14) shows the model is closer to
human uncertainty than a uniform guess but over-confident and agrees with human
consensus only 43.9% of the time. Trained decision models use Jev's answers or
human labels on real agent states. Strata has no such pipeline yet.

**What are we training?** A LoRA adapter (B). The target can come from three
sources: self-distillation (C, no data needed), teacher distillation (a stronger
model's or Jev's answers on your states), or human/agent labels (E, most aligned
to your workflow). B is *how* you adapt; E is *what* you adapt toward. They are
orthogonal. C is a way to get labels for B without E — which is why the handoff
says real trajectories are "last in priority order".

### F. Tokenization strategy — Strata is ahead here, with a cost

Most open readouts use **letter labels (A/B/C)** to dodge multi-token option names.
Strata reads option tokens directly and handles collisions with a trie walk
(1–7 extra cached requests per colliding item).

- Letter readout: simpler, more robust, single token, no collisions. But requires
  the model to map letters to options, and the prompt must include the mapping.
- Strata's direct tokens + trie walk: more faithful to the model's own vocabulary,
  no artificial mapping, but costlier and more fragile (window misses, 502s from
  unparseable text — which Strata sidesteps by reading logits, not text).

Strata's trie walk is genuinely novel (no equivalent found in the papers). It is
what lets Strata report probabilities for every item (235/235) with 99% on the
fast path. A letter readout might simplify engineering but would not clearly
improve Capability. Worth benchmarking both, not urgent.

## 4. What Strata is NOT missing

- **"Is reading logits 'cheating' vs. a real decision model?"** No. The entire
  2026 System One literature is built on this exact readout. Jev itself is widely
  believed (per the papers) to be a trained version of this same readout over a
  base LLM.
- **"Does it need a separate head?"** No — and the papers confirm the head alone
  buys almost nothing. Not building one was correct.
- **"Is the prefill bottleneck fatal?"** For a new state it dominates (~1.6 s),
  and no readout removes it. But Strata's Stage A showed a fixed 2,048+ token
  preamble cuts it to ~532 ms via checkpoint reuse. That is an orchestrator-level
  win, not an engine change, and it is the right layer for it. Trained models have
  the same prefill cost.
- **"Is 'calibrated: true' with real probabilities actually better?"** Mixed.
  Real probabilities *hurt* Capability by 2.8 points because of over-confidence —
  but they are more honest, rank better (AUC 0.827 vs 0.809), and have better
  ECE (0.043 vs 0.062). The "true" decision models solve this with temperature
  scaling, which Strata has not applied yet. So the gap is calibration fitting, not
  the readout itself.

## 5. The LoRA-on-MoE question

Not all-or-nothing. Options, simplest to most involved:

1. **Attention-only LoRA** (W_q, W_k, W_v, W_o; FFN/MoE experts frozen). Standard
   PEFT recipe, tooling handles MoE models out of the box, no router logic. Likely
   captures most of the benefit for a decision task, which is attention-heavy.
2. **Attention + shared expert.** One extra module, still no router logic. Better
   for multi-task adaptation.
3. **LoRA on selected experts** (ESFT, MoE-Sieve, CEFT, DR-LoRA): offline router
   stats → identify busy experts → LoRA on them. Higher potential, but the MoE²-LoRA
   paper (2607.21978) flags the downside: experts you skip but that still fire at
   inference receive no update. Requires custom training code.
4. **MoE-style LoRA** (MoELoRA, MoE²-LoRA): add a new set of LoRA experts with a
   learned gating network. Highest potential, highest complexity, designed for
   multi-task. Probably overkill for Strata's single task.

Recommendation: start with (1), escalate only if measured. Decision readouts depend
more on attention than on FFN knowledge, so (1) likely captures most of the lift.
The "Releasing the Inner Jev" +4.8-point result was achieved with standard LoRA.

## 6. Verdict

- **Validity: sound.** Frozen first-token readout is a first-class member of the
  System One family, with measured results competitive with zero-shot published
  baselines. The logits are a legitimate decision signal.
- **Reliability: moderate**, with known soft spots: over-confidence (fixable with
  temperature fitting, item A), coherence failures under rewording/reordering (partly
  fixable with training, not fully solved anywhere), tokenization edge cases
  (handled but costly), and the concurrency/engine-state bugs already found and
  fixed (companion doc §12–§13).
- **Usefulness: high** for the stated goal. Single forward pass, reuses the loaded
  backbone, no second model, no weights changed, opt-in. It will not match a
  trained decision model on accuracy/calibration until item A (and ideally B/C) is
  added.

## 7. Sources

- Geuter, Cheng, Alvarez-Melis, "Releasing the Inner Jev: A Simple Recipe for
  Open Decision Models", arXiv 2610.17205 (2026). Zero-shot readout + LoRA;
  OpenDecisions benchmark; "training the decision head alone provides little to no
  benefit".
- "LLM-as-Jev: LLMs Are Already Jev-Style Decision Models", arXiv 2610.02076.
  Frozen readouts from next-token probs; matches community models.
- "General Decision Models: Benchmarking and Insights Beyond Jev",
  arXiv 2610.03935. First-token readout; reasoning-to-readout self-distillation;
  SII-Decision.
- "What do your logits know?", arXiv 2604.09885. Logits contain
  decision-relevant information.
- "JevAgentHarness", arXiv 2610.21412. Agent use of decision models; training on
  Jev's answers on agent-visited states.
- JevBench (jevbench.xyz, github.com/JevBench/jevbench): coherence benchmark;
  additivity weakness (~7% pass).
- "MoE²-LoRA", arXiv 2607.21978: critique of subset-expert selection.
- OpenDecisions (jgeuter/OpenDecisions v1.0.1) and jev-bench (Praveenrajus/jev-bench
  v0.1.1): the validation sets used in the companion doc.
- vLLM multi-LoRA docs, MOLA (MLX), Ray Serve multi-LoRA: per-request adapter
  application without reloading base weights.