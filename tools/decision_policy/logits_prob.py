"""tools/decision_policy/logits_prob.py - probabilities over answer options from next-token logits.

The server reads the model's next-token logits after a prefill and computes real
probabilities over answer options, instead of generating text and reporting
degenerate 1.0/0.0. Pure Python, stdlib only (math). No numpy, no torch.

An answer option is a named group of token sequences (surface forms). A sequence
may be a single token (one token id) or a multi-token sequence (multiple token ids):

- Single-token:  P(option) = softmax(logits)[token_id].  One logits vector.
- Multi-token:   P(option) = P(t1) * P(t2|t1) * ...  via the chain rule.  One logits
  vector per position, each conditioned on the previous tokens.

An option's probability is the sum over its surface forms (logsumexp in log space).
The distribution over options is normalized via a final logsumexp.  All computation
is in log-space with max-subtraction, so large logits (e.g. 1000.0) do not overflow.

This mirrors three question types - noul (yes/no groups), choice (N option groups),
score (N level groups) - generalized here as "named groups of token sequences".

Design note: `logits_per_position` is shared across every sequence of a call.  For
single-token options (the common case) every sequence reads position 0.  For
multi-token options the caller must supply enough positions for the longest
sequence, each conditioned on that sequence's prefix; a shorter sequence consumes
only the positions it needs (position i is always conditioned on the question for
i=0, and on the sequence's own tokens for i>0).
"""
from __future__ import annotations

import math

__all__ = [
    "softmax",
    "logsumexp",
    "sequence_logprob",
    "answer_logprob",
    "distribution",
    "decide",
    "build_response",
]


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a logits vector."""
    if not logits:
        raise ValueError("softmax: logits must not be empty")
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def logsumexp(log_probs: list[float]) -> float:
    """Stable log-sum-exp."""
    if not log_probs:
        raise ValueError("logsumexp: log_probs must not be empty")
    m = max(log_probs)
    return m + math.log(sum(math.exp(x - m) for x in log_probs))


def _log_softmax(logits: list[float]) -> list[float]:
    """Log of softmax, stably: x_i - logsumexp(x)."""
    if not logits:
        raise ValueError("_log_softmax: logits must not be empty")
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


def sequence_logprob(token_ids: list[int],
                     logits_per_position: list[list[float]]) -> float:
    """log P(token_1, ..., token_n) via the chain rule.

    `logits_per_position[i]` is the logits vector at position i, conditioned on
    tokens 0..i-1.  For a single-token sequence, `logits_per_position` has one
    element.  An empty sequence has log-probability 0.0 (probability 1).
    """
    if len(logits_per_position) < len(token_ids):
        raise ValueError(
            f"sequence_logprob: {len(token_ids)} token(s) but only "
            f"{len(logits_per_position)} logits vector(s)")
    total = 0.0
    for i, tid in enumerate(token_ids):
        lp = _log_softmax(logits_per_position[i])
        if not 0 <= tid < len(lp):
            raise ValueError(
                f"sequence_logprob: token id {tid} out of range "
                f"[0, {len(lp)}) at position {i}")
        total += lp[tid]
    return total


def answer_logprob(sequences: list[list[int]],
                   logits_per_position: list[list[float]]) -> float:
    """log P(answer) = logsumexp of the sequence log-probabilities.

    An answer may have multiple surface forms (e.g. "yes" / "YES"); each is a
    token sequence, and the answer's probability is the sum over its forms.
    """
    if not sequences:
        raise ValueError("answer_logprob: sequences must not be empty")
    return logsumexp([sequence_logprob(s, logits_per_position) for s in sequences])


def distribution(answers: dict[str, list[list[int]]],
                  logits_per_position: list[list[float]]) -> dict[str, float]:
    """Normalized P(answer) for each answer name; the values sum to 1.0."""
    if not answers:
        raise ValueError("distribution: answers must not be empty")
    logps = {name: answer_logprob(seqs, logits_per_position)
             for name, seqs in answers.items()}
    total = logsumexp(list(logps.values()))
    return {name: math.exp(lp - total) for name, lp in logps.items()}


def decide(answers: dict[str, list[list[int]]],
           logits_per_position: list[list[float]]) -> tuple[str, float, dict[str, float]]:
    """-> (argmax answer, its probability, full distribution).

    Ties break toward the first answer in iteration (insertion) order.
    """
    dist = distribution(answers, logits_per_position)
    pick = None
    best = -1.0
    for name, p in dist.items():
        if p > best:
            best = p
            pick = name
    return pick, dist[pick], dist


def build_response(qtype: str, answers: dict[str, list[list[int]]],
                   distribution: dict[str, float]) -> dict:
    """The reference response shape (serve/systemone.py build_response) with real
    probabilities instead of degenerate 1.0/0.0.

    - noul:   {"type": "noul", "noul": P(yes)}
    - choice: {"type": "choice", "choice": pick, "probabilities": {...},
               "confidence": p}
    - score:  {"type": "score", "score": idx, "legend": {...},
               "probabilities": {str(i): ...}, "confidence": p}

    `answers` supplies the ordering (insertion order) that score's index and
    legend are built from.  The caller sets the body-level "calibrated": True.
    """
    if qtype not in ("noul", "choice", "score"):
        raise ValueError(f"build_response: unknown qtype {qtype!r}")
    if qtype == "noul":
        return {"type": "noul", "noul": float(distribution.get("yes", 0.0))}
    pick = None
    best = -1.0
    for name, p in distribution.items():
        if p > best:
            best = p
            pick = name
    confidence = float(best)
    if qtype == "choice":
        return {"type": "choice", "choice": pick,
                "probabilities": {k: float(v) for k, v in distribution.items()},
                "confidence": confidence}
    # score
    names = list(answers)
    idx = names.index(pick) if pick in names else 0
    legend = {str(i): name for i, name in enumerate(names)}
    probs = {str(i): float(distribution.get(name, 0.0))
             for i, name in enumerate(names)}
    return {"type": "score", "score": float(idx), "legend": legend,
            "probabilities": probs, "confidence": confidence}
