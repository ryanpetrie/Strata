"""serve/logits_first_token.py - first-token probability lookup from next-token logits.

For a question whose answer options have unique first tokens, the model's next-token
logits (after the prompt) give real probabilities over the options: softmax over the
option first-token logits, normalized. This avoids the degenerate 1.0/0.0 of a greedy
decode. Falls back to the decode path when any option's first token is outside the
engine's top-k, or when first tokens collide.

Pure Python: no Service, no engine, no I/O. server.py's Service.systemone wires it to
the model; serve/test_systemone.py covers it without a GPU.
"""
from __future__ import annotations

import math

from tools.decision_policy import logits_prob


def first_token_ids(options: list[str], tok) -> dict[str, int]:
    """-> {option_name: first_token_id} for each option, tokenized in the answer context.

    The endpoint's answer format is `name: answer`, so the answer follows a leading
    space. Tokenize " " + option_name and take the first token ID.
    """
    out = {}
    for opt in options:
        ids = tok.encode(" " + opt, parse_special=True)
        if not ids:
            raise ValueError(f"first_token_ids: option {opt!r} tokenizes to nothing")
        out[opt] = ids[0]
    return out


def option_names_for_question(q: dict) -> list[str]:
    """The answerable option names for a normalized question definition."""
    qtype, crit = q["type"], q["criteria"]
    if qtype == "noul":
        return ["yes", "no"]
    if qtype == "choice":
        return list(crit.keys())
    # score
    return crit["levels"]


def try_first_token_probs(questions: list[tuple[str, dict]], logits: dict[int, float],
                          tok) -> dict[str, dict] | None:
    """-> {question_name: response_dict} from first-token probabilities, or None to fall back.

    `logits` is {token_id: logit_value} from the engine's top-k. Returns None when any
    option's first token is outside the top-k, or when first tokens collide (two options
    sharing a first token cannot be distinguished by first-token logits alone).
    """
    # Build the answers dict for each question and check for collisions / missing tokens
    per_question = []
    for name, q in questions:
        options = option_names_for_question(q)
        ftids = first_token_ids(options, tok)
        # Collision check: two options with the same first token
        if len(set(ftids.values())) != len(ftids):
            return None
        # Missing from top-k check
        for tid in ftids.values():
            if tid not in logits:
                return None
        per_question.append((name, q, options, ftids))

    # All checks passed: compute probabilities
    out = {}
    for name, q, options, ftids in per_question:
        qtype = q["type"]
        # Build a reduced logits vector with remapped token IDs for logits_prob.distribution
        # The distribution function normalizes via logsumexp, so passing only the option
        # first-token logits gives the softmax over the options.
        opt_logits = [logits[ftids[opt]] for opt in options]
        id_to_idx = {ftids[opt]: i for i, opt in enumerate(options)}
        answers = {opt: [[id_to_idx[ftids[opt]]]] for opt in options}
        dist = logits_prob.distribution(answers, [opt_logits])
        resp = logits_prob.build_response(qtype, answers, dist)
        out[name] = resp
    return out
