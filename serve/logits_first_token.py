"""serve/logits_first_token.py - first-token probability lookup from next-token logits.

For a question whose answer options have unique first tokens, the model's next-token
logits (at the answer position) give real probabilities over the options: a softmax
over the option first-token logits, normalized. This avoids the degenerate 1.0/0.0 of
a greedy decode. Falls back to the decode path when any option's first token is
outside the engine's top-k, or when first tokens collide (two options sharing a first
token cannot be separated by first-token logits alone).

THE ANSWER POSITION. The request's assistant turn is prefilled with the answer line's
prefix ("name: "), so the next token the model writes IS the answer's first token.
Without that prefix the model starts its turn by writing the question's NAME (the
answer format is `name: answer`), and the next-token logits describe the name, not the
answer - measured: the name was the top token in every probe.

Pure Python: no Service, no engine, no I/O. server.py's Service.systemone wires it to
the model; serve/test_logits_first_token.py covers it without a GPU.
"""
from __future__ import annotations

import json
import math

from tools.decision_policy import logits_prob

# How many logits the engine returns for the answer position.  It has to be generous:
# an option the model is very unlikely to pick can sit far down the vocabulary's tail
# (a confident "no" puts " yes" well below the first few hundred), and an option whose
# first token is outside the window cannot be scored at all.  1024 is the engine's
# maximum and costs one short pipe write per request.
LOGITS_TOP_K = 1024


def answer_text(prefix: str, opt: str) -> str:
    """The text the model writes for this option.

    The prefix is what the assistant turn was prefilled with.  It ends at the answer
    line's colon ("name:"), and the answer follows it with a space - the space is part
    of the first token the model emits (the merged " no"), not a token of its own.
    That matters: a prompt ending in a lone space token after the colon is
    OFF-DISTRIBUTION (in training "name: no" tokenizes as ['name', ':', ' no']), and
    the model reacts to it by wanting to end the turn instead of answering - measured:
    with the trailing space, <|im_end|> was the top token at the answer position and
    the answer tokens came second.
    """
    if not prefix:
        return opt                     # at the start of a turn the model writes the bare word
    return prefix + opt if prefix.endswith(" ") else prefix + " " + opt


def first_token_ids(options: list[str], tok, prefix: str = "") -> dict[str, int]:
    """-> {option_name: first_token_id}: the token the answer starts with.

    The token is found by tokenizing the whole text the model would write and taking
    the first token that is not part of the prefix: that is the tokenization the model
    sees, so it is the answer's first token IN CONTEXT (the merged " word" when the
    answer follows a space, the bare word when it does not - the two are different
    tokens, and scoring the wrong one scores the wrong thing).
    """
    pre = list(tok.encode(prefix, parse_special=True)) if prefix else []
    out = {}
    for opt in options:
        ids = list(tok.encode(answer_text(prefix, opt), parse_special=True))
        n = 0
        while n < len(pre) and n < len(ids) and pre[n] == ids[n]:
            n += 1
        if n >= len(ids):
            raise ValueError(f"first_token_ids: option {opt!r} adds no token after the prefix")
        out[opt] = ids[n]
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


def end_of_turn_ids(tok) -> set[int]:
    """The ids of the tokens that end a turn (the end-of-turn and end-of-text control
    tokens).  Their weight is the model's "rather not answer"."""
    from serve.systemone import end_of_turn_ids as _ids
    return _ids(tok)


def first_token_plan(questions: list[tuple[str, dict]], tok, prefix: str = ""):
    """-> [(name, question, options, {option: first_token_id})] per question, or None.

    None when two options of one question share their first token: the first-token
    logits cannot separate them, so the question needs the chain rule (or the decode
    path).  The plan is what the readings are computed from, and the reason a plan
    does not exist is what the diagnostics report.
    """
    per_question = []
    for name, q in questions:
        options = option_names_for_question(q)
        ftids = first_token_ids(options, tok, prefix)
        # Collision check: two options with the same first token
        if len(set(ftids.values())) != len(ftids):
            return None
        per_question.append((name, q, options, ftids))
    return per_question


def fallback_reason(questions: list[tuple[str, dict]], logits: dict[int, float], tok,
                    prefix: str = "", logz: float | None = None) -> str:
    """Why the first-token lookup cannot answer this question, in a few words.

    The reason is what tells the two ways of failing apart: "collision" means the
    options share a first token and the chain rule would separate them, while
    "outside the window" or "the model wanted something else" mean the question needs
    the decode path.  It is a diagnostic, so it never changes the answer.
    """
    plan = first_token_plan(questions, tok, prefix)
    if plan is None:
        return "collision: two options share their first token"
    if not logits:
        return "no logits from the engine"
    for name, q, options, ftids in plan:
        missing = [o for o in options if ftids[o] not in logits]
        if missing:
            return f"outside the window: {missing[0]!r} is not in the engine's top-k"
    if max(logits, key=lambda t: logits[t]) not in {t for _, _, _, f in plan for t in f.values()}:
        return "the model's top token is not one of the options"
    return "no reason: the readings should have been computed"


def first_token_probs(questions: list[tuple[str, dict]], logits: dict[int, float], tok,
                       prefix: str = "", logz: float | None = None) -> dict[str, dict] | None:
    """-> {question_name: reading} or None to fall back to the decode path.

    `logits` is {token_id: logit_value} from the engine's top-k; `logz` is the
    full-vocabulary log-sum-exp the engine reports (the normalizer that turns a logit
    into the model's own probability).  None when the engine did not report one.

    Each reading holds:
      closed  - the distribution over the options alone: the FORCED-CHOICE reading.  The
                weight the model puts on anything else (ending the turn, a digit) is not
                counted, so the options add up to 1.
      open    - the model's OWN probability of each option, P = exp(logit - logz): the
                options add up to less than 1 and the rest is what it wanted to say
                instead.
      stop    - the probability of ending the turn rather than answering.
      outside - everything that is not one of the options.

    The two readings differ by exactly the mass outside the options.  The ANSWER is the
    same under both (a renormalization does not reorder them); only the confidence
    differs, so which one is right is a calibration question, measured rather than
    decided.  For a multi-token option the open probability is the FIRST token's, an
    upper bound on the whole sequence.

    Returns None when any option's first token is outside the top-k, or when first
    tokens collide (two options sharing a first token cannot be separated by first-token
    logits alone).
    """
    per_question = first_token_plan(questions, tok, prefix)
    if per_question is None:
        return None
    for name, q, options, ftids in per_question:
        # Missing from top-k check
        for tid in ftids.values():
            if tid not in logits:
                return None
        # THE MODEL'S OWN CHOICE.  The first-token probabilities describe the answer only
        # if the model is choosing among the options: if its top token is something else
        # (a digit it wants to write, the end of the turn), the option distribution is
        # about a choice it is not making, and the decode path answers it properly.
        # This is also what keeps a question that asks for a probability (the model
        # writes "0.7") on the decode path, where the number can be read.
        if max(logits, key=lambda t: logits[t]) not in set(ftids.values()):
            return None

    if logz is None:
        # the engine reported no normalizer: normalize over the tokens it did send
        top = max(logits.values())
        logz = top + math.log(sum(math.exp(v - top) for v in logits.values()))
    stop_ids = end_of_turn_ids(tok)
    out = {}
    for name, q, options, ftids in per_question:
        vals = [logits[ftids[opt]] for opt in options]
        # closed: the softmax over the options alone
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps)
        closed = {opt: exps[i] / s for i, opt in enumerate(options)}
        # open: the model's own probability of each option
        openp = {opt: math.exp(logits[ftids[opt]] - logz) for opt in options}
        stop = sum(math.exp(logits[t] - logz) for t in stop_ids if t in logits)
        outside = max(0.0, 1.0 - sum(openp.values()))
        out[name] = {"type": q["type"], "options": options, "closed": closed, "open": openp,
                     "stop": stop, "outside": outside}
    return out


def reading_response(reading: dict) -> dict:
    """The reference response shape for one question, from the CLOSED reading."""
    answers = {opt: [[i]] for i, opt in enumerate(reading["options"])}
    dist = {opt: reading["closed"][opt] for opt in reading["options"]}
    return logits_prob.build_response(reading["type"], answers, dist)


def reading_diagnostics(reading: dict) -> dict:
    """The alternative reading, for the comparison: the model's own probability of each
    option, the probability of ending the turn, and everything that is not an option."""
    pick = max(reading["closed"], key=lambda o: reading["closed"][o])
    return {"closed_confidence": reading["closed"][pick], "open_confidence": reading["open"][pick],
            "open": reading["open"], "stop": reading["stop"], "outside": reading["outside"]}


def debug_prompt(ids: list[int], tok, path: str, n: int = 48) -> None:
    """Append the tail of the rendered prompt to `path` (decoded).  The answer position
    is the last token of this prompt, so the tail shows exactly what the model was
    asked to continue - in particular whether the assistant turn really was prefilled
    with the answer line's prefix."""
    tail = list(ids)[-n:]
    try:
        text = tok.decode(tail)
    except Exception:
        text = "?"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"prompt tail ({len(tail)} of {len(ids)} tokens): {text!r}\n")


def debug_report(questions: list[tuple[str, dict]], logits: dict[int, float], tok, path: str,
                  prefix: str = "", logz: float | None = None, top_n: int = 12) -> None:
    """Append a diagnosis of the answer position to `path`: every option's first-token
    id with whether it fell inside the engine's top-k, both readings of the
    probabilities, and the top-n logits with their decoded text.

    This is what settles the bare-vs-merged tokenization question (the top-1 token's
    text says which form the model actually emits) and the noise-vs-uncertainty one
    (how much weight the model put on ending the turn instead of answering).
    """
    lines = []
    readings = first_token_probs(questions, logits, tok, prefix, logz)
    for name, q in questions:
        options = option_names_for_question(q)
        ftids = first_token_ids(options, tok, prefix)
        lines.append(f"question {name} ({q['type']}): " + ", ".join(
            f"{opt!r}=id{ftids[opt]}{'' if ftids[opt] in logits else ' MISSING'}"
            for opt in options))
        if readings and name in readings:
            r = readings[name]
            lines.append(f"  closed {json.dumps(r['closed'])}")
            lines.append(f"  open   {json.dumps(r['open'])}")
            lines.append(f"  stop={r['stop']:.4f} outside={r['outside']:.4f}")
    items = sorted(logits.items(), key=lambda kv: -kv[1])[:top_n]
    decoded = []
    for tid, val in items:
        try:
            text = tok.decode([tid])
        except Exception:
            text = "?"
        decoded.append(f"{tid}:{val:.4f}:{text!r}")
    lines.append(f"top {len(items)}: " + " | ".join(decoded))
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n---\n")
