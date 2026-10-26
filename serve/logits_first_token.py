"""serve/logits_first_token.py - first-token probability lookup from next-token logits.

For a question whose answer options have unique first tokens, the model's next-token
logits (at the answer position) give real probabilities over the options: a softmax
over the option first-token logits, normalized. This avoids the degenerate 1.0/0.0 of
a greedy decode. Falls back to the decode path when any option's first token is
outside the engine's top-k.

THE TRIE. When two options share a first token ("rule_1" and "rule_2" both start with
" rule") the first-token row cannot separate them, but the options' token sequences
form a TRIE: the row at a node of the trie tells the options that pass through it
apart. The walk stops where the options are told apart, so it is short - measured on
the 235-item OpenDecisions sample: one extra engine request for 50 of the 72
colliding items, seven for the worst.

THE ANSWER POSITION. The request's assistant turn is prefilled with the answer line's
prefix ("name: "), so the next token the model writes IS the answer's first token.
Without that prefix the model starts its turn by writing the question's NAME (the
answer format is `name: answer`), and the next-token logits describe the name, not the
answer - measured: the name was the top token in every probe.

THE TEMPERATURE.  A temperature divides the logits before the softmax, so the reported
distribution over the options is flattened (T > 1) or sharpened (T < 1).  It is
opt-in: the config's "systemone": {"temperature": {...}}, and 1.0 - the default -
leaves every number byte-identical to the path before it.  The PICK cannot change: the
map is monotone, so the answer is the same under every temperature.

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

# The logit of a token the engine did not send: the window holds the highest logits,
# so a token outside it is below every token inside it.  Its probability is therefore
# below the window's tail, which is what OUT_OF_WINDOW stands for - a token the model
# has all but ruled out, not an unknown.
OUT_OF_WINDOW = float("-inf")


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


def option_token_ids(options: list[str], tok, prefix: str = "") -> dict[str, list[int]]:
    """-> {option_name: the token ids the model writes for it}, the prefix's own ids
    dropped.

    The ids are the ones the model would actually emit: the option's text is
    tokenized in the context of the whole line (the prefix first), and the prefix's
    tokens are cut off at the first one that differs.  `first_token_ids` is this,
    keeping only the first id.
    """
    pre = list(tok.encode(prefix, parse_special=True)) if prefix else []
    out = {}
    for opt in options:
        ids = list(tok.encode(answer_text(prefix, opt), parse_special=True))
        n = 0
        while n < len(pre) and n < len(ids) and pre[n] == ids[n]:
            n += 1
        if n >= len(ids):
            raise ValueError(f"option_token_ids: option {opt!r} adds no token after the prefix")
        out[opt] = ids[n:]
    return out


def first_token_ids(options: list[str], tok, prefix: str = "") -> dict[str, int]:
    """-> {option_name: first_token_id}: the token the answer starts with.

    The token is found by tokenizing the whole text the model would write and taking
    the first token that is not part of the prefix: that is the tokenization the model
    sees, so it is the answer's first token IN CONTEXT (the merged " word" when the
    answer follows a space, the bare word when it does not - the two are different
    tokens, and scoring the wrong one scores the wrong thing).
    """
    try:
        seqs = option_token_ids(options, tok, prefix)
    except ValueError as e:
        raise ValueError(str(e).replace("option_token_ids:", "first_token_ids:")) from None
    return {opt: ids[0] for opt, ids in seqs.items()}


def option_names_for_question(q: dict) -> list[str]:
    """The answerable option names for a normalized question definition."""
    qtype, crit = q["type"], q["criteria"]
    if qtype == "noul":
        return ["yes", "no"]
    if qtype == "choice":
        return list(crit.keys())
    # score
    return crit["levels"]


def trie_plan(options: list[str], tok, prefix: str = "") -> dict | None:
    """-> the plan for a question whose options do not have unique first tokens, or None.

THE TRIE.  The options' token sequences form a trie: the root is the answer position
and every option is a path from the root.  Two options that share a first token share
a node, and the row at that node tells them apart.  The plan holds:

  tokens      {option: the option's token ids after the prefix}
  resolution  {option: the depth at which the option is told apart from the others}
  nodes       {node: the options told apart at that node}   a node is a tuple of ids
  outcomes    {node: the token ids the options at that node claim}
  walk        the nodes (other than the root) whose row the walk has to fetch
  stop_ids    the ids that end a turn

THE RESOLUTION.  An option is told apart at the shallowest node where it is alone:
either its next token is one no other option through that node has, or its text ENDS
there (its last token, and another option continues through that node).  The first
case needs nothing but the row; the second needs the probability of the turn ENDING
there, because a model that stops after the shorter option has answered with it.

THE ROOT IS CHECKED TOO.  The root is the answer position, and the options claim
their first tokens there, so the row at the answer position is held to the same rule
the first-token path holds it to: the model's top token there must be one the options
claim, or the model was going to say something else and the reading is about a choice
it is not making.

None when the options' tokenization does not continue the prefix's: the model's
context and the option's own text would then be different sequences, and the whole
reading would be about a different prompt.
    """
    try:
        seqs = option_token_ids(options, tok, prefix)
    except ValueError:
        return None
    if not seqs:
        return None
    pre = list(tok.encode(prefix, parse_special=True)) if prefix else []
    for opt in options:
        ids = list(tok.encode(answer_text(prefix, opt), parse_special=True))
        if ids[:len(pre)] != pre:
            return None

    # THE RESOLUTION: the shallowest node at which the option is alone.
    resolution: dict[str, int] = {}
    for opt in options:
        ids = seqs[opt]
        depth = len(ids)
        for i in range(len(ids)):
            through = [o for o in options
                       if len(seqs[o]) > i and seqs[o][:i] == ids[:i]]
            if not any(seqs[o][i] == ids[i] for o in through if o != opt):
                depth = i
                break
        resolution[opt] = depth

    # THE NODES: the options told apart at each node.
    nodes: dict[tuple, list[str]] = {}
    for opt in options:
        nodes.setdefault(tuple(seqs[opt][:resolution[opt]]), []).append(opt)

    # THE OUTCOMES: what the options at a node claim.  An option that continues
    # through the node claims the token it continues with; an option whose text ends
    # there claims the end of the turn.
    #
    # THE ROOT IS ALWAYS ONE OF THEM.  The root is the answer position, and the
    # options claim their first tokens there, so the row at the answer position is
    # checked the way the first-token path checks it: if the model's top token there
    # is not one the options claim, the model was going to say something else, and
    # the reading (which is about a context the model is not in) must not be used.
    stop_ids = end_of_turn_ids(tok)
    outcomes: dict[tuple, set] = {}
    for node in set(nodes) | {()}:
        claimed: set = set()
        ends = False
        for opt in options:
            ids = seqs[opt]
            if len(ids) < len(node) or ids[:len(node)] != list(node):
                continue
            if len(ids) > len(node):
                claimed.add(ids[len(node)])        # the option continues here
            else:
                ends = True                         # the option's text ends here
        if ends:
            claimed |= stop_ids
        outcomes[node] = claimed

    # THE WALK: the nodes whose row is needed, the shallowest first (the prompt is in
    # the conversation cache, so a deeper node is cheaper once its parent was read).
    # Every node on the way to a resolution is needed: the weight of an option is the
    # product of a factor per node (the token that comes next there), and the outcome
    # at the resolution node is read from that node's row.
    walk_set: set[tuple] = set()
    for opt in options:
        ids = seqs[opt]
        for i in range(1, resolution[opt] + 1):
            walk_set.add(tuple(ids[:i]))
    walk = sorted(walk_set, key=lambda n: (len(n), n))
    return {"tokens": seqs, "resolution": resolution, "nodes": nodes,
            "outcomes": outcomes, "walk": walk, "stop_ids": stop_ids}


def trie_probs(plan: dict, rows: dict[tuple, tuple[dict[int, float], float]]) -> dict | None:
    """-> {option: the weight of this option}, or None when it cannot be computed.

`rows[node]` is the (top-k, normalizer) at that node of the trie.  A node's row is
the distribution over the token that comes next, so the weight of an option is the
product of its token probabilities up to the node where it is told apart, times the
outcome it claims there:

  the option continues   -> the probability of the token it continues with
  the option's text ends -> the probability that the turn ENDS there

The second factor is what makes options of different lengths comparable: without it
the shorter option ("frontier") would carry the whole weight of the shared prefix and
always beat the longer one ("frontier_reasoning").

A token the row does not carry is not a failure: the row holds the highest logits, so
a token outside it is below every token inside it, and the option's probability is
below the window's tail.  Such an option is scored as zero - it cannot be the pick -
and the question is still answered.
    """
    tokens, resolution = plan["tokens"], plan["resolution"]
    out: dict[str, float] = {}
    for opt, ids in tokens.items():
        r = resolution[opt]
        p = 1.0
        for i in range(r):
            topk, logz = rows[tuple(ids[:i])]
            tid = ids[i]
            if tid not in topk:
                p = 0.0                  # the option is below the window's tail
                break
            p *= math.exp(topk[tid] - logz)
        node = tuple(ids[:r])
        topk, logz = rows[node]
        if r < len(ids):
            tid = ids[r]
            if tid not in topk:
                p = 0.0                  # the option is below the window's tail
            else:
                p *= math.exp(topk[tid] - logz)
        else:
            # the option's text ends here: the model answered with it only if the turn
            # ends here.  A turn-end token outside the window has a probability below
            # the window's tail, so zero is the honest estimate, not a failure.
            s = 0.0
            for t in plan["stop_ids"]:
                if t in topk:
                    s += math.exp(topk[t] - logz)
            p *= s
        out[opt] = p
    return out


def trie_readings(q: dict, plan: dict, rows: dict[tuple, tuple[dict[int, float], float]],
                   temperature: float = 1.0) -> dict | None:
    """-> the reading of a trie question, or None when it cannot be computed.

    `temperature` scales the closed distribution (see `apply_temperature`): it is
    applied to the OPTION-LEVEL effective logits, not to the per-trie-row logits.

The reading has the same shape as the first-token one (closed, open, stop, outside),
so the two can be compared and the response is built the same way.

None when the model was going to say something else at a node where the options are
told apart (including the answer position itself), or when no option carries any
probability at all.
    """
    options = option_names_for_question(q)
    probs = trie_probs(plan, rows)
    if probs is None:
        return None
    # THE MODEL'S OWN CHOICE.  At every node where the options are told apart, the
    # row's top token should be one of the outcomes the options claim there - or the
    # model was going to say something else and the option distribution is about a
    # choice it is not making.  That is not a reason to refuse: the weights already
    # carry it (an option the model was not going to write has a small weight), and
    # the open reading is the confidence.  The answer position is the one that
    # matters for the caller's fallback order, so it is reported as a flag.
    model_picked = True
    for node, claimed in plan["outcomes"].items():
        topk, _ = rows[node]
        if claimed and max(topk, key=lambda t: topk[t]) not in claimed:
            if not node:
                model_picked = False
    total = sum(probs.values())
    if total <= 0.0:
        return None
    closed = {opt: probs[opt] / total for opt in options}
    pick = max(closed, key=closed.get)
    # the end-of-turn mass where the pick was told apart: the model's "rather not
    # answer" at the position the answer was decided
    node = tuple(plan["tokens"][pick][:plan["resolution"][pick]])
    topk, logz = rows[node]
    stop = sum(math.exp(topk[t] - logz) for t in plan["stop_ids"] if t in topk)
    reading = {"type": q["type"], "options": options, "closed": closed, "open": dict(probs),
               "stop": stop, "outside": max(0.0, 1.0 - total), "model_picked": model_picked}
    return apply_temperature(reading, temperature)


def end_of_turn_ids(tok) -> set[int]:
    """The ids of the tokens that end a turn (the end-of-turn and end-of-text control
    tokens).  Their weight is the model's "rather not answer"."""
    from serve.systemone import end_of_turn_ids as _ids
    return _ids(tok)


def apply_temperature(reading: dict, temperature: float = 1.0) -> dict:
    """-> the reading with the temperature applied to the closed distribution.

    THE CLOSED READING - THE PART THAT IS SCALED.  A temperature divides the logits
    before the softmax, so the distribution over the options is flattened (T > 1) or
    sharpened (T < 1) and renormalized:

        closed_T(i) = normalize(closed_1(i) ** (1/T))

    That is exact, and it is why a temperature fitted offline on a stored run log
    predicts what the endpoint reports.  For the TRIE WALK the temperature is applied
    HERE, to the option-level effective logits (the closed distribution), and NOT to
    the per-trie-row logits: an option's weight is a PRODUCT of per-node token
    probabilities, and the per-row normalizers differ per option (the options' paths
    have different numbers of nodes), so "divide each node's logit by T" is not the
    same thing.  For a single-row question the two forms are identical.

    THE OPEN READING - WHAT IT IS, PLAINLY.  The open reading is the model's OWN
    probability of each option, P = exp(logit - logz), with the full-vocabulary
    normalizer logz.  At temperature T it would be exp(logit/T) / Z(T), and Z(T) is
    NOT available: the server has the engine's top-k window and the T = 1 normalizer,
    not the whole 248,320-token vocabulary.  So the mass the model put on the options
    (the sum of the open reading, measured at T = 1) is kept, and the closed reading
    at T is what it is scaled by:

        open_T(i) = closed_T(i) * M,    M = the mass on the options at T = 1

    That is NOT the model's own probability at temperature T, and the difference is
    stated in the docs rather than hidden.  It keeps the three properties the endpoint
    promises: the confidence is in [0, 1], it never exceeds the reported
    distribution's maximum (M <= 1), and a larger T lowers it (the maximum of a
    softened distribution falls towards 1/k).

    The mass outside the options ("stop", "outside") is left at its T = 1 value: the
    temperature reshapes the distribution OVER the options, and the model's mass
    elsewhere is not rescaled.

    The PICK cannot change: the map is monotone, so the answer is the same under
    every temperature.
    """
    if temperature == 1.0:
        return reading
    closed = reading["closed"]
    inv = 1.0 / temperature
    weights = {k: (v ** inv if v > 0.0 else 0.0) for k, v in closed.items()}
    total = sum(weights.values())
    if total <= 0.0:
        return reading
    scaled = {k: v / total for k, v in weights.items()}
    mass = min(1.0, max(0.0, sum(reading["open"].values())))
    out = dict(reading)
    out["closed"] = scaled
    out["open"] = {k: scaled[k] * mass for k in scaled}
    out["temperature"] = temperature
    return out


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
                    prefix: str = "", logz: float | None = None,
                    trie_reason: str | None = None) -> str:
    """Why the probabilities could not be read, in a few words.

    The reason is what tells the two ways of failing apart: "collision" means the
    options share a first token and the trie walk answers it, while "outside the
    window" or "the model wanted something else" mean the question needs the decode
    path.  It is a diagnostic, so it never changes the answer.
    """
    plan = first_token_plan(questions, tok, prefix)
    if plan is None:
        # a collision: the trie walk is what answers it, so its reason is the one
        return trie_reason or "collision: two options share their first token"
    if not logits:
        return "no logits from the engine"
    for name, q, options, ftids in plan:
        missing = [o for o in options if ftids[o] not in logits]
        if missing and len(missing) == len(options):
            return f"outside the window: {missing[0]!r} is not in the engine's top-k"
    if max(logits, key=lambda t: logits[t]) not in {t for _, _, _, f in plan for t in f.values()}:
        return "the model's top token is not one of the options"
    return "no reason: the readings should have been computed"


def trie_failure_reason(plan: dict, rows: dict[tuple, tuple[dict[int, float], float]]) -> str:
    """Why the trie walk could not answer, in a few words.

    It is a diagnostic: the answer is never a guess.  The three reasons are the ones
    that mean different things for a run - the model wanted something else (the
    option distribution is about a choice it is not making), an option's token was
    outside the engine's window, or the options carry no probability at all.
    """
    for node, claimed in plan["outcomes"].items():
        topk, _ = rows[node]
        if claimed and max(topk, key=lambda t: topk[t]) not in claimed:
            return "the model wanted something else"
    for opt, ids in plan["tokens"].items():
        r = plan["resolution"][opt]
        for i in range(r):
            topk, _ = rows[tuple(ids[:i])]
            if ids[i] not in topk:
                return f"outside the window: {opt!r} is not in the engine's top-k"
        topk, _ = rows[tuple(ids[:r])]
        if r < len(ids) and ids[r] not in topk:
            return f"outside the window: {opt!r} is not in the engine's top-k"
    return "the options carry no probability"


def first_token_probs(questions: list[tuple[str, dict]], logits: dict[int, float], tok,
                       prefix: str = "", logz: float | None = None,
                       temperature: float = 1.0) -> dict[str, dict] | None:
    """-> {question_name: reading} or None to fall back to the decode path.

    `logits` is {token_id: logit_value} from the engine's top-k; `logz` is the
    full-vocabulary log-sum-exp the engine reports (the normalizer that turns a logit
    into the model's own probability).  None when the engine did not report one.

    `temperature` scales the closed distribution (see `apply_temperature`).

    Each reading holds:
      closed  - the distribution over the options alone: the FORCED-CHOICE reading.  The
                weight the model puts on anything else (ending the turn, a digit) is not
                counted, so the options add up to 1.
      open    - the model's OWN probability of each option, P = exp(logit - logz): the
                options add up to less than 1 and the rest is what it wanted to say
                instead.
      stop    - the probability of ending the turn rather than answering.
      outside - everything that is not one of the options.
      model_picked - the model's top token at the answer position is one of the
                options' first tokens, i.e. the model was choosing among them.  False
                means it was going to write something else (a digit, a phrase), and
                the caller's answer then comes from the model's own text if that text
                is a valid answer, and from these readings if it is not.

    The two readings differ by exactly the mass outside the options.  The ANSWER is the
    same under both (a renormalization does not reorder them); only the confidence
    differs, so which one is right is a calibration question, measured rather than
    decided.  For a multi-token option the open probability is the FIRST token's, an
    upper bound on the whole sequence.

AN OPTION OUTSIDE THE WINDOW.  The engine sends the top k logits, not the whole
    vocabulary, so an option's first token can be outside it.  The window holds the
    highest logits, so such an option is below every option inside it: it cannot be
    the pick, and its probability is below the window's tail.  It is therefore scored
    as zero - the answer cannot change - instead of giving up on the question.

    Returns None when first tokens collide (two options sharing a first token cannot
    be separated by first-token logits alone) or when no option is in the window at
    all.  The model wanting to say something else is NOT one of those: the readings
    are computed and flagged, and the doubt they carry is the confidence.
    """
    per_question = first_token_plan(questions, tok, prefix)
    if per_question is None:
        return None
    top = max(logits, key=lambda t: logits[t]) if logits else None
    for name, q, options, ftids in per_question:
        # An option outside the window is scored as zero, not as a failure: it is
        # below every option inside the window, so it cannot be the pick.
        if not any(ftids[opt] in logits for opt in options):
            return None               # nothing to score: the decode path

    if logz is None:
        # the engine reported no normalizer: normalize over the tokens it did send
        top_logit = max(logits.values())
        logz = top_logit + math.log(sum(math.exp(v - top_logit) for v in logits.values()))
    stop_ids = end_of_turn_ids(tok)
    out = {}
    for name, q, options, ftids in per_question:
        vals = [logits.get(ftids[opt], OUT_OF_WINDOW) for opt in options]
        # closed: the softmax over the options alone
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps)
        closed = {opt: exps[i] / s for i, opt in enumerate(options)}
        # open: the model's own probability of each option
        openp = {opt: math.exp(logits.get(ftids[opt], OUT_OF_WINDOW) - logz) for opt in options}
        stop = sum(math.exp(logits[t] - logz) for t in stop_ids if t in logits)
        outside = max(0.0, 1.0 - sum(openp.values()))
        out[name] = apply_temperature({"type": q["type"], "options": options, "closed": closed,
                                       "open": openp, "stop": stop, "outside": outside,
                                       "model_picked": top in set(ftids.values())}, temperature)
    return out


def reading_response(reading: dict) -> dict:
    """The reference response shape for one question.

    The distribution is the CLOSED reading (the options renormalized over
    themselves): the shape's probabilities have to add up to 1, and that is what they
    mean.

    The confidence is the OPEN reading - the model's OWN probability of the pick.
    It is the better predictor of a correct answer, measured on the 235-item
    OpenDecisions sample: the AUC against correctness was 0.83 for the open reading
    and 0.75 for the closed one on the choice items (0.79 against 0.71 over every
    fast-path item).  The open probability is never larger than the closed one (it
    is the closed one times the mass the model put on the options at all), so the
    confidence never exceeds the distribution's maximum.

    A yes/no keeps the closed reading for its VALUE: the value is the probability of
    "yes" among the two answers, so the value and the answer always agree, and it
    measured better there (the expected calibration error 0.093 against 0.116).  It
    now carries a confidence too - the same open reading the other types report, the
    model's own probability of the side it picked.  So a confident "no" reports
    {"noul": 0.03, "confidence": 0.95}: the value is the probability of yes, the
    confidence is how sure the model is that the answer is "no"."""
    answers = {opt: [[i]] for i, opt in enumerate(reading["options"])}
    dist = {opt: reading["closed"][opt] for opt in reading["options"]}
    pick = max(reading["closed"], key=reading["closed"].get)
    return logits_prob.build_response(reading["type"], answers, dist,
                                      confidence=reading["open"].get(pick, dist[pick]))


def reading_diagnostics(reading: dict) -> dict:
    """The alternative reading, for the comparison: the model's own probability of each
    option, the probability of ending the turn, and everything that is not an option."""
    pick = max(reading["closed"], key=lambda o: reading["closed"][o])
    return {"closed_confidence": reading["closed"][pick], "open_confidence": reading["open"][pick],
            "open": reading["open"], "stop": reading["stop"], "outside": reading["outside"],
            "model_picked": reading.get("model_picked", True)}


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
