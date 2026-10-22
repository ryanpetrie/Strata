"""serve/systemone_letters.py - the letter readout: the options are shown as A, B,
C ... and the model's answer is the letter.

WHY.  Most decision readouts in the literature show the options as lettered choices
and read the probability of the letter: the OpenDecisions set's own reference
implementation (predict_letters.py) reads the logits of the letters A, B, C ... after
one forward pass.  Strata instead shows the options by their own names and reads the
probability of the option's own tokens, and when two options share their first token
("rule_1" and "rule_2" both start with " rule") it walks the options' trie to tell
them apart - one to seven extra engine requests for the colliding item (measured on
the 235-item OpenDecisions sample; tools/decision_policy/trie_costs.py).

THE LETTER PATH.  The options are shown with a letter each and the model answers with
the letter, so the answer is ONE token and the letters cannot collide: the row at the
answer position tells them apart, the trie walk is not needed, and one engine request
answers the question.  The letters are one token each in this tokenizer, bare ("A")
and space-merged (" A") alike (measured with the model's own tokenizer:
tools/strata_tokenizer.py - "A" is [32], " A" is [357]).

THE CLIENT DOES NOT SEE THE LETTERS.  The response shape is the one the endpoint
already has: a choice's probabilities are keyed by the option names, a score's by the
level indices, a yes/no's value is the probability of yes.  The letter is an internal
detail of the readout.

THE PROMPT.  It is Strata's own prompt with the options lettered, so the comparison
against the direct path is about the readout and not about two different prompts.
(The OpenDecisions reference implementation uses another shape entirely - a JSON
message with "evidence", "criterion" and "options" - which would have changed the
prompt and the readout at the same time.)

PAST 26 OPTIONS.  There are 26 letters, so a question with more options than that has
no letter for every option: such a question is shown and answered by name, the way the
direct path shows it, and no letter is read for it.  The letter set is not extended
with "AA" or "AB": those are two tokens, and a two-token answer is the trie walk's
problem again.

This module is pure: no Service, no engine, no I/O.  serve/server.py's
Service._systemone_single wires it to the model; serve/test_systemone.py covers it
without a GPU.
"""
from __future__ import annotations

import math

from serve import logits_first_token as lft
from serve import systemone as sysone

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LETTERED_OPTIONS = len(LETTERS)

# The letters are one token each and no two of them are the same token, so the
# options can never collide.  The trie walk is for the case where two options share
# their first token, and that cannot happen here.
TRIE = False

SYSTEM = """You are a classification module. Read the state, then answer every question on its own line as `name: answer`, with no other text.
Copy each question name exactly. Do not repeat the question, the type, or the instructions.
- Every question takes exactly one letter: the letter of the option you choose."""

# the letter set grows with the question: a question with more options than letters
# is answered by name, and the model is told so
TOO_MANY = "\n- A question with more options than letters is answered with the option's name."


def letter_options(q: dict) -> list[str] | None:
    """The option names in letter order (A, B, C ...), or None when the question has
    more options than there are letters."""
    options = lft.option_names_for_question(q)
    return options if len(options) <= MAX_LETTERED_OPTIONS else None


def system_message(questions) -> str:
    """The system message.  The same text for every question, except that a question
    with more options than letters is named as the exception."""
    if any(letter_options(q) is None for _, q in questions):
        return SYSTEM + TOO_MANY
    return SYSTEM


def question_lines(name: str, q: dict) -> list[str]:
    """One question's lines: the header, the lettered options, and the answer line's
    prefix."""
    options = letter_options(q)
    if options is None:
        # More options than letters: the question is shown and answered by name, the
        # way the direct path shows it.  No letter is read for it.
        return sysone.question_lines(name, q)
    lines = [f'[{name} | {q["type"]}] {q["instructions"]}']
    crit = q["criteria"]
    if q["type"] == "noul":
        for letter, opt in zip(LETTERS, options):
            meaning = crit.get("true") if opt == "yes" else crit.get("false")
            lines.append(f'  {letter}) {opt} means "{meaning}"' if meaning else f'  {letter}) {opt}')
    else:
        descs = sysone.score_descs(q) if q["type"] == "score" else crit
        for letter, opt in zip(LETTERS, options):
            desc = descs.get(opt)
            lines.append(f'  {letter}) {opt}' + (f' ("{desc}")' if desc else ""))
    lines.append("  answer with exactly one of these letters: " + ", ".join(LETTERS[:len(options)]))
    lines.append(f'  start your answer line with exactly "{name}:"')
    return lines


def render_answer_prompt(state_text: str, name: str, q: dict,
                         variant: str = "base") -> str:
    """A single-question prompt with the options lettered.  The answer line's prefix
    ("name: ") is NOT here: it prefills the ASSISTANT's turn, so the next token the
    model writes is the answer itself (the letter).

    `variant` is accepted for the same call shape as serve/systemone.py and is not
    used: the per-type prompt variants are the DIRECT readout's lines, and the letter
    template is its own.  serve/server.py refuses a start-up where a variant is
    configured and the letter readout is on, so this can never silently ignore one.
    """
    parts = ["State:\n" + state_text + "\n",
             "\nAnswer the question on its own line as `name: answer`, and nothing else. "
             "Copy each question name exactly. Do not repeat the question, the type, or the instructions.\n",
             "\nQuestions:"]
    parts.extend(question_lines(name, q))
    return "\n".join(parts)


def render_prompt(state_text: str, questions: list[tuple[str, dict]],
                  variants: dict | None = None) -> str:
    """One prompt for the whole request: the state once, then every question, with the
    options lettered.

    `variants` is accepted for the same call shape as serve/systemone.py and is not
    used (see render_answer_prompt)."""
    parts = ["State:\n" + state_text + "\n",
             "\nAnswer every question on its own line as `name: answer`, and nothing else. "
             "Copy each question name exactly. Do not repeat the question, the type, or the instructions.\n",
             "\nQuestions:"]
    for name, q in questions:
        parts.extend(question_lines(name, q))
    parts.append("\nValid line starts: " + ", ".join(n for n, _ in questions)
                + ". Reply with exactly one line per question and nothing else.")
    return "\n".join(parts)


def letter_token_ids(letters, tok, prefix: str = "") -> dict[str, list[int]]:
    """-> {letter: the token ids of that letter at the answer position}.

    Two surface forms of the same letter are read: the space-merged one (" A", the
    form that follows the answer line's prefix "q:") and the bare one ("A").  They
    are the same answer, so their probabilities are added: the reading must not
    depend on which of the two the model happens to write.

    A form is used only when the letter is ONE token in it.  A letter that takes two
    tokens is the trie walk's problem again, and the letter path is here to avoid it.
    """
    out = {letter: [] for letter in letters}
    for form in ((prefix, "") if prefix else ("",)):
        try:
            seqs = lft.option_token_ids(list(letters), tok, form)
        except ValueError:
            continue
        for letter, ids in seqs.items():
            if len(ids) == 1:
                out[letter].append(ids[0])
    return {letter: sorted(set(ids)) for letter, ids in out.items()}


def readings(questions: list[tuple[str, dict]], logits: dict[int, float], tok,
             prefix: str = "", logz: float | None = None,
             temperature: float = 1.0) -> dict | None:
    """-> {question_name: reading} or None to fall back to the decode path.

    The reading has the shape the direct path produces (serve/logits_first_token.py),
    with the OPTIONS' names as the keys: the letters never leave this module.

      closed  - the distribution over the options alone (they add up to 1)
      open    - the model's OWN probability of each option (they add up to less than 1)
      stop    - the probability of ending the turn rather than answering
      outside - everything that is not one of the options
      model_picked - the model's top token at the answer position is one of the
                letters, i.e. the model was choosing among them

    `temperature` scales the closed distribution, exactly as it does on the direct
    path (serve/logits_first_token.py's apply_temperature): the letter arm and the
    token arm must report the same thing, or the comparison between them would be a
    comparison of the temperatures.

    None when the question cannot be read this way: more options than letters, a
    letter that is not one token here, two letters that share a token, or no letter
    inside the engine's window.  The caller then answers from the decode path.
    """
    per_question = []
    for name, q in questions:
        options = letter_options(q)
        if options is None:
            return None              # more options than letters: the direct path answers it
        letters = LETTERS[:len(options)]
        ids = letter_token_ids(letters, tok, prefix)
        if not all(ids.values()):
            return None              # a letter is not one token here: the direct path answers it
        flat = [i for letter in letters for i in ids[letter]]
        if len(set(flat)) != len(flat):
            return None              # two letters share a token: they would collide
        per_question.append((name, q, options, letters, ids))
    if not per_question or not logits:
        return None
    if logz is None:
        # the engine reported no normalizer: normalize over the tokens it did send
        top_logit = max(logits.values())
        logz = top_logit + math.log(sum(math.exp(v - top_logit) for v in logits.values()))
    stop_ids = lft.end_of_turn_ids(tok)
    top = max(logits, key=lambda t: logits[t])
    # the largest logit in the window: the closed reading subtracts it, so the
    # exponentials cannot overflow
    biggest = max(logits.values())
    out = {}
    for name, q, options, letters, ids in per_question:
        closed_vals = {}
        openp = {}
        for letter, opt in zip(letters, options):
            in_window = [logits[t] for t in ids[letter] if t in logits]
            closed_vals[opt] = sum(math.exp(v - biggest) for v in in_window)
            openp[opt] = sum(math.exp(v - logz) for v in in_window)
        total = sum(closed_vals.values())
        if total <= 0.0:
            return None              # nothing to score: the decode path
        closed = {opt: closed_vals[opt] / total for opt in options}
        stop = sum(math.exp(logits[t] - logz) for t in stop_ids if t in logits)
        outside = max(0.0, 1.0 - sum(openp.values()))
        reading = {"type": q["type"], "options": options, "closed": closed, "open": openp,
                   "stop": stop, "outside": outside,
                   "model_picked": top in {i for letter in letters for i in ids[letter]}}
        out[name] = lft.apply_temperature(reading, temperature)
    return out


def fallback_reason(questions: list[tuple[str, dict]], logits: dict[int, float], tok,
                    prefix: str = "", logz: float | None = None,
                    trie_reason: str | None = None) -> str:
    """Why the probabilities could not be read, in a few words.  A diagnostic: it
    never changes the answer."""
    if not logits:
        return "no logits from the engine"
    every = set()
    for name, q in questions:
        options = letter_options(q)
        if options is None:
            return "more options than letters"
        letters = LETTERS[:len(options)]
        ids = letter_token_ids(letters, tok, prefix)
        if not all(ids.values()):
            return "a letter is not one token here"
        missing = [l for l in letters if not (set(ids[l]) & set(logits))]
        if len(missing) == len(letters):
            return f"outside the window: the letter {missing[0]!r} is not in the engine's top-k"
        every |= {i for letter in letters for i in ids[letter]}
    if max(logits, key=lambda t: logits[t]) not in every:
        return "the model's top token is not a letter"
    return "no reason: the readings should have been computed"


def _letter_of(value: str) -> str | None:
    """The letter the value is, when it is one: "A", "a", "A)", "(A)" - and nothing
    else.  "Apple" is not the letter A, and "no" is not the letter N."""
    v = value.strip()
    if v.startswith("("):
        v = v[1:].strip()
    if not v or v[0].upper() not in LETTERS:
        return None
    rest = v[1:].strip()
    if rest and rest[0] not in ")].,:;":
        return None
    return v[0].upper()


def _letter_value(value: str, q: dict) -> str:
    """The option name the letter names, or the value unchanged.

    The letter wins when the value is one: the prompt asked for a letter, so a letter
    is what the model means.  A value that is not a letter is left alone, so the
    direct path's own rules (the option's name, its description, a rate for a yes/no)
    still apply to it.
    """
    options = letter_options(q)
    if not options:
        return value
    letter = _letter_of(value)
    if letter is None:
        return value
    index = LETTERS.index(letter)
    if index >= len(options):
        return value
    return options[index]


def parse_answer_text(text: str, questions: list[tuple[str, dict]]) -> dict:
    """The reply of a single-question request, read at the answer position: the
    letter, mapped back to the option it names.

    The letter is turned into the option's name and the direct path's parser reads
    the result, so the two paths accept and refuse exactly the same answers.
    """
    from serve.structured import StructuredOutputError
    name, q = questions[0]
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise StructuredOutputError("/v1/systemone: the model wrote no answer")
    if ":" in lines[0]:
        return parse_answers(text, questions)          # the model wrote the whole line
    if len(lines) > 1:
        raise StructuredOutputError(f"/v1/systemone: the answer is one line, got {len(lines)}")
    value = _letter_value(lines[0], q)
    return sysone.parse_answer_text(value, questions)


def parse_answers(text: str, questions: list[tuple[str, dict]]) -> dict:
    """The reply's answer lines: the letter in each, mapped back to the option it
    names.  The letters are turned into the options' names and the direct path's
    parser reads the result, so the two paths accept and refuse the same answers."""
    by_name = dict(questions)
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        head, sep, tail = line.partition(":")
        name = head.strip()
        if sep and name in by_name:
            lines.append(f"{name}: {_letter_value(tail.strip(), by_name[name])}")
        else:
            lines.append(line)
    return sysone.parse_answers("\n".join(lines), questions)
