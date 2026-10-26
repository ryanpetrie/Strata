"""serve/systemone.py - POST /v1/systemone: classification without text generation.

The request shape follows the TypeSafe SystemOne contract: a `state` (a string, or a
JSON object/array passed as it is) plus 1-10 named `questions` in three modes -
`noul` (yes/no), `choice` (one of up to 255 named options), `score` (2-10 ordered
levels). It is served by Strata's own loaded chat model through the existing chat
path (Service.prepare -> openai_chunks -> openai_collect). No second model, no
heads, no training: the model reads the state and each question's literal
instructions and answers every question on its own `name: answer` line.

HONESTY NOTE. One greedy decode yields the argmax, not a calibrated distribution.
A yes/no/score pick is reported degenerate (1.0 on the pick, 0.0 elsewhere) and
every response says `"calibrated": false`. The one exception: a noul answer given
as a number in [0, 1] is reported as given - on items whose state states an explicit
rate, the model transcribes the rate (measured: '0.7', '0.8875', '0.92' on
calibration items), which is reading comprehension, not sampling noise. General
thresholds on these numbers are still meaningless; the safety mechanism is
abstention instead: model output that does not parse is a 502
(StructuredOutputError), the same rule serve/structured.py applies to failed
structured generation - a bad answer is an error, never a fabricated distribution.

This module is pure: no Service, no engine, no I/O. server.py's Service.systemone
wires it to the model; serve/test_systemone.py covers it without a GPU.
"""
from __future__ import annotations

import json
import math
import time
import uuid

OBJECT = "systemone.classification"

MAX_STATE_CHARS = 128000          # serialized state, like the reference API
MAX_QUESTIONS_CHARS = 128000      # serialized questions object, like the reference API
MAX_QUESTIONS = 10
MAX_NAME_LEN = 128
MAX_OPTIONS = 255
MIN_LEVELS = 2
MAX_LEVELS = 10
MAX_TOKENS_PER_QUESTION = 32      # one short answer line each, with headroom for long option names
MAX_TOKENS_CAP = 320

TYPES = ("noul", "choice", "score")

SYSTEM = """You are a classification module. Read the state, then answer every question on its own line as `name: answer`, with no other text.
Copy each question name exactly. Do not repeat the question, the type, or the instructions.
- A yes/no question takes exactly `yes`, `no`, or a probability between 0 and 1 (for example `0.7`).
- A choice question takes exactly one option name, copied exactly.
- A score question takes exactly one level name, copied exactly."""


def _is_str(x) -> bool:
    return isinstance(x, str)


def _no_newline(s: str) -> bool:
    return "\n" not in s and "\r" not in s


def _check_name(name, where: str) -> str:
    if not _is_str(name) or not name.strip():
        raise ValueError(f"{where}: names must be non-empty strings")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"{where} {name!r}: names are at most {MAX_NAME_LEN} characters")
    if not _no_newline(name) or ":" in name:
        # Strata-specific: the answers come back one per line as `name: answer`, so a
        # name holding a newline or a colon would make the reply unparseable.
        raise ValueError(f"{where} {name!r}: names hold no newlines and no colons")
    return name


def serialize_state(state) -> str:
    """The state as text. Objects and arrays are canonicalized (sorted keys) so the
    same record in a different key order is the same prompt - and the same cache."""
    if _is_str(state):
        if not state.strip():
            raise ValueError("state: must not be empty")
        text = state
    elif isinstance(state, dict):
        if not state:
            raise ValueError("state: must not be empty")
        text = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    elif isinstance(state, list):
        if not state:
            raise ValueError("state: must not be empty")
        text = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    else:
        raise ValueError("state: a string, a JSON object or a JSON array")
    if len(text) > MAX_STATE_CHARS:
        raise ValueError(f"state: {len(text)} characters, at most {MAX_STATE_CHARS}")
    return text


def validate_questions(questions) -> list[tuple[str, dict]]:
    """The questions map, in request order, each as (name, normalized definition)."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions: a map of 1 to %d named questions" % MAX_QUESTIONS)
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"questions: at most {MAX_QUESTIONS} questions per request, got {len(questions)}")
    if len(json.dumps(questions, ensure_ascii=False)) > MAX_QUESTIONS_CHARS:
        raise ValueError(f"questions: at most {MAX_QUESTIONS_CHARS} characters after serialization")
    out = []
    for name, q in questions.items():
        _check_name(name, "questions")
        if not isinstance(q, dict):
            raise ValueError(f'question "{name}": an object with "type" and "instructions"')
        qtype = q.get("type")
        if qtype not in TYPES:
            raise ValueError(f'question "{name}": type is one of {", ".join(TYPES)}, got {qtype!r}')
        instructions = q.get("instructions")
        if not _is_str(instructions) or not instructions.strip():
            # The reference API rejects an uninstructed question: the model reads
            # instructions literally and cannot answer without them.
            raise ValueError(f'question "{name}": needs a non-empty "instructions" string')
        criteria = q.get("criteria")
        out.append((name, {"type": qtype, "instructions": instructions,
                           "criteria": _check_criteria(name, qtype, criteria)}))
    return out


def _check_criteria(name: str, qtype: str, criteria):
    if qtype == "noul":
        if criteria is None:
            return {}
        if not isinstance(criteria, dict) or any(k not in ("true", "false") for k in criteria):
            raise ValueError(f'question "{name}": noul criteria is {{"true": ..., "false": ...}} or omitted')
        for k, v in criteria.items():
            if not _is_str(v):
                raise ValueError(f'question "{name}": noul criteria.{k} is a string')
        return dict(criteria)
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f'question "{name}": choice criteria is an object of 1 to {MAX_OPTIONS} options')
        if len(criteria) > MAX_OPTIONS:
            raise ValueError(f'question "{name}": at most {MAX_OPTIONS} options, got {len(criteria)}')
        for opt, desc in criteria.items():
            _check_name(opt, f'question "{name}" option')
            if desc is not None and not _is_str(desc):
                raise ValueError(f'question "{name}" option "{opt}": a description string or null')
        return dict(criteria)
    levels, descs = _check_levels(name, criteria)
    return {"levels": levels, "descriptions": descs}


def _check_levels(name: str, criteria):
    """Score levels: a bare list of names, or {name: description|null} like choice
    options (the answer is always the short name; the description is shown, not said)."""
    if isinstance(criteria, dict):
        if not (MIN_LEVELS <= len(criteria) <= MAX_LEVELS):
            raise ValueError(f'question "{name}": score criteria is {MIN_LEVELS} to {MAX_LEVELS} levels')
        for level, desc in criteria.items():
            _check_name(level, f'question "{name}" level')
            if desc is not None and not _is_str(desc):
                raise ValueError(f'question "{name}" level "{level}": a description string or null')
        if len(set(criteria)) != len(criteria):
            raise ValueError(f'question "{name}": score levels are distinct')
        return list(criteria), {k: v for k, v in criteria.items() if v}
    if not isinstance(criteria, list) or not (MIN_LEVELS <= len(criteria) <= MAX_LEVELS):
        raise ValueError(f'question "{name}": score criteria is an array of {MIN_LEVELS} to {MAX_LEVELS} '
                         "level names, lowest first, or an object of level names to descriptions")
    for level in criteria:
        if not _is_str(level) or not level.strip():
            raise ValueError(f'question "{name}": score levels are non-empty strings')
        if not _no_newline(level):
            raise ValueError(f'question "{name}" level {level!r}: no newlines')
    if len(set(criteria)) != len(criteria):
        raise ValueError(f'question "{name}": score levels are distinct')
    return list(criteria), None


def validate_request(req: dict) -> tuple[str, list[tuple[str, dict]]]:
    """-> (state_text, [(name, definition)]). ValueError (the server answers 400)."""
    if not isinstance(req, dict):
        raise ValueError("send a JSON object")
    if "state" not in req:
        raise ValueError("state: missing")
    if "questions" not in req:
        raise ValueError("questions: missing")
    return serialize_state(req["state"]), validate_questions(req["questions"])


def score_names(q: dict) -> list:
    """The answerable level names of a normalized score question."""
    return q["criteria"]["levels"]


def score_descs(q: dict) -> dict:
    return q["criteria"]["descriptions"] or {}


def _question_lines(name: str, q: dict) -> list[str]:
    """One question's lines: header, criteria, and the `name:` answer prefix."""
    lines = [f'[{name} | {q["type"]}] {q["instructions"]}']
    crit, qtype = q["criteria"], q["type"]
    if qtype == "noul":
        if crit:
            lines.append('  yes means "%s" ; no means "%s"'
                         % (crit.get("true", "yes"), crit.get("false", "no")))
            lines.append("  a number between 0 and 1 answers with that probability of yes")
        else:
            lines.append("  answer yes, no, or a number between 0 and 1")
    elif qtype == "choice":
        lines.append("  answer with exactly one of these option names")
        for opt, desc in crit.items():
            lines.append(f'  - {opt}' + (f' ("{desc}")' if desc else ""))
    else:
        lines.append("  answer with exactly one of these level names, lowest first")
        descs = score_descs(q)
        for level in score_names(q):
            lines.append(f'  - {level}' + (f' ("{descs[level]}")' if descs.get(level) else ""))
    lines.append(f'  start your answer line with exactly "{name}:"')
    return lines


def render_prompt(state_text: str, questions: list[tuple[str, dict]]) -> str:
    """One prompt for the whole request: the state once, then every question. One
    model call answers all questions, so the prefill (and its cache) is shared.

    The question lines deliberately hold no colons: the reply's `name: answer` lines
    are the only colon shape in the prompt, so the model cannot echo a header into
    a well-formed-looking answer (an echo fails closed as a 502, never misparses)."""
    parts = ["State:\n" + state_text + "\n",
             "\nAnswer every question on its own line as `name: answer`, and nothing else. "
             "Copy each question name exactly. Do not repeat the question, the type, or the instructions.\n",
             "\nQuestions:"]
    for name, q in questions:
        parts.extend(_question_lines(name, q))
    parts.append("\nValid line starts: " + ", ".join(n for n, _ in questions)
                + ". Reply with exactly one line per question and nothing else.")
    return "\n".join(parts)


def render_logits_prompt(state_text: str, name: str, q: dict) -> str:
    """A single-question prompt that ends at the answer position (`name: `), so the
    next-token logits are the answer's first token. The logits path
    (serve/logits_first_token.py) reads those logits directly instead of generating
    text. Milestone 1 handles one question per request; the multi-question prompt
    stays on the decode path."""
    parts = ["State:\n" + state_text + "\n",
             "\nAnswer the question on its own line as `name: answer`, and nothing else. "
             "Copy the question name exactly. Do not repeat the question, the type, or the instructions.\n",
             "\nQuestions:"]
    parts.extend(_question_lines(name, q))
    parts.append(f"\n{name}: ")
    return "\n".join(parts)


def parse_answers(text: str, questions: list[tuple[str, dict]]) -> dict:
    """The reply's answer lines -> {name: normalized value}. Anything else is a
    StructuredOutputError (the server answers 502): a missing, duplicated or
    unknown line, or a value outside the question's criteria."""
    from serve.structured import StructuredOutputError
    seen: dict[str, str] = {}
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        head, sep, tail = line.partition(":")
        name, value = head.strip(), tail.strip() if sep else ""
        known = next((n for n, _ in questions if n == name), None)
        if known is None or not sep or not value:
            raise StructuredOutputError(
                f"/v1/systemone: the reply is one `name: answer` line per question, got {raw.strip()!r}")
        if known in seen:
            raise StructuredOutputError(f'/v1/systemone: question "{known}" answered twice')
        seen[known] = value
    missing = [n for n, _ in questions if n not in seen]
    if missing:
        raise StructuredOutputError(f"/v1/systemone: no answer for {missing[0]!r}")
    by_name = dict(questions)
    out = {}
    for name, value in seen.items():
        qtype = by_name[name]["type"]
        crit = by_name[name]["criteria"]
        if qtype == "noul":
            low = value.lower()
            if low in ("yes", "no"):
                out[name] = low == "yes"
                continue
            try:
                num = float(value)
            except ValueError:
                num = float("nan")
            if not math.isfinite(num) or not 0.0 <= num <= 1.0:
                raise StructuredOutputError(
                    f'/v1/systemone: question "{name}" takes yes, no, or a number 0-1, got {value!r}')
            out[name] = num
        elif qtype == "choice":
            if value not in crit:
                raise StructuredOutputError(
                    f'/v1/systemone: question "{name}" takes one of {sorted(crit)}, got {value!r}')
            out[name] = value
        else:
            levels = score_names(by_name[name])
            if value not in levels:
                raise StructuredOutputError(
                    f'/v1/systemone: question "{name}" takes one of {levels}, got {value!r}')
            out[name] = levels.index(value)
    return out


def build_response(questions: list[tuple[str, dict]], answers: dict, model: str, usage: dict,
                   timings=None) -> dict:
    """The reference API's response shape. The distributions are degenerate (one
    greedy sample): 1.0 on the pick, 0.0 elsewhere - hence "calibrated": false."""
    out_answers = {}
    for name, q in questions:
        qtype, crit = q["type"], q["criteria"]
        if qtype == "noul":
            v = answers[name]
            out_answers[name] = {"type": "noul",
                                 "noul": 1.0 if v is True else (0.0 if v is False else float(v))}
        elif qtype == "choice":
            pick = answers[name]
            out_answers[name] = {"type": "choice", "choice": pick,
                                 "probabilities": {o: 1.0 if o == pick else 0.0 for o in crit},
                                 "confidence": 1.0}
        else:
            idx = answers[name]
            levels = score_names(q)
            out_answers[name] = {"type": "score", "score": float(idx),
                                 "legend": {str(i): level for i, level in enumerate(levels)},
                                 "probabilities": {str(i): 1.0 if i == idx else 0.0
                                                   for i in range(len(levels))},
                                 "confidence": 1.0}
    body = {"id": "systemone-" + uuid.uuid4().hex, "object": OBJECT, "created": int(time.time()),
            "model": model, "calibrated": False, "answers": out_answers,
            "usage": {"prompt_tokens": int(usage.get("prompt_tokens", 0)),
                      "completion_tokens": int(usage.get("completion_tokens", 0)),
                      "total_tokens": int(usage.get("total_tokens", 0)),
                      # the part of the prompt the conversation cache already held
                      "prompt_tokens_details": {"cached_tokens": int(
                          (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))}}}
    if timings:
        body["timings"] = timings
    return body
