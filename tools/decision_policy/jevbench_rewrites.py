#!/usr/bin/env python3
"""tools/decision_policy/jevbench_rewrites.py - JevBench's coherence metric over /v1/systemone.

WHAT THIS IS. JevBench (https://huggingface.co/datasets/JevBench/jevbench, Apache-2.0) is
a metamorphic test suite for typed probabilistic decision models: a case is one state and
a batch of typed questions, and each of its 50 relations rewrites the QUESTIONS (never
the state) and checks one law of probability over the two answer distributions.  The
official metric is in the published package (https://github.com/JevBench/jevbench):

  jevbench/score.py       the scoring: per relation
                            score  = 1 - (violated tests / judged tests)     (binary)
                            graded = 1 - mean(max(0, deviation - tolerance) / (1 - tolerance))
                          per dimension (pillar) = the mean of its relations' scores,
                          overall = the mean of the dimensions' scores,
                          with a bootstrap interval over the cases.
  jevbench/relations/*.py one law per relation, with the deviation it measures
                          (rep.py, mea.py, log.py are the ones this harness uses).
  jevbench/schema.py      the answer shape: a Noul is {"true": p, "false": 1 - p},
                          a Choice is {option id: p}, a Score is {"0": p0, ...}.

This harness re-implements those laws and that scoring (the sources are named beside
each law below) and runs the endpoint's answers through them.  It does NOT import the
package: the package is not installed here, and a tool that needs `pip install jevbench`
cannot be run from a checkout.

THE THREE VARIANT CLASSES.  A class says what the rewrite did to the question:
  reword        the question is re-expressed - its wording, its option ids, its key or
                its answer type - while the event it asks about stays the same
                (the official REP dimension).
  negation     the question is negated, and the two probabilities must sum to one
                (the official MEA.complement group).
  combination  two or more propositions - questions or options - are combined, or a
                question is derived from another (MEA.partition, MEA.marginalisation,
                LOG.bounds, LOG.monotonicity, and De Morgan's law, which the official
                taxonomy files under REP.invariance because its law is an equality).
The classes are the harness's own grouping; the official group of every relation is
kept in the sample and printed with it.

WHAT IS MEASURED, AND WHAT IS NOT.  Every question here is asked on its own, one
request per question.  The official run asks a whole case in one request, so its
"base" distribution is the answer inside a batch and ours is the answer alone.  Both
readings of a law here are solo, so like is compared with like; what is lost is the
batch dimension (BAT), which is not measured at all.  (The endpoint CAN answer
several questions in one request, each with its own probability - see
docs/research/shared_backbone_decision_policy.md section 11 - but a metamorphic law
compares a question with its rewrite, and asking both alone keeps the two sides
identical in everything but the question.)

THE TWO READINGS OF THE CONFIDENCE (docs/research/shared_backbone_decision_policy.md
§10.4).  The endpoint reports the closed reading as `probabilities` (the options
renormalized over themselves, so they add up to 1) and the open reading as
`confidence` (the model's own probability of the pick).  The metric uses the
distribution - the closed reading - on both sides of every law, and the run log keeps
both readings per request so the other one can be compared offline.

HONESTY RULES, enforced by the code:
- One attempt per request.  A non-200 reply (502 on unparseable output, truncation
  included) is not a distribution: the request is left out of the predictions file and
  the test it serves is not judged.  Nothing is retried or repaired.  Under the
  official rules a relation whose judged tests cover less than 95% of the tests it was
  planned for is not scored at all, and the report says so.
- Resume-safe: a request whose answer is already in the predictions file is not asked
  again, so an interrupted run continues with the same --out.  A request that FAILED
  produced no answer at all, so it is asked again by a later run; an answer that was
  produced is never overwritten.
- The tolerance is the official one: max(2 x repeat noise, 0.05), re-judged at a fixed
  0.05 (jevbench/score.py, `tolerance="fixed"`).  With one repeat per request the
  noise is 0, so 0.05 is the floor the official floor is.
- The sample is a SUBSET of the frozen plan, so the coverage rule is applied to the
  sample (the sample is the plan of this run).  The numbers are a demonstration on
  this sample, not the benchmark's official score.
- Nothing is fabricated: a missing answer lowers the coverage, it never becomes a
  distribution.

USAGE
  python tools/decision_policy/jevbench_rewrites.py --items <sample.jsonl> --out <predictions.jsonl>
  python tools/decision_policy/jevbench_rewrites.py --items ... --out ... --limit 40 --dry-run
  python tools/decision_policy/jevbench_rewrites.py --items ... --out ... --report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "strata-unsloth-ud-iq4_xs.json"

# The official tolerance: max(2 x the model's repeat noise, 0.05), re-judged at a fixed
# value for every model (jevbench/score.py, `tolerance="fixed"`, fixed=0.05).  One
# repeat per request means no measured noise, so the floor is the tolerance.
TOLERANCE = 0.05
MIN_COVERAGE = 0.95        # jevbench/score.py: a relation below it is not scored
MIN_N = 5                  # jevbench/score.py: a relation needs at least this many judged tests

# The relation -> its class and its official group (jevbench/taxonomy.py).
CLASSES = {
    "reword": [
        ("instruction_paraphrase", "REP.invariance"),
        ("description_paraphrase", "REP.invariance"),
        ("typo_noise", "REP.invariance"),
        ("verbosity", "REP.invariance"),
        ("key_rename", "REP.invariance"),
        ("cross_lingual", "REP.invariance"),
        ("option_permutation", "REP.equivariance"),
        ("option_rename", "REP.equivariance"),
        ("clone_symmetry", "REP.equivariance"),
        ("key_instruction_conflict", "REP.binding"),
        ("name_description_conflict", "REP.binding"),
        ("description_swap", "REP.binding"),
        ("noul_as_choice", "REP.type"),
        ("noul_as_score", "REP.type"),
    ],
    "negation": [
        ("negation_wrapper", "MEA.complement"),
        ("natural_negation", "MEA.complement"),
        ("choice_complement", "MEA.complement"),
        ("score_complement", "MEA.complement"),
        ("compound_negation", "MEA.complement"),
    ],
    "combination": [
        ("de_morgan", "REP.invariance"),
        ("frechet_and_upper", "LOG.bounds"),
        ("inclusion_exclusion", "MEA.marginalisation"),
        ("entailment_strength", "LOG.monotonicity"),
        ("quantifier_monotonicity", "LOG.monotonicity"),
        ("count_upper", "LOG.bounds"),
        ("count_lower", "LOG.bounds"),
        ("nary_and_upper", "LOG.bounds"),
        ("nary_and_lower", "LOG.bounds"),
        ("nary_or_upper", "LOG.bounds"),
        ("nary_or_lower", "LOG.bounds"),
        ("choice_indicator", "MEA.marginalisation"),
        ("score_cumulative", "MEA.marginalisation"),
        ("pair_choice_marginals", "MEA.marginalisation"),
        ("option_merge", "MEA.partition"),
        ("option_split", "MEA.partition"),
        ("option_clone", "MEA.partition"),
        ("similarity_effect", "MEA.partition"),
        ("partition_hierarchy", "MEA.partition"),
        ("score_granularity", "MEA.partition"),
        ("extreme_level", "MEA.partition"),
        ("option_closure", "MEA.partition"),
    ],
}
RELATION_CLASS = {r: c for c, rows in CLASSES.items() for r, _ in rows}
RELATION_GROUP = {r: g for rows in CLASSES.values() for r, g in rows}


# ------------------------------------------------------------------ the endpoint

def load_config(path: Path) -> dict:
    """The launch config: the port, the model and the API key.  The key is never printed."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"port": int(cfg.get("port", 8080)), "model": cfg.get("model_name"),
            "api_key": cfg.get("api_key", "")}


def post(url: str, key: str, payload: dict, timeout: float):
    """One POST to the endpoint.  -> (http status, body, milliseconds)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), \
                (time.perf_counter() - t0) * 1000.0
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:                                       # noqa: BLE001
            body = {}
        return e.code, body, (time.perf_counter() - t0) * 1000.0


def request_id(case_id: str, key: str, question: dict) -> str:
    """The id of one request: the case, the question's key and the question itself.

    Content addressed, so the same question asked by two relations is one request
    (the official plan shares requests between relations that read the same
    questions, and asking them twice would only cost time).
    """
    blob = json.dumps(question, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "%s|%s|%s" % (case_id, key, hashlib.sha1(blob).hexdigest()[:12])


def to_distribution(answer: dict, question: dict) -> dict[str, float]:
    """The endpoint's answer -> the official distribution over option ids.

    jevbench/schema.py, read_distribution: a Noul is {"true": p, "false": 1 - p}, a
    Choice is one probability per option id, a Score one per level index; the last
    two are normalized over the options they carry.
    """
    qtype = question["type"]
    if qtype == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    probs = answer.get("probabilities") or {}
    if qtype == "choice":
        raw = {k: float(probs.get(k, 0.0)) for k in question["criteria"]}
    else:
        raw = {str(i): float(probs.get(str(i), 0.0)) for i in range(len(question["criteria"]))}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("the probabilities sum to %r" % total)
    return {k: v / total for k, v in raw.items()}


# ------------------------------------------------------------------ the laws

def tv(a: dict[str, float], b: dict[str, float]) -> float:
    """Total variation distance over the union of the option ids (jevbench/schema.py)."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _base(row: dict, key: str) -> dict[str, float]:
    return row["_base"][key]


def _variant(row: dict, key: str) -> dict[str, float]:
    return row["_variant"][key]


def _vkey(row: dict, key: str) -> str:
    """The variant's key for a base question.

    Most relations keep the key; a few rename it (the official KeyRename and
    KeyInstructionConflict), and the sample says which variant key stands for
    which base question.
    """
    return row["_vmap"].get(key, key)


def _true(row: dict, key: str) -> float:
    """P(yes) of a Noul's distribution."""
    return _base(row, key)["true"]


def _vtrue(row: dict, key: str) -> float:
    """P(yes) of a Noul's variant distribution."""
    return _variant(row, key)["true"]


def law_tv(row: dict):
    """The two distributions must be equal: TV(base, variant) = 0.

    jevbench/relations/rep.py: InstructionParaphrase, DescriptionParaphrase,
    TypoNoise, Verbosity, CrossLingual (the same law as KeyRename, whose variant
    carries a new key).
    """
    key = row["targets"][0]
    return ("tv", tv(_variant(row, _vkey(row, key)), _base(row, key)))


def law_option_permutation(row: dict):
    """Reordering the options leaves each option's probability where it was.

    jevbench/relations/rep.py, OptionPermutation.  A Score's levels are reversed
    (params["order"] == "reversed"), so the variant's level i is the original level
    n-1-i and the distribution is read back in the original order before the distance
    is taken.
    """
    key = row["targets"][0]
    d = dict(_variant(row, _vkey(row, key)))
    if row["params"].get("order") == "reversed":
        n = len(d)
        d = {str(n - 1 - int(k)): v for k, v in d.items()}
    return ("tv", tv(d, _base(row, key)))


def law_option_rename(row: dict):
    """Neutral option ids, the same descriptions: the probabilities follow the ids back.

    jevbench/relations/rep.py, OptionRename (params["mapping"]).
    """
    key = row["targets"][0]
    mapping = row["params"]["mapping"]
    d = {mapping[k]: v for k, v in _variant(row, _vkey(row, key)).items()}
    return ("tv", tv(d, _base(row, key)))


def law_clone_symmetry(row: dict):
    """Two identical options get the same probability: |P(o) - P(clone)| = 0.

    jevbench/relations/rep.py, CloneSymmetry.
    """
    key = row["targets"][0]
    d = _variant(row, _vkey(row, key))
    o, c = row["params"]["option"], row["params"]["clone"]
    return ("clone_symmetry", abs(d[o] - d[c]))


def law_swap_back(row: dict):
    """The descriptions moved, so the probabilities move with them.

    jevbench/relations/rep.py: DescriptionSwap and NameDescriptionConflict (both
    read the answer back by description over params["pair"]).
    """
    key = row["targets"][0]
    a, b = row["params"]["pair"]
    d = dict(_variant(row, _vkey(row, key)))
    d[a], d[b] = d[b], d[a]
    return ("tv", tv(d, _base(row, key)))


def law_key_conflict(row: dict):
    """A misleading key changes nothing: the answer follows the instructions.

    jevbench/relations/rep.py, KeyInstructionConflict (params["keys"]: the key each
    question was sent under).
    """
    out = []
    for key, sent_as in row["params"]["keys"].items():
        out.append(("tv", tv(_variant(row, sent_as), _base(row, key))))
    return out


def law_noul_as_choice(row: dict):
    """The same yes/no question as a Noul and as a two-option Choice: one P(yes).

    jevbench/relations/rep.py, NoulAsChoice (check "abs_diff").
    """
    key = row["targets"][0]
    return ("abs_diff", abs(_variant(row, _vkey(row, key))["yes"] - _base(row, key)["true"]))


def law_noul_as_score(row: dict):
    """The same yes/no question as a Noul and as a two-level Score: one P(yes).

    jevbench/relations/rep.py, NoulAsScore: the levels are ("No", "Yes"), so "yes"
    is level 1.
    """
    key = row["targets"][0]
    return ("abs_diff", abs(_variant(row, _vkey(row, key))["1"] - _base(row, key)["true"]))


def law_complement(row: dict):
    """P(Q) + P(not Q) = 1.

    jevbench/relations/mea.py: NaturalNegation, NegationWrapper (P(Q) against the
    negated Noul), ChoiceComplement (P(o) against a Noul about "not o") and
    ScoreComplement (P(level k) against a Noul about "not level k").
    """
    key = row["targets"][0]
    p = row["params"]
    if "negated" in p:                                  # a Noul and its negation
        a = _true(row, key)
        b = _vtrue(row, p["negated"])
    elif "option" in p:                                # a Choice and a Noul about "not o"
        a = _base(row, key)[p["option"]]
        b = _vtrue(row, p["probe"])
    else:                                              # a Score and a Noul about "not level k"
        a = _base(row, key)[p["level"]]
        b = _vtrue(row, p["probe"])
    return ("complement_gap", abs(a + b - 1.0))


def law_compound_negation(row: dict):
    """P(not (A and B)) + P(A and B) = 1.

    jevbench/relations/mea.py, CompoundNegation.
    """
    p = row["params"]
    return ("complement_gap", abs(_vtrue(row, p["nand"]) + _vtrue(row, p["and"]) - 1.0))


def law_de_morgan(row: dict):
    """P(not (A and B)) = P(not A or not B).

    jevbench/relations/rep.py, DeMorgan (check "nand_vs_or_not").
    """
    p = row["params"]
    return ("nand_vs_or_not", abs(_vtrue(row, p["nand"]) - _vtrue(row, p["or_not"])))


def law_coarsened(row: dict):
    """The options are merged, split or refined: the mass of every original event stays.

    jevbench/relations/mea.py: OptionMerge, OptionSplit, OptionClone,
    SimilarityEffect, PartitionHierarchy, ScoreGranularity, ExtremeLevel and
    OptionClosure.  Each collapses the variant's distribution back onto the
    original's options and takes the distance to the base.
    """
    key = row["targets"][0]
    base = _base(row, key)
    d = dict(_variant(row, _vkey(row, key)))
    p = row["params"]
    if "merged" in p:                                   # two options merged into one
        a, b = p["parts"]
        coarse = {k: v for k, v in base.items() if k not in (a, b)}
        coarse[p["merged"]] = base[a] + base[b]
        return ("tv_to_coarsened", tv(d, coarse))
    if "parts" in p:                                   # one option split in two
        o = p["option"]
        d[o] = sum(d.pop(x) for x in p["parts"])
        return ("tv_after_collapse", tv(d, base))
    if "clone" in p:                                   # a clone of an option
        o = p["option"]
        d[o] += d.pop(p["clone"])
        return ("tv_after_collapse", tv(d, base))
    if "twin" in p:                                    # a near-duplicate of an option
        o = p["option"]
        d[o] += d.pop(p["twin"])
        return ("tv_after_collapse", tv(d, base))
    if "blocks" in p and isinstance(p["blocks"], dict):  # the options grouped into blocks
        coarse = {block: sum(base[o] for o in fine) for block, fine in p["blocks"].items()}
        return ("tv_to_coarsened", tv(d, coarse))
    if "blocks" in p:                                  # the levels merged in pairs
        coarse = {str(j): sum(base[str(i)] for i in b) for j, b in enumerate(p["blocks"])}
        return ("tv_to_coarsened", tv(d, coarse))
    if "added" in p:                                   # a level beyond the top
        added = p["added"]
        top = str(len(base) - 1)
        d[top] += d.pop(added)
        return ("tv_after_collapse", tv(d, base))
    o, nota = p["replaced"], p["nota"]                 # one option replaced by a catch-all
    expected = {(nota if k == o else k): v for k, v in base.items()}
    return ("catch_all_tv", tv(d, expected))


def law_score_cumulative(row: dict):
    """"level k or higher?" equals the Score's upper tail, for every k.

    jevbench/relations/mea.py, ScoreCumulative (check "tail_gap_max").
    """
    key = row["targets"][0]
    score = _base(row, key)
    n = len(score)
    out = []
    for k, probe in row["params"]["probes"].items():
        tail = sum(score[str(i)] for i in range(int(k), n))
        out.append(abs(_vtrue(row, probe) - tail))
    return ("tail_gap_max", max(out))


def law_choice_indicator(row: dict):
    """"is it o?" equals P(o), for every option o.

    jevbench/relations/mea.py, ChoiceIndicator (check "indicator_gap").
    """
    key = row["targets"][0]
    base = _base(row, key)
    gaps = [abs(_vtrue(row, probe) - base[o]) for o, probe in row["params"]["probes"].items()]
    return ("indicator_gap", max(gaps))


def law_pair_marginals(row: dict):
    """The four-way Choice over two Nouls has the Nouls as its marginals.

    jevbench/relations/mea.py, PairChoiceMarginals (check "marginal_gap").
    """
    a_key, b_key = row["targets"]
    cell = _variant(row, row["params"]["pair"])
    pa, pb = cell["10"] + cell["11"], cell["01"] + cell["11"]
    return ("marginal_gap", max(abs(pa - _true(row, a_key)), abs(pb - _true(row, b_key))))


def law_inclusion_exclusion(row: dict):
    """P(A and B) + P(A or B) = P(A) + P(B).

    jevbench/relations/mea.py, InclusionExclusion.
    """
    a_key, b_key = row["targets"]
    p = row["params"]
    got = _vtrue(row, p["conj"]) + _vtrue(row, p["disj"])
    return ("inclusion_exclusion", abs(got - _true(row, a_key) - _true(row, b_key)))


def law_frechet_and_upper(row: dict):
    """P(A and B) <= min(P(A), P(B)).

    jevbench/relations/log.py, FrechetAndUpper (check "excess").
    """
    a_key, b_key = row["targets"]
    p = row["params"]
    a, b = _true(row, a_key), _true(row, b_key)
    return ("excess", max(0.0, _vtrue(row, p["conj"]) - min(a, b)))


def _nary_ps(row: dict) -> list[float]:
    return [_true(row, k) for k in row["targets"]]


def law_nary(row: dict):
    """The bounds of a conjunction and a disjunction of three Nouls.

    jevbench/relations/log.py: NaryAndUpper, NaryAndLower (Bonferroni), NaryOrUpper
    (the union bound) and NaryOrLower.
    """
    ps = _nary_ps(row)
    p = row["params"]
    p_all, p_any = _vtrue(row, p["all"]), _vtrue(row, p["any"])
    name = row["relation"]
    if name == "nary_and_upper":
        gap = p_all - min(ps)
    elif name == "nary_and_lower":
        gap = sum(ps) - (len(ps) - 1) - p_all
    elif name == "nary_or_upper":
        gap = p_any - min(1.0, sum(ps))
    else:
        gap = max(ps) - p_any
    return ("excess", max(0.0, gap))


def law_count(row: dict):
    """The bounds of "at least two of three".

    jevbench/relations/log.py: CountUpper (Markov) and CountLower.
    """
    ps = _nary_ps(row)
    s = sum(ps)
    p_two = _vtrue(row, row["params"]["two"])
    if row["relation"] == "count_upper":
        gap = p_two - s / 2.0
    else:
        gap = (s - 1.0) / (len(ps) - 1) - p_two
    return ("excess", max(0.0, gap))


def law_entailment_strength(row: dict):
    """A stronger claim cannot be more probable than the weaker one it entails.

    jevbench/relations/log.py, EntailmentStrength (check "chain").
    """
    key = row["targets"][0]
    p = row["params"]
    mid = _true(row, key)
    strong = _vtrue(row, p["strong"])
    weak = _vtrue(row, p["weak"])
    return ("chain", max(0.0, strong - mid, mid - weak, strong - weak))


def law_quantifier(row: dict):
    """P(all) <= P(at least two) <= P(at least one).

    jevbench/relations/log.py, QuantifierMonotonicity (check "chain").
    """
    p = row["params"]
    p_all, p_two, p_any = _vtrue(row, p["all"]), _vtrue(row, p["two"]), _vtrue(row, p["any"])
    return ("chain", max(0.0, p_all - p_two, p_two - p_any, p_all - p_any))


LAWS = {
    "instruction_paraphrase": law_tv,
    "description_paraphrase": law_tv,
    "typo_noise": law_tv,
    "verbosity": law_tv,
    "key_rename": law_tv,
    "cross_lingual": law_tv,
    "option_permutation": law_option_permutation,
    "option_rename": law_option_rename,
    "clone_symmetry": law_clone_symmetry,
    "description_swap": law_swap_back,
    "name_description_conflict": law_swap_back,
    "key_instruction_conflict": law_key_conflict,
    "noul_as_choice": law_noul_as_choice,
    "noul_as_score": law_noul_as_score,
    "negation_wrapper": law_complement,
    "natural_negation": law_complement,
    "choice_complement": law_complement,
    "score_complement": law_complement,
    "compound_negation": law_compound_negation,
    "de_morgan": law_de_morgan,
    "option_merge": law_coarsened,
    "option_split": law_coarsened,
    "option_clone": law_coarsened,
    "similarity_effect": law_coarsened,
    "partition_hierarchy": law_coarsened,
    "score_granularity": law_coarsened,
    "extreme_level": law_coarsened,
    "option_closure": law_coarsened,
    "score_cumulative": law_score_cumulative,
    "choice_indicator": law_choice_indicator,
    "pair_choice_marginals": law_pair_marginals,
    "inclusion_exclusion": law_inclusion_exclusion,
    "frechet_and_upper": law_frechet_and_upper,
    "nary_and_upper": law_nary,
    "nary_and_lower": law_nary,
    "nary_or_upper": law_nary,
    "nary_or_lower": law_nary,
    "count_upper": law_count,
    "count_lower": law_count,
    "entailment_strength": law_entailment_strength,
    "quantifier_monotonicity": law_quantifier,
}


# ------------------------------------------------------------------ the scoring

def judge(deviation: float, tolerance: float = TOLERANCE) -> tuple[bool, float]:
    """(violated, graded) of one deviation - jevbench/score.py, `_judge`.

    A deviation past the tolerance is a violation; the graded score falls with how
    far past it the deviation went, scaled to [0, 1].
    """
    if deviation != deviation:                       # NaN: a violation, never a pass
        return True, 0.0
    excess = max(0.0, deviation - tolerance) / (1.0 - tolerance) if tolerance < 1.0 else 0.0
    return not (deviation <= tolerance), 1.0 - min(1.0, excess)


def score_rows(rows: list[dict]) -> dict:
    """The official scoring over the judged tests.

    jevbench/score.py: a test is violated when any of its outcomes is, and its graded
    score is the worst one; a relation's score is 1 - the violated share of its
    judged tests and its graded score the mean of theirs; a class (a dimension of the
    official taxonomy) is the mean of its relations' scores; the overall is the mean
    of the classes'.  A relation with fewer than MIN_N judged tests, or with less
    than MIN_COVERAGE of its planned tests judged, is not scored.
    """
    per_rel: dict[str, dict] = defaultdict(lambda: {"tests": 0, "judged": 0, "violated": 0,
                                                     "graded": [], "devs": [], "planned": 0})
    for row in rows:
        rel = row["relation"]
        cell = per_rel[rel]
        cell["planned"] += 1
        try:
            outcomes = LAWS[row["relation"]](row)
        except (KeyError, TypeError, ValueError) as e:
            cell.setdefault("errors", []).append("%s: %s" % (row["id"], e))
            continue
        if isinstance(outcomes, tuple):
            outcomes = [outcomes]
        cell["tests"] += 1
        judged = [judge(dev) for _, dev in outcomes]
        cell["judged"] += 1
        cell["violated"] += 1 if any(v for v, _ in judged) else 0
        cell["graded"].append(min(g for _, g in judged))
        cell["devs"].extend(dev for _, dev in outcomes)
    relations = {}
    for rel, cell in sorted(per_rel.items()):
        n = cell["judged"]
        coverage = n / cell["planned"] if cell["planned"] else 0.0
        entry = {"n": n, "planned": cell["planned"], "coverage": coverage,
                 "violated": cell["violated"],
                 "score": (1.0 - cell["violated"] / n) if n else None,
                 "graded": mean(cell["graded"]) if cell["graded"] else None,
                 "mean_deviation": mean(cell["devs"]) if cell["devs"] else None,
                 "class": RELATION_CLASS.get(rel, "?"),
                 "group": RELATION_GROUP.get(rel, "?")}
        if cell.get("errors"):
            entry["errors"] = cell["errors"][:3]
        if n < MIN_N:
            entry["excluded"] = "n < %d" % MIN_N
        elif coverage < MIN_COVERAGE:
            entry["excluded"] = "coverage %.0f%% < %.0f%%" % (100 * coverage, 100 * MIN_COVERAGE)
        relations[rel] = entry
    classes = {}
    for cls in CLASSES:
        scores = [relations[r]["score"] for r, _ in CLASSES[cls]
                  if r in relations and relations[r].get("score") is not None]
        graded = [relations[r]["graded"] for r, _ in CLASSES[cls]
                  if r in relations and relations[r].get("graded") is not None]
        tests = sum(relations[r]["n"] for r, _ in CLASSES[cls] if r in relations)
        classes[cls] = {"score": mean(scores) if scores else None,
                        "graded": mean(graded) if graded else None,
                        "relations": len(scores), "relations_planned": len(CLASSES[cls]),
                        "tests": tests}
    scored = [c["score"] for c in classes.values() if c["score"] is not None]
    graded_cls = [c["graded"] for c in classes.values() if c["graded"] is not None]
    overall = mean(scored) if len(scored) == len(classes) else None
    return {"relations": relations, "classes": classes, "overall": overall,
            "graded_overall": mean(graded_cls) if len(graded_cls) == len(classes) else None,
            "tests": len(rows), "judged": sum(c["judged"] for c in per_rel.values())}


def bootstrap(rows: list[dict], dists: dict[str, dict[str, float]], draws: int = 1000,
              seed: int = 0) -> dict:
    """The interval over the cases - jevbench/score.py's bootstrap.

    The cases are resampled with replacement and the overall is recomputed over the
    same relations; with few cases the interval is wide and should be read as such.
    """
    import random
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    case_ids = sorted(by_case)
    if len(case_ids) < 2 or not draws:
        return {}
    rng = random.Random(seed)
    overall, graded = [], []
    for _ in range(draws):
        sample = [rng.choice(case_ids) for _ in case_ids]
        picked = [row for c in sample for row in by_case[c]]
        res = score_rows(picked)
        if res["overall"] is not None:
            overall.append(res["overall"])
        if res["graded_overall"] is not None:
            graded.append(res["graded_overall"])
    out = {}
    for name, vals in (("overall", overall), ("graded_overall", graded)):
        if vals:
            vals.sort()
            out[name] = [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]
    return out


# ------------------------------------------------------------------ the sample

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_sample(path: Path) -> list[dict]:
    """The sample rows, with the distributions of the base and the variant attached.

    The rows carry the questions; the answers live in the predictions file, joined by
    the request id.  A row whose base or variant answer is missing cannot be judged:
    it is kept, but the report counts it as unjudged (the official coverage rule).
    """
    rows = read_jsonl(path)
    return rows


def attach(rows: list[dict], dists: dict[str, dict[str, float]]) -> list[dict]:
    """Give every row the distributions of its base and its variant questions.

    A row whose base or variant answer is missing cannot be judged: it is kept, but
    the report counts it as unjudged (the official coverage rule).
    """
    for row in rows:
        row["_base"] = {}
        row["_variant"] = {}
        row["_missing"] = []
        # which variant key stands for which base question (a few relations rename it)
        row["_vmap"] = {b: k for k, b in (row.get("variant_keys") or {}).items() if b}
        for role, table in (("base", row["_base"]), ("variant", row["_variant"])):
            for e in row[role]:
                if e["id"] in dists:
                    table[e["key"]] = dists[e["id"]]
                else:
                    row["_missing"].append(e["id"])
    return rows


def judged_rows(rows: list[dict]) -> list[dict]:
    """The rows whose every question was answered: the ones a law can be read on."""
    return [row for row in rows if not row["_missing"]]


def read_predictions(path: Path) -> dict[str, dict[str, float]]:
    """The predictions file -> {request id: the distribution over the option ids}."""
    out = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        try:
            out[row["id"]] = row["probs"]
        except (KeyError, TypeError):
            continue
    return out


# ------------------------------------------------------------------ the run

def plan_requests(rows: list[dict]) -> list[dict]:
    """Every request the sample needs, deduplicated by its content-addressed id.

    One request per question: the endpoint's answer position is per question, so a
    request with several questions would leave the fast path.
    """
    seen, requests = {}, []
    for row in rows:
        for role in ("base", "variant"):
            for e in row[role]:
                rid = e.get("id") or request_id(row["case_id"], e["key"], e["question"])
                if rid in seen:
                    seen[rid]["roles"].add(role)
                    continue
                req = {"id": rid, "case_id": row["case_id"], "key": e["key"],
                       "question": e["question"], "state": row["state"], "roles": {role},
                       "relations": {row["relation"]}}
                seen[rid] = req
                requests.append(req)
    for row in rows:
        for role in ("base", "variant"):
            for e in row[role]:
                rid = e.get("id") or request_id(row["case_id"], e["key"], e["question"])
                if rid in seen:
                    seen[rid]["relations"].add(row["relation"])
    return requests


def run_requests(requests: list[dict], url: str, key: str, timeout: float,
                 out_path: Path, log_path: Path) -> tuple[int, int]:
    """Ask the endpoint.  -> (answered, failed)."""
    n_ok = n_fail = 0
    t_start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as fout, log_path.open("a", encoding="utf-8") as flog:
        for n, req in enumerate(requests, 1):
            payload = {"state": req["state"], "questions": {req["key"]: req["question"]}}
            try:
                status, body, ms = post(url, key, payload, timeout)
            except Exception as e:                                # noqa: BLE001 - connection loss etc.
                status, body, ms = -1, {"error": str(e)}, 0.0
            got = body.get("answers", {}) if status == 200 else {}
            answer = got.get(req["key"])
            record = {"id": req["id"], "case_id": req["case_id"], "key": req["key"],
                      "type": req["question"]["type"], "roles": sorted(req["roles"]),
                      "relations": sorted(req["relations"]),
                      "http": status, "latency_ms": round(ms, 1),
                      "fast_path": bool(body.get("calibrated"))}
            if status == 200 and answer is not None:
                try:
                    probs = to_distribution(answer, req["question"])
                except (KeyError, TypeError, ValueError) as e:
                    record["predicted"] = False
                    record["error"] = "unmappable answer: %s" % e
                    n_fail += 1
                else:
                    fout.write(json.dumps({"id": req["id"], "probs": probs,
                                           "case_id": req["case_id"], "key": req["key"],
                                           "type": req["question"]["type"]},
                                          ensure_ascii=False) + "\n")
                    fout.flush()
                    record["predicted"] = True
                    record["probabilities"] = answer.get("probabilities")
                    record["confidence"] = answer.get("confidence")
                    if req["question"]["type"] == "choice":
                        record["pick"] = answer.get("choice")
                    elif req["question"]["type"] == "noul":
                        record["pick"] = "yes" if float(answer.get("noul", 0.0)) >= 0.5 else "no"
                    else:
                        record["pick"] = answer.get("score")
                    usage = body.get("usage") or {}
                    record["prompt_tokens"] = usage.get("prompt_tokens")
                    record["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    diag = (body.get("diagnostics") or {}).get(req["key"])
                    if diag:
                        record["closed_confidence"] = diag.get("closed_confidence")
                        record["open_confidence"] = diag.get("open_confidence")
                        record["stop_probability"] = diag.get("stop")
                        record["outside_probability"] = diag.get("outside")
                        record["open_probs"] = diag.get("open")
                    if diag and diag.get("reason"):
                        record["fallback_reason"] = diag["reason"]
                        record["fast_path"] = False
                    n_ok += 1
            else:
                record["predicted"] = False
                if status == 200:
                    record["error"] = "no answer for %r (the reply did not parse)" % req["key"]
                else:
                    record["error"] = json.dumps(body.get("error", {}), ensure_ascii=False)[:300]
                n_fail += 1
            flog.write(json.dumps(record, ensure_ascii=False) + "\n")
            flog.flush()
            if n % 25 == 0 or n == len(requests):
                el = time.perf_counter() - t_start
                print("[jevbench_rewrites] %d/%d requests ok=%d fail=%d elapsed=%.0fs"
                      % (n, len(requests), n_ok, n_fail, el), flush=True)
    return n_ok, n_fail


# ------------------------------------------------------------------ the report

def report(rows: list[dict], dists: dict[str, dict[str, float]], draws: int = 1000) -> dict:
    """The metric, computed offline from the predictions."""
    attach(rows, dists)
    judged = judged_rows(rows)
    res = score_rows(judged)
    res["ci"] = bootstrap(judged, dists, draws=draws)
    res["unjudged"] = len(rows) - len(judged)
    return res


def print_report(res: dict) -> None:
    """The report: per relation, per class, and the overall."""
    print("\nJevBench coherence, %d tests judged (%d unjudged), tolerance %.2f"
          % (res["judged"], res["unjudged"], TOLERANCE))
    for cls in CLASSES:
        print("\n%s" % cls.upper())
        print("  %-28s %-22s %5s %8s %9s %8s" % ("relation", "group", "n", "coverage", "score", "graded"))
        for rel, _ in CLASSES[cls]:
            cell = res["relations"].get(rel)
            if not cell:
                continue
            mark = " (%s)" % cell["excluded"] if cell.get("excluded") else ""
            print("  %-28s %-22s %5d %7.0f%% %9s %8s%s"
                  % (rel, cell["group"], cell["n"], 100 * cell["coverage"],
                     "%.3f" % cell["score"] if cell["score"] is not None else "-",
                     "%.3f" % cell["graded"] if cell["graded"] is not None else "-", mark))
        c = res["classes"][cls]
        print("  %-28s %-22s %5d %8s %9s %8s"
              % ("CLASS", "%d of %d relations scored" % (c["relations"], c["relations_planned"]),
                 c["tests"], "",
                 "%.3f" % c["score"] if c["score"] is not None else "-",
                 "%.3f" % c["graded"] if c["graded"] is not None else "-"))
    ci = res.get("ci") or {}
    if res["overall"] is None:
        print("\noverall: not scored (a class has no relation that met the rules)")
    else:
        print("\noverall %.3f%s"
              % (res["overall"],
                 ("  95%% CI [%.3f, %.3f]" % tuple(ci["overall"])) if "overall" in ci else ""))
    if res.get("graded_overall") is not None:
        print("graded %.3f%s"
              % (res["graded_overall"],
                 ("  95%% CI [%.3f, %.3f]" % tuple(ci["graded_overall"])) if "graded_overall" in ci else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the sample: jevbench_rewrites_sample.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="predictions.jsonl to append")
    ap.add_argument("--log", type=Path, default=None, help="per-request JSONL log (default: <out>.runlog.jsonl)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only this many requests")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="list what would run, send nothing")
    ap.add_argument("--report", action="store_true",
                    help="compute the metric from the predictions file, send nothing")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap draws for the interval")
    args = ap.parse_args()

    rows = load_sample(args.items)
    cases = sorted({r["case_id"] for r in rows})
    relations = sorted({r["relation"] for r in rows})
    requests = plan_requests(rows)

    if args.report:
        if not args.out.exists():
            print("no predictions file at %s: nothing to score" % args.out, file=sys.stderr)
            return 1
        dists = read_predictions(args.out)
        res = report(rows, dists, draws=args.bootstrap)
        print_report(res)
        return 0

    cfg = load_config(args.config)
    port = args.port or cfg["port"]
    key = args.api_key or cfg["api_key"] or os.environ.get("STRATA_API_KEY", "")
    url = "http://%s:%d/v1/systemone" % (args.host, port)
    if not key:
        print("no API key: pass --api-key or set STRATA_API_KEY", file=sys.stderr)
        return 2

    done = set(read_predictions(args.out))
    todo = [r for r in requests if r["id"] not in done]
    per_class: dict[str, int] = defaultdict(int)
    for row in rows:
        per_class[RELATION_CLASS.get(row["relation"], "?")] += 1
    print("[jevbench_rewrites] %d cases, %d relations, %d tests, %d requests "
          "(+%d already done), %s" % (len(cases), len(relations), len(rows),
                                      len(requests), len(done), url), flush=True)
    print("  tests per class: %s" % ", ".join("%s %d" % (c, per_class[c]) for c in CLASSES
                                               if per_class.get(c)))
    if args.dry_run:
        per_rel: dict[str, int] = defaultdict(int)
        for row in rows:
            per_rel[row["relation"]] += 1
        print("  tests per relation: %s"
              % ", ".join("%s %d" % (r, per_rel[r]) for r in relations))
        print("  %d requests to send, one question each" % len(todo))
        for req in todo[:10]:
            print("  %-22s %-6s %s" % (req["key"], req["question"]["type"],
                                       ",".join(sorted(req["roles"]))))
        if len(todo) > 10:
            print("  ... %d more" % (len(todo) - 10))
        return 0

    if args.limit:
        todo = todo[:args.limit]
    log_path = args.log or args.out.with_suffix(".runlog.jsonl")
    n_ok, n_fail = run_requests(todo, url, key, args.timeout, args.out, log_path)
    print("[jevbench_rewrites] done: ok=%d fail=%d, predictions=%s" % (n_ok, n_fail, args.out))
    print("  the metric: python %s --items %s --out %s --report"
          % (Path(__file__).name, args.items, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
