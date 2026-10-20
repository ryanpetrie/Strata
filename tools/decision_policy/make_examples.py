#!/usr/bin/env python3
"""tools/decision_policy/make_examples.py - build the frozen decision example set.

This research task is an EXPLORATION. It does not change any default behaviour, and it does not train a policy
against real user trajectories. It measures whether a small decision prompt is fast and accurate enough to be
worth a hidden-state classifier later.

WHAT THE LABELS ARE. Every record's ``target_action`` is produced by ``decide`` below, a rule written out in plain
Python and applied to the same state the model sees. These are therefore SYNTHETIC, RULE-BASED labels, not reviewed
real agent trajectories. That is stated in every record's ``label_source`` and in the report. Real labels are
Phase 2's job; until then nothing here may be read as a learned-policy quality claim.

WHY A RULE AND NOT HAND LABELS. A hand label that disagrees with the rule would make the accuracy number
meaningless, and a rule that disagrees with its own labels is a bug. So the generator ASSERTS that the rule and the
stated target agree, and refuses to write a set where they do not. Change the rule, re-run, and the set changes
with it - the set is a function of the rule, not a pile of independent guesses.

The action vocabulary is the one the handoff suggests, kept small and stable:
    CONTINUE, CALL_TOOL, ANSWER, ASK_USER, STOP

The state fields are the handoff's, plus three that make the rule decidable:
    goal                      what the agent is trying to do (context only)
    last_action               the previous step (a tool name, or CONTINUE/COMPOSE)
    new_information_available  a tool result added something new
    answer_confidence         low | medium | high
    steps_remaining           steps left before the budget is spent
    available_tools           the tools that are usable right now
    last_tool_succeeded       the previous tool call worked
    evidence_sufficient       enough evidence to answer confidently
    user_intent_ambiguous     the request is under-specified
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "examples.jsonl"

ACTIONS = ["CONTINUE", "CALL_TOOL", "ANSWER", "ASK_USER", "STOP"]


def decide(state: dict) -> str:
    """The documented policy, in priority order. This is the label rule, not a model."""
    if int(state["steps_remaining"]) <= 0:
        return "STOP"
    if bool(state["user_intent_ambiguous"]):
        return "ASK_USER"
    if state["answer_confidence"] == "high" and bool(state["evidence_sufficient"]):
        return "ANSWER"
    tools = state["available_tools"]
    if tools and (bool(state["new_information_available"]) or not bool(state["last_tool_succeeded"])):
        return "CALL_TOOL"
    return "CONTINUE"


def S(goal, last_action, new_info, confidence, steps, tools, tool_ok, sufficient, ambiguous):
    return {"goal": goal, "last_action": last_action, "new_information_available": bool(new_info),
            "answer_confidence": confidence, "steps_remaining": int(steps),
            "available_tools": list(tools), "last_tool_succeeded": bool(tool_ok),
            "evidence_sufficient": bool(sufficient), "user_intent_ambiguous": bool(ambiguous)}


# (id, state, intended target, note). The intended target is checked against decide(state) below.
SCENARIOS = [
    # ---- ANSWER: high confidence and the evidence is already sufficient ----
    ("ex-0001", S("Summarize the API documentation", "READ_DOCUMENT", True, "high", 2, ["SEARCH", "READ_DOCUMENT"], True, True, False), "ANSWER", "the document is read and answers the goal"),
    ("ex-0002", S("Answer what version is installed", "SEARCH", True, "high", 1, ["SEARCH"], True, True, False), "ANSWER", "search found the version"),
    ("ex-0003", S("Produce the final changelog", "COMPOSE", False, "high", 1, [], True, True, False), "ANSWER", "drafting is done, nothing more is needed"),
    ("ex-0004", S("Confirm the test results", "RUN_TESTS", True, "high", 2, ["RUN_TESTS"], True, True, False), "ANSWER", "tests passed and the result is known"),
    ("ex-0005", S("Explain the error cause", "READ_LOG", True, "high", 1, ["READ_LOG"], True, True, False), "ANSWER", "the log names the cause"),
    ("ex-0006", S("Report the benchmark numbers", "BENCHMARK", True, "high", 3, ["BENCHMARK"], True, True, False), "ANSWER", "the run finished and printed the numbers"),
    ("ex-0007", S("State the license", "SEARCH", True, "high", 1, ["SEARCH"], True, True, False), "ANSWER", "the license is on the page"),
    ("ex-0008", S("Summarize the meeting notes", "READ_NOTES", True, "high", 2, ["READ_NOTES"], True, True, False), "ANSWER", "the notes are in hand"),
    ("ex-0009", S("Give the release date", "SEARCH", True, "high", 1, ["SEARCH"], True, True, False), "ANSWER", "the date is published"),
    ("ex-0010", S("Write the conclusion section", "COMPOSE", False, "high", 1, [], True, True, False), "ANSWER", "the conclusion is written"),

    # ---- CALL_TOOL: new information, or the last tool failed and can be retried ----
    ("ex-0011", S("Locate the bug", "SEARCH", True, "low", 4, ["SEARCH", "READ_FILE"], True, False, False), "CALL_TOOL", "a hit points at a file to read"),
    ("ex-0012", S("Locate the bug", "READ_FILE", True, "medium", 3, ["SEARCH", "READ_FILE"], True, False, False), "CALL_TOOL", "the file names another to open"),
    ("ex-0013", S("Gather the facts", "SEARCH", True, "medium", 2, ["SEARCH"], True, False, False), "CALL_TOOL", "a result suggests another query"),
    ("ex-0014", S("Check the config", "READ_FILE", False, "medium", 2, ["READ_FILE"], False, False, False), "CALL_TOOL", "the read failed; retry it"),
    ("ex-0015", S("Check the config", "READ_FILE", False, "medium", 2, ["READ_FILE"], False, False, False), "CALL_TOOL", "retry after a transient error"),
    ("ex-0016", S("Fetch the page", "FETCH", False, "low", 3, ["FETCH"], False, False, False), "CALL_TOOL", "the fetch timed out"),
    ("ex-0017", S("Run the query", "QUERY", True, "medium", 2, ["QUERY"], True, False, False), "CALL_TOOL", "the query returned a new fact"),
    ("ex-0018", S("Inspect the log", "READ_LOG", True, "medium", 3, ["READ_LOG", "SEARCH"], True, False, False), "CALL_TOOL", "the log points at another entry"),
    ("ex-0019", S("Compare versions", "SEARCH", True, "medium", 2, ["SEARCH"], True, False, False), "CALL_TOOL", "one version's page links another"),
    ("ex-0020", S("Read the spec", "SEARCH", True, "low", 4, ["SEARCH", "READ_DOCUMENT"], True, False, False), "CALL_TOOL", "a section is still unread"),
    ("ex-0021", S("Read the spec", "READ_DOCUMENT", True, "medium", 3, ["SEARCH", "READ_DOCUMENT"], True, False, False), "CALL_TOOL", "the document continues"),
    ("ex-0022", S("Validate the input", "VALIDATE", True, "medium", 2, ["VALIDATE"], True, False, False), "CALL_TOOL", "validation found a field to check"),
    ("ex-0023", S("Validate the input", "VALIDATE", False, "medium", 2, ["VALIDATE"], False, False, False), "CALL_TOOL", "validation failed; retry"),
    ("ex-0024", S("Scan the repo", "SCAN", True, "low", 3, ["SCAN"], True, False, False), "CALL_TOOL", "the scan listed files to open"),
    ("ex-0025", S("Scan the repo", "SCAN", False, "medium", 2, ["SCAN"], False, False, False), "CALL_TOOL", "the scan errored; retry"),
    ("ex-0026", S("Fetch dependencies", "FETCH", True, "medium", 2, ["FETCH"], True, False, False), "CALL_TOOL", "a dependency lists another"),
    ("ex-0027", S("Fetch dependencies", "FETCH", False, "medium", 2, ["FETCH"], False, False, False), "CALL_TOOL", "the fetch failed; retry"),
    ("ex-0028", S("Probe the endpoint", "PROBE", True, "medium", 2, ["PROBE"], True, False, False), "CALL_TOOL", "the probe returned data to use"),
    ("ex-0029", S("Probe the endpoint", "PROBE", False, "medium", 2, ["PROBE"], False, False, False), "CALL_TOOL", "the probe was refused; retry"),
    ("ex-0030", S("Search the docs", "SEARCH", True, "medium", 2, ["SEARCH"], True, False, False), "CALL_TOOL", "the docs are still being searched"),

    # ---- CONTINUE: keep going, but there is nothing new to act on ----
    ("ex-0031", S("Draft the answer", "COMPOSE", False, "medium", 2, [], True, False, False), "CONTINUE", "still drafting"),
    ("ex-0032", S("Draft the answer", "CONTINUE", False, "medium", 2, [], True, False, False), "CONTINUE", "no new input"),
    ("ex-0033", S("Refine the summary", "COMPOSE", False, "low", 3, [], True, False, False), "CONTINUE", "keep refining"),
    ("ex-0034", S("Refine the summary", "CONTINUE", False, "medium", 2, [], True, False, False), "CONTINUE", "no new input"),
    ("ex-0035", S("Polish the text", "COMPOSE", False, "medium", 1, [], True, False, False), "CONTINUE", "one pass left"),
    ("ex-0036", S("Wait for the tool", "CONTINUE", False, "medium", 2, [], True, False, False), "CONTINUE", "nothing to do yet"),
    ("ex-0037", S("Think through the plan", "CONTINUE", False, "low", 3, [], True, False, False), "CONTINUE", "planning continues"),
    ("ex-0038", S("Think through the plan", "CONTINUE", False, "medium", 2, [], True, False, False), "CONTINUE", "planning continues"),
    ("ex-0039", S("Organize the findings", "COMPOSE", False, "medium", 2, [], True, False, False), "CONTINUE", "reorganizing"),
    ("ex-0040", S("Organize the findings", "CONTINUE", False, "low", 3, [], True, False, False), "CONTINUE", "reorganizing"),
    ("ex-0041", S("Review the draft", "COMPOSE", False, "medium", 2, [], True, False, False), "CONTINUE", "still reviewing"),
    ("ex-0042", S("Review the draft", "CONTINUE", False, "medium", 2, [], True, False, False), "CONTINUE", "still reviewing"),

    # ---- ASK_USER: the request is under-specified ----
    ("ex-0043", S("Fix the build", "READ_LOG", True, "low", 3, ["READ_LOG"], True, False, True), "ASK_USER", "which build is meant is unclear"),
    ("ex-0044", S("Update the docs", "SEARCH", True, "medium", 2, ["SEARCH"], True, False, True), "ASK_USER", "which docs are meant is unclear"),
    ("ex-0045", S("Deploy the change", "PROBE", False, "low", 2, ["PROBE"], False, False, True), "ASK_USER", "the target environment is unclear"),
    ("ex-0046", S("Refactor the module", "SCAN", True, "medium", 3, ["SCAN"], True, False, True), "ASK_USER", "the scope of the refactor is unclear"),
    ("ex-0047", S("Answer the question", "SEARCH", True, "low", 2, ["SEARCH"], True, False, True), "ASK_USER", "the question is ambiguous"),
    ("ex-0048", S("Choose the library", "SEARCH", True, "medium", 3, ["SEARCH"], True, False, True), "ASK_USER", "the constraints are unclear"),
    ("ex-0049", S("Set the timeout", "READ_FILE", False, "medium", 2, ["READ_FILE"], True, False, True), "ASK_USER", "which timeout is meant is unclear"),
    ("ex-0050", S("Set the timeout", "READ_FILE", True, "low", 2, ["READ_FILE"], True, False, True), "ASK_USER", "which timeout is meant is unclear"),

    # ---- STOP: the budget is spent, whatever else is true ----
    ("ex-0051", S("Find the relevant documentation", "SEARCH", True, "medium", 0, ["SEARCH"], True, False, False), "STOP", "no steps left"),
    ("ex-0052", S("Find the relevant documentation", "READ_DOCUMENT", True, "high", 0, ["SEARCH", "READ_DOCUMENT"], True, True, False), "STOP", "budget spent even though the answer is ready"),
    ("ex-0053", S("Summarize the API", "SEARCH", True, "high", 0, ["SEARCH"], True, True, False), "STOP", "budget spent"),
    ("ex-0054", S("Answer the question", "CONTINUE", False, "medium", 0, [], True, False, False), "STOP", "budget spent"),
    ("ex-0055", S("Answer the question", "SEARCH", False, "low", 0, ["SEARCH"], False, False, False), "STOP", "budget spent with a failed search"),
    ("ex-0056", S("Write the report", "COMPOSE", False, "medium", 0, [], True, False, False), "STOP", "budget spent"),
    ("ex-0057", S("Write the report", "COMPOSE", False, "high", 0, [], True, True, False), "STOP", "budget spent"),
    ("ex-0058", S("Locate the bug", "READ_FILE", True, "medium", 0, ["READ_FILE"], True, False, False), "STOP", "budget spent"),
    ("ex-0059", S("Locate the bug", "SEARCH", False, "low", 0, ["SEARCH"], False, False, False), "STOP", "budget spent"),
    ("ex-0060", S("Validate the input", "VALIDATE", True, "medium", 0, ["VALIDATE"], True, False, False), "STOP", "budget spent"),
]


def main() -> int:
    bad = []
    records = []
    for rid, state, target, note in SCENARIOS:
        derived = decide(state)
        if derived != target:
            bad.append((rid, target, derived))
        records.append({
            "id": rid,
            "state": state,
            "target_action": target,
            "label_source": "agent_authored_rule",
            "split_group": "synthetic-v1",
            "note": note,
        })
    if bad:
        for rid, target, derived in bad:
            print("MISMATCH %s: stated %s, rule says %s" % (rid, target, derived), file=sys.stderr)
        return 1
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    counts = {a: sum(1 for r in records if r["target_action"] == a) for a in ACTIONS}
    print("wrote %s (%d examples): %s" % (OUT, len(records), counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
