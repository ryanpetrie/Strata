#!/usr/bin/env python3
"""tools/decision_policy/retemperature.py - re-scale a run's distributions.

A run's predictions file holds the CLOSED reading at the temperature the server
was serving with.  This script re-scales it: it undoes the served temperature and
applies another one, and writes a predictions file the official scorer reads.

    python tools/decision_policy/retemperature.py --items <items.jsonl> \
        --predictions <run.jsonl> --served 0.62,0.67,0.77 --to 0.55,0.70,0.80 \
        --out <new.jsonl> [--runlog <run.runlog.jsonl>]

WHY IT EXISTS.  The temperature is a server-side setting, so changing it means
restarting the server - and a restarted engine is a different engine, and this
sample's run-to-run spread is about 0.6 Capability points.  An unpaired
comparison of two runs (one with the old constants, one with the new) is
therefore not decisive.  This script makes the comparison PAIRED: the two arms
come out of ONE run, so they share the engine, the picks, the logits and the
state, and the only thing that differs is the arithmetic.  The 1.1 measurement
did the same thing by hand (the "temperature undone" arm was the same run with
normalize(closed_T ** T) put back).

The transform is the server's own: serve/logits_first_token.py::apply_temperature
is called, not re-implemented.  The inverse of "apply T" is "apply 1/T", because
f_T(p) = normalize(p ** (1/T)) and f_T^-1(q) = normalize(q ** T) = f_{1/T}(q).

The open reading (the reported confidence) is NOT reconstructed: it is the closed
reading at T scaled by the mass on the options, measured at T = 1, and the mass
is not in the predictions file.  The scorer reads the closed reading, so the
scorer's numbers are exact; a run log written with --runlog keeps the original
open reading, which is the one field this script cannot rebuild.

Pure standard library.  Nothing is written inside the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.logits_first_token import apply_temperature as _server_apply_temperature  # noqa: E402

TYPES = ("choice", "noul", "score")


def read_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_temperatures(text: str) -> dict[str, float]:
    """'choice=0.62,noul=0.67,score=0.77' or '0.62,0.67,0.77' -> {type: T}."""
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, value = part.split("=", 1)
            name = name.strip()
        else:
            name, value = None, part
        try:
            t = float(value)
        except ValueError:
            raise SystemExit("not a temperature: %r" % part)
        if t <= 0.0:
            raise SystemExit("a temperature must be above 0: %r" % part)
        if name is None:
            types = [t for t in TYPES if t not in out]
            if not types:
                raise SystemExit("more temperatures than question types: %r" % text)
            out[types[0]] = t
        else:
            if name not in TYPES:
                raise SystemExit("not a question type: %r" % name)
            out[name] = t
    return out


def scale_closed(closed: dict[str, float], temperature: float) -> dict[str, float]:
    """The closed reading at temperature T, as the server computes it."""
    reading = {"closed": closed, "open": closed}
    return _server_apply_temperature(reading, temperature)["closed"]


def undo(closed: dict[str, float], temperature: float) -> dict[str, float]:
    """The closed reading at T = 1, from the closed reading at T.

    The inverse of "apply T" is "apply 1/T" (see the module docstring).
    """
    if temperature == 1.0:
        return dict(closed)
    return scale_closed(closed, 1.0 / temperature)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="the scorer-shaped items file")
    ap.add_argument("--predictions", type=Path, required=True, help="the run's predictions")
    ap.add_argument("--served", required=True,
                    help="the temperature the run was served with: choice=0.62,noul=0.67,score=0.77")
    ap.add_argument("--to", required=True, dest="target",
                    help="the temperature to apply: the same form")
    ap.add_argument("--out", type=Path, required=True, help="the predictions file to write")
    ap.add_argument("--runlog", type=Path, default=None,
                    help="the run's run log; it is copied with the new probabilities")
    args = ap.parse_args()

    served = parse_temperatures(args.served)
    target = parse_temperatures(args.target)
    items = {it["id"]: it for it in read_jsonl(args.items)}
    predictions = read_jsonl(args.predictions)
    runlog = {r["id"]: r for r in read_jsonl(args.runlog)} if args.runlog else {}

    out_rows = []
    arms = []
    for pred in predictions:
        iid = pred["id"]
        task = items.get(iid)
        if task is None:
            raise SystemExit("the item %r is not in the items file" % iid)
        qtype = task["question"]["type"]
        probs = pred.get("probs") or {}
        if qtype == "noul":
            p = float(probs["yes"])
            closed = {"yes": p, "no": 1.0 - p}
        else:
            closed = {lab: float(probs.get(lab, 0.0)) for lab in task["labels"]}
        at_one = undo(closed, served.get(qtype, 1.0))
        scaled = scale_closed(at_one, target.get(qtype, 1.0))
        if qtype == "noul":
            out_rows.append({"id": iid, "probs": {"yes": scaled["yes"]}})
        else:
            out_rows.append({"id": iid, "probs": {lab: scaled[lab] for lab in task["labels"]}})
        arms.append((iid, task, scaled))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
                        encoding="utf-8")
    print("[retemperature] wrote %d items to %s" % (len(out_rows), args.out))
    print("[retemperature] served %s -> %s" % (served, target))

    if args.runlog:
        log_rows = []
        for iid, task, scaled in arms:
            row = dict(runlog.get(iid, {"id": iid}))
            row["id"] = iid
            qtype = task["question"]["type"]
            row["probabilities"] = ({"yes": scaled["yes"]} if qtype == "noul"
                                    else {lab: scaled[lab] for lab in task["labels"]})
            original_open = runlog.get(iid, {}).get("open_probs")
            if original_open:
                # the mass the model put on the options, measured at T = 1: the open
                # reading is the closed reading at T scaled by it (the server's own
                # formula), and the closed reading sums to 1
                mass = min(1.0, max(0.0, sum(float(v) for v in original_open.values())))
                row["open_probs"] = {k: scaled[k] * mass for k in scaled}
                pick = max(scaled, key=lambda k: scaled[k])
                row["closed_confidence"] = scaled[pick]
                row["open_confidence"] = row["open_probs"][pick]
            log_rows.append(row)
        out_log = args.runlog.with_suffix(args.runlog.suffix + ".retemp")
        out_log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log_rows),
                           encoding="utf-8")
        print("[retemperature] wrote %d run log rows to %s" % (len(log_rows), out_log))
        print("[retemperature] the open reading is rebuilt from the mass on the options "
              "(the server's own formula); the mass outside the options is not in the "
              "predictions file and is copied as it was")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
