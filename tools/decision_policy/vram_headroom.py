"""tools/decision_policy/vram_headroom.py - how much VRAM was left during a run.

The engine dies when a verify window does not fit: under WDDM the driver then pages
GPU memory to system RAM and a window may never finish (#2065).  It warns at every
request below 1024 MiB free and says what to do about it.  This scans the engine log
for the free-VRAM figures and reports the minimum, so the conversation-cache slots and
the VRAM reserve can be tuned against a measurement instead of a guess.

    python tools/decision_policy/vram_headroom.py
    python tools/decision_policy/vram_headroom.py --since-line 6468
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_LOG = Path(r"C:\tools\strata\strata-unsloth-ud-iq4_xs.log")
VRAM = re.compile(r"(\d+) MiB of VRAM free")
START = re.compile(r"\[strata\].*engine started")
PARK = re.compile(r"parked=(\d+) bytes=(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--since-line", type=int, default=0,
                    help="only consider log lines after this one (default: the last engine start)")
    ap.add_argument("--last-start", action="store_true",
                    help="only consider the last engine start (the default)")
    args = ap.parse_args()

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    start = args.since_line
    if args.since_line == 0:
        for i, ln in enumerate(lines):
            if START.search(ln):
                start = i + 1
    if start >= len(lines):
        print("no engine log after the start line")
        return 1

    samples = []          # (line number, MiB free)
    for i in range(start, len(lines)):
        m = VRAM.search(lines[i])
        if m:
            samples.append((i + 1, int(m.group(1))))

    if not samples:
        print("no VRAM figures in the log after that point")
        return 1

    values = [v for _, v in samples]
    lo = min(samples, key=lambda t: t[1])
    hi = max(samples, key=lambda t: t[1])
    below = [(n, v) for n, v in samples if v < 1024]

    print(f"log lines considered: {start + 1}..{len(lines)}  ({len(lines) - start})")
    print(f"VRAM free samples   : {len(samples)}")
    print(f"minimum             : {lo[1]} MiB   (line {lo[0]})")
    print(f"maximum             : {hi[1]} MiB   (line {hi[0]})")
    print(f"at load             : {values[0]} MiB")
    print(f"final               : {values[-1]} MiB")
    print(f"below the 1024 MiB warning threshold: {len(below)} samples"
          + (f"  (lowest {min(v for _, v in below)} MiB)" if below else ""))

    # the tail: where the headroom was actually spent
    print("\nthe last 12 readings:")
    for n, v in samples[-12:]:
        print(f"  line {n}: {v} MiB")

    # the conversation cache's footprint
    parks = []
    for i in range(start, len(lines)):
        m = PARK.search(lines[i])
        if m:
            parks.append((i + 1, int(m.group(1)), int(m.group(2))))
    if parks:
        print(f"\nconversation cache: {len(parks)} park events, "
              f"slots seen {max(p[1] for p in parks)}, "
              f"largest snapshot {max(p[2] for p in parks) / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
