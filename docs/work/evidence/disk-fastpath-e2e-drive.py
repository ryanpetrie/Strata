"""disk-fastpath-e2e-drive.py - the Phase 3 fast path through the live serve, two conversations.

Against an idle serve started with --conversation-cache-disk-only --conversation-cache-disk-cow
--conversation-cache-disk-fast-path and a device-resident K/V (no --kv / --kv-resident: the flag's help says 0 means
"all of it in VRAM").  One conversation stays live, so a park (and a restore) needs the other one:

  T1  A cold                                      (A live)
  T2  B cold                                      parks A to the disk tier (B live)
  T3  A continued: assistant(A's reply) + a new user turn
                                                  parks B, then restores A - the log must say path=directstorage
  T4  B continued                                 parks A, restores B - a second fast restore
  ---- three data bytes of three of A's pages are flipped (hashes taken from A's manifest) ----
  T5  A continued again                           the restore must fail closed, the prompt be re-read, and the
                                                  request must still answer
  ---- the three bytes are put back ----
  T6  B continued                                 parks A, restores B
  T7  the same messages as T5                     a cold read of the same prompt: its reply must equal T5's

It prints a JSON report (and writes it with --output) holding the replies, per-turn spans of the engine log
(restore/spill lines, the prompt's reused/read split, STATE_HASH, failures), and metric deltas.

  python docs/work/evidence/disk-fastpath-e2e-drive.py --config docs/work/evidence/disk-fastpath-e2e-config.json
"""
import argparse
import json
import pathlib
import re
import struct
import sys
import time
import urllib.request

FILLER = ("the quick brown fox jumps over the lazy dog near the river bank while the morning light "
          "spreads slowly across the quiet valley and the distant hills fade into a pale blue sky")
A_ASKS = [
    "Summarize the order of the lines above in five words.",
    "Now state how many numbered lines appear above, as one number.",
    "Name the animal that appears in every line above, in one word.",
    "Give the color of the sky mentioned in the lines above, in one word.",
]
B_ASKS = ["Reply with the word ready.", "Reply with the word busy.", "Reply with the word idle."]


def a_messages(lines):
    text = "\n".join(f"{i:04d}: {FILLER}" for i in range(1, lines + 1))
    return [{"role": "system", "content": "You are a test harness answering briefly."},
            {"role": "user", "content": text + "\n\n" + A_ASKS[0]}]


def b_messages():
    return [{"role": "system", "content": "You are a small helper."},
            {"role": "user", "content": "Reply with the word ready."}]


class Client:
    def __init__(self, base, key, model):
        self.base = base.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = "Bearer " + key

    def get(self, path, timeout=60):
        request = urllib.request.Request(self.base + path, headers=self.headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def chat(self, messages, max_tokens=8):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers=self.headers)
        started = time.time()
        with urllib.request.urlopen(request, timeout=1800) as response:
            reply = json.load(response)
        return reply, time.time() - started


def wait_idle(client, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            metrics = client.get("/metrics")
        except Exception:
            time.sleep(1.0)
            continue
        live = metrics.get("live") or {}
        if live.get("state") == "idle" and not live.get("queued"):
            return metrics
        time.sleep(0.5)
    raise RuntimeError("the server did not become idle in time")


def index_entries(pool_dir):
    """pages.idx: magic(8) version(4) generation(8) count(8), count x {hash u64, bytes u32, pad u32, offset u64},
    then an 8-byte checksum.  Returns [(hash, bytes, offset)]."""
    raw = (pathlib.Path(pool_dir) / "pool" / "pages.idx").read_bytes()
    if len(raw) < 36 or raw[:8] != b"STRPIDX1":
        raise RuntimeError("pages.idx is not readable")
    count = struct.unpack_from("<Q", raw, 20)[0]
    return [struct.unpack_from("<QIIQ", raw, 28 + i * 24) for i in range(count)]


def manifest_pages(pool_dir):
    """The newest *.bin under manifests/ belongs to the conversation parked last (A, before its restore request);
    the pages it names are the hashes in pages.idx that its bytes contain (a manifest lists hash and length per
    page)."""
    manifests = sorted((pathlib.Path(pool_dir) / "manifests").glob("*.bin"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise RuntimeError("no manifest was written")
    newest = manifests[-1]
    manifest = newest.read_bytes()
    pages = []
    for hash_, bytes_, _pad, offset in index_entries(pool_dir):
        if struct.pack("<Q", hash_) in manifest:
            pages.append((int(hash_), int(bytes_), int(offset)))
    if not pages:
        raise RuntimeError("the manifest names no page of the pool index")
    print(f"manifest {newest.name}: {len(pages)} pages of the pool")
    return pages


def flip(page_bin, offset):
    with open(page_bin, "r+b") as f:
        f.seek(offset)
        old = f.read(1)[0]
        f.seek(offset)
        f.write(bytes([old ^ 0x01]))
    return old


def put_back(page_bin, offset, old):
    with open(page_bin, "r+b") as f:
        f.seek(offset)
        f.write(bytes([old]))


RESTORE = re.compile(r"restored (\d+) tokens from disk in ([\d.]+) ms \(([\d.]+) MiB/s\); tier=disk(.*)$")
SPILLED = re.compile(r"spilled (\d+) tokens to disk in ([\d.]+) ms;(.*)$")
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                    r"(\d+) generated")
STATE_HASH = re.compile(r"STATE_HASH L=(\d+) .*?kv=([0-9a-f]+)")


def slice_log(log_path, start):
    with open(log_path, "rb") as f:
        f.seek(start)
        data = f.read()
    return data.decode("utf-8", "replace"), start + len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="the run config the server was started with")
    parser.add_argument("--lines", type=int, default=240, help="filler lines in conversation A's prompt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text())
    base = f"http://{config.get('host', '127.0.0.1')}:{config.get('port', 8095)}"
    client = Client(base, config.get("api_key"), config["model_name"])
    log_path = config["log"]
    pool_dir = None
    for i, a in enumerate(config.get("args", [])):
        if a == "--conversation-cache-spill-dir":
            pool_dir = config["args"][i + 1]
    if pool_dir is None:
        raise SystemExit("the config names no --conversation-cache-spill-dir")
    pool_bin = str(pathlib.Path(pool_dir) / "pool" / "pages.bin")

    metrics = wait_idle(client)
    info = metrics.get("engine") or {}
    report = {"config": args.config, "base": base,
              "engine": {k: info.get(k) for k in ("conversation_disk_fast_path", "conversation_disk_paged",
                                                  "conversation_disk_mib", "conversation_cache_mib")}}
    print("engine:", report["engine"])
    if str(report["engine"].get("conversation_disk_fast_path")) not in ("1", "True"):
        print("WARNING: the INFO line does not report the fast path on", file=sys.stderr)

    log_pos = pathlib.Path(log_path).stat().st_size
    turns = []
    conv_a = a_messages(args.lines)
    conv_b = b_messages()
    last = {}   # conversation -> its latest reply

    def turn(name, conversation, expect, key=None):
        nonlocal log_pos
        before = client.get("/metrics")
        reply, wall = client.chat(conversation)
        text = reply["choices"][0]["message"].get("content") or ""
        after = wait_idle(client)
        time.sleep(1.5)   # let the park's own log line land before the slice
        log, end = slice_log(log_path, log_pos)
        log_pos = end
        entry = {"turn": name, "expect": expect, "wall_s": round(wall, 3), "reply": text,
                 "usage": reply.get("usage"), "cache_lines":
                     [l.strip() for l in log.splitlines()
                      if "conversation cache" in l or "STATE_HASH" in l or "fast path" in l or
                      re.search(r": prompt \d+ tokens", l)]}
        cache_before = before.get("conversation_cache") or {}
        cache_after = after.get("conversation_cache") or {}
        entry["metric_delta"] = {k: (cache_after.get(k) or 0) - (cache_before.get(k) or 0)
                                 for k in ("disk_saves", "disk_restores", "disk_errors", "disk_refusals")}
        lines = entry["cache_lines"]
        entry["restore_line"] = next((m.group(0) for m in map(RESTORE.search, lines) if m), None)
        entry["spilled_line"] = next((m.group(0) for m in map(SPILLED.search, lines) if m), None)
        m = next((m for m in map(PROMPT.search, lines) if m), None)
        entry["prompt"] = ({"tokens": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)),
                            "prefill_ms": int(m.group(4)), "generated": int(m.group(6))} if m else None)
        entry["state_hashes"] = [{"L": int(m.group(1)), "kv": m.group(2)}
                                 for m in map(STATE_HASH.search, lines) if m]
        entry["failures"] = [l for l in lines if "failed" in l.lower() or "refus" in l.lower()]
        turns.append(entry)
        print(f"\n--- {name}: {wall:.2f}s  ({expect})")
        for l in lines:
            if "restored" in l or "spilled" in l or "failed" in l.lower() or "refus" in l.lower() or ": prompt" in l:
                print("    log:", l[:200])
        print("    reply:", text[:120].replace("\n", " "))
        sys.stdout.flush()
        return entry

    def extend(conversation, ask, key):
        conversation.append({"role": "assistant", "content": last.get(key, "")})
        conversation.append({"role": "user", "content": ask})

    t1 = turn("A-cold", conv_a, "cold read; A parks when B arrives", "A")
    last["A"] = t1["reply"]
    t2 = turn("B-cold", conv_b, "cold read; parks A to the disk tier", "B")
    last["B"] = t2["reply"]
    extend(conv_a, A_ASKS[1], "A")
    t3 = turn("A-restore", conv_a, "restore of A, path=directstorage", "A")
    last["A"] = t3["reply"]
    extend(conv_b, B_ASKS[1], "B")
    t4 = turn("B-restore", conv_b, "restore of B, path=directstorage", "B")
    last["B"] = t4["reply"]

    pages = manifest_pages(pool_dir)
    chosen = [pages[len(pages) // 4], pages[len(pages) // 2], pages[(3 * len(pages)) // 4]]
    flipped = []
    for hash_, bytes_, offset in chosen:
        at = offset + bytes_ // 2
        flipped.append((at, flip(pool_bin, at)))
    print(f"\nflipped {len(flipped)} of A's pages of {len(pages)} in the manifest: "
          + ", ".join(str(a) for a, _ in flipped))
    extend(conv_a, A_ASKS[2], "A")
    t5 = turn("A-corrupt", conv_a, "the restore must fail closed and the prompt be re-read", "A")
    last["A"] = t5["reply"]
    for at, old in flipped:
        put_back(pool_bin, at, old)
    print("page bytes restored")
    extend(conv_b, B_ASKS[2], "B")
    t6 = turn("B-restore-2", conv_b, "restore of B (parks A)", "B")
    last["B"] = t6["reply"]

    log_pos = pathlib.Path(log_path).stat().st_size
    t7 = turn("A-repeat", list(conv_a), "a cold read of the same messages as A-corrupt")
    report["turns"] = turns
    report["flipped_bytes"] = [at for at, _ in flipped]
    report["fail_closed_reply_matches_cold"] = t5["reply"] == t7["reply"]
    print("\nA-corrupt reply equals A-repeat reply:", report["fail_closed_reply_matches_cold"])
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("report written to", args.output)


if __name__ == "__main__":
    main()
