"""tools/decision_policy/lora_spike.py - roadmap item 2.2, step 0: the feasibility spike.

What it measures (docs/research/phase2_engine_lora_plan.md, §4 "Step 0" and §6.1):

  **S0.1 - the engine's run-to-run determinism.**  The engine is started **standalone** (not through
  the server: the server holds the card, and only one engine instance may run), with the launch
  config's own `exe` and `args` plus `--serve`.  It takes one request per line on stdin.  The same
  `GEN` request is sent three times with an unrelated request in between, and the same `LOGITS`
  request twice.  The `T` lines and the `L`/`Z` lines are compared byte for byte.  Two more probes
  tell "not deterministic" apart from "order-dependent": does the ORDER of the requests change the
  answer, and does a chat-shaped request (a growing conversation) between two identical requests
  change it?

  **S0.2 - the adapter's arithmetic.**  `y += s * B @ (A @ x)` in the plan's on-disk format (the
  `index.txt` + `lora.bin` of §2.1) against a numpy host reference on random tensors.  No GPU, no
  model: `--only s0.2` runs it on its own.

Usage:

    python tools/decision_policy/lora_spike.py --config C:\\tools\\strata\\strata-unsloth-ud-iq4_xs.json --only s0.2
    python tools/decision_policy/lora_spike.py --config C:\\tools\\strata\\strata-unsloth-ud-iq4_xs.json --only s0.1 --out s01.json

The engine's log and the results go to `--work` (default: a temp folder).  Nothing is written next to
the model store.

Every number this script prints comes from a command it ran.  Nothing here changes the engine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import numpy as np  # noqa: E402

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import StrataEngine, child_env  # noqa: E402

DEFAULT_CONFIG = Path(r"C:\tools\strata\strata-unsloth-ud-iq4_xs.json")
DEFAULT_WORK = Path(os.environ.get("TEMP", r"C:\Users\ryanp\AppData\Local\Temp")) / "opencode" / "lora_spike"

# The two prompts.  They are fixed text, so the ids are the same in every run.  The first is the
# request under test; the second is the unrelated request that goes between two identical ones.
PROMPT_A = (
    "A calibration record for a small tide gauge. The gauge stands on the north pier of a harbour "
    "and reports the water's height above the chart datum every six minutes. In the first week the "
    "readings rose steadily, because a storm was building in the western ocean, and the operators "
    "watched the trace climb from three metres to nearly five. Then the wind turned, the storm's "
    "centre passed to the north, and the readings fell back towards two metres in a single day. "
    "The instrument's own logbook records a short interruption on the third night, when the "
    "electrical supply to the pier was switched for maintenance, and a brief climb in the "
    "thermometer on the fourth afternoon, which the engineers attributed to a warm current near the "
    "surface. Over the whole month the average height was four metres, and the highest reading was "
    "five and a quarter metres at ten past three in the afternoon of the ninth. The logbook ends "
    "with a note that the gauge would be replaced in the spring, because the old mechanism had "
    "begun to stick in cold weather and the readings had become unreliable after heavy rain. "
    "Please summarise this record in three sentences, naming the highest reading and its date."
)
PROMPT_B = (
    "A recipe for a winter soup that uses what is left in the cupboard. Take two onions, a stick "
    "of celery and a handful of carrots, and cut them small enough that they will soften in twenty "
    "minutes. Fry them in a little butter until the edges take colour, then add a cup of red "
    "lentils, a tin of chopped tomatoes, a bay leaf and enough stock to cover everything by two "
    "fingers. Simmer it for half an hour, until the lentils have collapsed and the soup has turned "
    "the colour of the tomatoes. Finish it with a spoonful of yoghurt or a knob of butter, and "
    "season it at the very end, because the lentils take more salt than you would expect. If the "
    "soup is too thick, loosen it with a little more hot stock; if it is too thin, let it bubble "
    "for another ten minutes without a lid. It keeps for three days in the fridge and improves on "
    "the second day. Serve it with bread, or with rice and a spoonful of yoghurt. Please rewrite "
    "this recipe as a numbered list for someone who has never cooked before."
)


# ---------------------------------------------------------------------------- the tokenizer


def load_tokenizer(pack: Path) -> ST.Tokenizer:
    """The model's own tokenizer, the way serve/server.py builds it (the piece list ordered by id)."""
    vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
    pieces = [None] * len(vocab)
    for token, index in vocab.items():
        pieces[index] = token
    merges = (pack / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((pack / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(pieces, merges, types)


# ---------------------------------------------------------------------------- the VRAM reading


def vram_free_mib() -> int | None:
    """The card's free VRAM right now, from nvidia-smi.  None when nvidia-smi is not there.

    The engine logs its own free-VRAM figure only when it drops below its own warning threshold
    (src/program/generate.cpp:11680-11696), so the per-request figure is read from outside.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------- the engine


class Engine:
    """The engine, started standalone, spoken to over its stdin.

    The shipped StrataEngine does the start-up (the READY handshake, the log file, the INFO line);
    this adds the raw request line and the raw answer lines, because the measurement is about the
    bytes.
    """

    def __init__(self, cfg: dict, log: Path, env_extra: dict | None = None):
        args = list(cfg["args"])                      # the launch config's own arguments
        exe = cfg["exe"]
        cwd = cfg.get("cwd")
        env = child_env(cfg)                         # the server's own environment for the engine
        for k, v in (env_extra or {}).items():
            env[str(k)] = str(v)
        self.cmd = [exe, "--serve", *args]
        self.env_extra = dict(env_extra or {})
        self.engine = StrataEngine(exe, args, cwd=cwd, log=str(log), env=env)
        self.log = log
        self.info = dict(self.engine.info)

    def request(self, line: str, label: str, timeout_s: float = 900.0) -> dict:
        """One request line; the answer's raw lines, byte for byte."""
        vram_before = vram_free_mib()
        t0 = time.monotonic()
        try:
            self.engine.proc.stdin.write(line + "\n")
            self.engine.proc.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"the engine's pipe is gone: {e}") from None
        rec = {"label": label, "request": line, "vram_free_mib_before": vram_before,
               "T": [], "L": [], "Z": None, "DONE": None, "other": []}
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                raw = self.engine.lines.get(timeout=max(0.05, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"{label}: the engine said nothing for {timeout_s:.0f} s") from None
            if raw is None:
                raise RuntimeError(f"{label}: the engine ended (its stdout closed)")
            if raw.startswith("T "):
                rec["T"].append(raw)
            elif raw.startswith("L "):
                rec["L"].append(raw)
            elif raw.startswith("Z "):
                rec["Z"] = raw
            elif raw.startswith("DONE"):
                rec["DONE"] = raw
                break
            elif raw.startswith("ERR"):
                rec["ERR"] = raw
                break
            else:
                rec["other"].append(raw)
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        return rec

    def close(self):
        self.engine.close()


# ---------------------------------------------------------------------------- the comparison


def t_bytes(rec: dict) -> bytes:
    return b"".join(line.encode("utf-8") for line in rec["T"])


def l_bytes(rec: dict) -> bytes:
    return b"".join(line.encode("utf-8") for line in rec["L"]) + (rec["Z"] or "").encode("utf-8")


def parse_l(rec: dict) -> list[tuple[int, float]]:
    rows = []
    for line in rec["L"]:
        f = line.split()
        if len(f) >= 3:
            rows.append((int(f[1]), float(f[2])))
    return rows


def parse_z(rec: dict) -> float | None:
    if not rec["Z"]:
        return None
    try:
        return float(rec["Z"].split()[1])
    except (IndexError, ValueError):
        return None


def compare(a: dict, b: dict) -> dict:
    """The byte comparison of two answers, and (for LOGITS) the numeric difference."""
    out = {
        "a": a["label"], "b": b["label"],
        "T_bytes_equal": t_bytes(a) == t_bytes(b),
        "T_a": a["T"], "T_b": b["T"],
        "L_bytes_equal": l_bytes(a) == l_bytes(b),
        "Z_a": a["Z"], "Z_b": b["Z"],
    }
    if a["L"] or b["L"]:
        ra, rb = parse_l(a), parse_l(b)
        out["L_rows"] = len(ra)
        if len(ra) == len(rb):
            diffs = [abs(x[1] - y[1]) for x, y in zip(ra, rb)]
            ids_equal = [x[0] for x in ra] == [y[0] for y in rb]
            out["L_ids_equal"] = ids_equal
            out["L_max_abs_diff"] = max(diffs) if diffs else 0.0
            out["L_n_rows_differing"] = sum(1 for d in diffs if d != 0.0)
            out["L_max_rel_diff"] = (out["L_max_abs_diff"] / max(abs(x[1]) for x in ra)) if ra else 0.0
        else:
            out["L_rows_a"], out["L_rows_b"] = len(ra), len(rb)
        za, zb = parse_z(a), parse_z(b)
        out["Z_abs_diff"] = abs(za - zb) if (za is not None and zb is not None) else None
        out["numeric"] = logits_numeric_diff(a, b)
    return out


def done_fields(rec: dict) -> dict:
    """The engine's DONE line, parsed: DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish>
    <drafts accepted> <drafts offered> <reused> <hits> <lookups> <ram blobs> <file blobs> <file MB>
    <prompt tokens read> <offloaded> ..."""
    line = rec.get("DONE") or ""
    f = line.split()
    names = ("generated", "prompt", "prompt_ms", "decode_ms", "finish", "drafts_accepted", "drafts_offered",
             "reused", "hits", "lookups", "ram_blobs", "file_blobs", "file_mb", "prompt_tokens_read", "offloaded")
    out = {}
    for i, name in enumerate(names):
        if i < len(f) - 1:
            try:
                out[name] = float(f[i + 1]) if "." in f[i + 1] else int(f[i + 1])
            except ValueError:
                out[name] = f[i + 1]
    return out


def logits_numeric_diff(a: dict, b: dict) -> dict:
    """Two LOGITS rows compared token by token (not rank by rank): how far the two distributions moved."""
    ra = dict(parse_l(a))
    rb = dict(parse_l(b))
    za, zb = parse_z(a), parse_z(b)
    common = sorted(set(ra) & set(rb))
    out = {
        "rows_a": len(ra), "rows_b": len(rb),
        "ids_in_both": len(common),
        "ids_only_in_a": len(set(ra) - set(rb)),
        "ids_only_in_b": len(set(rb) - set(ra)),
        "top1_a": max(ra, key=ra.get) if ra else None,
        "top1_b": max(rb, key=rb.get) if rb else None,
        "dZ": (zb - za) if (za is not None and zb is not None) else None,
    }
    if common:
        diffs = [abs(ra[i] - rb[i]) for i in common]
        out["max_abs_dlogit"] = max(diffs)
        out["mean_abs_dlogit"] = sum(diffs) / len(diffs)
    if ra and rb and za is not None and zb is not None:
        ids = sorted(set(ra) | set(rb))
        pa = [math.exp(ra.get(i, -1e30) - za) for i in ids]
        pb = [math.exp(rb.get(i, -1e30) - zb) for i in ids]
        out["top1_prob_a"] = math.exp(ra[max(ra, key=ra.get)] - za)
        out["top1_prob_b"] = math.exp(rb[max(rb, key=rb.get)] - zb)
        # the total variation distance over the union of the two top-k rows; each side's missing
        # mass (below its own cut) is counted as zero, so this is an upper bound
        out["tv_upper_bound"] = 0.5 * sum(abs(p - q) for p, q in zip(pa, pb))
        out["missing_mass_a"] = max(0.0, 1.0 - sum(pa))
        out["missing_mass_b"] = max(0.0, 1.0 - sum(pb))
    return out


# ---------------------------------------------------------------------------- S0.1


def s0_1(cfg: dict, work: Path, max_new: int, logits_k: int, out_json: Path | None) -> dict:
    pack = Path(cfg["tokenizer"])
    tok = load_tokenizer(pack)
    tpl = ChatTemplate(pack / "chat_template.jinja")

    def chat(messages) -> list[int]:
        return tok.encode(tpl.render(messages), parse_special=True)

    a_ids = chat([{"role": "user", "content": PROMPT_A}])
    b_ids = chat([{"role": "user", "content": PROMPT_B}])
    print(f"prompt A: {len(a_ids)} tokens   prompt B: {len(b_ids)} tokens")

    work.mkdir(parents=True, exist_ok=True)
    log = work / "engine-s01.log"
    engine = Engine(cfg, log)
    print(f"engine started: {' '.join(engine.cmd)}")
    print(f"engine log: {log}")

    ids = ",".join(str(t) for t in a_ids)
    ids_b = ",".join(str(t) for t in b_ids)
    gen_line = f"GEN {max_new} ckpt=0 {ids}"
    log_line = f"LOGITS {logits_k} {max_new} ckpt=0 {ids}"

    records: list[dict] = []
    try:
        # 1. the request under test
        records.append(engine.request(gen_line, "gen_a1"))
        # 2. an unrelated request between two identical ones
        records.append(engine.request(f"GEN {max_new} ckpt=0 {ids_b}", "gen_b1"))
        # 3. the request under test again
        records.append(engine.request(gen_line, "gen_a2"))
        # 4. the request under test again, with nothing between it and the last one
        records.append(engine.request(gen_line, "gen_a3"))
        # 5. another unrelated request
        records.append(engine.request(f"GEN {max_new} ckpt=0 {ids_b}", "gen_b2"))
        # 6. the request under test a third time (the order probe)
        records.append(engine.request(gen_line, "gen_a4"))
        # 7-8. the LOGITS request, twice
        records.append(engine.request(log_line, "logits_a1"))
        records.append(engine.request(log_line, "logits_a2"))
        # 9. a chat-shaped request: the same conversation, one turn longer
        reply = tok.decode([int(line.split()[1]) for line in records[0]["T"]], errors="replace")
        conv_ids = chat([{"role": "user", "content": PROMPT_A},
                         {"role": "assistant", "content": reply},
                         {"role": "user", "content": "Now name the two dates the logbook mentions, "
                                                    "and say which reading they belong to."}])
        print(f"chat-shaped prompt: {len(conv_ids)} tokens")
        records.append(engine.request(f"GEN {max_new} ckpt=0 " + ",".join(str(t) for t in conv_ids),
                                      "gen_chat"))
        # 10-11. the same two requests again, after the conversation
        records.append(engine.request(gen_line, "gen_a5"))
        records.append(engine.request(log_line, "logits_a3"))
    finally:
        engine.close()

    by_label = {r["label"]: r for r in records}
    comparisons = {
        # the same GEN request, with an unrelated request in between
        "gen_a1_vs_gen_a2 (an unrelated request between)": compare(by_label["gen_a1"], by_label["gen_a2"]),
        "gen_a2_vs_gen_a4 (an unrelated request between)": compare(by_label["gen_a2"], by_label["gen_a4"]),
        "gen_a1_vs_gen_a4 (two unrelated requests between)": compare(by_label["gen_a1"], by_label["gen_a4"]),
        # the same GEN request, back to back
        "gen_a2_vs_gen_a3 (nothing between)": compare(by_label["gen_a2"], by_label["gen_a3"]),
        # the same LOGITS request, twice
        "logits_a1_vs_logits_a2 (nothing between)": compare(by_label["logits_a1"], by_label["logits_a2"]),
        # after a chat-shaped request
        "gen_a1_vs_gen_a5 (a conversation between)": compare(by_label["gen_a1"], by_label["gen_a5"]),
        "logits_a1_vs_logits_a3 (a conversation between)": compare(by_label["logits_a1"], by_label["logits_a3"]),
        # the unrelated request against itself
        "gen_b1_vs_gen_b2 (an unrelated request, twice)": compare(by_label["gen_b1"], by_label["gen_b2"]),
    }

    result = {
        "what": "S0.1: the engine's run-to-run determinism",
        "config": str(cfg.get("_path", "")),
        "engine_cmd": engine.cmd,
        "engine_log": str(log),
        "max_new": max_new,
        "logits_k": logits_k,
        "prompt_a_tokens": len(a_ids),
        "prompt_b_tokens": len(b_ids),
        "prompt_ids_sha": {
            "a": __import__("hashlib").sha256(ids.encode("utf-8")).hexdigest(),
            "b": __import__("hashlib").sha256(ids_b.encode("utf-8")).hexdigest(),
        },
        "records": records,
        "comparisons": comparisons,
    }
    if out_json is not None:
        out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"results: {out_json}")
    return result


# ---------------------------------------------------------------------------- S0.3 (the follow-up probe)


def s0_3(cfg: dict, work: Path, max_new: int, logits_k: int, repeats: int,
         env_extra: dict, tag: str, out_json: Path | None) -> dict:
    """The same LOGITS request, `repeats` times, back to back: does the row converge?

    S0.1 found that the generated tokens are byte-identical across repeats while the logits row
    is not.  This probe asks why: the engine's own DONE line says where the experts came from
    (hits in the GPU cache, lookups, the RAM copy, the files), so the row's movement can be read
    against the data path that produced it.  `env_extra` can switch the engine's own knobs (for
    example STRATA_PREFILL_CPU_SHARE=0, the CPU share of the prompt path) to see whether a
    deterministic mode exists.
    """
    pack = Path(cfg["tokenizer"])
    tok = load_tokenizer(pack)
    tpl = ChatTemplate(pack / "chat_template.jinja")
    a_ids = tok.encode(tpl.render([{"role": "user", "content": PROMPT_A}]), parse_special=True)
    ids = ",".join(str(t) for t in a_ids)
    line = f"LOGITS {logits_k} {max_new} ckpt=0 {ids}"

    work.mkdir(parents=True, exist_ok=True)
    log = work / f"engine-s03-{tag}.log"
    engine = Engine(cfg, log, env_extra=env_extra)
    print(f"\n=== S0.3 [{tag}]: {repeats} identical LOGITS requests, back to back ===")
    print(f"engine env overrides: {env_extra or '{}'}")
    print(f"engine log: {log}")
    records = []
    try:
        for i in range(repeats):
            records.append(engine.request(line, f"rep{i + 1}"))
    finally:
        engine.close()

    first = records[0]
    rows = []
    for rec in records[1:]:
        cmp_ = compare(first, rec)
        cmp_["numeric"] = logits_numeric_diff(first, rec)
        rows.append(cmp_)
    identical = sum(1 for c in rows if c["L_bytes_equal"])
    result = {
        "what": "S0.3: the same LOGITS request, back to back",
        "tag": tag,
        "engine_env_overrides": env_extra,
        "engine_cmd": engine.cmd,
        "engine_log": str(log),
        "repeats": repeats,
        "records": records,
        "comparisons_against_the_first": rows,
        "logits_rows_byte_identical": f"{identical} of {len(rows)}",
        "t_lines_byte_identical": f"{sum(1 for c in rows if c['T_bytes_equal'])} of {len(rows)}",
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"results: {out_json}")
    print(f"  the logits rows: {result['logits_rows_byte_identical']} byte-identical to the first")
    print(f"  the T lines:     {result['t_lines_byte_identical']} byte-identical to the first")
    for rec, cmp_ in zip(records[1:], rows):
        d = done_fields(rec)
        n = cmp_["numeric"]
        print(f"  {rec['label']:6s} L bytes equal: {str(cmp_['L_bytes_equal']):5s}  "
              f"top1 {n['top1_a']}/{n['top1_b']}  max|dlogit| {n.get('max_abs_dlogit', 0):.4g}  "
              f"dZ {n['dZ']:+.4g}  TV {n.get('tv_upper_bound', 0):.3g}  "
              f"hits {d.get('hits')}/{d.get('lookups')}  ram {d.get('ram_blobs')}  file {d.get('file_blobs')}")
    return result


# ---------------------------------------------------------------------------- S0.2


def engine_read(flat: np.ndarray, ne0: int, ne1: int) -> np.ndarray:
    """The engine's own rule for a tensor in its file: ne0 is the contiguous dimension.

    The shape is (ne0, ne1) in the engine's order, ne0 contiguous: element (i, j) sits at
    i + ne0*j.  A flat buffer of ne0*ne1 elements is therefore ne1 rows of ne0.
    """
    return flat.reshape(ne1, ne0).T


def s0_2(seed: int = 20261021) -> dict:
    """The adapter's arithmetic: y += s * B @ (A @ x), in the plan's on-disk format.

    The geometry is the plan's own example (docs/research/phase2_engine_lora_plan.md §2.1):
    blk.3.attn_q.weight is {in 2560, out 12288} (include/strata/core/geometry.hpp), the adapter is
    rank 16 with alpha 16, and the index.txt rows are

        blk.3.attn_q.weight.lora_a  16  16.0     16  2560     0   81920
        blk.3.attn_q.weight.lora_b  16  16.0  12288    16  81920  393216

    so lora_a is ne0=rank, ne1=in and lora_b is ne0=out, ne1=rank - "the tensor's shape in the
    engine's order (dim 0 contiguous)".
    """
    rank, n_in, n_out, alpha = 16, 2560, 12288, 16.0
    s = alpha / rank
    rng = np.random.default_rng(seed)
    # PEFT's tensors: A is [rank, in] and B is [out, rank], as safetensors stores them (the LAST
    # dimension contiguous: A's `in`, B's `rank`).
    a_peft = rng.standard_normal((rank, n_in), dtype=np.float32)
    b_peft = rng.standard_normal((n_out, rank), dtype=np.float32)
    x = rng.standard_normal(n_in, dtype=np.float32)

    # the reference: the plan's formula, in PEFT's mathematical convention
    ref = s * (b_peft @ (a_peft @ x))
    # the same thing in float64: the yardstick for the float32 rounding below
    ref64 = s * (b_peft.astype(np.float64) @ (a_peft.astype(np.float64) @ x.astype(np.float64)))

    # the plan's file format: ne0 contiguous, A as ne0=rank/ne1=in, B as ne0=out/ne1=rank.
    # The converter writes the bytes; the loader reads them back with the engine's own rule.
    a_file = np.ascontiguousarray(a_peft.T)            # (in, rank) row-major: rank contiguous
    b_file = np.ascontiguousarray(b_peft.T)            # (rank, out) row-major: out contiguous
    a_read = engine_read(a_file.reshape(-1), rank, n_in)
    b_read = engine_read(b_file.reshape(-1), n_out, rank)
    got = s * (b_read @ (a_read @ x))

    # the two ways to get it wrong, both plausible and both silent
    a_not_transposed = engine_read(np.ascontiguousarray(a_peft).reshape(-1), rank, n_in)
    b_not_transposed = engine_read(np.ascontiguousarray(b_peft).reshape(-1), n_out, rank)
    wrong_a = s * (b_read @ (a_not_transposed @ x))
    wrong_b = s * (b_not_transposed @ (a_read @ x))

    # the other storage convention (the plan's orientation sentence: the engine's B is [rank, out],
    # rank contiguous, which is PEFT's own byte layout) - the same math, a different expression
    b_alt = engine_read(np.ascontiguousarray(b_peft).reshape(-1), rank, n_out)   # ne0=rank, ne1=out
    alt = s * ((a_read @ x) @ b_alt)

    def stats(delta: np.ndarray) -> dict:
        return {
            "max_abs_diff_from_the_reference": float(np.max(np.abs(delta - ref))),
            "max_abs_diff_from_the_float64_reference": float(np.max(np.abs(delta.astype(np.float64) - ref64))),
            "mean_abs_diff_from_the_reference": float(np.mean(np.abs(delta - ref))),
            "cosine_similarity_with_the_reference": float(
                np.dot(delta, ref) / (np.linalg.norm(delta) * np.linalg.norm(ref))),
            "l2_norm": float(np.linalg.norm(delta)),
        }

    return {
        "what": "S0.2: the adapter's arithmetic",
        "geometry": {"rank": rank, "in": n_in, "out": n_out, "alpha": alpha, "scaling_s": s,
                     "seed": seed},
        "the_plan_s_formula": "y += s * B @ (A @ x), A = [rank, in], B = [out, rank]",
        "the_plan_s_index_rows": {
            "lora_a": {"ne0": rank, "ne1": n_in, "bytes": rank * n_in * 2},
            "lora_b": {"ne0": n_out, "ne1": rank, "bytes": n_out * rank * 2},
        },
        "the_plan_s_convention": (
            "index.txt's ne0/ne1 are the tensor's shape in the engine's order (dim 0 contiguous): "
            "lora_a is ne0=rank/ne1=in (rank contiguous) and lora_b is ne0=out/ne1=rank (out "
            "contiguous).  The engine's rule for reading such a tensor: element (i, j) at i + ne0*j, "
            "so the mathematical matrix read back is A = [rank, in] and B = [out, rank] - PEFT's "
            "matrices.  PEFT's safetensors keep the LAST dimension contiguous (A's `in`, B's "
            "`rank`), so the converter's bytes are the transpose of PEFT's for BOTH A and B."),
        "reference": {"max_abs": float(np.max(np.abs(ref))), "l2_norm": float(np.linalg.norm(ref)),
                      "float64_max_abs": float(np.max(np.abs(ref64))),
                      "float32_rounding_max_abs_diff": float(np.max(np.abs(ref.astype(np.float64) - ref64)))},
        "as_the_plan_s_format_does_it": {
            "A_read_back_shape": list(a_read.shape), "B_read_back_shape": list(b_read.shape),
            "A_equals_PEFT_A": bool(np.array_equal(a_read, a_peft)),
            "B_equals_PEFT_B": bool(np.array_equal(b_read, b_peft)),
            "delta": stats(got),
            "matches_the_reference": bool(np.array_equal(got, ref)),
            "max_abs_diff": float(np.max(np.abs(got - ref))),
        },
        "if_A_is_not_transposed": stats(wrong_a),
        "if_B_is_not_transposed": stats(wrong_b),
        "the_other_storage_convention_rank_contiguous_B": {
            "B_read_back_shape": list(b_alt.shape),
            "expression": "y += s * ((A @ x) @ B), B = [rank, out], rank contiguous (PEFT's bytes)",
            "delta": stats(alt),
            "matches_the_reference": bool(np.allclose(alt, ref, rtol=0, atol=0)),
            "max_abs_diff": float(np.max(np.abs(alt - ref))),
        },
        "base_path_untouched": "the delta is a separate add; the base path's bytes do not change",
    }


# ---------------------------------------------------------------------------- the report


def print_s0_1(result: dict) -> None:
    print("\n--- S0.1: the requests, and how long each took ---")
    for rec in result["records"]:
        done = rec.get("DONE", "")
        toks = len(rec["T"])
        print(f"  {rec['label']:12s} {rec['elapsed_s']:8.3f} s  T lines {toks:3d}  "
              f"vram free before {rec['vram_free_mib_before']} MiB   {done[:80]}")
    print("\n--- S0.1: the byte comparisons ---")
    for name, cmp_ in result["comparisons"].items():
        print(f"  {name}")
        print(f"      T bytes equal: {cmp_['T_bytes_equal']}   L/Z bytes equal: {cmp_['L_bytes_equal']}")
        if "L_max_abs_diff" in cmp_:
            print(f"      L rows: {cmp_['L_rows']}  ids equal: {cmp_.get('L_ids_equal')}  "
                  f"rows differing: {cmp_['L_n_rows_differing']}  "
                  f"max |dlogit|: {cmp_['L_max_abs_diff']:.6g}  "
                  f"rel: {cmp_['L_max_rel_diff']:.3g}  |dZ|: {cmp_.get('Z_abs_diff')}")
        if not cmp_["T_bytes_equal"]:
            print(f"      T a: {cmp_['T_a']}")
            print(f"      T b: {cmp_['T_b']}")


def print_s0_2(result: dict) -> None:
    print("\n--- S0.2: the adapter's arithmetic ---")
    g = result["geometry"]
    print(f"  geometry: rank {g['rank']}, in {g['in']}, out {g['out']}, alpha {g['alpha']}, "
          f"s = alpha/rank = {g['scaling_s']}")
    print(f"  the plan's formula: {result['the_plan_s_formula']}")
    print(f"  the plan's index rows: {json.dumps(result['the_plan_s_index_rows'])}")
    ok = result["as_the_plan_s_format_does_it"]
    print(f"  as the plan's format does it: A {ok['A_read_back_shape']} (PEFT's A: {ok['A_equals_PEFT_A']}), "
          f"B {ok['B_read_back_shape']} (PEFT's B: {ok['B_equals_PEFT_B']})")
    print(f"      matches the reference: {ok['matches_the_reference']}  "
          f"max |diff|: {ok['max_abs_diff']:.6g}  "
          f"(the float32 rounding of the reference itself: "
          f"{result['reference']['float32_rounding_max_abs_diff']:.6g})")
    for key in ("if_A_is_not_transposed", "if_B_is_not_transposed"):
        s = result[key]
        print(f"  {key}: max |diff| {s['max_abs_diff_from_the_reference']:.6g}  "
              f"cosine {s['cosine_similarity_with_the_reference']:.6f}  "
              f"||delta|| {s['l2_norm']:.6g}  (reference ||delta|| {result['reference']['l2_norm']:.6g})")
    alt = result["the_other_storage_convention_rank_contiguous_B"]
    print(f"  the other storage convention: {alt['expression']}")
    print(f"      matches the reference: {alt['matches_the_reference']}  "
          f"max |diff|: {alt['max_abs_diff']:.6g}")


# ---------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="the launch config (exe, args, tokenizer)")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK, help="where the engine's log goes")
    ap.add_argument("--only", choices=("s0.1", "s0.2", "s0.3", "both"), default="both")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--logits-k", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=8, help="S0.3: how many identical LOGITS requests")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="S0.3: an engine environment override (repeatable)")
    ap.add_argument("--tag", default="default", help="S0.3: a name for the run (the log file's name)")
    ap.add_argument("--out", type=Path, default=None, help="where the JSON result goes")
    args = ap.parse_args()

    if args.only in ("s0.2", "both"):
        s2 = s0_2()
        print_s0_2(s2)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(s2, indent=2) + "\n", encoding="utf-8")
            print(f"\nresults: {args.out}")

    if args.only in ("s0.1", "both"):
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        cfg["_path"] = str(args.config)
        s1 = s0_1(cfg, args.work, args.max_new, args.logits_k, args.out)
        print_s0_1(s1)

    if args.only == "s0.3":
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        cfg["_path"] = str(args.config)
        env_extra = {}
        for item in args.env:
            k, _, v = item.partition("=")
            env_extra[k] = v
        s0_3(cfg, args.work, args.max_new, args.logits_k, args.repeats, env_extra, args.tag, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
