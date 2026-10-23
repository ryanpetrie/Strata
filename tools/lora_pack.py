"""tools/lora_pack.py - a PEFT LoRA adapter, converted into the engine's own format (one text index, one fp16 blob).

    python tools/lora_pack.py --adapter <peft dir> --out <dir>
    python tools/lora_pack.py --adapter <peft dir> --out <dir> --validate      # convert, then read the result back
    python tools/lora_pack.py --adapter <peft dir> --show                      # the rows it would write, no files

The engine does not parse JSON or safetensors (`include/strata/core/weights.hpp:25-30`, and
`include/strata/core/lora_pack.hpp`): a parser in the engine is a new, unaudited component whose failure mode is a
wrong byte offset, which decodes to a plausible weight.  The conversion therefore happens here, and the engine
only checks what this file wrote against the model's own layout table.

THE FORMAT (docs/research/phase2_engine_lora_plan.md 2.1, corrected by phase2_lora_spike.md 3):

    <out>/index.txt
    # strata lora index v1
    # name rank alpha ne0 ne1 offset bytes
    blk.3.attn_k.weight.lora_a 16 16.0 16 512 0 16384
    blk.3.attn_k.weight.lora_b 16 16.0 512 16 16384 16384
    <out>/lora.bin            fp16, little-endian, in the index's order

`ne0` is the engine's CONTIGUOUS axis, and that is the orientation trap this tool exists to survive.  PEFT stores
A [rank, in] and B [out, rank] with the LAST axis contiguous; the engine's rule is element (i, j) at i + ne0 * j,
so BOTH matrices are written TRANSPOSED here (A^T is [in, rank], B^T is [rank, out]).  The spike measured what
the two mistakes cost: a delta of the same magnitude as the right answer (27175 and 24515 against a norm of
24344) with cosines of -0.103 and 0.015, and no error anywhere.

WHAT IT REFUSES, by name: a module it cannot map (the 36 GDN layers' Hugging Face module names are open question
1 of the plan's 7 and are deliberately NOT guessed here), a routed-expert adapter (the engine refuses it for
size; the message carries the arithmetic), `use_rslora` (a different scaling, alpha / sqrt(r), that the engine's
formula would get wrong), a rank above the engine's maximum, a rank/alpha the two halves disagree on, a value
fp16 cannot hold, and a tensor that is not a pair of matrices.

Read-only on its inputs.  No GPU, no model, no engine: this is step 1 of the plan's 4.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys
import tempfile

import numpy as np

VERSION = "# strata lora index v1"
HEADER = "# name rank alpha ne0 ne1 offset bytes"
MAX_RANK = 64          # include/strata/core/lora_pack.hpp: kMaxLoraRank

#: PEFT's module name -> the engine's tensor suffix.  Only what this repository can confirm: the QSA projections
#: (`bench/results/2026-10-07-community-rtx-5080-docker-wsl2/mtp-inventory.md:31-35`) and the shared expert (a
#: dense FFN, the plan's 3.2).  The GDN layers' names are NOT here on purpose - a guess is a silently wrong
#: adapter, the one failure the format cannot detect downstream.
MODULES = {
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight",
    "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj": "ffn_down_shexp.weight",
    "mlp.gate_proj": "ffn_gate_shexp.weight",
    "mlp.up_proj": "ffn_up_shexp.weight",
    "mlp.down_proj": "ffn_down_shexp.weight",
}
ROUTED = re.compile(r"^mlp\.experts\.(?:\d+\.)?(?:gate|up|down|gate_up)_proj$")
GLOBAL = {"lm_head": "output.weight", "model.lm_head": "output.weight"}

#: The suffix order the index writes: q, k, v, o, the shared expert, the head.  A stable order means one adapter
#: always converts to the same bytes, which the engine's session digest relies on.
ORDER = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
         "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "output.weight"]

LAYERS = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
TENSOR = re.compile(r"^(?P<stem>.+)\.lora_(?P<half>[AB])(?:\.(?P<adapter>[A-Za-z0-9_]+))?\.weight$")

DTYPES = {"F16": ("<f2", 2), "F32": ("<f4", 4), "BF16": ("<u2", 2)}


class LoraPackError(Exception):
    """A refusal: the message names the tensor and what is wrong with it."""


# ------------------------------------------------------------------------------------------------ safetensors
def safetensors_header(path):
    """(header dict, the offset of the first data byte) - the format is 8 bytes of length, then JSON."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def _shape_of(info, name):
    shape = [int(d) for d in info["shape"]]
    while len(shape) > 2 and shape[0] == 1:   # some trainers write [1, rank, in]
        shape.pop(0)
    if not shape:
        shape = [1]
    return shape


def read_tensor(path, header, base, name):
    """One tensor as a float64 array in its own row-major order (axes are the caller's business)."""
    info = header[name]
    dtype, width = DTYPES.get(info["dtype"], (None, 0))
    if dtype is None:
        raise LoraPackError(f"{name}: dtype {info['dtype']} is not F16, BF16 or F32")
    shape = _shape_of(info, name)
    start, end = info["data_offsets"]
    want = width
    for d in shape:
        want *= d
    if end - start != want:
        raise LoraPackError(f"{name}: {end - start} data bytes for {info['dtype']} {shape} (wanted {want})")
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    if info["dtype"] == "BF16":
        wide = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view("<f4")
        return wide.reshape(shape).astype(np.float64)
    return np.frombuffer(raw, dtype).astype(np.float64).reshape(shape)


# ------------------------------------------------------------------------------------------------ the name map
def engine_name(stem, rank):
    """PEFT's module stem -> the engine's base tensor name, or a refusal that names the tensor."""
    for prefix in ("base_model.model.", "base_model."):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if stem in GLOBAL:
        return GLOBAL[stem]
    m = LAYERS.search(stem)
    if not m:
        raise LoraPackError(f"{stem}: not a `layers.N.<module>` name and not lm_head - cannot tell which layer it is")
    layer, module = int(m.group(1)), m.group(2)
    if ROUTED.match(module):
        raise LoraPackError(
            f"{stem}: a routed-expert adapter is refused.  At rank {rank} one adapter per expert over the three "
            "expert tensors of all 48 layers is 7.55 GB, and a shared one over the fused weights is 1.52 GB; the "
            "engine holds the adapter beside the base weights inside 58 MiB of headroom "
            "(docs/research/phase2_engine_lora_plan.md 3.3).  QSA attention-only is the MoE-compatible adapter.")
    if module not in MODULES:
        raise LoraPackError(
            f"{stem}: no confirmed mapping for `{module}`.  The QSA projections (self_attn.q_proj, .k_proj, "
            ".v_proj, .o_proj) and the shared expert (mlp.*, mlp.shared_expert.*) are mapped; the 36 GDN layers' "
            "Hugging Face module names are open question 1 of docs/research/phase2_engine_lora_plan.md 7 and are "
            "deliberately not guessed (a guess here is a silently wrong adapter).")
    return f"blk.{layer}.{MODULES[module]}"


def rank_of(header, name, half):
    """The rank a PEFT tensor declares: A is [rank, in] and B is [out, rank]."""
    shape = _shape_of(header[name], name)
    if len(shape) != 2:
        raise LoraPackError(f"{name}: shape {shape}; a LoRA matrix is 2-D")
    return shape[0] if half == "a" else shape[1]


def group(header, adapter_name=None):
    """{the engine's base tensor: {"a": name, "b": name}} - the mapping, before any bytes are read."""
    found = {}
    names = [n for n in header if n.endswith(".weight") and ".lora_" in n]
    if not names:
        raise LoraPackError("no .lora_A / .lora_B tensors in the adapter")
    adapters = set()
    for name in names:
        m = TENSOR.match(name)
        if not m:
            raise LoraPackError(f"{name}: not a PEFT LoRA tensor (`...lora_A.weight` / `...lora_B[.name].weight`)")
        adapters.add(m.group("adapter"))
    adapters.discard(None)
    if adapter_name is None and len(adapters) > 1:
        raise LoraPackError(f"the adapter carries several adapters ({', '.join(sorted(adapters))}); "
                            "choose one with --adapter-name")
    if adapter_name is not None and adapters and adapter_name not in adapters:
        raise LoraPackError(f"--adapter-name {adapter_name!r} is not in the file ({', '.join(sorted(adapters))})")
    chosen = adapter_name if adapters else None
    for name in names:
        m = TENSOR.match(name)
        if chosen is not None and m.group("adapter") not in (None, chosen):
            continue
        half = m.group("half").lower()
        base = engine_name(m.group("stem"), rank_of(header, name, half))
        slot = found.setdefault(base, {})
        if half in slot:
            raise LoraPackError(f"{base}: {slot[half]} and {name} both map to its lora_{half.upper()} - two source "
                                "tensors for one engine tensor (the shared expert is spelled both mlp.* and "
                                "mlp.shared_expert.*; only one of them is the checkpoint's)")
        slot[half] = name
    return found


# ------------------------------------------------------------------------------------------------ conversion
def convert(adapter_dir, out_dir, *, safetensors=None, config=None, adapter_name=None, max_rank=MAX_RANK,
            validate=False, log=print):
    """Writes <out>/index.txt and <out>/lora.bin.  Returns (rows, blob bytes).  Raises LoraPackError on a refusal."""
    src = pathlib.Path(adapter_dir)
    st_path = pathlib.Path(safetensors) if safetensors else src / "adapter_model.safetensors"
    cfg_path = pathlib.Path(config) if config else src / "adapter_config.json"
    if not st_path.exists():
        raise LoraPackError(f"no {st_path} (pass --safetensors)")
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if cfg.get("use_rslora"):
        raise LoraPackError("use_rslora: the scaling is alpha / sqrt(r) and the engine's formula is alpha / r; "
                            "convert a standard adapter or change the training recipe")
    header, data_base = safetensors_header(st_path)
    header.pop("__metadata__", None)

    grouping = group(header, adapter_name)
    alpha_cfg = cfg.get("lora_alpha")
    cfg_rank = cfg.get("r")

    rows, blob, problems = [], bytearray(), []
    def sort_key(base):
        layer, suffix = (int(base.split(".")[1]), base.split(".", 2)[2]) if base.startswith("blk.") else (10 ** 6, base)
        return (layer, ORDER.index(suffix) if suffix in ORDER else 99, base)

    for base in sorted(grouping, key=sort_key):
        slot = grouping[base]
        missing = [f"{base}.lora_{h.upper()} is missing" for h in ("a", "b") if h not in slot]
        if missing:
            problems.extend(missing)
            continue
        a = read_tensor(st_path, header, data_base, slot["a"])
        b = read_tensor(st_path, header, data_base, slot["b"])
        if a.ndim != 2 or b.ndim != 2:
            raise LoraPackError(f"{base}: A is {list(a.shape)} and B is {list(b.shape)}; a LoRA is two 2-D matrices")
        rank, in_dim = a.shape
        out_dim, b_rank = b.shape
        if rank != b_rank:
            raise LoraPackError(f"{base}: A is {list(a.shape)} and B is {list(b.shape)}: the ranks disagree "
                                f"({rank} against {b_rank})")
        if rank > max_rank:
            raise LoraPackError(f"{base}: rank {rank} is above this engine's maximum ({max_rank})")
        alpha = alpha_cfg
        if isinstance(alpha, dict):
            # A per-module alpha is keyed by the module name, sometimes with its parents and sometimes bare.
            module = slot["a"].split(".lora_")[0]
            for key in (module, module.split(".")[-1], module.split("layers.")[-1].split(".", 1)[-1]):
                if key in alpha:
                    alpha = alpha[key]
                    break
            else:
                raise LoraPackError(f"{base}: the config's lora_alpha has no entry for {module}")
            if isinstance(alpha, dict):   # a nested {'alpha': ...} shape some trainers write
                alpha = alpha.get("alpha")
        alpha = float(rank if alpha is None else alpha)
        if isinstance(cfg_rank, (int, float)) and int(cfg_rank) != rank:
            raise LoraPackError(f"{base}: the tensors are rank {rank} and adapter_config.json says r = {cfg_rank}")
        for label, arr in (("A", a), ("B", b)):
            if not np.all(np.isfinite(arr)):
                raise LoraPackError(f"{base}: {label} has a value that is not finite")
            biggest = float(np.max(np.abs(arr), initial=0.0))
            if biggest > 65504:
                raise LoraPackError(f"{base}: {label}'s largest value {biggest:.6g} does not fit fp16 (the engine's "
                                    "adapter format is fp16)")
        # THE ORIENTATION: A^T is [in, rank] and B^T is [rank, out]; the index's ne0 is the contiguous axis.
        for half, arr, ne0, ne1 in (("a", a.T, rank, in_dim), ("b", b.T, out_dim, rank)):
            data = np.ascontiguousarray(arr, dtype="<f2").tobytes()
            rows.append((f"{base}.lora_{half}", rank, alpha, ne0, ne1, len(blob), len(data)))
            blob += data
    if problems:
        raise LoraPackError("; ".join(problems))
    if not rows:
        raise LoraPackError("nothing to convert: no pair of A and B tensors was mapped")

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = [VERSION, HEADER]
    for name, rank, alpha, ne0, ne1, off, size in rows:
        text.append(f"{name} {rank} {alpha:.1f} {ne0} {ne1} {off} {size}")
    (out / "index.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    (out / "lora.bin").write_bytes(bytes(blob))
    if log:
        log(f"[lora] {len(rows) // 2} target(s), {len(blob) / 1e6:.2f} MB of fp16, wrote {out / 'index.txt'} and "
            f"{out / 'lora.bin'}")
    if validate:
        validate_pack(out, grouping, st_path, header, data_base, log=log)
    return rows, len(blob)


def read_pack(out_dir):
    """The engine's view of a written adapter: (rows, blob).  Used by --validate and by the tests."""
    out = pathlib.Path(out_dir)
    lines = (out / "index.txt").read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith(VERSION):
        raise LoraPackError(f"{out / 'index.txt'} does not start with `{VERSION}`")
    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 7:
            raise LoraPackError(f"not `name rank alpha ne0 ne1 offset bytes`: {line!r}")
        name, rank, alpha, ne0, ne1, off, size = f
        rows.append((name, int(rank), float(alpha), int(ne0), int(ne1), int(off), int(size)))
    return rows, (out / "lora.bin").read_bytes()


def validate_pack(out_dir, grouping, st_path, header, data_base, log=print):
    """Reads what was written back and compares it with the source, through the engine's own element rule."""
    rows, blob = read_pack(out_dir)
    by_name = {r[0]: r for r in rows}
    checked = 0
    for base in grouping:
        for half in ("a", "b"):
            src = read_tensor(st_path, header, data_base, grouping[base][half])
            name, rank, alpha, ne0, ne1, off, size = by_name[f"{base}.lora_{half}"]
            if size != ne0 * ne1 * 2:
                raise LoraPackError(f"{name}: {size} bytes for [{ne0}, {ne1}] fp16")
            got = np.frombuffer(blob[off:off + size], "<f2").reshape(ne1, ne0).T   # the file holds the transpose
            if got.shape != src.shape:
                raise LoraPackError(f"{name}: read back {list(got.shape)}, wrote {list(src.shape)}")
            err = float(np.max(np.abs(got.astype(np.float64) - src), initial=0.0))
            if err != 0.0:
                raise LoraPackError(f"{name}: the round trip is not exact (max |difference| {err:.3g}); the source "
                                    "is not fp16-representable")
            checked += 1
    if log:
        log(f"[lora] --validate: {checked} matrices read back from the file and equal to the source, exactly")
    return checked


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="the PEFT adapter directory (adapter_config.json + .safetensors)")
    ap.add_argument("--out", help="where to write index.txt and lora.bin")
    ap.add_argument("--safetensors", help="the .safetensors file, when not <adapter>/adapter_model.safetensors")
    ap.add_argument("--config", help="the adapter_config.json, when not <adapter>/adapter_config.json")
    ap.add_argument("--adapter-name", help="which PEFT adapter to convert when the file carries several")
    ap.add_argument("--max-rank", type=int, default=MAX_RANK, help=f"the engine's maximum rank (default {MAX_RANK})")
    ap.add_argument("--validate", action="store_true", help="read the written file back and compare with the source")
    ap.add_argument("--show", action="store_true", help="print the rows it would write, writing nothing")
    a = ap.parse_args(argv)
    if not a.show and not a.out:
        ap.error("--out is required unless --show")
    try:
        if a.show:
            with tempfile.TemporaryDirectory() as tmp:
                rows, _ = convert(a.adapter, a.out or tmp, safetensors=a.safetensors, config=a.config,
                                  adapter_name=a.adapter_name, max_rank=a.max_rank, validate=a.validate,
                                  log=(lambda s: print(s)) if a.out else None)
            print(VERSION)
            print(HEADER)
            for name, rank, alpha, ne0, ne1, off, size in rows:
                print(f"{name} {rank} {alpha:.1f} {ne0} {ne1} {off} {size}")
            return 0
        convert(a.adapter, a.out, safetensors=a.safetensors, config=a.config, adapter_name=a.adapter_name,
                max_rank=a.max_rank, validate=a.validate)
        return 0
    except LoraPackError as e:
        print(f"[lora] refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
