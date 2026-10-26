"""tools/test_lora_pack.py - the PEFT -> engine adapter converter, on synthetic adapters (no GPU, no model).

The byte layout is pinned TWICE, here and in src/core/lora_pack_test.cpp, with the same six A values and six B
values: this file checks what tools/lora_pack.py writes, that test checks what the engine reads, and the two
agreeing is what says the converter and the reader share one format (the orientation is the thing that would
otherwise fail silently).

    python -m unittest tools.test_lora_pack
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import lora_pack as L  # noqa: E402

# The one small adapter both tests pin.  PEFT: A [rank 2, in 3], B [out 3, rank 2], alpha 2, so the scale is 1.
A_PEFT = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
B_PEFT = np.array([[0.5, -0.5], [1.5, 1.0], [-2.0, 0.25]], dtype=np.float64)
#: The file's twelve fp16 bits, in file order: A^T (3 rows of 2) then B^T (2 rows of 3).
GOLDEN_BITS = [0x3C00, 0x4400, 0x4000, 0x4500, 0x4200, 0x4600,   # 1, 4, 2, 5, 3, 6
               0x3800, 0x3E00, 0xC000, 0xB800, 0x3C00, 0x3400]   # .5, 1.5, -2, -.5, 1, .25
LAYER = "base_model.model.model.layers.3.self_attn.k_proj"


def shard(tensors):
    """A safetensors file: [u64 header length][JSON header][data].  `tensors` is [(name, array, dtype)]."""
    header, data = {}, b""
    for name, arr, dtype in tensors:
        arr = np.asarray(arr)
        if dtype == "BF16":
            raw = (arr.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raw = arr.astype({"F16": "<f2", "F32": "<f4"}[dtype]).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data


class Adapter:
    """A temporary PEFT adapter directory (adapter_config.json + adapter_model.safetensors)."""

    def __init__(self, tensors, cfg=None, name="adapter_model.safetensors"):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="lora-pack-test-"))
        (self.dir / name).write_bytes(shard(tensors))
        (self.dir / "adapter_config.json").write_text(json.dumps(cfg if cfg is not None else
                                                                 {"r": 2, "lora_alpha": 2, "peft_type": "LORA"}))
        self.out = self.dir / "pack"

    def convert(self, **kw):
        return L.convert(self.dir, self.out, log=None, **kw)


def golden():
    return Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.weight", B_PEFT, "F16")])


class TheFormat(unittest.TestCase):
    def test_the_rows_and_the_bytes(self):
        a = golden()
        rows, size = a.convert()
        self.assertEqual(rows, [("blk.3.attn_k.weight.lora_a", 2, 2.0, 2, 3, 0, 12),
                                ("blk.3.attn_k.weight.lora_b", 2, 2.0, 3, 2, 12, 12)])
        self.assertEqual(size, 24)
        bits = list(np.frombuffer((a.out / "lora.bin").read_bytes(), "<u2"))
        self.assertEqual(bits, GOLDEN_BITS)

    def test_the_index_text_is_the_documented_shape(self):
        a = golden()
        a.convert()
        lines = (a.out / "index.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "# strata lora index v1")
        self.assertEqual(lines[1], "# name rank alpha ne0 ne1 offset bytes")
        self.assertEqual(lines[2], "blk.3.attn_k.weight.lora_a 2 2.0 2 3 0 12")
        self.assertEqual(lines[3], "blk.3.attn_k.weight.lora_b 2 2.0 3 2 12 12")

    def test_already_packed_rows_read_back(self):
        a = golden()
        a.convert()
        rows, blob = L.read_pack(a.out)
        got = np.frombuffer(blob[0:12], "<f2").reshape(3, 2).T      # A's file is [in][rank]
        np.testing.assert_array_equal(got, A_PEFT)
        got = np.frombuffer(blob[12:24], "<f2").reshape(2, 3).T     # B's file is [rank][out]
        np.testing.assert_array_equal(got, B_PEFT)

    def test_conversion_is_deterministic(self):
        a = golden()
        a.convert()
        first = (a.out / "index.txt").read_bytes(), (a.out / "lora.bin").read_bytes()
        a.convert()
        self.assertEqual(first, ((a.out / "index.txt").read_bytes(), (a.out / "lora.bin").read_bytes()))


class Validating(unittest.TestCase):
    def test_validate_reads_the_file_back_and_agrees(self):
        a = golden()
        rows, _ = a.convert(validate=True)
        self.assertEqual(len(rows), 2)

    def test_validate_catches_a_transposed_B(self):
        """The orientation bug, on purpose: the file holds B^T, so writing B unchanged must be caught."""
        a = golden()
        a.convert()
        rows, blob = L.read_pack(a.out)
        broken = bytearray(blob)
        broken[12:24] = np.ascontiguousarray(B_PEFT, dtype="<f2").tobytes()   # NOT transposed
        (a.out / "lora.bin").write_bytes(bytes(broken))
        header, base = L.safetensors_header(a.dir / "adapter_model.safetensors")
        header.pop("__metadata__", None)
        with self.assertRaises(L.LoraPackError) as e:
            L.validate_pack(a.out, {"blk.3.attn_k.weight": {"a": f"{LAYER}.lora_A.weight",
                                                            "b": f"{LAYER}.lora_B.weight"}},
                            a.dir / "adapter_model.safetensors", header, base, log=None)
        self.assertIn("not exact", str(e.exception))

    def test_an_f32_source_that_fp16_cannot_hold_exactly_is_refused_by_validate(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT + 0.1, "F32"),
                     (f"{LAYER}.lora_B.weight", B_PEFT, "F32")], cfg={"r": 2, "lora_alpha": 2})
        with self.assertRaises(L.LoraPackError) as e:
            a.convert(validate=True)
        self.assertIn("not fp16-representable", str(e.exception))


class TheNameMap(unittest.TestCase):
    def test_every_confirmed_module_maps_to_the_engine_tensor(self):
        # One spelling of the shared expert per module: mlp.* and mlp.shared_expert.* are two names for one tensor
        # on a checkpoint, so a file that carried both would be a collision (see the test below).
        modules = {m: s for m, s in L.MODULES.items() if not m.startswith("mlp.shared_expert.")}
        tensors, want = [], []
        for module, suffix in modules.items():
            stem = f"base_model.model.model.layers.7.{module}"
            tensors += [(f"{stem}.lora_A.weight", A_PEFT, "F16"), (f"{stem}.lora_B.weight", B_PEFT, "F16")]
            want += [f"blk.7.{suffix}.lora_a", f"blk.7.{suffix}.lora_b"]
        tensors += [("base_model.model.lm_head.lora_A.weight", np.ones((2, 3)), "F16"),
                    ("base_model.model.lm_head.lora_B.weight", np.ones((4, 2)), "F16")]
        want += ["output.weight.lora_a", "output.weight.lora_b"]
        a = Adapter(tensors)
        rows, _ = a.convert()
        self.assertEqual(len(rows), len(want))
        self.assertEqual({r[0] for r in rows}, set(want))
        self.assertEqual(rows[-1][0], "output.weight.lora_b")    # the head sorts after every layer

    def test_an_unconfirmed_module_is_refused_by_name_and_says_why(self):
        a = Adapter([("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight", A_PEFT, "F16"),
                     ("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight", B_PEFT, "F16")])
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        msg = str(e.exception)
        self.assertIn("in_proj_qkv", msg)
        self.assertIn("question 1", msg)

    def test_a_routed_expert_is_refused_with_the_size(self):
        a = Adapter([("base_model.model.model.layers.5.mlp.experts.0.gate_proj.lora_A.weight", A_PEFT, "F16"),
                     ("base_model.model.model.layers.5.mlp.experts.0.gate_proj.lora_B.weight", B_PEFT, "F16")])
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("7.55 GB", str(e.exception))

    def test_two_adapters_in_one_file_need_a_name(self):
        a = Adapter([(f"{LAYER}.lora_A.first.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.first.weight", B_PEFT, "F16"),
                     (f"{LAYER}.lora_A.second.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.second.weight", B_PEFT, "F16")])
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("--adapter-name", str(e.exception))
        rows, _ = a.convert(adapter_name="second")
        self.assertEqual(len(rows), 2)

    def test_a_name_that_is_not_a_lora_tensor_is_refused(self):
        a = Adapter([(f"{LAYER}.weight", A_PEFT, "F16")])
        with self.assertRaises(L.LoraPackError):
            a.convert()

    def test_both_spellings_of_the_shared_expert_collide_by_name(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.weight", B_PEFT, "F16"),
                     ("base_model.model.model.layers.3.mlp.gate_proj.lora_A.weight", A_PEFT, "F16"),
                     ("base_model.model.model.layers.3.mlp.gate_proj.lora_B.weight", B_PEFT, "F16"),
                     ("base_model.model.model.layers.3.mlp.shared_expert.gate_proj.lora_A.weight", A_PEFT, "F16"),
                     ("base_model.model.model.layers.3.mlp.shared_expert.gate_proj.lora_B.weight", B_PEFT, "F16")])
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("both map to its lora_A", str(e.exception))


class Refusals(unittest.TestCase):
    def test_rslora(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.weight", B_PEFT, "F16")],
                    cfg={"r": 2, "lora_alpha": 2, "use_rslora": True})
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("use_rslora", str(e.exception))

    def test_a_rank_above_the_engine_maximum(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", np.ones((128, 3)), "F16"),
                     (f"{LAYER}.lora_B.weight", np.ones((3, 128)), "F16")], cfg={"r": 128, "lora_alpha": 128})
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("maximum (64)", str(e.exception))

    def test_a_rank_the_config_disagrees_with(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.weight", B_PEFT, "F16")],
                    cfg={"r": 8, "lora_alpha": 8})
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("rank 2", str(e.exception))

    def test_a_missing_half(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "F16")])
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("lora_B is missing", str(e.exception))

    def test_a_value_fp16_cannot_hold(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", np.full((2, 3), 1e30), "BF16"),
                     (f"{LAYER}.lora_B.weight", B_PEFT, "BF16")], cfg={"r": 2, "lora_alpha": 2})
        with self.assertRaises(L.LoraPackError) as e:
            a.convert()
        self.assertIn("does not fit fp16", str(e.exception))

    def test_bf16_converts_and_round_trips(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "BF16"), (f"{LAYER}.lora_B.weight", B_PEFT, "BF16")])
        rows, _ = a.convert(validate=True)
        self.assertEqual(rows[0][0], "blk.3.attn_k.weight.lora_a")

    def test_alpha_as_a_per_module_dict(self):
        a = Adapter([(f"{LAYER}.lora_A.weight", A_PEFT, "F16"), (f"{LAYER}.lora_B.weight", B_PEFT, "F16")],
                    cfg={"r": 2, "lora_alpha": {"k_proj": 8}})
        rows, _ = a.convert()
        self.assertEqual(rows[0][2], 8.0)


if __name__ == "__main__":
    unittest.main()
