"""tools/phase2/test_teacher_capture.py - tests for the 2.3 teacher capture.

    C:\\tools\\strata\\.venv\\Scripts\\python.exe -m unittest tools.phase2.test_teacher_capture -v

No GPU, no engine binary: a fake tokenizer (word pieces, think markers always
matched like the pack's type-4 tokens), the real chat template, and a scripted
stand-in for the engine's line protocol. The suite pins (1) the tool's
thinking-on rendering byte-identical to Service.prepare(..., enable_thinking
=True), (2) the turn-2 id layout (no second blank line), and (3) one row
captured end to end through a mocked engine: L/Z parsing, the readout's own
renormalization, resume, the manifest, and the out-of-repo refusal.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serve import frontend as fe  # noqa: E402
from serve import logits_first_token as lft  # noqa: E402
from serve import systemone as sysone  # noqa: E402
from tools.phase2 import teacher_capture as tc  # noqa: E402


class FakeTok:
    """Word pieces with the pack's special-token behavior where it matters.

    Specials for the template (<|im_start|> etc.) match only with
    parse_special=True; the think markers match either way, as the pack's
    type-4 tokens do. A leading space merges into the following word, as real
    BPE does (" yes" is one piece), so first-token readings work. Plain spans
    (a </think> quoted in message text) encode as ordinary text.
    """

    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>",
                "<|image_pad|>", "<|vision_end|>"]
    ALWAYS = ["<think>", "</think>"]

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.pieces: list[str] = []

    @property
    def control_tokens(self):
        return list(self.SPECIALS)

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.pieces.append(piece)
        return self.vocab[piece]

    def _split_plain(self, text: str) -> list[str]:
        import re as _re
        return _re.findall(r" ?[A-Za-z]+| ?\d+|\s+|.", text, flags=_re.DOTALL)

    def encode(self, text, parse_special=False, plain=()):
        spans = sorted(plain or [])
        specials = sorted((list(self.SPECIALS) if parse_special else []) + self.ALWAYS,
                          key=len, reverse=True)

        def in_plain(i):
            return any(a <= i < b for a, b in spans)

        out: list[int] = []
        buf: list[str] = []
        pos, n = 0, len(text)
        while pos < n:
            hit = None
            if not in_plain(pos):
                for s in specials:
                    if text.startswith(s, pos):
                        hit = s
                        break
            if hit is not None:
                if buf:
                    out.extend(self._id(p) for p in self._split_plain("".join(buf)))
                    del buf[:]
                out.append(self._id(hit))
                pos += len(hit)
            else:
                buf.append(text[pos])
                pos += 1
        if buf:
            out.extend(self._id(p) for p in self._split_plain("".join(buf)))
        return out

    def decode(self, ids, errors="replace"):
        return "".join(self.pieces[i] for i in ids)

    def token_bytes(self, i: int) -> bytes:
        return self.pieces[i].encode("utf-8")


def real_template():
    return fe.ChatTemplate(REPO_ROOT / "serve" / "chat_template.jinja")


def sample_row(qtype="noul"):
    if qtype == "noul":
        question = {"type": "noul", "instructions": "Is this comment toxic?",
                    "criteria": {"true": "Rude", "false": "Civil"}}
    elif qtype == "choice":
        question = {"type": "choice", "instructions": "Pick one.",
                    "criteria": {"alpha": "the first", "beta": "the second"}}
    else:
        question = {"type": "score", "instructions": "Rate it.",
                    "criteria": ["low", "high"]}
    return {"id": "cfg/train/1", "state": "a short state",
            "question": question, "config": "cfg", "primitive": qtype}


class FakeEngine:
    max_context = 262144
    known_ctx = 262144
    starting = False


class RenderFidelityTests(unittest.TestCase):
    def test_thinking_on_matches_prepare_byte_identical(self):
        tok, template = FakeTok(), real_template()
        for qtype in ("noul", "choice", "score"):
            row = sample_row(qtype)
            state = sysone.serialize_state(row["state"])
            messages, _questions = tc.build_messages(sysone, state, "q", row["question"], "base")
            mine = tc.render_turn1_ids(messages, tok, template)
            from serve.server import Service
            svc = Service(FakeEngine(), tok, template, model_name="t")
            ids, thinking, _max_new = svc.prepare(messages, None, {"enable_thinking": True}, 64)
            self.assertTrue(thinking)
            self.assertEqual(mine, ids)

    def test_thinking_off_prefix_is_prepare_force(self):
        tok, template = FakeTok(), real_template()
        row = sample_row("noul")
        state = sysone.serialize_state(row["state"])
        messages, _questions = tc.build_messages(sysone, state, "q", row["question"], "base")
        from serve.server import Service
        svc = Service(FakeEngine(), tok, template, model_name="t")
        off_ids, thinking, _m = svc.prepare(messages, None, {"enable_thinking": False},
                                            64, force="q:")
        self.assertFalse(thinking)
        prompt = template.render(messages, tools=None, enable_thinking=False)
        self.assertEqual(off_ids, list(tok.encode(prompt, parse_special=True))
                         + list(tok.encode("q:", parse_special=True)))

    def test_turn2_layout_has_no_second_blank_line(self):
        tok = FakeTok()
        prompt = [1, 2, 3]
        close = tok.encode("</think>", parse_special=True)
        self.assertEqual(len(close), 1)
        think = tok.encode(" some thought", parse_special=True) + close
        turn2, sep, prefix = tc.build_turn2_ids(prompt, think, tok)
        self.assertEqual(turn2, prompt + think + sep + prefix)
        self.assertEqual(tok.decode(sep), "\n\n")
        # The thinking already ends with the closing blank line: none is added.
        think2 = think + sep
        turn2b, _s2, _p2 = tc.build_turn2_ids(prompt, think2, tok)
        self.assertEqual(turn2b, prompt + think2 + prefix)


class MockEngineClient(tc.EngineClient):
    """The engine's line protocol, scripted: no subprocess."""

    def __init__(self, script: dict):
        self.script = script
        self.sent: list[str] = []
        self.stall_s = 5

    def _send(self, text: str):
        self.sent.append(text)

    def _next(self, timeout: float, what: str) -> str:
        line = self.script["lines"].pop(0)
        if line == "TIMEOUT":
            raise tc.EngineError("engine-silent: no line during %s" % what)
        return line

    def start(self):
        return 4096

    def stop(self):
        pass


def scripted_engine(tok, think_text=" some thought"):
    think = tok.encode(think_text, parse_special=True)
    close = tok.encode("</think>", parse_special=True)
    gen_lines = ["T %d" % t for t in think + close]
    gen_lines.append("DONE %d 100 10.0 20.0 stop 0 0 0 0 0 0 0 0.0 0 chunk=0" % len(gen_lines))
    return {"lines": gen_lines}


class MockedCaptureTests(unittest.TestCase):
    def test_one_row_end_to_end(self):
        tok, template = FakeTok(), real_template()
        row = sample_row("noul")
        state = sysone.serialize_state(row["state"])
        messages, questions = tc.build_messages(sysone, state, "q", row["question"], "base")
        prompt_ids = tc.render_turn1_ids(messages, tok, template)
        think = tok.encode(" some thought", parse_special=True)
        close = tok.encode("</think>", parse_special=True)
        turn2, _sep, _prefix = tc.build_turn2_ids(prompt_ids, think + close, tok)
        first = lft.first_token_ids(["yes", "no"], tok, "q:")
        topk = {first["yes"]: 2.0, first["no"]: 1.0, 7: -5.0}
        logz = math.log(sum(math.exp(v) for v in topk.values()))
        script = scripted_engine(tok)
        script["lines"].append("Z %.9g" % logz)
        for tid, val in sorted(topk.items(), key=lambda kv: -kv[1]):
            script["lines"].append("L %d %.9g" % (tid, val))
        script["lines"].append("DONE 1 %d 5.0 1.0 stop 0 0 0 0 0 0 0 0.0 0 chunk=0" % len(turn2))
        eng = MockEngineClient(script)
        rec = tc.capture_row(row, tok, template, sysone, "base", 1.0, eng, 4096)
        self.assertEqual(rec["id"], row["id"])
        self.assertAlmostEqual(sum(rec["target"]), 1.0, delta=1e-6)
        self.assertGreater(rec["target"][0], 0.5)  # "yes" carried the higher logit
        self.assertEqual(rec["thinking_tokens"], len(think) + len(close))
        self.assertEqual(rec["prompt_tokens"], len(turn2))
        self.assertTrue(any(s.startswith("GEN ") for s in eng.sent))
        self.assertTrue(any(s.startswith("LOGITS 1024 1 park=0 ") for s in eng.sent))
        self.assertTrue(any(s == "STOP\n" for s in eng.sent))

    def test_failures_are_recorded(self):
        tok, template = FakeTok(), real_template()
        row = sample_row("noul")
        eng = MockEngineClient({"lines": ["DONE 0 10 1.0 0.5 length 0 0 0 0 0 0 0 0.0 0 chunk=0"]})
        with self.assertRaises(tc.CaptureError) as ctx:
            tc.capture_row(row, tok, template, sysone, "base", 1.0, eng, 4096)
        self.assertIn("thinking-unclosed", str(ctx.exception))
        eng2 = MockEngineClient({"lines": ["ERR bad request: max_new"]})
        with self.assertRaises(tc.EngineError):
            tc.capture_row(row, tok, template, sysone, "base", 1.0, eng2, 4096)


class RunTests(unittest.TestCase):
    def _items_file(self, tmp: Path) -> Path:
        rows = []
        for i, qtype in enumerate(("noul", "choice", "score")):
            row = sample_row(qtype)
            row["id"] = "cfg/train/%d" % i
            row["config"] = "cfg"
            rows.append(row)
        path = tmp / "items.jsonl"
        with io.open(str(path), "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def _run_capture(self, tmp: Path, items: Path, out: Path, **kwargs):
        class FakeRunEngine:
            def __init__(self, *a, **k):
                pass

            def start(self):
                return 4096

            def stop(self):
                pass

            def gen_thinking(self, ids, max_new, close_ids):
                return [9] + (list(close_ids or [])), {"finish": "stop"}, True

            def logits_row(self, ids, k):
                return {0: 0.0}, 0.0, {"finish": "stop"}

        with mock.patch.object(tc, "EngineClient", FakeRunEngine), \
                mock.patch.object(tc, "load_tokenizer", return_value=FakeTok()), \
                mock.patch.object(tc, "load_template",
                                    return_value=(real_template(), "fake-template")), \
                mock.patch.object(tc, "target_from_reading", return_value=([0.6, 0.4], ["a", "b"])), \
                mock.patch.object(tc, "sha256_file", return_value="0" * 64):
            args = tc.build_parser().parse_args(
                ["--items", str(items), "--out", str(out), "--configs", "cfg",
                 "--config", str(items), "--engine", sys.executable]
                + kwargs.get("extra_args", []))
            with mock.patch.object(tc, "load_ship_config",
                                   return_value={"exe": sys.executable, "args": [],
                                                 "cwd": str(tmp), "tokenizer": str(tmp)}):
                return tc.run(args)

    def test_resume_manifest_and_out_refusal(self):
        tmp = Path(tempfile.mkdtemp(prefix="tc-test-"))
        items = self._items_file(tmp)
        out = tmp / "cap"
        out.mkdir()
        pre = {"id": "cfg/train/0", "state": "s", "question": {}, "target": [1.0],
               "config": "cfg", "type": "noul", "thinking_tokens": 1,
               "prompt_tokens": 2, "wall_s": 0.1}
        with io.open(str(out / "capture.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(pre) + "\n")

        rc = self._run_capture(tmp, items, out)
        self.assertEqual(rc, 0)
        recs = []
        with io.open(str(out / "capture.jsonl"), encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        self.assertEqual(sorted(r["id"] for r in recs),
                         ["cfg/train/0", "cfg/train/1", "cfg/train/2"])
        with io.open(str(out / "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        # The resumed row counts: 1 carried over plus 2 captured this run.
        self.assertEqual(manifest["rows_completed"], 3)
        self.assertEqual(manifest["rows_failed"], 0)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["prompt_variant"],
                         {"noul": "base", "choice": "base", "score": "base"})

    def test_resume_carries_prior_failure_counts(self):
        # A resumed run recounts capture.jsonl: the pre-existing ok row and
        # the pre-existing ok:false row both appear in the final manifest.
        tmp = Path(tempfile.mkdtemp(prefix="tc-test-"))
        items = self._items_file(tmp)
        out = tmp / "cap"
        out.mkdir()
        pre_ok = {"id": "cfg/train/0", "state": "s", "question": {},
                  "target": [1.0], "config": "cfg", "type": "noul",
                  "thinking_tokens": 1, "prompt_tokens": 2, "wall_s": 0.1}
        pre_fail = {"id": "cfg/train/1", "ok": False,
                    "error": "thinking-unclosed: no </think> in 10 tokens",
                    "config": "cfg", "type": "choice"}
        with io.open(str(out / "capture.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(pre_ok) + "\n")
            f.write(json.dumps(pre_fail) + "\n")

        rc = self._run_capture(tmp, items, out)
        # One carried failure remains, so the run reports failure.
        self.assertEqual(rc, 1)
        with io.open(str(out / "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        # 1 prior ok + 1 new ok; the prior failure is carried, not dropped.
        self.assertEqual(manifest["rows_completed"], 2)
        self.assertEqual(manifest["rows_failed"], 1)
        self.assertEqual([e["id"] for e in manifest["failures"]], ["cfg/train/1"])
        self.assertEqual(manifest["status"], "complete")

    def test_tally_capture_counts_ok_and_failures(self):
        tmp = Path(tempfile.mkdtemp(prefix="tc-test-"))
        cap = tmp / "capture.jsonl"
        rows = [
            {"id": "a", "target": [1.0]},
            {"id": "b", "ok": False, "error": "boom"},
            {"id": "c", "target": [0.5, 0.5]},
        ]
        with io.open(str(cap), "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        completed, failed, failures = tc.tally_capture(cap)
        self.assertEqual((completed, failed), (2, 1))
        self.assertEqual(failures, [{"id": "b", "error": "boom"}])
        self.assertEqual(tc.tally_capture(tmp / "missing.jsonl"), (0, 0, []))

    def test_out_inside_repo_refused(self):
        args = tc.build_parser().parse_args(["--out", str(REPO_ROOT / "cap")])
        with self.assertRaises(tc.CaptureError):
            tc.run(args)

    def test_help_documents_flags(self):
        text = tc.build_parser().format_help()
        for flag in ("--items", "--out", "--limit", "--seed", "--configs", "--engine",
                     "--config", "--prompt-variant", "--temperature", "--stall-s",
                     "--retry-failures"):
            self.assertIn(flag, text)


if __name__ == "__main__":
    unittest.main()
