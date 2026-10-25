"""tools/phase2/teacher_capture.py - the 2.3 teacher capture: thinking-on answer-position targets.

WHAT IT DOES. For each training row it renders the decision prompt with thinking
ON using the server's own functions, runs two engine turns, and writes the
answer-position distribution the endpoint would report:

  turn 1  render system_message + render_answer_prompt through the pack's chat
          template with enable_thinking=True (the same ids Service.prepare
          would send), then GEN the thinking, stopping at the </think> marker.
  turn 2  append the thinking through the template's closing blank line plus
          the answer prefix ("q:") and send LOGITS with park=0 (a one-shot
          read), then renormalize with the readout's own readings() call.

Output is outside the repository: <out>/capture.jsonl (append-only, keyed by
row id, so a rerun resumes) plus <out>/manifest.json (engine binary hash, the
shipping config copy and diff, seed, prompt variant, temperature, row counts).

Run it by hand; nothing imports it. One invocation starts exactly one fresh
engine subprocess (the program's fresh-engine-per-run rule) and nothing else
may hold the card while it runs.

WHAT IT REFUSES.
  * It never edits serve/, src/, or any config file: the shipping config is
    read read-only and the engine command line is built from its "args".
  * It never trains anything and never writes an adapter.
  * It never touches C++ or rebuilds the engine.
  * --out inside the repository is refused (a capture next to the code would
    be committed by accident).
  * A row that cannot be read (context overflow, unclosed thinking, no logits,
    no renormalization, a target that does not sum to 1) is a recorded
    FAILURE, never a silent skip.

Environment: the server venv (C:\\tools\\strata\\.venv), which has jinja2 and
regex for the template and the reference tokenizer. No torch.

    python tools/phase2/teacher_capture.py --out <dir-outside-repo> --limit 100
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serve import frontend as fe  # noqa: E402
from serve import logits_first_token as lft  # noqa: E402
from serve import systemone as sysone  # noqa: E402
from tools.strata_tokenizer import Tokenizer as RefTokenizer  # noqa: E402

QUESTION_NAME = "q"
THINK_CLOSE_TEXT = "</think>"
CTX_SLACK = 8  # serve/server.py: the same margin the server keeps past prompt + max_new
MIN_THINK_ROOM = 16  # fewer free tokens than this is not a thinking pass: overflow
GEN_MAX_NEW_CAP = 32768  # a thinking pass longer than this is recorded unclosed, not runaway
TOP_K = lft.LOGITS_TOP_K
FINISH_OK = ("stop", "length", "cancel")  # DONE words accepted once </think> was seen
FAILURE_ABORT_AFTER = 3  # consecutive row failures of the same engine kind abort the run


class CaptureError(RuntimeError):
    """A row that cannot be captured: recorded, never silently dropped."""


class EngineError(RuntimeError):
    """The engine answered ERR, died, or stayed silent past the stall budget."""


# ------------------------------------------------------------------ paths
def default_data_root() -> Path:
    base = os.environ.get("STRATA_DECISION_DATA") or os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Temp", "opencode")
    return Path(base)


def default_items() -> Path:
    return default_data_root() / "decision_policy_training" / "items" / "train_soft.jsonl"


def default_ship_config() -> Path:
    env = os.environ.get("STRATA_SHIP_CONFIG")
    if env:
        return Path(env)
    return REPO_ROOT.parent / "strata" / "strata-unsloth-ud-iq4_xs.json"


def refuse_repo_path(out: Path) -> Path:
    """--out must exist outside the repository; a capture inside it would be committed."""
    resolved = out.resolve()
    root = REPO_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise CaptureError(
            "--out %s is inside the repository (%s): captures live outside it" % (out, root))
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ config / tokenizer / template
def load_ship_config(path: Path) -> dict:
    if not path.exists():
        raise CaptureError(
            "shipping config not found at %s: pass --config (it is read, never edited)" % path)
    with io.open(str(path), encoding="utf-8") as f:
        return json.load(f)


def scrub_config(cfg: dict) -> dict:
    """The manifest's config copy: everything except secrets."""
    return {k: v for k, v in cfg.items() if k not in ("api_key",)}


def load_tokenizer(tokenizer_dir: Path):
    """The pack's tokenizer, the same way serve/server.py loads it (vocab + merges + types)."""
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for text, i in vocab.items():
        tokens[i] = text
    merges = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8"))
    return RefTokenizer(tokens, merges, types)


def load_template(tokenizer_dir: Path, repo_root: Path):
    """The pack's chat template when it has one, else the server's own (as server.py does)."""
    tpl = tokenizer_dir / "chat_template.jinja"
    if not tpl.exists():
        tpl = repo_root / "serve" / "chat_template.jinja"
    return fe.ChatTemplate(tpl), str(tpl)


# ------------------------------------------------------------------ items
def load_items(path: Path) -> list[dict]:
    rows = []
    with io.open(str(path), encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise CaptureError("%s line %d: not JSON (%s)" % (path, lineno, e))
    if not rows:
        raise CaptureError("%s: no rows" % path)
    return rows


def select_rows(rows: list[dict], configs: list[str], limit: int | None, seed: int) -> list[dict]:
    """Stratified pick for the pilot, file order otherwise.

    With --limit: per config, seeded-shuffle its rows (file order before that)
    and take ceil/remainder quotas in --configs order (limit // C each, the
    first limit % C configs take one more), then seeded-shuffle the take
    again. limit=100 over three configs with seed 20261023 is 34/33/33.
    Without --limit: --configs still filters, and --seed shuffles everything.
    """
    by_config: dict[str, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row.get("config", ""), []).append(row)
    unknown = [c for c in configs if c not in by_config]
    if unknown:
        raise CaptureError("configs not in %s: %s" % (sorted(by_config), ", ".join(unknown)))
    if limit is not None:
        if limit <= 0:
            raise CaptureError("--limit takes a whole number of 1 or more")
        rng = random.Random(seed)
        quotas: dict[str, int] = {}
        per, extra = divmod(limit, len(configs))
        for i, cfg in enumerate(configs):
            quotas[cfg] = per + (1 if i < extra else 0)
        take: list[dict] = []
        for cfg in configs:
            pool = list(by_config[cfg])
            rng.shuffle(pool)
            take.extend(pool[:quotas[cfg]])
        rng2 = random.Random(seed + 1)
        rng2.shuffle(take)
        return take
    picked = [row for cfg in configs for row in by_config[cfg]]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(picked)
    return picked


# ------------------------------------------------------------------ rendering (the server's own functions only)
def build_messages(readout, state_text: str, name: str, q: dict, variant: str) -> tuple[list[dict], list]:
    questions = readout.validate_questions({name: {
        "type": q["type"], "instructions": q["instructions"], "criteria": q.get("criteria")}})
    messages = [{"role": "system", "content": readout.system_message(questions)},
                {"role": "user", "content": readout.render_answer_prompt(
                    state_text, name, questions[0][1], variant)}]
    return messages, questions


def render_turn1_ids(messages: list[dict], tok, template) -> list[int]:
    """The thinking-on prompt ids, exactly as Service.encode_prompt builds them.

    Mirrors serve/server.py Service.encode_prompt minus the PromptEncoder (an
    exact cache) and the image path (decision rows have no pictures): the same
    literal marking, the same template call with enable_thinking=True, the same
    encode. The fidelity test pins this against Service.prepare.
    """
    tags = fe.literal_tags(getattr(tok, "control_tokens", ()))
    marked, _, changed = fe.mark_think_literals(messages, None, tags)
    prompt = template.render(marked, tools=None, enable_thinking=True)
    if changed:
        prompt, plain = fe.unmark_think_literals(prompt, tags)
        return list(tok.encode(prompt, parse_special=True, plain=plain)) if plain else \
            list(tok.encode(prompt, parse_special=True))
    return list(tok.encode(prompt, parse_special=True))


def special_ids(tok, texts: list[str]) -> dict[str, list[int]]:
    return {t: list(tok.encode(t, parse_special=True)) for t in texts}


def build_turn2_ids(prompt_ids: list[int], think_ids: list[int], tok) -> tuple[list[int], list[int], list[int]]:
    """prompt + thinking (through </think>) + the template's closing blank line + prefix.

    The blank line is appended ONLY when the thinking does not already end
    with it: no second "\\n\\n". Returns (turn2_ids, sep_ids, prefix_ids).
    """
    prefix_ids = list(tok.encode(QUESTION_NAME + ":", parse_special=True))
    sep_ids = list(tok.encode("\n\n", parse_special=True))
    tail = think_ids[-len(sep_ids):] if sep_ids else []
    sep = [] if (sep_ids and list(tail) == list(sep_ids)) else list(sep_ids)
    return list(prompt_ids) + list(think_ids) + sep + list(prefix_ids), sep, prefix_ids


def target_from_reading(readout, questions: list, topk: dict, tok, prefix: str,
                        logz: float | None, temperature: float) -> tuple[list[float], list[str]]:
    """The endpoint's own renormalization, and nothing else, over the options only."""
    readings = readout.readings(questions, topk, tok, prefix, logz, temperature=temperature)
    if not readings or questions[0][0] not in readings:
        reason = readout.fallback_reason(questions, topk or {}, tok, prefix, logz)
        raise CaptureError("renormalization-none: %s" % reason)
    q = questions[0][1]
    options = lft.option_names_for_question(q)
    closed = readings[questions[0][0]]["closed"]
    target = [float(closed[o]) for o in options]
    if abs(sum(target) - 1.0) > 1e-6:
        raise CaptureError("target-sum: options sum to %.9f, not 1" % sum(target))
    return target, options


# ------------------------------------------------------------------ engine subprocess (line protocol only)
class EngineClient:
    """One engine subprocess speaking GEN / LOGITS / STOP / QUIT on stdio.

    Only the line protocol is reused here (serve/server.py's Engine is the
    reference for the reading side); the rendering above it is the tool's own
    two-turn sequencing. One instance per run: started, used for every row,
    stopped at the end.
    """

    def __init__(self, argv: list[str], cwd: str, env: dict, stall_s: float, log_path: Path):
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.stall_s = stall_s
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.max_context = 0

    def start(self) -> int:
        log = open(self.log_path, "ab")
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, env=self.env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                bufsize=1)
        except OSError as e:
            raise EngineError("cannot start %s: %s" % (self.argv[0], e))
        threading.Thread(target=self._pump, daemon=True).start()
        line = self._next(timeout=900.0, what="READY")
        while line.startswith("INFO"):
            line = self._next(timeout=900.0, what="READY")
        if not line.startswith("READY"):
            raise EngineError("expected READY, got %r" % line[:120])
        try:
            self.max_context = int(line.split()[1])
        except (IndexError, ValueError):
            raise EngineError("unparseable READY line: %r" % line[:120])
        if self.max_context <= 0:
            raise EngineError("engine reports no context: %r" % line[:120])
        return self.max_context

    def _pump(self):
        try:
            assert self.proc is not None and self.proc.stdout is not None
            for raw in self.proc.stdout:
                try:
                    self.lines.put(raw.decode("utf-8", errors="replace"))
                except Exception:
                    self.lines.put("")
        finally:
            self.lines.put(None)

    def _send(self, text: str):
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write(text.encode("utf-8"))
            self.proc.stdin.flush()
        except OSError:
            raise EngineError("the engine stopped (exit code %s)" % self.proc.poll())

    def _next(self, timeout: float, what: str) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise EngineError("engine-silent: no line during %s for %.0f s" % (what, timeout))
        if line is None:
            code = self.proc.poll() if self.proc else None
            raise EngineError("the engine stopped (exit code %s) during %s" % (code, what))
        return line.strip()

    @staticmethod
    def _done_fields(line: str) -> dict:
        parts = line.split()
        try:
            return {"generated": int(parts[1]), "prompt_tokens": int(parts[2]),
                    "prompt_ms": float(parts[3]), "decode_ms": float(parts[4]),
                    "finish": parts[5] if len(parts) > 5 else ""}
        except (IndexError, ValueError):
            return {"generated": 0, "prompt_tokens": 0, "prompt_ms": 0.0,
                    "decode_ms": 0.0, "finish": ""}

    def gen_thinking(self, ids: list[int], max_new: int, close_ids: list[int] | None) -> tuple[list[int], dict, bool]:
        """Turn 1: GEN the thinking; stop the engine once the close marker is seen.

        Returns (think_ids through </think>, DONE fields, closed). An
        unclosed pass (DONE without the marker) is the caller's failure, not
        an exception: its rows are still recorded.
        """
        csv = ",".join(str(int(t)) for t in ids)
        self._send("GEN %d %s\n" % (int(max_new), csv))
        tokens: list[int] = []
        done: dict = {}
        closed = False
        stop_sent = False
        while True:
            line = self._next(self.stall_s, what="GEN")
            if line.startswith("T "):
                try:
                    tid = int(line[2:])
                except ValueError:
                    continue
                tokens.append(tid)
                if close_ids is not None and not closed and len(tokens) >= len(close_ids):
                    if tokens[-len(close_ids):] == close_ids:
                        closed = True
                        self._send("STOP\n")
                        stop_sent = True
            elif line.startswith("DONE"):
                done = self._done_fields(line)
                break
            elif line.startswith("ERR"):
                raise EngineError("GEN: %s" % line[4:].strip())
            # PP / RESUME / REUSED lines are progress, not content.
        _ = stop_sent
        if close_ids is not None and closed:
            idx = next(i for i in range(len(tokens) - len(close_ids) + 1)
                       if tokens[i:i + len(close_ids)] == close_ids)
            tokens = tokens[:idx + len(close_ids)]
        return tokens, done, closed

    def logits_row(self, ids: list[int], k: int) -> tuple[dict[int, float], float | None, dict]:
        """Turn 2: LOGITS with park=0 (a one-shot read); returns (topk, logz, DONE fields)."""
        csv = ",".join(str(int(t)) for t in ids)
        self._send("LOGITS %d 1 park=0 %s\n" % (int(k), csv))
        topk: dict[int, float] = {}
        logz: float | None = None
        done: dict = {}
        while True:
            line = self._next(self.stall_s, what="LOGITS")
            if line.startswith("L "):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        topk[int(parts[1])] = float(parts[2])
                    except ValueError:
                        pass
            elif line.startswith("Z "):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        logz = float(parts[1])
                    except ValueError:
                        pass
            elif line.startswith("DONE"):
                done = self._done_fields(line)
                break
            elif line.startswith("ERR"):
                raise EngineError("LOGITS: %s" % line[4:].strip())
            # T / PP / RESUME lines carry no row.
        return topk, logz, done

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(b"QUIT\n")
                proc.stdin.flush()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


# ------------------------------------------------------------------ capture
def capture_row(row: dict, tok, template, readout, variant: str, temperature: float,
                engine: EngineClient, ctx: int) -> dict:
    t0 = time.monotonic()
    q = row["question"]
    state_text = readout.serialize_state(row["state"])
    messages, questions = build_messages(readout, state_text, QUESTION_NAME, q, variant)
    prompt_ids = render_turn1_ids(messages, tok, template)
    room = ctx - CTX_SLACK - len(prompt_ids)
    if room < MIN_THINK_ROOM:
        raise CaptureError("context-overflow: prompt %d tokens leaves %d of %d" % (
            len(prompt_ids), room, ctx))
    close = special_ids(tok, [THINK_CLOSE_TEXT])[THINK_CLOSE_TEXT]
    think_ids, done1, closed = engine.gen_thinking(
        prompt_ids, min(room, GEN_MAX_NEW_CAP), close if len(close) == 1 else None)
    if close is not None and len(close) != 1:
        # A vocabulary where </think> is not one token: search the decoded text.
        text = tok.decode(think_ids)
        at = text.find(THINK_CLOSE_TEXT)
        if at < 0:
            raise CaptureError("thinking-unclosed: no %s in %d tokens (finish %s)" % (
                THINK_CLOSE_TEXT, len(think_ids), done1.get("finish", "?")))
        head = tok.encode(text[:at + len(THINK_CLOSE_TEXT)], parse_special=True)
        think_ids = list(head)
        closed = True
    if not closed:
        raise CaptureError("thinking-unclosed: no %s in %d tokens (finish %s)" % (
            THINK_CLOSE_TEXT, len(think_ids), done1.get("finish", "?")))
    turn2_ids, _sep, _prefix = build_turn2_ids(prompt_ids, think_ids, tok)
    if len(turn2_ids) + 1 + CTX_SLACK > ctx:
        raise CaptureError("context-overflow: turn 2 needs %d tokens of %d" % (
            len(turn2_ids), ctx))
    topk, logz, _done2 = engine.logits_row(turn2_ids, TOP_K)
    if not topk:
        raise CaptureError("no-logits: turn 2 returned no L lines")
    if logz is None:
        top = max(topk.values())
        logz = top + math.log(sum(math.exp(v - top) for v in topk.values()))
    prefix = QUESTION_NAME + ":"
    target, _options = target_from_reading(
        readout, questions, topk, tok, prefix, logz, temperature)
    wall_s = time.monotonic() - t0
    return {"id": row["id"], "state": row.get("state"), "question": dict(questions[0][1]),
            "target": target, "config": row.get("config"), "type": questions[0][1]["type"],
            "thinking_tokens": len(think_ids), "prompt_tokens": len(turn2_ids),
            "wall_s": round(wall_s, 3)}


def failure_row(row: dict, error: str) -> dict:
    q = row.get("question") or {}
    return {"id": row.get("id"), "ok": False, "error": error,
            "config": row.get("config"), "type": q.get("type")}


# ------------------------------------------------------------------ run
def read_done_ids(capture_path: Path, retry_failures: bool) -> set[str]:
    done: set[str] = set()
    if not capture_path.exists():
        return done
    with io.open(str(capture_path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if retry_failures and rec.get("ok") is False:
                continue
            if rec.get("id") is not None:
                done.add(rec["id"])
    return done


def tally_capture(capture_path: Path) -> tuple[int, int, list[dict]]:
    """Recount rows already in capture.jsonl so a resumed run keeps full totals.

    Keyed by row id (last record wins); a record counts as failed only when
    it says ok:false, otherwise it counts as completed.
    """
    by_id: dict[str, dict] = {}
    if not capture_path.exists():
        return 0, 0, []
    with io.open(str(capture_path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("id") is not None:
                by_id[rec["id"]] = rec
    completed = sum(1 for rec in by_id.values() if rec.get("ok") is not False)
    failures = [{"id": rid, "error": rec.get("error", "")}
                for rid, rec in by_id.items() if rec.get("ok") is False]
    return completed, len(failures), failures


def write_manifest(path: Path, manifest: dict):
    tmp = path.with_suffix(".tmp")
    with io.open(str(tmp), mode="w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def run(args) -> int:
    out = refuse_repo_path(Path(args.out))
    out.mkdir(parents=True, exist_ok=True)
    capture_path = out / "capture.jsonl"
    manifest_path = out / "manifest.json"

    rows = load_items(Path(args.items))
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] if args.configs \
        else sorted({r.get("config", "") for r in rows})
    selected = select_rows(rows, configs, args.limit, args.seed)

    ship = load_ship_config(Path(args.config or default_ship_config()))
    for qtype in ("noul", "choice", "score"):
        known = sysone.PROMPT_VARIANTS[qtype]
        if args.prompt_variant not in known:
            raise CaptureError("unknown %s prompt variant %r: use one of %s" % (
                qtype, args.prompt_variant, ", ".join(repr(v) for v in known)))
    tokenizer_dir = Path(ship.get("tokenizer", ""))
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = (Path(ship.get("cwd") or REPO_ROOT) / tokenizer_dir)
    tok = load_tokenizer(tokenizer_dir)
    template, template_path = load_template(tokenizer_dir, REPO_ROOT)
    readout = sysone  # the direct readout; the shipping config uses no letters readout

    engine_exe = args.engine or ship.get("exe", "")
    # serve/server.py:1030 starts the engine as [exe, "--serve", *args]: the
    # shipping config's "args" never carry the subcommand, so the tool adds it
    # exactly as the server does (without it the binary runs its generate CLI).
    argv = [str(engine_exe), "--serve"] + [str(a) for a in ship.get("args", [])]
    cwd = str(ship.get("cwd") or REPO_ROOT)
    env = dict(os.environ)
    for lib in ship.get("lib_dirs", []) or []:
        env["PATH"] = str(lib) + os.pathsep + env.get("PATH", "")
    env.update({k: str(v) for k, v in (ship.get("env") or {}).items()})
    engine = EngineClient(argv, cwd, env, args.stall_s, out / "engine.log")
    ctx = engine.start()

    manifest: dict = {
        "tool": "tools/phase2/teacher_capture.py",
        "engine": {"exe": str(engine_exe), "sha256": sha256_file(Path(engine_exe))},
        "engine_argv": argv, "engine_cwd": cwd,
        "config_source": str(Path(args.config or default_ship_config())),
        "config": scrub_config(ship),
        "config_diff": {
            "exe": {"config": ship.get("exe"), "used": str(engine_exe)},
            "args": "as in the shipping config, with --serve inserted after the exe "
                     "(serve/server.py:1030 starts it the same way; no other additions, no removals)",
            "note": "server-level keys (port, host, api_key) are not engine arguments "
                    "and were not used; the config file itself was never edited",
        },
        "tokenizer": str(tokenizer_dir), "template": template_path,
        "seed": args.seed, "prompt_variant_requested": args.prompt_variant,
        "prompt_variant": {t: args.prompt_variant for t in ("noul", "choice", "score")},
        "prompt_variant_note": "base for ALL types by owner decision (2026-10-23); "
                               "the served noul uses criteria-first, so the teacher "
                               "sees the base wording there",
        "temperature": args.temperature, "question_name": QUESTION_NAME,
        "sampling": {"turn1": "engine default (greedy: no sampling keys on the GEN line)",
                     "turn2": "answer-position row only (LOGITS, park=0)"},
        "limit": args.limit, "configs": configs, "items": str(args.items),
        "engine_per_run": "one fresh engine subprocess per invocation; nothing else on the card",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "running", "rows_completed": 0, "rows_failed": 0, "failures": [],
    }
    write_manifest(manifest_path, manifest)

    done_ids = read_done_ids(capture_path, args.retry_failures)
    prior_completed, prior_failed, prior_failures = tally_capture(capture_path)
    manifest["rows_completed"] = prior_completed
    manifest["rows_failed"] = prior_failed
    manifest["failures"] = list(prior_failures)
    prior_failed_ids = {e["id"] for e in prior_failures}
    consecutive = 0
    with io.open(str(capture_path), encoding="utf-8", mode="a", newline="\n") as f:
        for row in selected:
            if row.get("id") in done_ids:
                continue
            if row.get("id") in prior_failed_ids:
                # --retry-failures re-attempts this id: retract its stale tally
                # entry first so the recount below does not double count it.
                prior_failed_ids.discard(row.get("id"))
                manifest["rows_failed"] -= 1
                manifest["failures"] = [e for e in manifest["failures"]
                                        if e["id"] != row.get("id")]
            try:
                rec = capture_row(row, tok, template, readout, args.prompt_variant,
                                  args.temperature, engine, ctx)
            except CaptureError as e:
                rec = failure_row(row, str(e))
                manifest["rows_failed"] += 1
                manifest["failures"].append({"id": row.get("id"), "error": str(e)})
                consecutive = 0
            except EngineError as e:
                rec = failure_row(row, "engine-error: %s" % e)
                manifest["rows_failed"] += 1
                manifest["failures"].append({"id": row.get("id"), "error": "engine-error: %s" % e})
                consecutive += 1
                if consecutive >= FAILURE_ABORT_AFTER:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    manifest["status"] = "aborted: %d consecutive engine errors (%s)" % (
                        consecutive, e)
                    write_manifest(manifest_path, manifest)
                    engine.stop()
                    print("[capture] aborted: %s" % e)
                    return 3
            else:
                manifest["rows_completed"] += 1
                consecutive = 0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            write_manifest(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_manifest(manifest_path, manifest)
    engine.stop()
    print("[capture] %d rows captured, %d failed -> %s" % (
        manifest["rows_completed"], manifest["rows_failed"], capture_path))
    return 0 if manifest["rows_failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="2.3 teacher capture: thinking-on answer-position targets, two engine turns per row.")
    ap.add_argument("--items", default=str(default_items()),
                    help="training rows JSONL (default: STRATA_DECISION_DATA/decision_policy_training/items/train_soft.jsonl)")
    ap.add_argument("--out", required=True,
                    help="REQUIRED: output directory OUTSIDE the repository (capture.jsonl + manifest.json)")
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot mode: capture this many rows, stratified by --configs (100 -> 34/33/33)")
    ap.add_argument("--seed", type=int, default=20261023,
                    help="sampling seed, recorded in the manifest (default 20261023, the pilot's)")
    ap.add_argument("--configs", default=None,
                    help="comma-separated configs to capture (default: every config in --items)")
    ap.add_argument("--engine", default=None,
                    help="engine executable (default: the shipping config's exe)")
    ap.add_argument("--config", default=None,
                    help="shipping config to read engine args from, read-only (default: the checkout's)")
    ap.add_argument("--prompt-variant", default="base",
                    help="rendering variant for ALL types, recorded in the manifest (default base, the owner decision)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="readout temperature for the renormalization (default 1.0, the endpoint's value)")
    ap.add_argument("--stall-s", type=float, default=900.0,
                    help="abandon a row when the engine says nothing for this long (seconds, default 900)")
    ap.add_argument("--retry-failures", action="store_true",
                    help="re-attempt rows recorded ok:false (default: skip every id already present)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (CaptureError, EngineError) as e:
        print("[capture] %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
