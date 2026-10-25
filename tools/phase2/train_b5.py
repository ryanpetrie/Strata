"""tools/phase2/train_b5.py - the Phase 2 decision-adapter training recipe (Stream B, unit B5).

The pre-registration (docs/research/phase2_training_arms.md 4-5) fixes the recipe;
the values below were fixed 2026-10-25, before any adapter number exists:

    rank 16, alpha 16.0, QSA projections only (layers 3, 7, ..., 47),
    AdamW lr 1e-4, batch 8, seed 20261025, 1-3 epochs chosen by the develop set.
    The ONLY difference between the three arms is the target tensor:
      teacher: the capture's thinking-on distribution (targets_teacher.jsonl, 5997)
      human:   people's distribution (train_soft.jsonl target_probs, 24000)
      control: the teacher vectors under a seeded permutation (targets_control.jsonl)
    Loss: soft cross-entropy (KL) of the model's answer-position distribution over
    the question's answer options vs the target - the endpoint's own closed reading
    (serve/logits_first_token.py first_token_probs): softmax over the options'
    first-token logits, renormalized over the options only. No hard-label loss.

Inputs are rendered with the endpoint's own functions (serve.systemone, read-only):
render_answer_prompt(variant base, name "q") and first_token_plan with prefix "q:",
the capture's own convention (tools/phase2/teacher_capture.py:473). Measured
2026-10-25 (b5-training/option_mapping.json): zero first-token collisions on all
5997 + 24000 rows, so the loss is defined on every row.

Refusals (fail loud, no silent subset): a modelled layer outside the 12 QSA
layers, anything but 48 LoRA targets, use_rslora, an option collision, a target
that is not a distribution, a tokenizer mismatch on the option strings.

Modes:
    --smoke            toy model + synthetic rows: proves the loss, the PEFT
                       targeting (layers_to_transform on the QSA pattern), the
                       save format and the converter path, on CPU in seconds.
    --arm {teacher,human,control} --epochs N
                       the real run. Needs the HF checkpoint
                       (Qwen/Qwen3.8-Flash-Next, ~360 GB in BF16) and a card that
                       holds it: NOT run here, see the B5 report for the numbers.

    C:\\tools\\strata-train\\Scripts\\python.exe tools/phase2/train_b5.py --smoke
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEED = 20261025
RANK = 16
ALPHA = 16.0
QSA_LAYERS = [l for l in range(48) if l % 4 == 3]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
N_TARGETS = len(QSA_LAYERS) * len(TARGET_MODULES)  # 48
LR = 1e-4
BATCH = 8
QUESTION_NAME = "q"
PREFIX = QUESTION_NAME + ":"

DATA = pathlib.Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode\b5-training")
TRAIN_ITEMS = pathlib.Path(r"C:\Users\ryanp\AppData\Local\Temp\opencode"
                           r"\decision_policy_training\items\train_soft.jsonl")


def load_rows(arm):
    """(prompt_inputs, option_ids, targets): lists aligned by row."""
    if arm == "human":
        rows = [json.loads(l) for l in io.open(TRAIN_ITEMS, encoding="utf-8") if l.strip()]
        by_id = {}
        for r in rows:
            tp, labels = r["target_probs"], r["labels"]
            by_id[r["id"]] = ([tp.get(lb, 0.0) for lb in labels], r)
        return by_id
    src = DATA / ("targets_teacher.jsonl" if arm == "teacher" else "targets_control.jsonl")
    by_id = {}
    for line in io.open(src, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            by_id[d["id"]] = (d["target"], None)
    return by_id


def normalize_question(raw):
    """The endpoint's normalized question (validated, or the capture's stored form)."""
    from serve import systemone as sysone
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    try:
        return sysone.validate_questions({QUESTION_NAME: raw})[0][1]
    except Exception:
        crit = raw.get("criteria") if isinstance(raw, dict) else None
        if (isinstance(raw, dict) and isinstance(crit, dict)
                and isinstance(crit.get("levels"), list)):
            return raw  # the capture's normalized shape; option_names reads it
        raise


def soft_ce(logits, target):
    """-sum(t * log softmax(l)) over options; logits [B, O] (padded), target [B, O]."""
    import torch
    logp = logits - logits.amax(dim=1, keepdim=True)
    logp = logp - logp.exp().sum(dim=1, keepdim=True).log()
    return (-target * logp).sum(dim=1).mean()


# ----------------------------------------------------------------------- smoke
def build_toy(vocab=64, width=32, layers=48, seed=SEED):
    """A toy with the QSA module paths, so the PEFT targeting is the real one."""
    import numpy as np
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(width, width)
            self.k_proj = nn.Linear(width, width)
            self.v_proj = nn.Linear(width, width)
            self.o_proj = nn.Linear(width, width)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([Layer() for _ in range(layers)])
            self.language_model.embed = nn.Embedding(vocab, width)
            self.language_model.head = nn.Linear(width, vocab, bias=False)

        def forward(self, ids):
            h = self.language_model.embed(ids)
            for layer in self.language_model.layers:
                a = layer.self_attn
                h = h + a.o_proj(a.q_proj(h) + a.k_proj(h) + a.v_proj(h))
            return self.language_model.head(h[:, -1:, :])

    m = Toy()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.from_numpy(
                (rng.standard_normal(p.shape) * 0.1).astype("float32")))
    return m


def smoke(out):
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    model = build_toy()
    cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                     target_modules=TARGET_MODULES,
                     layers_to_transform=list(QSA_LAYERS),
                     layers_pattern="language_model.layers",
                     use_rslora=False)
    model = get_peft_model(model, cfg)
    names = [n for n, _ in model.named_parameters() if ".lora_A." in n]
    n_targets = len(names)
    assert n_targets == N_TARGETS, f"{n_targets} LoRA targets, not {N_TARGETS}"
    assert model.peft_config["default"].r == RANK
    assert not model.peft_config["default"].use_rslora

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(SEED)
    losses = []
    model.train()
    for step in range(4):
        ids = torch.randint(0, 64, (BATCH, 6))
        logits = model(ids).squeeze(1)  # [B, vocab]
        opt_ids = torch.tensor(rng.choice(64, size=(BATCH, 3), replace=False))
        opt_logits = logits.gather(1, opt_ids)
        tgt = torch.from_numpy(rng.dirichlet([1, 1, 1], size=BATCH).astype("float32"))
        loss = soft_ce(opt_logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(v == v and abs(v) < 1e6 for v in losses), losses

    # the engine's format is fp16: cast A/B before save, and --validate below
    # pins that the file equals the source exactly. A value fp16 cannot hold
    # is refused by the converter instead of being silently rounded.
    for n, p in model.named_parameters():
        if ".lora_" in n:
            p.data = p.data.to(torch.float16)
    ado = pathlib.Path(out) / "smoke-adapter"
    model.save_pretrained(ado)
    saved = json.loads((ado / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["r"] == RANK and not saved.get("use_rslora"), saved

    sys.path.insert(0, str(REPO / "tools"))
    import lora_pack
    rows, nbytes = lora_pack.convert(str(ado), str(pathlib.Path(out) / "smoke-packed"),
                                     validate=True, log=None)
    assert len(rows) == 2 * N_TARGETS, len(rows)
    print(f"[smoke] {N_TARGETS} targets, losses "
          + " ".join(f"{v:.3f}" for v in losses) + ", packed {0} bytes".format(nbytes))
    return losses


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=["teacher", "human", "control"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default=str(DATA / "adapters"))
    a = ap.parse_args(argv)
    if a.smoke:
        smoke(a.out)
        return 0
    if not a.arm:
        ap.error("--arm is required without --smoke")
    if not 1 <= a.epochs <= 3:
        ap.error("the pre-registration expects 1-3 epochs; the develop set chooses")
    print(f"[{a.arm}] rank {RANK} QSA-only x{N_TARGETS}, AdamW lr {LR}, "
          f"batch {BATCH}, epochs {a.epochs}, seed {SEED}")
    print("the real run needs the HF checkpoint and a card that holds it; not run here.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
