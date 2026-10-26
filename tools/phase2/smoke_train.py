"""tools/phase2/smoke_train.py - the trivial training run roadmap 2.1 asks for, with no download and no pretrained
weights: a random two-layer GPT, a LoRA on one projection, eight rows of made-up token ids, three steps.

    python tools/phase2/smoke_train.py            # CPU, the default here: the card carries the user's model
    python tools/phase2/smoke_train.py --cuda     # the same on the GPU, when it is free (2.1's exit criterion)

What it proves: torch, transformers, peft, accelerate and datasets import and work together; a PEFT adapter can be
built on a transformers model, trained, and its trainable-parameter count is what the LoRA config says; and the
resulting adapter is exactly what `tools/lora_pack.py` converts (PEFT's own `.lora_A`/`.lora_B` names - the
converter's test uses the same shape).

What it does NOT prove: that the model or the data are real, that the adapter is useful, or that a full-size
training run fits in VRAM.  2.1's exit criterion is the GPU variant of this run; it needs the card free.
"""
from __future__ import annotations

import argparse
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

VOCAB = 64


def rows(n: int, length: int = 16) -> Dataset:
    """A dataset of made-up token ids: no tokenizer, no download, and nothing to learn - which is the point."""
    g = torch.Generator().manual_seed(1234)
    out = []
    for _ in range(n):
        ids = torch.randint(0, VOCAB, (length,), generator=g).tolist()
        out.append({"input_ids": ids, "attention_mask": [1] * length, "labels": list(ids)})
    return Dataset.from_list(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cuda", action="store_true", help="train on the GPU (2.1's exit criterion; needs the card free)")
    ap.add_argument("--out", default="phase2-smoke-adapter")
    a = ap.parse_args(argv)

    device = "cpu"
    if a.cuda:
        if not torch.cuda.is_available():
            print("[smoke] --cuda asked for, but torch cannot see a CUDA device")
            return 2
        device = "cuda"
    base = GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=32, n_embd=32, n_layer=2, n_head=2))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["c_attn"]))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("[smoke] torch %s, device %s, trainable tensors %d, trainable params %d"
          % (torch.__version__, device, len(trainable), sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("[smoke] the first PEFT names: %s" % ", ".join(trainable[:4]))

    args = TrainingArguments(output_dir=a.out, max_steps=3, per_device_train_batch_size=2, learning_rate=1e-2,
                             logging_steps=1, save_strategy="no", report_to=[], use_cpu=device == "cpu",
                             dataloader_num_workers=0, seed=1234)
    trainer = Trainer(model=model, args=args, train_dataset=rows(8))
    trainer.train()
    model.save_pretrained(a.out)
    losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    print("[smoke] trained 3 steps on %s; losses %s; adapter written to %s"
          % (device, ", ".join("%.3f" % x for x in losses) or "n/a", a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
