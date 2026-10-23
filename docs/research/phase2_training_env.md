# Phase 2.1: the training environment (2026-10-22)

**Status: DONE (2026-10-22).**  Installed and smoke-tested, and the exit criterion is met: on 2026-10-22 the
trivial run completed **on the 4090** (`--cuda`: 3 steps, losses 3.877, 3.886, 3.875, adapter written), with the
server stopped for the run.  This is roadmap item
[2.1](decision_policy_roadmap.md#21-training-infrastructure-prerequisite) - the prerequisite for 2.2's
verification, 2.3 and 2.3b.  Nothing here touches the server's own environment.

## Where it is, and why there

| | |
|---|---|
| the environment | a **separate venv outside the repository** (`C:\tools\strata-train` on this machine) |
| why separate | `C:\tools\strata\.venv` is the **server's** environment: it is pinned to what the engine's host process needs, and the engine is the user's running program. Adding torch, transformers and trl to it would change what every server start imports. |
| the repo's venv | unchanged; `import torch` there still fails, which is the check that it stayed unchanged |
| disk | 4.6 GB |

## What was installed

| package | version |
|---|---|
| torch | **2.11.0+cu128** (`torch.cuda.is_available()` is True, the device is the RTX 4090) |
| transformers | 5.20.2 |
| peft | 0.21.2 |
| trl | 1.16.0 |
| datasets | 5.1.0 |
| accelerate | 1.15.0 |
| numpy | 2.5.4 |

The CUDA wheel comes from `https://download.pytorch.org/whl/cu128`, not from PyPI: PyPI's Windows `torch` is a
CPU build, and a CPU build would train nothing in reasonable time.

## How to reproduce

```powershell
pwsh tools/phase2/env_setup.ps1                     # venv, torch (cu128), then transformers/peft/trl/accelerate/datasets
C:\tools\strata-train\Scripts\python.exe tools/phase2/smoke_train.py            # CPU: the stack, 3 steps, ~1 s
C:\tools\strata-train\Scripts\python.exe tools/phase2/smoke_train.py --cuda     # 2.1's exit criterion, needs the card
```

`env_setup.ps1` is idempotent: it creates the venv only when it is missing and re-runs pip on top.

## The trivial run, measured on the card (2026-10-22)

```
[smoke] torch 2.11.0+cu128, device cuda, trainable tensors 4, trainable params 1024
{'train_runtime': '0.4424', 'train_steps_per_second': '6.781', 'train_loss': '3.879'}
[smoke] trained 3 steps on cuda; losses 3.877, 3.886, 3.875; adapter written to <out>
```

That is 2.1's exit criterion: one successful trivial training run, on the 4090, recorded and reproducible with
the two commands above.  It says the stack trains on this card.  It does not say a real adapter fits in VRAM -
see below.

## What the CPU run proves, measured (2026-10-22)

```
[smoke] torch 2.11.0+cu128, device cpu, trainable tensors 4, trainable params 1024
[smoke] the first PEFT names: base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight, ...
[smoke] trained 3 steps on cpu; losses 3.893, 3.929, 3.906; adapter written to <out>
```

* torch, transformers, peft, accelerate and datasets import and work **together** on Python 3.12 - which is the
  version this venv was built with, because the server's 3.14 has no torch wheel.
* A PEFT adapter can be attached to a transformers model and trained; the trainable-parameter count is the LoRA's
  own arithmetic (4 tensors, rank 4 on a 32-wide projection: 2 layers x 4 x (32 + 96) = 1,024).
* `save_pretrained` writes exactly the files `tools/lora_pack.py` reads: `adapter_config.json` and
  `adapter_model.safetensors` with PEFT's `...lora_A.weight` / `...lora_B.weight` names.  The **module names** are
  the model's (here `transformer.h.0.attn.c_attn`, a GPT-2 stand-in), and the converter maps per model - for
  Qwen3.8-Flash-Next it maps `self_attn.{q,k,v,o}_proj` and the shared expert, and refuses the rest by name
  (`phase2_engine_lora_plan.md` step 1, §7 question 1).
* The adapter it writes is **F32**; the converter accepts F16, BF16 and F32 and refuses a value fp16 cannot hold.

## What it does not prove

* **Nothing about the real model.** The smoke test trains a random two-layer GPT-2 on made-up token ids: no
  download, no tokenizer, no pretrained weights, nothing to learn. It tests the plumbing, not a recipe.
* **Nothing about VRAM.** A 4-parameter adapter on a 32-wide model says nothing about what a rank-16 adapter on
  48 layers costs to train, or whether Adam's state fits beside the user's model. That is 2.3's question.

## The rule that still applies

Training data is governed by [decision_policy_training_data.md](decision_policy_training_data.md): the licence
record, and the rules that follow from it.  Nothing in this environment changes them - OpenDecisions stays
validation-only, jev-bench stays the human yardstick, and a training run uses the licence-clean pool.
