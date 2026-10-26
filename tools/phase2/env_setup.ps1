# tools/phase2/env_setup.ps1 - the Phase 2.1 training environment (roadmap 2.1).
# A SEPARATE venv, deliberately: the repo's .venv is the server's and must not gain torch.
$ErrorActionPreference = "Stop"
$venv = "C:\tools\strata-train"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    py -3.12 -m venv $venv
}
$py = "$venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
# CUDA wheels: PyPI's Windows torch is CPU-only, so the index matters.
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Output "cu128 index failed; trying cu126"
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
}
& $py -m pip install transformers peft trl accelerate datasets safetensors numpy
& $py -c "import torch, peft, trl, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
