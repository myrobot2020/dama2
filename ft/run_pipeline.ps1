# Cursor chat exports -> SFT JSONL -> LoRA (see requirements-finetune.txt).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python ft/prepare_dataset.py
python ft/train_sft.py @args
