# Cursor chat exports -> SFT JSONL -> optional LoRA (requirements-finetune.txt).
# Training runs only if CURSOR_OLLAMA_PIPELINE=1 or you pass -Train.
param(
    [switch]$Train,
    [double]$MaxFileAgeHours = 0
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$prepareArgs = @()
if ($MaxFileAgeHours -gt 0) {
    $prepareArgs += @("--max-file-age-hours", [string]$MaxFileAgeHours)
}
python ft/prepare_dataset.py @prepareArgs

$enableTrain = $Train -or ($env:CURSOR_OLLAMA_PIPELINE -eq "1")
if (-not $enableTrain) {
    Write-Host "LoRA train skipped. Set CURSOR_OLLAMA_PIPELINE=1 or pass -Train."
    exit 0
}

python ft/train_sft.py @args
