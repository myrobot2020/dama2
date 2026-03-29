# Cursor Agent transcripts (DB + ~/.cursor/.../agent-transcripts) -> cursor chats/
# -> SFT JSONL -> optional LoRA (requirements-finetune.txt).
# Training runs only if CURSOR_OLLAMA_PIPELINE=1 or -Train.
param(
    [switch]$Train,
    [double]$MaxFileAgeHours = 0,
    [switch]$SkipExport,
    [ValidateSet("workspace", "all")]
    [string]$ExportMode = "workspace",
    [switch]$NoCleanExport
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $SkipExport) {
    $exportCmd = @("ft/export_cursor_db.py", "--mode", $ExportMode)
    if (-not $NoCleanExport) {
        $exportCmd += "--clean"
    }
    python @exportCmd
}

$prepareArgs = @("--val-files", "0")
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
