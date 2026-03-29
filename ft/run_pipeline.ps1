# Cursor Agent transcripts -> cursor chats/ -> train.jsonl -> Ollama Modelfile + `ollama create`.
# No PyTorch in this script. Default: SYSTEM-only (no weight change). Set OLLAMA_ADAPTER_GGUF
# for partial weight adjustment (GGUF adapter on top of FROM).
# Ollama step runs only if CURSOR_OLLAMA_PIPELINE=1 or -Train.
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

$enableOllama = $Train -or ($env:CURSOR_OLLAMA_PIPELINE -eq "1")
if (-not $enableOllama) {
    Write-Host "Ollama create skipped. Set CURSOR_OLLAMA_PIPELINE=1 or pass -Train."
    exit 0
}

python ft/build_ollama_modelfile.py --create @args
