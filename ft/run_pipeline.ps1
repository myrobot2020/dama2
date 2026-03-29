# Cursor Agent transcripts -> cursor chats/ -> train.jsonl -> [LoRA] -> [GGUF] -> Ollama Modelfile + `ollama create`.
# Default: no PyTorch; SYSTEM-only Ollama unless you enable LoRA / adapter paths.
#
# Switches / env:
#   -LoRa              Run HF LoRA (train_sft.py). Same as DAMA_HF_LORA=1
#   -Train             Run `ollama create` (same as CURSOR_OLLAMA_PIPELINE=1). Does NOT mean HF training.
#   -Adapter           After LoRA, try optional GGUF conversion (ft/convert_lora_to_gguf.py). Same as DAMA_CONVERT_ADAPTER_GGUF=1
#   DAMA_HF_MODEL      HF id for training (default Qwen/Qwen2.5-0.5B-Instruct); keep in sync with OLLAMA_MODEL / Ollama FROM
#   OLLAMA_MODEL       Ollama base for FROM (default qwen2.5:0.5b-instruct)
#   OLLAMA_ADAPTER_GGUF  Path to PEFT output dir or .gguf file (set automatically after -LoRa if unset)
param(
    [switch]$Train,
    [switch]$LoRa,
    [switch]$Adapter,
    [double]$MaxFileAgeHours = 0,
    [switch]$SkipExport,
    [ValidateSet("workspace", "all")]
    [string]$ExportMode = "workspace",
    [switch]$NoCleanExport
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

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

$doLoRa = $LoRa -or ($env:DAMA_HF_LORA -eq "1")
if ($doLoRa) {
    $hfModel = if ($env:DAMA_HF_MODEL) { $env:DAMA_HF_MODEL } else { "Qwen/Qwen2.5-0.5B-Instruct" }
    Write-Host "Running LoRA training: $hfModel"
    $trainCmd = @("ft/train_sft.py", "--model", $hfModel)
    python @trainCmd
    $stem = $hfModel -replace '/', '_'
    $adapterDir = Join-Path (Join-Path $PSScriptRoot "runs") $stem
    if (Test-Path $adapterDir) {
        if (-not $env:OLLAMA_ADAPTER_GGUF) {
            $env:OLLAMA_ADAPTER_GGUF = (Resolve-Path $adapterDir).Path
            Write-Host "Set OLLAMA_ADAPTER_GGUF=$($env:OLLAMA_ADAPTER_GGUF)"
        }
    } else {
        Write-Warning "Expected LoRA output dir not found: $adapterDir"
    }

    $doConvert = $Adapter -or ($env:DAMA_CONVERT_ADAPTER_GGUF -eq "1")
    if ($doConvert) {
        if (-not (Test-Path $adapterDir)) {
            Write-Warning "Skipping GGUF conversion: missing $adapterDir"
        } else {
            $ggufOut = Join-Path $PSScriptRoot "data\adapter.gguf"
            python ft/convert_lora_to_gguf.py --adapter-dir "$adapterDir" --out "$ggufOut"
            if (Test-Path $ggufOut) {
                $env:OLLAMA_ADAPTER_GGUF = (Resolve-Path $ggufOut).Path
                Write-Host "Using GGUF adapter: $($env:OLLAMA_ADAPTER_GGUF)"
            }
        }
    }
}

$enableOllama = $Train -or ($env:CURSOR_OLLAMA_PIPELINE -eq "1")
if (-not $enableOllama) {
    Write-Host "Ollama create skipped. Set CURSOR_OLLAMA_PIPELINE=1 or pass -Train."
    exit 0
}

python ft/build_ollama_modelfile.py --create @args
