$ErrorActionPreference = "Stop"

$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $appDir "start_dama_rag.bat"

if (-not (Test-Path $target)) {
  throw "Missing launcher: $target"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Dama RAG (Local).lnk"

$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut($shortcutPath)
$sc.TargetPath = $target
$sc.WorkingDirectory = $appDir

# Use a built-in Windows icon (no image creation needed).
# You can change the index number to pick a different default icon.
$sc.IconLocation = "$env:SystemRoot\System32\shell32.dll, 23"

$sc.Save()

Write-Host "Created shortcut:" $shortcutPath
Write-Host "Icon source:" $sc.IconLocation
