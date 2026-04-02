# Copies Cursor chat-related data to Desktop\cuso for backup.
# Safe to run multiple times (it will overwrite destination files).

param(
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$srcHistory = Join-Path $env:USERPROFILE "AppData\Roaming\Cursor\User\History"
$srcWorkspaceStorage = Join-Path $env:USERPROFILE "AppData\Roaming\Cursor\User\workspaceStorage"
$srcProjectsRoot = Join-Path $env:USERPROFILE ".cursor\projects"
$dstRoot = Join-Path $env:USERPROFILE "Desktop\cuso"
$logFile = Join-Path $dstRoot "cursor_backup.log"

function Ensure-Dir([string]$path) {
  if ($DryRun) { return }
  # Create (or validate) directory without relying on Test-Path.
  # This avoids edge cases where Test-Path/aliases behave unexpectedly.
  New-Item -ItemType Directory -Path $path -Force | Out-Null
}

function Robocopy([string]$source, [string]$destination) {
  Ensure-Dir $destination
  if ($DryRun) {
    Write-Host "DRYRUN: robocopy `"$source`" `"$destination`" /E /R:2 /W:5"
    return
  }
  # /E copies subdirectories including empty ones.
  # /R /W set retries and wait between retries.
  # Copying is overwrite-by-default for same paths.
  # /TEE prints to console and also logs via /LOG+.
  # /NFL /NDL reduce noise (no file/dir names).
  # Call robocopy.exe explicitly to avoid recursion with our Robocopy function.
  & robocopy.exe $source $destination /E /R:2 /W:5 /TEE /LOG+:$logFile /NFL /NDL
}

Ensure-Dir $dstRoot
if (-not $DryRun) {
  $null = (New-Item -ItemType File -Path $logFile -Force -ErrorAction SilentlyContinue)
  Add-Content -Path $logFile -Value ("`n----- {0:u} -----" -f (Get-Date))
  Add-Content -Path $logFile -Value "Starting Cursor backup copy."
}

Write-Host "Backing up Cursor chat history..."
Robocopy $srcHistory (Join-Path $dstRoot "CursorUI\User\History")

Write-Host "Backing up Cursor workspaceStorage..."
Robocopy $srcWorkspaceStorage (Join-Path $dstRoot "CursorUI\User\workspaceStorage")

Write-Host "Backing up agent-transcripts from Cursor projects..."
$agentDirs = Get-ChildItem -Path $srcProjectsRoot -Recurse -Directory -Filter "agent-transcripts" -Force -ErrorAction SilentlyContinue
foreach ($agentDir in $agentDirs) {
  # Example:
  #   $rel = c-Users-ADMIN-Desktop-dama\agent-transcripts
  $rel = $agentDir.FullName.Substring($srcProjectsRoot.Length).TrimStart('\')
  $dstAgentDir = Join-Path $dstRoot ("CursorProjects\" + $rel)
  Robocopy $agentDir.FullName $dstAgentDir
}

Write-Host "Cursor backup complete: $dstRoot"

