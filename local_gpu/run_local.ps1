# PowerShell launcher for local GPU training.
# Runs training detached so it survives closing this shell.
#
# Usage (from inside lstm-qgan-local folder):
#   .\run_local.ps1
#
# To see live progress:
#   Get-Content outputs\train.log -Wait -Tail 50
#
# To stop it:
#   Get-Process python | Where-Object { $_.Path -match "lstm-qgan-local" } | Stop-Process

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

# Adjust these to taste.
$epochs = 200
$batch  = 32
$subset = 5000
$loss   = "wasserstein"
$ncritic = 3
$lr     = "2e-4"

New-Item -ItemType Directory -Force -Path "$repo\outputs" | Out-Null
$logfile = "$repo\outputs\train.log"

Write-Host "Launching training in background..."
Write-Host "Log file: $logfile"
Write-Host "Working dir: $repo"

$python = "python"

$args = @(
    "main.py",
    "--mode", "train",
    "--epochs", $epochs,
    "--batch_size", $batch,
    "--subset", $subset,
    "--loss", $loss,
    "--n_critic", $ncritic,
    "--lr", $lr,
    "--device", "cuda",
    "--resume",
    "--ckpt_interval", "5"
)

$proc = Start-Process -FilePath $python -ArgumentList $args `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $logfile `
    -RedirectStandardError "$repo\outputs\train.err" `
    -WindowStyle Hidden -PassThru

Write-Host ""
Write-Host "Started PID: $($proc.Id)"
Write-Host ""
Write-Host "Follow live output with:"
Write-Host "    Get-Content `"$logfile`" -Wait -Tail 50"
Write-Host ""
Write-Host "Stop with:"
Write-Host "    Stop-Process -Id $($proc.Id)"
