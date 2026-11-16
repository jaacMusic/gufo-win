# Long-context sweep for Qwen3.8-Flash-Next: prefill and decode at 8K..256K.
# Runs one gufo process at a time (the 96 GB carve holds one Q4 model + state).
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\windows\bench-flash-next-context.ps1
#   ... -Sizes 8192,32768 -Repetitions 1        quick subset
#   ... -Exe C:\dev\bubo\bin\bubo.exe -OutDir artifacts\bench-context-bubo
# Results: one log per run under artifacts\bench-context\ (git-ignored).
param(
  [string]$Model = "",
  [string]$Mtp = "",
  [int[]]$Sizes = @(8192, 32768, 131072, 262144),
  [int]$Repetitions = 2,
  [string]$OutDir = "",
  [string]$Exe = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$bin = if ($Exe) { $Exe } else { Join-Path $root "build\release\gufo.exe" }
$env:PATH = "$(Split-Path $bin -Parent);$(Join-Path $root 'build\release');C:\TheRock\build\bin;$env:PATH"
if (-not (Test-Path $bin)) { throw "bench executable not found: $bin" }
if (-not $OutDir) { $OutDir = Join-Path $root "artifacts\bench-context" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

# Newest snapshot holding both the target and the MTP sidecar.
$cache = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
         elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
         else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
$snapshots = @(Get-ChildItem (Join-Path $cache "models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots") `
  -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
$target = "UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
$lastShard = "UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf"
$mtpFile = "MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
if (-not $Model) {
  # Newest snapshot with every shard (a partial download keeps only 00001) and the sidecar.
  $snap = $snapshots | Where-Object { (Test-Path (Join-Path $_.FullName $target)) -and (Test-Path (Join-Path $_.FullName $lastShard)) -and
    (Test-Path (Join-Path $_.FullName $mtpFile)) } | Select-Object -First 1
  if (-not $snap) { throw "UD-Q4_K_XL + shared MTP not found under $cache" }
  $Model = Join-Path $snap.FullName $target
}
if (-not $Mtp) {
  $Mtp = @($snapshots | ForEach-Object { Join-Path $_.FullName $mtpFile } | Where-Object { Test-Path $_ }) |
    Select-Object -First 1
  if (-not $Mtp) { throw "MTP sidecar not found under $cache" }
}
if (-not (Test-Path $Model)) { throw "model not found: $Model" }

function Invoke-Bench([string]$Name, [string[]]$BenchArgs) {
  $log = Join-Path $OutDir "$Name.log"
  Write-Host "`n=== $Name ===" -ForegroundColor Cyan
  # cmd-level merge keeps native stderr out of PowerShell's error records.
  $argline = ($BenchArgs | ForEach-Object { if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ } }) -join ' '
  cmd /c "`"$bin`" $argline 2>&1" | Tee-Object -FilePath $log
  if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE); see $log" }
}

$common = @("bench", "--model", $Model, "--n-gen", "128",
  "--repetitions", "$Repetitions", "--temperature", "0")
foreach ($size in $Sizes) {
  if ($size -gt 262144) { throw "size $size exceeds the 262144-token native context" }
  $name = "$([int]($size / 1024))k"
  # The next token needs one slot, so a full 262K prompt is 262143 tokens and
  # decode stops 128 generated tokens + 1 short of the context.
  $prompt = [Math]::Min($size, 262143)
  $depth = [Math]::Min($size, 262144 - 129)
  Invoke-Bench "pp-$name"  ($common + @("--n-prompt", "$prompt", "--n-depth", "0"))
  Invoke-Bench "tg-ar-$name" ($common + @("--n-prompt", "1", "--n-depth", "$depth"))
  Invoke-Bench "tg-mtp-$name" ($common + @("--n-prompt", "1", "--n-depth", "$depth",
    "--speculative", "mtp", "--mtp-model", $Mtp, "--verbose"))
}
Write-Host "`nlogs: $OutDir" -ForegroundColor Green
