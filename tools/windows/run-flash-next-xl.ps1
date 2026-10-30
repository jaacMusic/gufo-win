# Qwen3.8-Flash-Next on the Windows build of Gufo. Scriptable twin of
# run-flash-next.ps1: same model, plus the server options the engine_console
# and start-gufo-flash-next-xl.ps1 feed it.
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next-xl.ps1               # serve: UD-Q4_K_XL, thinking, adaptive MTP
#   ... -Think off                              instruct mode (temp 0.7/top-p 0.8)
#   ... -Draft mtp3                             MTP capped at 3 drafts
#   ... -Draft off                              autoregressive baseline
#   ... -Mode bench                             pp/tg at depths 0..128K
#   ... -DraftVocab -Survival -Lookup           the opt-in MTP options (docs\models\qwen3.8-flash-next)
#
# Files come from the Hugging Face cache (see docs\WINDOWS.md for the
# download); -Snapshot points at another copy of unsloth/Qwen3.8-Flash-Next-GGUF.
# Ready when the log shows event=load_completed and GET /health returns 200.
param(
  [ValidateSet("serve", "bench")] [string]$Mode = "serve",
  [ValidateSet("on", "off")] [string]$Think = "on",
  # Server default for how much the model reasons before answering
  # (gufo --reasoning-effort). "auto" = the model's own default (xhigh on
  # Flash-Next) and is passed as no flag at all. gufo refuses an explicit
  # effort while --think is off, so callers gate on -Think.
  [ValidateSet("auto", "minimal", "low", "medium", "high", "xhigh", "max")] [string]$ReasoningEffort = "auto",
  [ValidateSet("mtp", "mtp3", "off")] [string]$Draft = "mtp",
  [switch]$Greedy,
  # Latin draft vocab: English/code only; non-Latin just loses the MTP speedup.
  [switch]$DraftVocab,
  # Confidence-stop MTP policy (gufo --mtp-policy survival).
  [switch]$Survival,
  [switch]$Lookup,
  # 131072 = what fits comfortably in the 96 GB carve with Q4_K_XL + MTP + mmproj
  # (256K was ~89 GiB and got reverted in the launcher); 8888 = the engine port
  # this box's firewall rule and clients use - 8080 is the user's Open WebUI,
  # never take it (the wrapper passes both explicitly; these are the safe defaults).
  [int]$Context = 131072,
  # Session pool for serve mode. Total reservation is sessions*context tokens
  # of state: 8x40960 (~75 GiB with the trunk) fits the 96 GB carve, 8x131072
  # does not. 1 = the historical single-conversation default.
  [int]$Sessions = 1,
  # Per-client-IP pending-request cap (gufo --max-pending-per-client; 4 = the
  # engine default). Every local client is 127.0.1, so with local agents the
  # per-client cap binds BEFORE --sessions does: raise it to match the agent
  # count or concurrent decode requests 429 (client_queue_full).
  [int]$MaxPendingPerClient = 4,
  # Max prompt tokens between active decode rounds (gufo --prefill-chunk).
  # 0 = the engine default (512). Larger = faster prefill, coarser decode slots.
  [int]$PrefillChunk = 0,
  # Prompt tokens per model prefill pass (gufo --prefill-batch).
  # 0 = the engine default (8192); clamped to the context size.
  [int]$PrefillBatch = 0,
  [int]$Port = 8888,
  [string]$BindHost = "0.0.0.0",
  [string]$Snapshot = "",
  [string]$ModelFile = "",
  [string]$MmprojFile = "",
  [string]$MtpModel = "",
  # Text-only: skip the --mmproj sidecar entirely (image requests are rejected).
  [switch]$NoVision,
  # gufo rejects requests whose model string != --served-model-name; default matches
  # the existing OpenCode config entry so the SAME model id works with both engines.
  [string]$ServedName = "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL"
)
$ErrorActionPreference = "Stop"
$bin = Join-Path $PSScriptRoot "..\..\build\release"
# TheRock runtime DLLs (rocblas, amdhip, ...) live next to the toolchain; without
# this gufo.exe dies with 0xC0000135 (status dll not found) before printing anything.
$env:PATH = "$bin;C:\TheRock\build\bin;$env:PATH"
if (-not (Test-Path "$bin\gufo.exe")) {
  throw "gufo.exe not found in $bin; build first: powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1"
}

if (-not $ModelFile) {
  $ModelFile = "UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
}
if (-not $MmprojFile) { $MmprojFile = "mmproj-BF16.gguf" }
$mtpFile = "MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
# Each Hugging Face revision is its own snapshot folder, holding only the files
# downloaded at that revision: pick the newest one that has each file.
$snapshots = @()
if (-not $Snapshot) {
  $cache = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
           elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
           else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
  $snapshots = @(Get-ChildItem (Join-Path $cache "models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots") `
    -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
    Select-Object -ExpandProperty FullName)
  $Snapshot = $snapshots | Where-Object { Test-Path (Join-Path $_ $ModelFile) } | Select-Object -First 1
  if (-not $Snapshot) {
    throw "$ModelFile not found in unsloth/Qwen3.8-Flash-Next-GGUF under $cache; see docs\WINDOWS.md or pass -Snapshot"
  }
}
$model = Join-Path $Snapshot $ModelFile
if (-not (Test-Path $model)) { throw "model not found: $model" }
if (-not $MtpModel) {
  $MtpModel = @($Snapshot) + $snapshots | ForEach-Object { Join-Path $_ $mtpFile } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $MtpModel) { $MtpModel = Join-Path $Snapshot $mtpFile }
}
if ($Draft -ne "off" -and -not (Test-Path $MtpModel)) { throw "MTP model not found: $MtpModel (or use -Draft off)" }

# Vision: gufo's vision stack requires the BF16 sidecar (mmproj-BF16.gguf,
# same unsloth repo - the F16 copy errors with "invalid BF16 Qwen vision
# tensor"). Serve mode only (bench takes no --mmproj). -NoVision = text-only.
$Mmproj = ""
if ($Mode -eq "serve" -and -not $NoVision) {
  $Mmproj = @($Snapshot) + $snapshots | ForEach-Object { Join-Path $_ $MmprojFile } |
    Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $Mmproj) { Write-Warning "$MmprojFile not found in any snapshot; serving text-only" }
}

# Never greedy by default. Thinking uses the sampler embedded in the GGUF
# (general.sampling.*: temp 1.0, top-p 0.95, top-k 20); instruct uses Qwen's
# non-thinking convention (0.7 / 0.8 / 20). Clients may override per request.
$sampling = if ($Think -eq "on") {
  @("--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0")
} else {
  @("--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0")
}
# -Greedy: temperature 0, the method of gufo's published tables, for
# like-for-like comparisons only.
if ($Greedy) { $sampling = @("--temperature", "0") }
$speculative = switch ($Draft) {
  "mtp" { @("--speculative", "mtp", "--mtp-model", $MtpModel) }
  "mtp3" { @("--speculative", "mtp", "--mtp-model", $MtpModel, "--draft-tokens", "3") }
  "off" { @() }
}
if ($Draft -ne "off" -and $Mode -eq "serve") {
  if ($DraftVocab) { $speculative += @("--mtp-draft-vocab", "latin") }
  if ($Survival) { $speculative += @("--mtp-policy", "survival") }
  if ($Lookup) { $speculative += "--prompt-lookup" }
}

if ($Mode -eq "serve") {
  # 32 MiB requests: base64 images next to a long conversation.
  $arguments = @("serve", "--host", $BindHost, "--port", "$Port", "--sessions", "$Sessions",
    "--max-request-bytes", "33554432",
    "llm", "--model", $model, "--served-model-name", $ServedName,
    "--context", "$Context", "--max-output-bytes", "8388608",
    "--think", $Think) + $sampling + $speculative
  if ($ReasoningEffort -ne "auto") { $arguments += @("--reasoning-effort", $ReasoningEffort) }
  if ($PrefillChunk -gt 0) { $arguments += @("--prefill-chunk", "$PrefillChunk") }
  if ($PrefillBatch -gt 0) { $arguments += @("--prefill-batch", "$PrefillBatch") }
  if ($Mmproj) { $arguments += @("--mmproj", $Mmproj) }
} else {
  # Same shape as upstream's single-user table: pp2048/tg128 per depth.
  # (bench has no thinking switch; -Think only picks the sampler here.)
  $arguments = @("bench", "--model", $model, "--n-prompt", "2048", "--n-gen", "128",
    "--n-depth", "0,4096,16384,32768,65536,131072", "--repetitions", "2") +
  $sampling + $speculative
}

# GUFO_* variables (GUFO_PLATFORM_TUNING, diagnostics) reach the server; show
# them, since one left in a shell changes what runs.
Get-ChildItem Env: | Where-Object { $_.Name -like "GUFO_*" } | ForEach-Object {
  Write-Host "using $($_.Name)=$($_.Value) from this shell" -ForegroundColor Yellow
}
Write-Host "gufo $($arguments -join ' ')'"
# gufo logs to stderr; any PS-level 2>&1 wrap (ErrorRecords) makes the outer
# script/tee die on the first line. Merge stderr at cmd level instead: PS sees
# plain stdout strings only. Args with spaces get cmd-style quoting.
$ErrorActionPreference = 'Continue'
$cmdline = ($arguments | ForEach-Object { if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ } }) -join ' '
cmd /c "`"$bin\gufo.exe`" $cmdline 2>&1"
exit $LASTEXITCODE
