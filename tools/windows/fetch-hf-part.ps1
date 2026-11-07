# Fetch one byte range of a remote file, appending across interruptions.
# curl's -o truncates, so each attempt goes to a temp file that is appended to
# Out on success; a stall or drop therefore costs only that attempt.
param(
  [Parameter(Mandatory)] [string]$Url,
  [Parameter(Mandatory)] [string]$Out,
  [Parameter(Mandatory)] [int64]$Start,
  [Parameter(Mandatory)] [int64]$End
)

$len = $End - $Start + 1
for ($try = 0; $try -lt 500; $try++) {
  $have = if (Test-Path $Out) { (Get-Item $Out).Length } else { 0 }
  if ($have -eq $len) { exit 0 }
  if ($have -gt $len) { Remove-Item $Out -Force; $have = 0 }

  $tmp = "$Out.tmp"
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  # Abort a stalled stream instead of hanging forever on a dead connection.
  & curl.exe -fsS -L --retry 2 --retry-delay 3 --connect-timeout 20 `
    --speed-limit 51200 --speed-time 30 --max-time 300 `
    -r "$($Start + $have)-$End" -o $tmp $Url

  if (Test-Path $tmp) {
    if ((Get-Item $tmp).Length -gt 0) {
      $dst = [IO.File]::Open($Out, [IO.FileMode]::Append, [IO.FileAccess]::Write)
      $src = [IO.File]::OpenRead($tmp)
      try { $src.CopyTo($dst, 4MB) } finally { $src.Dispose(); $dst.Dispose() }
    }
    Remove-Item $tmp -Force
  }
}
exit 1
