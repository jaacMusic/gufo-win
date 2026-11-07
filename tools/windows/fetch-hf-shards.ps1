# Resumable fetch of large model files from the Hugging Face Hub.
#
# `hf download` cannot resume: every attempt writes a new
# `<blob>.<random>.incomplete` and deletes it when the transfer fails, and a
# stale `.locks` entry can block a file for half an hour. This script instead
# pulls each file as independent byte ranges into `.parts`, so a rerun fetches
# only the ranges that are missing or short, and an interruption costs one
# chunk at most.
#
# Sizes and digests come from the Hub API, so a finished file is checked before
# it keeps its name. An exclusive lock keeps it to one run, which lets Task
# Scheduler keep relaunching it: a dropped connection, a wedged worker, a
# reboot or a killed shell simply resumes on the next start.
param(
  [Parameter(Mandatory)] [string]$Repo,       # e.g. unsloth/Qwen3.8-Flash-Next-GGUF
  [string]$Revision = 'main',                 # commit hash, tag or branch
  [Parameter(Mandatory)] [string[]]$Files,    # repo paths, wildcards allowed
  [Parameter(Mandatory)] [string]$Dest,       # plain directory, not the hf cache
  [int]$Workers = 3,
  [int]$ChunkMiB = 64
)

$ErrorActionPreference = 'Stop'

# Task Scheduler relaunches this every 5 minutes; the lock keeps it to one run.
New-Item -ItemType Directory -Force $Dest | Out-Null
$partsRoot = Join-Path $Dest '.parts'
New-Item -ItemType Directory -Force $partsRoot | Out-Null
try {
  $lock = [IO.File]::Open((Join-Path $partsRoot 'fetch.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch { exit 0 }

$chunk = [int64]$ChunkMiB * 1MB

function Say($m) {
  $line = "{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $m
  $line
  # Detached runs have no console to capture; keep the log beside the parts.
  Add-Content -Path (Join-Path $partsRoot 'fetch.log') -Value $line
}

# Sleep would silently pause a multi-hour transfer; hold the system awake while
# this process lives (the flag is released when it exits).
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(int f);'
$power = Add-Type -Namespace Win32 -Name Power -MemberDefinition $sig -PassThru
$null = $power::SetThreadExecutionState(0x80000001)

function Stop-Tree($id) {
  Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Tree $_.ProcessId }
  Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

# The Hub API reports the exact byte size and, for LFS files, the SHA-256 of the
# content, so no size table has to be maintained by hand.
$api = "https://huggingface.co/api/models/$Repo/revision/$Revision`?blobs=true"
$listing = (Invoke-RestMethod -Uri $api -TimeoutSec 60).siblings
$targets = @()
foreach ($s in $listing) {
  foreach ($pattern in $Files) {
    if ($s.rfilename -like $pattern) {
      $sha = if ($s.lfs) { if ($s.lfs.sha256) { $s.lfs.sha256 } else { $s.lfs.oid } } else { $null }
      $targets += [pscustomobject]@{ n = $s.rfilename; size = [int64]$s.size; sha = $sha }
      break
    }
  }
}
if ($targets.Count -eq 0) { throw "no file in $Repo@$Revision matches: $($Files -join ', ')" }
Say ("{0}@{1}: {2} files, {3:N1} GB" -f $Repo, $Revision, $targets.Count, (($targets | Measure-Object size -Sum).Sum / 1GB))

$base = "https://huggingface.co/$Repo/resolve/$Revision"
$worker = Join-Path $PSScriptRoot 'fetch-hf-part.ps1'

$attempt = 0
while ($true) {
  try {
    foreach ($s in $targets) {
      $final = Join-Path $Dest $s.n
      $marker = Join-Path $partsRoot ($s.n + '.ok')
      if ((Test-Path $marker) -and (Test-Path $final) -and (Get-Item $final).Length -eq $s.size) {
        Say "skip $($s.n) (already assembled)"
        continue
      }

      $partsDir = Join-Path $partsRoot $s.n
      New-Item -ItemType Directory -Force $partsDir | Out-Null
      New-Item -ItemType Directory -Force (Split-Path $final) | Out-Null
      $count = [int][math]::Ceiling($s.size / $chunk)

      $queue = New-Object System.Collections.ArrayList
      for ($i = 0; $i -lt $count; $i++) {
        $start = [int64]$i * $chunk
        $end = [math]::Min($start + $chunk - 1, $s.size - 1)
        $len = $end - $start + 1
        $p = Join-Path $partsDir ('{0:D5}' -f $i)
        if ((Test-Path $p) -and (Get-Item $p).Length -eq $len) { continue }
        [void]$queue.Add([pscustomobject]@{ start = $start; end = $end; len = $len; path = $p; tries = 0 })
      }
      Say "$($s.n): $($count) chunks, $($queue.Count) to fetch"

      $failed = @()
      while ($queue.Count -gt 0) {
        $procs = @()
        while ($queue.Count -gt 0 -and $procs.Count -lt $Workers) {
          $job = $queue[0]
          $queue.RemoveAt(0)
          # Start-Process joins the list without quoting, so quote every value:
          # the repository path (this script's own directory) may contain spaces.
          $a = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`" " +
               "-Url `"$base/$($s.n)`" -Out `"$($job.path)`" " +
               "-Start $($job.start) -End $($job.end)"
          $procs += [pscustomobject]@{
            p = (Start-Process -FilePath powershell.exe -ArgumentList $a -PassThru -WindowStyle Hidden)
            job = $job
            started = Get-Date
          }
        }

        $lastSay = Get-Date
        $prev = (Get-ChildItem $partsDir -File | Measure-Object Length -Sum).Sum
        while (@($procs | Where-Object { -not $_.p.HasExited }).Count -gt 0) {
          Start-Sleep -Milliseconds 1000
          # A worker that never returns would block the batch forever; each curl
          # attempt is capped at 5 min, so anything past 12 min is wedged.
          foreach ($e in $procs) {
            if (-not $e.p.HasExited -and ((Get-Date) - $e.started).TotalMinutes -gt 12) {
              Say "watchdog: killing wedged worker for $([IO.Path]::GetFileName($e.job.path))"
              Stop-Tree $e.p.Id
            }
          }
          if (((Get-Date) - $lastSay).TotalSeconds -ge 60) {
            $have = (Get-ChildItem $partsDir -File | Measure-Object Length -Sum).Sum
            Say ("{0}: {1:N1}/{2:N1} GB  {3:N2} MB/s  queued={4}" -f $s.n, ($have / 1GB), ($s.size / 1GB), (($have - $prev) / ((Get-Date) - $lastSay).TotalSeconds / 1MB), $queue.Count)
            $prev = $have
            $lastSay = Get-Date
          }
        }

        foreach ($e in $procs) {
          $ok = (Test-Path $e.job.path) -and (Get-Item $e.job.path).Length -eq $e.job.len
          if (-not $ok) {
            $e.job.tries++
            # The worker already retries internally; a chunk that keeps failing
            # means the connection, not the chunk, so back off before requeueing.
            Start-Sleep -Seconds 15
            if ($e.job.tries -lt 20) { [void]$queue.Add($e.job) }
            else { $failed += $e.job.path; Say "GIVE UP $($e.job.path) after 20 tries" }
          }
        }
        if ($failed.Count -gt 0) { throw "chunks failed: $($failed -join ', ')" }
      }

      Say "assembling $($s.n)"
      $out = [IO.File]::Create($final)
      try {
        for ($i = 0; $i -lt $count; $i++) {
          $in = [IO.File]::OpenRead((Join-Path $partsDir ('{0:D5}' -f $i)))
          $in.CopyTo($out, 4MB)
          $in.Dispose()
        }
      } finally { $out.Dispose() }

      if ($s.sha) {
        $got = (Get-FileHash -Algorithm SHA256 $final).Hash.ToLower()
        if ($got -ne $s.sha) {
          Remove-Item $final -Force
          Remove-Item $partsDir -Recurse -Force
          throw "SHA256 mismatch for $($s.n): got $got, parts discarded for a clean refetch"
        }
        Say "ok $($s.n) sha256 matches"
      } else {
        Say "ok $($s.n) size matches (no published digest)"
      }
      Remove-Item $partsDir -Recurse -Force
      Set-Content -Path $marker -Value $(if ($s.sha) { $s.sha } else { $s.size })
    }
    break
  } catch {
    $attempt++
    Say "ERROR: $($_.Exception.Message) - retry #$attempt in 60 s"
    if ($attempt -ge 50) { throw }
    Start-Sleep -Seconds 60
  }
}

Say 'all files present and verified'
