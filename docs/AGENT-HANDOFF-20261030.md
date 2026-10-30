# Agent handoff 2026-10-30: confidence-gated MTP port needs engine validation

**VALIDATED 2026-10-30 — closed.** Results in "Port 2026-10-30:
confidence-gated MTP chains" in docs/LINUX-PERF-PORT.md: greedy
byte-identical PASS, decode bench parity, agent-session median decode
+2.1% (gains in late turns), sampled smoke clean, no batch-drop anomaly.
Worktree `gufo-base-5ffd` (baseline `5ffd6c84`) may be removed.

Port of upstream 5d66de8c (#882) is committed but NOT engine-validated: the
gufo engine hosted the session that made the change, so no model could run.

What changed: see "Port 2026-10-30: confidence-gated MTP chains" in
docs/LINUX-PERF-PORT.md. Summary: greedy MTP chains stop at the first draft
below 0.85 head confidence (ArgmaxConfidence kernel), sampled chains below
0.95 host q, planned batches drop below 0.6. Fork-only behaviors kept
(fast_sampling pricing, sampled_subset_cost re-pricing, mtp_survival,
prompt-lookup, gpu_verification = gpu_greedy).

Done already (no engine): cpu-test build, 109/109 ctest, gpu-test HIP
compile of gufo_qwen38_flash_next, clang-format clean in changed regions.

## Validation checklist (run in a fresh session, engine idle)

1. Byte-identical greedy gate: greedy (temperature 0) decode is argmax
   everywhere, so the gate must not change greedy outputs. Compare a fixed
   prompt set before/after this commit with `gufo bench` or the byte-
   identical check used for earlier ports (see docs/LINUX-PERF-PORT.md).
   The gate only shortens speculative chains, never the target token.
2. Decode bench: the fork's standard decode benchmark (AGENT-SESSION.md
   Bubo comparison setup). Upstream claims +0.5 greedy / +1.8 repetitive
   tok/s; expect the fork's MTP acceptance (35.1 median session) to close
   some of the gap to Bubo 64.0.
3. agent-session.py replay: confirm the acceptance-rate metric moves and
   no session-level regression on warm-turn scheduling.
4. Sampled paths: run a temperature>0 session briefly; GateStops reads
   proposals.back().probability, exercised only when sampled.
5. If the batch drop (kMtpBatchDropProbability 0.6) fires wrongly, the
   symptom is greedy batch peers losing their anchor speculation; watch
   per-request acceptance in the bench trace.

## How to run (Windows, engine idle)

`gufo-base-5ffd` is a worktree of `5ffd6c84` (= the port commit's parent) built
with the documented HIP flags; its `gufo.exe` is the pre-port baseline. The
candidate is `build\release\gufo.exe`. Model lives in the HF cache snapshot
dir (4-part UD-Q4_K_XL + `MTP` sidecar); use the snapshot the live server
uses. Stop the serving engine first; every number below must run with the
engine idle, baseline and candidate alternating.

```powershell
$SN = "$env:USERPROFILE\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\<snapshot>"
$M  = "$SN\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
$D  = "$SN\MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
$BASE = '<worktree>\gufo-base-5ffd\build\release\gufo.exe'
$CAND = 'build\release\gufo.exe'
$SPEC = '--speculative','mtp','--mtp-model',$D,'--prompt-lookup'

# 1. Greedy byte-identical gate (t=0 default; diff must be empty)
& $BASE prompt -m $M @SPEC -n 64 'Write a haiku about GPUs:' > base.txt
& $CAND prompt -m $M @SPEC -n 64 'Write a haiku about GPUs:' > cand.txt
compare (Get-Content base.txt) (Get-Content cand.txt)

# 2. Decode A/B (record acceptance= with -v; 3 alternating reps, take medians)
& $BASE bench --model $M @SPEC --n-prompt 2048 --n-gen 128 --n-depth 0 --repetitions 3 --temperature 0
& $CAND bench --model $M @SPEC --n-prompt 2048 --n-gen 128 --n-depth 0 --repetitions 3 --temperature 0

# 3. Session replay per binary: serve on 5815 with the live-server flags
#    (see `Get-CimInstance Win32_Process` for the running gufo.exe), then
python tools/serving/agent-session.py --url http://127.0.0.1:5815 --json

# 4. Sampled smoke: same prompt call with `-t 0.8 -s 7` on the candidate
```

