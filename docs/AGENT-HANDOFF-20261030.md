# Agent handoff 2026-10-30: confidence-gated MTP port needs engine validation

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
