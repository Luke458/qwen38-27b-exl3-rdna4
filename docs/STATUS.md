# Project status

This project targets EXL3 decode on the Radeon RX 9070 XT (`gfx1201`), using
the mixed-bit GestaltLabs Qwen3.8-27B EXL3 11.5 GB checkpoint. The optimized
kernel scope is the checkpoint's eligible 3-bit, mul1, FP16, single-row
projections. The checkpoint also contains other bitrates and modules; this is
not a whole-model 3-bpw claim.

Eager, batch-one model inference works with the pinned ROCm fork and gfx1201
compatibility fixes. A calibration run measured roughly 29 tokens/s steady
decode on a 128-token prompt; holdout prompt lengths of 128, 1024 and 4096
were also run. These are direct-model benchmark results, not an API server
throughput or concurrency measurement.

The best measured optimization stack is `0009b-unroll4` plus the
`EXL3_GEMV_FUSED_HAD=1` fusion from `0008-fused-had`. Across 10 alternating
baseline/candidate pairs, the primary 1024-prompt/256-decode workload had a
median paired speedup of **1.0416× (+4.16%)**, with a 95% bootstrap interval
of **[1.0396, 1.0425]**. The 128- and 4096-prompt workloads also improved in
the same run. See [compact trial evidence](results/paired-0009b.json) and the
retained experiment records under `experiments/0009b-unroll4/`.

This is a measured gain, but it did **not** pass the frozen primary-workload
promotion gate of median speedup at least 1.05. The immutable baseline remains
the champion. No five-pair confirmation trial was run for this failed decision
gate. The unroll-2 predecessor has a recorded baseline/candidate teacher-forced
comparison with zero logit difference and zero KL. The combined unroll-4 plus
fusion stack is described as bitwise equivalent in its experiment results, but
there is no separate `0009b` model-correctness artifact in this snapshot.

The bundled fork includes an OpenAI-compatible server, but this checkpoint on
this GPU has not yet passed an endpoint startup and request smoke test in this
project. Server behavior, public hosting readiness, and server throughput are
therefore unverified. The documented `EXL3_GEMV=0` fallback also reaches a
gfx1201 cooperative-GEMM trap; it needs a fix or a clear scope restriction
before claiming reliable fallback behavior.

For follow-on work and the validation gates, see [optimization scope](OPTIMIZATION.md).
