# Agent handoff: Qwen3.8-27B EXL3 on RX 9070 XT

## Objective and boundaries

Improve useful single-user inference on the **existing mixed-bit**
GestaltLabs/Qwen3.8-27B-EXL3-11.5GB checkpoint and gfx1201 GPU. First improve
ordinary text decode and multi-row verification/prefill. Then qualify native
MTP, evaluate a compatible DFlash drafter, and bring up the existing vision tower.
The checkpoint's eligible 3-bit mul1 projections remain the first kernel target;
this is not a mandate to requantize the whole model to three bits.

Use this document with [the execution loop](ITERATIVE_OPTIMIZATION_LOOP.md).
Additional source-level leads and cost calculations are in
[the efficiency investigation](EFFICIENCY_INVESTIGATION.md).
Do not infer current runtime state from old progress prose. Inspect source,
binary hashes, configuration and retained evidence before each campaign.

Preserve user edits, local models, all experiment records and immutable baseline
binaries. No driver changes, GPU resets, clocks, unrelated process termination,
new model downloads, public pushes or default-profile promotion without the
corresponding user authorization. Keep one owner of GPU execution. Parallel
agents may inspect source or write disjoint files, but must not benchmark together.

## Verified starting facts (reviewed 2026-09-24)

- The 0008 output-transform fusion + 0009b unroll-4 stack measured 1.0416x
  primary decode throughput over ten paired trials. It missed the frozen 1.05x
  promotion threshold. Five reserved confirmation pairs were not run.
- The 0010 extraction-mask rewrite failed correctness and was rejected. Do not
  reintroduce it as an untested shortcut: EXL3 trellis windows overlap.
- The 0011 gfx12 WMMA port has GEMM-vs-GEMV probes and recorded successful API
  smoke tests. It is a capability fix, not a measured throughput improvement.
  The frozen 8667f03d binary predates that fix and is not a serving baseline.
- The 0012 Python weight-concat plan is **invalid as written**. Retain its history,
  but supersede its assumptions rather than implementing its sketch literally.
- CPU reads of the actual checkpoint found 64 main-layer gate/up pairs: 35
  pairs at 3 bits, 28 at 2 bits, one at 1 bit. Each pair has equal bitrate and
  codebook multiplier, but **none has identical suh input scales**.
- Of 48 GDN qkv/z pairs, 35 are 2/4-bit, 12 are 4/4-bit, one is 3/4-bit.
  None has identical suh. Shared input dimensions do not mean shared transforms.
- With int8 disabled, current rocm_py enables gate/up MultiLinear. BC_GatedMLP
  routes through exl3_mgemm_gr to the multi-matrix GEMV at M=1 when eligible.
  This already schedules both matrices and preserves separate input transforms.
- Current ROCm GEMV dispatch precedes the fallback selector; the NVIDIA-like
  CC label is not evidence that these eligible M=1 calls use the wrong kernel.
- The checkpoint includes 39 MTP tensors (~0.198 GiB stored) and 987 vision
  tensors (~0.531 GiB stored). These are storage totals, **not runtime peaks**.
- The current API wrapper is text-only and disables speculative decoding.
  Serving smoke does not qualify MTP, vision, concurrency or arbitrary context.

Source anchors: `vendor/rocm_exl3/exllamav3/rocm_py/__init__.py`,
`modules/mlp.py`, `exllamav3_ext/libtorch/mlp.cpp`,
`exllamav3_ext/rocm/quant/exl3_gemm_rdna.hip`, `exl3_mgemv_rdna.hip`,
and `experiments/0011-wmma-gfx12/RESULTS.md` in the local research workspace.

## Phase 0 — reproducible controls and attribution

1. Read repository instructions and current Git state. Identify serving-capable
   baseline and best experimental binaries by hash; never substitute one for
   another under the same baseline identity.
2. Record model revision, all relevant source/header hashes, compiler flags,
   Python overrides, environment switches, cache settings and workload identities.
   A shared-header change requires rebuilding every dependent translation unit.
3. Verify GPU health, free VRAM and competing workloads. Stop a server only if
   it is this campaign's process or the user authorized stopping it.
4. Run baseline correctness and serving smoke. Archive old measurements and
   establish a new control for capability fixes that alter execution paths.
5. Capture actual dispatch counts and separate timings for gate/up transforms,
   grouped dot, output transform, activation, down, GDN projections, recurrent
   update, attention and lm_head. Cover eager warmup and internal graph replay.
6. Preserve existing text workload identities (128/1024/4096 prompt, 256 decode,
   batch one), and add a separately identified API workload. Do not compare
   HTTP timings directly to GPU-only benchmark timings.

Exit: source/binary provenance, live dispatch attribution, numerical reference,
repeatable timing and a serving-capable control. Report unknowns explicitly.

## Phase 1 — optimise the existing grouped gate/up path

First candidate: **two-matrix grouped GEMV specialization**, not a shared-suh
concatenation. Keep distinct transformed activation rows and output scales.

Candidate sequence, one mechanism per experiment:

1. Specialize the existing bszm=2, no-routing-weights, equal-width path. Measure
   whether generic indexing, launch geometry or intermediate stages matter.
2. Port the validated output-Hadamard epilogue fusion to this path, retaining
   fp16 rounding points and per-matrix scales.
3. Investigate pairing output tiles so the epilogue can perform the existing
   gate/up activation and multiplication without an extra intermediate pass.
   Preserve original rounding; changing precision is a separate numerical mode.
   A subsequent candidate may fuse down.suh + its input Hadamard into that
   128-wide epilogue, using an explicit pretransformed-input down contract.
4. Tune split-K/warps/prefetch only on the winning grouped implementation and
   the actual 5120->17408 shapes, not synthetic square GEMMs.

Do not assume equal-suh, uniform bitrate, zero bias or contiguous layouts:
prove the intended predicate from checkpoint and call-site data. Decline all
other calls without corrupting outputs. GDN mixed-bitrate grouping is a later
candidate needing separate segment metadata and per-projection transforms.

Exit: exact output equality where arithmetic is unchanged; intended dispatch
confirmed; stream and graph safety; paired layer and model gains. No 1.5x promise.

## Phase 2 — efficient multi-row verification and prefill

The gfx1201 Python override still forces multi-row LinearEXL3 reconstruction.
The new WMMA capability creates an opportunity to replace that workaround
selectively, not permission to remove it globally.

1. Measure native packed GEMM versus reconstruction at verifier rows 2, 3, 5,
   9 and representative prefill chunks, across real shapes and bitrates.
2. Validate outputs against independent/reference paths and compare complete
   teacher-forced logits on identical histories. WMMA rounding is not bitwise
   equivalent by assumption; establish its numerical budget before tuning.
3. Track temporary dequantization memory, scratch allocation and lm_head cost.
   The 248320-vocabulary head can create large multi-row buffers.
4. Introduce an explicit, narrow routing predicate only for measured winners;
   retain reconstruction as fallback. Recheck serving and memory at context limits.

Exit: verifier latency curve, quality gates, memory peaks, and routing evidence.
This is a prerequisite for meaningful speculative-decoding comparisons.

## Phase 3 — native MTP

Use the existing checkpoint MTP component first; avoid an external drafter until
there is a measured reason. Reuse target embedding/head as the architecture expects.

- Compare plain decode against draft lengths 1, 2 and 4 on identical prompts,
  sampling settings and cache budgets. Warm each execution shape.
- Record draft, verify, CPU/control, sampling/readback time; accepted and total
  committed tokens per round; rejection position; TTFT/TPOT and peak VRAM.
- Correctness: target verification, rejection/rollback of GDN recurrent states
  and KV pages, EOS, cache boundaries, cancellation, consecutive requests, and
  deterministic fixtures. Greedy text agreement alone is not sufficient.
- Verify that draft execution does not unnecessarily project all hidden states
  through lm_head or copy full-vocabulary tensors to the CPU.
- Adaptive draft length is a separate candidate after fixed-length measurements.

Break-even: `(draft_ms + verify_ms + control_ms) / committed_tokens_per_round`
must beat the corresponding plain-decode milliseconds per token. Count the bonus
token consistently. Higher acceptance does not itself establish a speedup.

Exit: optional MTP mode with measured net benefit, or an honest negative result.
Keep plain decode available and default until the applicable gates pass.

## Phase 4 — vision capability track

This can run independently of speculation after the runtime is stable.

1. Load the existing vision component; do not download another vision tower.
2. Start with one bounded-resolution image. Validate preprocessing, image-token
   positions, embeddings, multimodal prompt construction and text generation.
3. Measure tower weights, activation peaks, visual token count, encoder time,
   multimodal prefill and text decode. Storage size is not a fit guarantee.
4. Start with sequential encode/unload then text inference if simultaneous fit
   is uncertain. Consider a resident tower only after peak-memory measurement.
5. Add a bounded API image schema, decoded-byte/pixel/token limits, cancellation
   and image-input tests. Prefer local uploads/data inputs first; do not introduce
   arbitrary remote URL fetching without SSRF protections and an explicit design.
6. Test image requests followed by plain text requests for state leakage.

Exit: verified single-image API capability with stated limits. Do not claim it
accelerates text decode. Video/multi-image and speculation+vision are later gates.

## Phase 5 — compatible DFlash evaluation

Only after verifier profiling and memory accounting. Obtain user approval before
downloading a new drafter. Verify exact target compatibility, tokenizer/vocabulary,
hidden-state tap positions, block size and the pinned runtime's supported DFlash
variant; do not equate all DFlash versions.

Compare against **both** plain decode and the best MTP configuration on coding,
prose, structured and low-acceptance prompts. Account for drafter weights/cache,
target verification, hidden-state exports, head projections, CPU readbacks and
vision/context headroom. Retain only net wins without target-distribution errors.

## Deliverables and stop conditions

For every phase: tests, minimal source patch, exact reproduction command,
provenance manifest, raw measurements and a readable result with limitations.
Update current docs only after evidence exists. Do not rewrite old raw records.

Follow the existing 24-candidate/four-GPU-hour campaign ledger where applicable;
first reconcile its stale counters. Do not silently reset the budget. Separate
capability and speculative workloads need explicitly recorded policies, not a
retroactive relaxation of the original text-kernel threshold.

Stop a family after its declared stall condition or an unacceptable safety/
maintenance tradeoff. A GPU fault stops execution immediately: preserve logs,
check health before any resumption, and never auto-reset the device. A working
feature, correct kernel and promoted speedup are three different outcomes.
