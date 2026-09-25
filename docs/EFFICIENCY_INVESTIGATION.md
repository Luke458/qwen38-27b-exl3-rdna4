# Further efficiency investigation — 2026-09-24

Scope: read-only source inspection, existing recorded timings and CPU reads of
checkpoint metadata/tensors. **No new GPU benchmark or speedup is claimed here.**
The next agent should use [the plan](AGENT_OPTIMIZATION_PLAN.md) and
[bounded loop](ITERATIVE_OPTIMIZATION_LOOP.md) to test these hypotheses.

## 1. Concat does not create the assumed extra parallelism

The current M=1 gate/up path runs:

1. Per-matrix input transform.
2. Grouped split-K dot.
3. Per-matrix output transform.
4. Separate SiLU/multiply in BC_GatedMLP.
5. Down projection, including its input transform.

The weighted-reduction stage described in the generic multi-matrix source header
does **not** run here: gate/up passes no routing weights.

For K=5120, N=17408, two matrices:

- N tiles per matrix = 17408 / 16 = 1088.
- Grouped dot grid = 1088 x 2 = **2176 blocks**, four waves per block.
- Hypothetical concatenated N=34816 grid = 34816 / 16 = **2176 blocks**,
  also four waves under the current selector.

Thus doubling N by packing the pair does not itself improve this grid's
parallelism. Layout/locality differences remain testable, but the old occupancy
story is not supported. All 64 main-layer gate/up pairs have different input
scales, so the shared-suh sketch also fails numerical equivalence.

Anchors: `exllamav3_ext/rocm/quant/exl3_mgemv_rdna.hip:599-715`,
`exl3_gemv_rdna.hip:158-170`, `exllamav3_ext/libtorch/mlp.cpp:38-88`.

## 2. Strong structural candidate: fuse the MLP intermediate stages

The multi-matrix path does not have the single-matrix 0008 output-Hadamard
fusion. A narrower, simpler first step is to combine **Had-out for gate/up +
SiLU/multiply** into one epilogue, retaining the original fp16 rounding points.

An extended candidate can additionally apply **down.suh + Had-in for down**
in that epilogue and write the transformed down input directly. The intermediate
width is 17408, divisible by 128. All these transforms can operate within aligned
128-element groups; nonlinear activation is elementwise. This makes local fusion
plausible without requiring the dot kernel's cross-block completion scheme.

Required engineering:

- Consume separate gate/up scale vectors, perform each Hadamard independently,
  round outputs as before, then perform the identical activation/multiply.
- Preserve act_limit, output dtype and the next input-transform rounding.
- Do not call the existing down entry point unchanged: it would apply Had-in
  again. Add an explicit pretransformed-input contract with a safe fallback.
- Internal graph capture patches pointers through a prologue; bypassing its
  transform must still publish/patch all required pointers correctly.
- Start with two-stage epilogue fusion, then extend to down-input fusion as a
  separate candidate. Keep dot/Had-out cross-block fusion separate again.

At FP16 width 17408 over 64 layers, eliminating gate/up's transformed output
write+read removes **8,912,896 bytes** of logical intermediate traffic per step.
Eliminating the activation output write+read before down Had-in removes another
**4,456,448 bytes**. These are logical accesses, not measured DRAM bytes; some
may hit cache. They are small relative to model weight traffic. The main potential
is removing one or two kernel stages per MLP, not doubling weight bandwidth.
Measure epilogue/launch contribution before assigning a speed target.

## 3. Highest-priority speculative lever: unnecessary reconstruction

`rocm_py/__init__.py:103-107` still forces every **Python** LinearEXL3 call with
M>1 into reconstruction. This was necessary before gfx12 WMMA worked. BC C++
modules can bypass this override, so the cost depends on the actual call path.

The output head is a particularly concrete target. Checkpoint metadata:

| Item | Bytes |
|---|---:|
| Packed 4-bit lm_head trellis, 5120 x 248320 | 635,699,200 |
| Full equivalent FP16 matrix | 2,542,796,800 |
| Current 32768-column FP16 reconstruction slice | 335,544,320 |

The sliced implementation does not allocate the entire dense head simultaneously,
but reconstructing every slice still writes the full equivalent dense matrix and
feeds it to dense multiplication on every such call. Avoiding that temporary can
remove substantial work even when verification has only a few rows. This is an
algorithmic traffic observation, not a measured speedup or DRAM counter result.

Next test: compare packed native GEMM versus reconstruct+hgemm at M=2/3/5/9 on
this head, then on full identical-history verification. Keep the existing guard
for unqualified cases. Probe rounding/overflow/outliers and measure memory and
latency. Do not assume WMMA-vs-GEMV spot checks qualify complete speculative
verification.

Anchors: `modules/quant/exl3.py:132-139,182-211`,
`generator/generator.py:1013-1031`, `rocm_py/__init__.py:95-109`.

## 4. MTP's small weights do not mean a negligible draft cost

The MTP component has one layer and ~0.198 GiB of stored tensors, but **each
drafted position also evaluates the target's large lm_head**. A four-token draft
window can therefore invoke the ~0.592 GiB packed head four times, in addition
to the MTP layer, target verification and state updates. Measure the head
separately rather than treating the draft as a tiny isolated transformer.

Other costs visible in source:

- Optional confidence transfer to CPU during draft iterations.
- Target+draft rows passed to one verification forward.
- Serial acceptance/sampling and `.item()` checks.
- Recurrent/KV rollback at rejection and checkpoint boundaries.
- MTP prefill/export and accepted-state carry into the draft cache.

Potential optimization: a narrowly gated greedy verification path that evaluates
argmax/accepted-prefix on device and transfers compact results once. It must not
be used with filters, penalties or other stateful sampling features unless their
sequential semantics are preserved. Merely vectorizing the current loop is unsafe.

A related head+argmax fusion could avoid materializing logits for greedy-only
drafting, but the head's output is much smaller than its weight reads. Do not
assume avoiding the logits buffer solves the dominant cost. Retain full-logit
paths for quality tests and sampling modes that need distributions.

Anchors: `generator/generator.py:744-776,1065-1085,1139-1215,1285-1318`,
`architecture/qwen3_5_mtp.py:193-253`, `generator/job.py:1391-1432`.

## 5. DFlash has a different cost model

DFlash drafts a block in one forward, but current dynamic truncation happens
**after** computing that block. Reducing verification length may save target work;
it does not retroactively save draft computation. The runtime also projects
exported target states into draft KV and performs sampling/readbacks.

Select block length using measured total round cost and committed tokens, not
acceptance rate alone. Verify hidden-state tap mapping for the exact checkpoint.
No DFlash speedup or compatible downloaded drafter was established in this review.

Anchors: `generator/generator.py:790-882,1272-1283`,
`architecture/dflash.py:207-266`.

## 6. Vision: avoid adding permanent memory before measuring need

The checkpoint already contains the tower (~0.531 GiB stored tensors). Its
provided `vision.py` computes image embeddings and unloads the tower before
loading text. That is a useful first functional route, but repeatedly reloading
the entire text model would be poor interactive serving behavior.

After validating bounded images, compare resident tower, temporary GPU tower
with text kept resident, and staged/offloaded encoding where necessary. Measure
activation peaks and visual-token prefill, not just tower weight size. A bounded
image-embedding cache could amortize repeated-image conversations, keyed by image
content, processor configuration and model revision, with explicit memory limits.
Treat embeddings as user data; avoid cross-user cache leakage.

## 7. Correctness constraint on future concurrency

The multi-matrix path has a **per-device singleton parameter block** carrying
launch pointers. Independent concurrent streams could overwrite it between the
prologue and consumer kernels. This is a source-level risk, not a reproduced
failure in the current single-stream server. Do not improve throughput by simply
enabling parallel GPU request streams. Audit and isolate scratch/parameter state
first; test adversarial overlapping streams and graph replay.

Anchor: `exllamav3_ext/rocm/quant/exl3_mgemv_rdna.hip:104-121,213-218`.

## Priority queue

1. Trace current grouped path and verifier/head routing under the serving-capable
   build; establish the correct control and affected time fractions.
2. Grouped Had-out + activation fusion; optionally extend into down Had-in.
3. Qualify native packed multi-row head/verification and narrow reconstruction.
4. Tune actual grouped dot loads/prefetch/unroll against the current best stack.
5. Measure MTP windows and head/control cost; then narrowly optimize greedy paths.
6. Bring up bounded vision separately; evaluate DFlash only with compatibility,
   verification and memory prerequisites satisfied.

Every item above is a candidate, not a promised gain. No existing evidence
supports guaranteeing 21–24 ms per decode step from these changes.
