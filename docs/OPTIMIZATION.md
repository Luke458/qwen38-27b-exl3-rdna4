# Optimization scope after this release snapshot

**Current handoff:** [agent plan](AGENT_OPTIMIZATION_PLAN.md),
[iterative loop](ITERATIVE_OPTIMIZATION_LOOP.md), and
[further efficiency investigation](EFFICIENCY_INVESTIGATION.md).
The original queue below is historical: 0010's extraction-mask rewrite was
rejected, and 0012's shared-suh concat sketch is invalid for the actual checkpoint.
Use the linked handoff for new work.

Pause kernel tuning while packaging and validating the current baseline-backed
runtime. Preserve the frozen policy and raw paired measurements. The +4.16%
stack is reportable as an experiment; its missed 5% median gate does not make
it the champion.

The next implementation priorities, if optimization resumes, are:

1. Rewrite the direct-core trellis extraction chain with fewer bit operations,
   then prove the reconstructed windows and full outputs bit-identical.
2. Test explicit prefetch or greater software-pipeline depth against the
   `0009b` stack, measuring paired end-to-end decode as well as real-shape
   projection latency.
3. Extend the output-Hadamard fusion to the gate/up multi-matrix path, which
   the `0008` single-matrix fusion does not cover.

Each candidate must identify its rebuilt translation units and binary hash,
prove that the intended kernel executes, and pass oracle, dispatch, fallback,
stream, graph, and model-level correctness checks appropriate to its change.
Use identical teacher-forced histories for model comparisons. Measure the
primary and both holdout workloads with alternating fresh-process trials, retain
invalid samples with reasons, and apply the frozen primary gate: at least 10
decision pairs, median speedup ≥1.05, and a 95% bootstrap lower bound >1.00.
Only a decision-gate winner proceeds to the reserved five confirmation pairs
and final holdout qualification. Do not relax the gate after a failed candidate.

For an API release, separately check bounded cache and chunk settings on this
16 GB card, endpoint startup and health, a real OpenAI-compatible chat request,
and the documented fallback path. A direct-model benchmark cannot establish
those server properties.
