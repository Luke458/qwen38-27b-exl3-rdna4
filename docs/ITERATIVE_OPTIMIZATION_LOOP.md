# Bounded execution loop for the next agent

Read [the plan](AGENT_OPTIMIZATION_PLAN.md) first. Your job is to produce measured,
reproducible gains or a useful negative result, not to make every hypothesis win.

## Kickoff prompt

> Continue this repository's Qwen3.8-27B EXL3 / RX 9070 XT work. Read
> docs/AGENT_OPTIMIZATION_PLAN.md and this loop completely. Begin with Phase 0,
> then specialize the existing grouped gate/up path. Do not implement the old
> shared-suh concat sketch: checkpoint scales differ, and grouping already exists.
> Preserve all user changes and immutable evidence. Use isolated candidate builds,
> one GPU owner, strict numerical gates, verified dispatch and paired end-to-end
> measurements. Keep int8 disabled. No public push, model download, driver/clock
> change or default promotion without the relevant authorization. Report progress
> and blockers honestly. MTP, vision and DFlash are separately qualified phases;
> follow their prerequisites rather than enabling everything at once.

## One experiment = one falsifiable mechanism

Before editing, create a new experiment directory without replacing an existing
ID. Record:

```yaml
id: NEW_UNUSED_ID
parent_control: source-and-binary-hash
hypothesis: "specific expensive stage and proposed change"
supported_predicate: "architecture, shape, bitrate, codebook, layout, graph mode"
mechanism: "one change; explain saved traffic, instructions or launches"
arithmetic_mode: "same arithmetic or separately budgeted numerical change"
expected_scope: "kernel, whole decode, prefill, speculative round, or vision"
correctness_commands: []
dispatch_evidence_command: []
benchmark_commands: []
timeout_seconds: REQUIRED
budget_remaining: REQUIRED
rollback: "disable switch or select unchanged control artifact"
```

Record expectations as hypotheses, not predicted results. Use Amdahl's law to
bound plausible model gain from the measured affected fraction. Do not multiply
unrelated microbenchmark speedups or sum overlapping host/kernel timing spans.

## State machine

```text
PREFLIGHT -> IDENTITY_VERIFIED -> BUILT -> EXACT/NUMERIC_TESTS
                                        | failure -> INCORRECT
       -> DISPATCH_VERIFIED -> REAL_SHAPE_PAIRED_SCREEN
                                | no gain -> NO_GAIN
       -> MODEL_CORRECT -> 10-PAIR_END_TO_END_DECISION
                              | uncertain -> bounded remeasurement
                              | threshold missed -> NOT_QUALIFIED
       -> 5_RESERVED_CONFIRMATION_PAIRS -> FRESH_HOLDOUT -> REVIEW
```

Missing stages, stale binaries, absent metrics, fallback-only execution and
unsupported workloads must never produce PASS. A crash is not a noisy sample.
Use existing tools only after inspecting their current interfaces; do not fabricate
an experiment record to fit the comparator. Capability tests have their own exit
criteria rather than pretending to meet a speed threshold.

## Iteration procedure

1. **Preflight:** acquire the shared GPU lock, inspect device health and other
   owners, record environment and free memory. Never kill unrelated workloads.
2. **Identify:** hash model/config, all changed source/header files, compiler
   arguments, policy and binary. Verify the imported extension path in the child
   process. Rebuild every dependent TU when a shared header changes.
3. **Build isolated:** preserve control objects/binary. Retain stdout/stderr and
   enforce process timeouts. Do not score a partially linked or stale artifact.
4. **Correctness first:** exact trellis/transform tests, zero/impulse/random/
   outlier fixtures, actual matrix sizes, eligible and rejected calls. For exact
   arithmetic require bitwise equality. For changed arithmetic use predeclared
   calibrated limits and model-quality gates, never loosened after failure.
5. **Execution contract:** prove candidate dispatch actually occurs; exercise
   repeated calls, alternate streams, graph capture/replay, fallback, disable
   switch and scratch lifetime. Test both eager and BC graph paths.
6. **Screen cheaply:** compare real checkpoint tensors with rotating working sets,
   warmup and alternating AB/BA runs. Retain raw times. Inspect counters/ISA only
   in separate diagnostic runs; traces must not replace unprofiled timing.
7. **Model quality:** identical teacher-forced histories across calibration and
   reserved holdout data. Check logit errors, KL, loss and nonfinite values.
   Confirm candidate engagement during those histories. Tests that only exercise
   fallback validate fallback, not candidate quality.
8. **End-to-end:** for the existing text campaign, use ten paired decision trials
   on fixed 128/1024/4096-prompt, 256-decode workloads. Require primary median
   speedup >=1.05, 95% bootstrap lower bound >1, no holdout regression >3%, no
   faults and acceptable memory. Report sub-threshold gains without promotion.
9. **Confirm:** only the selected decision winner gets five reserved confirmation
   pairs and fresh-process holdout checks. Recheck endpoint behavior if runtime
   code changes. Never train the candidate selection on reserved holdouts.
10. **Retain or stop:** archive the result, explain why, update the remaining budget,
    and pick the next mechanism supported by evidence. Leave the stable control
    selected unless promotion is authorized and all required gates pass.

## Benchmark discipline

- Separate GPU-event time, CPU enqueue time, HTTP latency and total wall time.
- Event instrumentation itself adds overhead; use attribution to form hypotheses,
  then judge candidates with uninstrumented end-to-end trials.
- Reused small tensors can measure cache speed instead of model weight traffic.
- A low fraction of copy bandwidth does not prove the kernel is memory-bound.
- Generic profiling tools may hang this host/runtime. Use the existing safe
  event workflow first; install/enable additional profiling only when justified.
- Record invalid-trial reasons before examining performance; retain invalid
  records. Missing telemetry is uncertainty, not evidence of an idle GPU.
- For MTP/DFlash measure total committed tokens per round, not acceptance alone.
  Preserve target sampling/rejection semantics and recurrent-state rollback.
- For vision record pixels and visual tokens; do not compare different image
  resolutions under one workload identity.

## Agent orchestration

If the user authorizes subagents: one owns the isolated candidate, one reviews
numerical/dispatch safety, one reviews measurements. Assign disjoint files and
give **only one** agent permission to execute GPU work. The coordinator reads
required skill instructions, reconciles evidence and controls promotion.

Do not loop indefinitely on an incorrect design. Revisit source semantics after
one unexplained correctness failure and stop after repeated device faults.
When blocked by missing approval, assets or external state, provide the exact
next action rather than claiming completion.

## Required final handoff

- Working capabilities and limitations.
- Candidate/control source and binary identities.
- Numerical and dispatch evidence; skipped cases explicitly listed.
- Raw paired timing locations, median/CI and whole-model result.
- VRAM peak, faults, invalid samples and remaining budget.
- Promotion decision, restoration command and next hypothesis.

Never conflate "compiles", "runs", "correct", "faster in a layer benchmark",
"faster end-to-end" and "qualified for the default profile".
