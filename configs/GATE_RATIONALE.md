# Acceptance policy freeze

Frozen after full FP16 baseline generation and before this session's candidate model-output evaluation. Earlier development candidates were not part of a qualified campaign.

Same-arithmetic values come from the independent-oracle GPU parity measurements. Int8 approximation ceilings (2% NRMS, 10% normalized maximum error, cosine >= 0.9998) were declared before CPU calibration. `tools/calibrate_policy.py` verified them using actual checkpoint weight blocks, synthetic activations and the independent CPU equations, without candidate output. These samples establish feasibility, not whole-model quality.

Model ceilings are explicit engineering acceptance choices: maximum absolute logit difference <= 1, mean KL(baseline || candidate) <= 0.01, absolute perplexity increase <= 0.1 on identical teacher-forced histories. They are not measured guarantees and will not be increased if the candidate fails. The comparison also requires finite logits and exactly matching histories. Prompt repetition is a controlled regression workload, not a general quality benchmark.

Residual int8 is out of scope and remains unfrozen. Kernel implementation equivalence to the int8 oracle has the separate, tighter existing test bounds (NRMS < 0.002, cosine > 0.999999).

Memory gate: 15 GiB of tracked allocator high-water plus recorded device telemetry. This is not a claim that Torch accounts for every driver allocation. Current workload cache is sized from the workload set and fits below this gate; both compared runs must use identical capacity and code.

Baseline Python compatibility patch routes multi-row EXL3 Linears through the existing reconstruction path on gfx1201 because the dense WMMA implementation traps. Single-token packed decode remains enabled. This patch is identical in baseline and candidate. External torch graph capture is unsupported by the int8 kernel and takes the existing FP16 fallback; the fork may also use its internal graph paths.
