# Report

## Multi-Agent Architecture

The runtime pipeline is:

`SpecInterpreterAgent -> ProbePlannerAgent -> BenchmarkGeneratorAgent -> ProbeRunner -> NcuAnalystAgent -> ExecutionTriageAgent -> ResultAggregatorAgent`

Each logical agent has:

- a production prompt under `prompts/*.txt`
- a deterministic fallback implementation in `src/agents.py`
- strict JSON input/output expectations

The LLM path is optional. If `API_KEY` and `BASE_MODEL` are unavailable, if the response is malformed, or if a suggestion is unsafe, the deterministic fallback remains authoritative.

## Prompt System

`src/prompt_manager.py` loads prompts, validates required files, and logs missing-prompt problems without crashing the whole submission. Prompts were rewritten to include:

- explicit agent role
- exact JSON input schema
- exact JSON output schema
- measurement constraints
- anti-hardcoding instructions
- JSON-only output requirement
- explicit uncertainty handling

## Target Specification Handling

The runtime reads evaluation targets only from `/target/target_spec.json`. It does not fall back to `./target_spec.json`, `./test_spec.json`, or an internally synthesized "all supported metrics" request when that file is missing or malformed. The selected path is recorded in both `agent_logs` and `output.json` as `target_spec_path`.

The official target specification may contain either semantic hardware metric names or Nsight Compute / CUDA attribute style metric names. The agent supports both through a compatibility layer in `src/output_schema.py` and still preserves the original requested names in `output.json`.

## Deterministic Probe Suite

### Pointer-chasing latency probe

The pointer-chase probe uses randomized dependent loads across a working-set sweep. It includes:

- default cached mode
- `ld.global.cg` style mode where feasible to reduce L1 effects
- a cache-flush buffer between measurements

The analyzer estimates:

- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_cycles`
- `l2_cache_capacity_bytes`

using stable plateaus, large-working-set behavior, and the strongest sustained latency cliff.

### Global memory bandwidth probe

The global-bandwidth probe queries free memory with `cudaMemGetInfo`, chooses a streaming allocation size large enough to exceed cache when possible, retries with smaller sizes on allocation failure, and tests:

- read
- write
- copy
- vectorized copy

It reports the best stable median GB/s, not a single outlier.

The analyzer also derives compatibility metrics from the same probe:

- `dram_bytes_read_per_second`
- `dram_bytes_write_per_second`
- `gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed`

When exact Nsight Compute metrics are unavailable, the pipeline falls back to deterministic estimates and marks the lower confidence in the result text.

### Shared memory bandwidth probe

The shared-memory probe emphasizes repeated shared-memory accesses within the kernel and reports structured metadata including block size, iterations, modeled bytes, elapsed time, and GB/s.

### Core clock probe

The sustained-frequency probe uses a single long-running block so `clock64()` cycles and CUDA event time refer to the same work interval. Multiple durations are measured and used to infer a realized `actual_core_clock_mhz`.

### FP32 throughput probe

`peak_fp32_tflops` is measured by an FMA-heavy kernel timed with CUDA events. The operation count is known from the kernel structure, so TFLOP/s is derived from runtime work, not from vendor peak-spec formulas alone.

For compatibility with server-side metric requests, the same probe can also report `sm_throughput_pct_of_peak_sustained_elapsed`, preferring exact Nsight Compute data and otherwise using a conservative deterministic fallback based on sample stability.

### Shared-memory bank-conflict probe

The bank-conflict probe compares a conflict-free shared-memory access pattern with a stride-32 bank-conflicted pattern, repeats measurements inside the binary, and reports:

- `shared_memory_bank_conflict_penalty_cycles`
- `shared_memory_bank_conflict_penalty_ratio`

### Environment and SM-masking probe

The environment probe samples inline PTX `%smid` over multiple rounds and compares observed runtime SM IDs with API-reported `multiProcessorCount`. If observed active SMs are fewer than the API count, the output can flag `sm_masking_suspected`.

The same probe also records runtime device properties such as clock rate, memory clock rate, memory bus width, and L2 size so compatibility names like `device__attribute_max_gpu_frequency_khz` can resolve without static lookup tables.

## Analyzer And Triage

`src/analyzer.py` now separates:

- structured-line parsing
- parsed-measurement construction
- metric-specific aggregation
- triage feature extraction

Aggregation uses median-style robust statistics, simple MAD-based outlier rejection, and coefficient-of-variation style confidence reduction. Execution triage downgrades or fails results when variance, inconsistencies, or runtime/tool failures make the measurement unreliable.

## Anti-Hardcoding Design

- No static GPU spec tables are used.
- The project does not identify a GPU by name and then look up known numbers.
- API-reported properties are treated as auxiliary notes only.
- The prompt layer is not allowed to replace deterministic probe templates with free-form CUDA output.

## Failure Handling

The project always writes `output.json`, even when:

- `nvcc` is missing
- `ncu` is missing or blocked
- the CUDA runtime is unavailable
- the driver/runtime versions mismatch
- prompt files are missing
- LLM calls fail or return malformed JSON

In those cases metrics are marked `failed` or `partial` with explicit reasons and log paths.

## Known Limitations

- Plateau and cliff detection are still heuristic and may benefit from more adaptive sweeps on noisy systems.
- Runtime restrictions can prevent actual measurement; in that case the output remains structurally complete but metrics will fail explicitly.
- NCU summaries are auxiliary and may be unavailable on locked-down systems.
