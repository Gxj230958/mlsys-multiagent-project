# Autonomous GPU Hardware-Probing Agent

This project is a submit-ready MLSYS GPU reverse-engineering agent. It uses generated CUDA microbenchmarks plus a prompt-driven multi-agent orchestration layer, while preserving a deterministic local fallback when no OpenAI-compatible API is configured.

## Local Run

```bash
chmod +x run.sh
./run.sh
```

`run.sh` always runs from the project root, creates the required directories, invokes `python3 src/agent.py`, and exits `0` when `output.json` is generated even if some metrics fail.

## Inputs And Outputs

The only evaluation-target input is `/target/target_spec.json`.

The runtime does not read `./target_spec.json`, `./test_spec.json`, or any other local fallback file for target selection. If `/target/target_spec.json` is missing, malformed, or contains no recognizable requested metrics, the run records that failure in `output.json` instead of inventing substitute targets.

The official target specification may contain either semantic hardware metric names or Nsight Compute / CUDA attribute style metric names. The agent supports both through a compatibility layer and preserves the original requested names in `output.json`.

Generated outputs:

- `output.json` in the project root during local runs
- `/workspace/output.json` as well when `/workspace` exists and differs from the local root
- generated CUDA sources and binaries under `benchmarks/generated/`
- compile, run, and optional NCU logs under `logs/`

`output.json` includes `target_spec_path` so each run records which specification file was selected.

## Environment Variables

Optional OpenAI-compatible settings:

- `API_KEY`
- `BASE_MODEL`
- `BASE_URL`

No API key is required. Without those variables, the project uses deterministic local logic for target normalization, probe planning, benchmark selection, triage, and result aggregation.

## CUDA Tooling Notes

- `nvcc` is required to compile the generated CUDA benchmarks.
- `ncu` is optional and used only for auxiliary summaries.
- CUDA driver/runtime mismatches, missing GPU access, or missing tools do not abort the whole pipeline; they are reflected in `output.json` as `failed` or `partial` metrics with explicit reasons.

## Supported Metrics

The evaluator can request any supported subset of:

- `l1_latency_cycles`
- `l2_latency_cycles`
- `dram_latency_cycles`
- `global_memory_bandwidth_gbps`
- `dram_bytes_read_per_second`
- `dram_bytes_write_per_second`
- `shared_memory_bandwidth_gbps`
- `l2_cache_capacity_bytes`
- `actual_core_clock_mhz`
- `device_attribute_max_gpu_frequency_khz`
- `device_attribute_max_mem_frequency_khz`
- `device_attribute_fb_bus_width_bits`
- `peak_fp32_tflops`
- `sm_throughput_pct_of_peak_sustained_elapsed`
- `gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed`
- `shared_memory_bank_conflict_penalty_cycles`
- `shared_memory_bank_conflict_penalty_ratio`
- `observed_active_sm_count`

## Anti-Hardcoding

The project does not use static GPU specification tables or infer answers from device names. API-reported properties are logged only as auxiliary evidence; reported metrics are intended to come from runtime probes.
