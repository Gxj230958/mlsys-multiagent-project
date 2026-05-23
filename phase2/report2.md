# Architecture Report: Agentic Optimization for the Phase 2 LoRA Operator

## Executive Summary

This project implements an automated CUDA optimization agent for the Phase 2 LoRA-style operator:

```text
Y = W X + A(B^T X)
```

where all tensors are float32 CUDA tensors, the LoRA rank is fixed at 16, and the hidden evaluation dimension `d` may be any integer in the interval `[3584, 4608]`. The final submission artifact is not a manually edited one-off kernel. Instead, the project builds, validates, profiles, and selects `optimized_lora.cu` through a controlled optimization pipeline.

The core design goal is to combine performance exploration with submission safety. Every candidate must compile as a self-contained PyTorch extension, pass numerical correctness against the PyTorch reference implementation, and outperform the current best candidate under CUDA-event benchmarking before it can replace `optimized_lora.cu`.

## High-Level Architecture

The system is organized as a small agent framework with clear separation of responsibilities:

- `agent.py` is the orchestration layer. It loads configuration, builds the shape policy, runs candidate generation, compiles and validates candidates, benchmarks them, applies the selection policy, and writes the final reports.
- `src/candidates.py` defines deterministic candidate families and the candidate metadata schema. Each candidate includes source code, implementation family, tuning parameters, expected effect, risk, and a reproducible genome record.
- `src/harness.py` is responsible for compilation, correctness testing, and timing. It uses `torch.utils.cpp_extension.load` for candidate builds and CUDA events for median latency measurement.
- `src/selector.py` implements the replacement policy. Correctness is a hard gate, and the primary score is the median speedup across representative benchmark sizes.
- `src/shape_policy.py` protects against overfitting to a tiny set of dimensions by mixing endpoints, aligned sizes, near-boundary sizes, intermediate sizes, and seeded random dimensions.
- `src/llm_client.py`, `src/static_cuda_check.py`, and related prompt modules provide an optional LLM-assisted exploration path. LLM outputs are advisory only and must pass static checks, compilation, correctness, and benchmarking.
- `src/ncu_profiler.py` and `src/profile_critic.py` provide optional Nsight Compute profiling and bottleneck interpretation when the server environment allows it.
- `src/reporter.py` writes the structured `output.json` and this human-readable report.

This modular structure makes the project auditable: candidate generation, measurement, selection, and reporting are separate components rather than a single opaque script.

## Candidate Search Strategy

The search starts from a guaranteed safe baseline and then explores increasingly specialized implementations:

1. A conservative ATen baseline using standard matrix multiplications and addition.
2. An `addmm` accumulation variant that reduces unnecessary tensor operations.
3. Custom rank-16 CUDA update kernels that keep the large GEMMs in optimized library code while specializing the low-rank update `A(B^T X)`.
4. Thread-column variants that assign multiple neighboring columns to a thread to improve horizontal locality and reduce indexing overhead.
5. Optional LLM-proposed variants, including bounded template mutations and raw CUDA proposals after static validation.

The selected candidate in the current run is:

```text
custom_rank16_update_bx32_by8
```

This design is intentionally practical. The dominant work in this operator is still dense matrix multiplication, so the project avoids replacing all GEMM logic with fragile hand-written kernels. Instead, it focuses custom CUDA effort on the rank-16 update, where the structure is small enough to specialize safely and where launch overhead, indexing, and memory traffic can be improved.

## Correctness-First Evaluation

Correctness is treated as a non-negotiable gate. Each candidate is checked against:

```python
W @ X + A @ (B.transpose(0, 1).contiguous() @ X)
```

The validation policy covers both aligned and non-aligned dimensions, including interval endpoints and near-boundary sizes such as `3585` and `4607`. This matters because the official hidden tests may choose any integer in `[3584, 4608]`, not only common powers of two or multiples of 128.

Candidates that fail compilation or numerical validation are recorded in `output.json` but cannot be benchmarked or selected. This makes the final output conservative and robust even when exploratory candidates fail.

## Benchmark and Selection Policy

Benchmarking uses CUDA events with warmup iterations and median timing. For each sampled dimension, local speedup is computed as:

```text
torch_reference_median_ms / candidate_median_ms
```

The selection score is the median of per-dimension speedups across the benchmark set. This avoids selecting a candidate that performs well on only one dimension while regressing elsewhere. The report also preserves per-dimension timings, minimum speedups, and best-update traces so the result can be inspected after the run.

The current selected candidate improves the local median speedup while preserving correctness across the tested shape policy. The project therefore optimizes for a balanced hidden-set outcome rather than a single cherry-picked size.

## LLM Integration With Safety Boundaries

The optional LLM path is designed as a proposal mechanism, not as an authority. The LLM may suggest candidate specs, tuning directions, or raw CUDA code, but it cannot directly overwrite `optimized_lora.cu`.

LLM-proposed candidates must pass the same pipeline as deterministic candidates:

1. static source validation for raw CUDA proposals;
2. extension compilation;
3. numerical correctness checks;
4. CUDA-event benchmarking;
5. selection policy approval.

This gives the project the benefit of broader exploration while keeping the final artifact governed by measured evidence rather than by unverified generated code.

## Profiling and Feedback Loop

The project includes optional Nsight Compute profiling support. When available, NCU profiling can collect kernel-level evidence for bottleneck analysis, and the profile critic can convert that evidence into candidate-generation guidance.

This profiling path is deliberately optional because evaluation servers may restrict profiler access. If NCU is unavailable or fails to profile kernels, the deterministic and LLM-assisted search still completes and produces a valid `optimized_lora.cu`.

## Reproducibility and Auditability

The project writes several artifacts that make the optimization process reproducible:

- `optimized_lora.cu`: the final selected source file.
- `output.json`: structured compile, correctness, benchmark, selection, LLM, and profiling records.
- `generated/best_history/`: snapshots of every best-so-far candidate.
- `generated/llm_raw_candidates/`: raw LLM-generated CUDA candidates that passed static screening.
- `generated/static_check_reports/`: static validation reports for raw candidates.
- `build_cache/`: compiled extension build directories.

The output is not just a final score. It is a trace of why the selected candidate was chosen and why rejected candidates were not selected.

## Why This Architecture Is Strong

This architecture is strong because it balances three constraints that are often in tension in CUDA optimization:

- It is safe: a valid baseline is written immediately, and no candidate can replace it without passing correctness.
- It is performance-oriented: the search targets the real operator structure and uses measured CUDA timings rather than static heuristics.
- It is generalizable: the shape policy explicitly tests non-round and near-boundary dimensions to reduce overfitting to a few visible cases.
- It is extensible: deterministic templates, LLM-guided proposals, raw CUDA candidates, and profiler feedback all plug into the same validation and selection pipeline.
- It is auditable: every candidate decision is recorded with source hashes, parameters, correctness results, benchmark results, and selection rationale.

In short, the project is not merely a CUDA kernel submission. It is a measured optimization system that continuously protects correctness while searching for better implementations under realistic evaluation constraints.
