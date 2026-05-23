# Phase 2 LoRA Compact Optimizer

This project builds a single `optimized_lora.cu` for:

```text
Y = W @ X + A @ (B.T @ X)
```

where all tensors are contiguous CUDA float32 tensors, `A/B` have rank 16, and `d` is in the hidden range `[3584, 4608]`.

## Run

```bash
bash run.sh
```

Defaults are the final full path:

- `ENABLE_LLM=1`
- `ENABLE_NCU=1`
- `NCU_PROFILE=1`
- `FULL_SEARCH=1`
- `RANDOM_SIZES=0`
- `AGENT_TIME_BUDGET_SECONDS=1500`

`run.sh` asks the LLM for one compact predefined-family hint when API credentials are available, evaluates a small fixed candidate set, writes `optimized_lora.cu`, optionally runs a compact NCU profile on the selected candidate, then writes `output.json` and `report2.md`.

## Candidate Set

The search is focused on high-impact GEMM strategies:

- `safe_aten_basic`
- `aten_addmm_accumulate`
- `custom_rank16_update_best_only`
- `cublas_sgemm_sequential`
- `cublas_sgemm_single_stream_beta_accumulate`
- `cublas_gemmex_tf32`

The cuBLAS candidates use the current PyTorch CUDA stream and single-stream sequential GEMMs. The GemmEx candidate permits TF32-style acceleration but remains gated by official-style correctness.

## Correctness and Selection

Each candidate is compiled, checked at `d = 3584, 4096, 4608`, then benchmarked at the same sizes. Selection uses median speedup across those benchmark sizes. The selected candidate is finally checked at:

```text
3584, 3585, 4096, 4607, 4608
```

Official-style correctness accepts tiny relative L2 error (`<= 1e-5`) or the strict allclose-style check, while always rejecting NaN/Inf, shape mismatches, CUDA errors, and large numerical drift.

## Output

`output.json` is intentionally compact and contains only:

- final summary
- best candidate speedup/correctness summary
- one-line candidate statuses
- short notes

It should stay well under 20 KB.
