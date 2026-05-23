# Phase 2 Compact Run

- Best candidate: `cublas_sgemm_single_stream_beta_accumulate`
- Median speedup: `1.027409`
- Correct: `True`
- Candidates tested: `6`
- Compile ok / correct: `6` / `5`
- Speedup by d: `{'3584': 1.027409, '4096': 1.02051, '4608': 1.073508}`
- Correctness: `{'max_abs_err': 1e-06, 'rel_l2_err': 1e-06}`

## Candidates
- `safe_aten_basic` compile=True correct=True speedup=0.997718 reason=rejected
- `aten_addmm_accumulate` compile=True correct=True speedup=0.997474 reason=rejected
- `custom_rank16_update_best_only` compile=True correct=True speedup=1.012951 reason=rejected
- `cublas_sgemm_sequential` compile=True correct=True speedup=1.015354 reason=rejected
- `cublas_sgemm_single_stream_beta_accumulate` compile=True correct=True speedup=1.027409 reason=selected
- `cublas_gemmex_tf32` compile=True correct=False speedup=0.0 reason=correctness failed at d=[3584, 4096, 4608]

## Notes
- Full compact GEMM/cuBLAS run with LLM hinting and NCU probe enabled by default.
- CUDA available: True

## LLM
- `{'enabled': True, 'available': True, 'used': False, 'provider': 'deepseek', 'model': 'deepseek-v4-pro', 'try': None, 'reason': '', 'env_loaded_from': '/workspace/.env', 'key_present': {'DEEPSEEK_API_KEY': True, 'DeepSeek_API_KEY': False, 'API_KEY': True, 'OPENAI_API_KEY': False}}`

## NCU
- `{'enabled': True, 'available': True, 'path': '/usr/local/bin/ncu', 'version': 'NVIDIA (R) Nsight Compute Command Line Profiler', 'attempted': True, 'ok': True, 'candidate': 'cublas_sgemm_single_stream_beta_accumulate', 'd': 4096, 'log': 'generated/ncu_compact/profile_selected.log', 'returncode': 0}`
