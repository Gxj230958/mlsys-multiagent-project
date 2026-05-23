from __future__ import annotations

import math
import statistics
import traceback
from pathlib import Path
from typing import Any

from .utils import summarize_exception

_REFERENCE_TIME_CACHE: dict[tuple[int, int, int, int], float] = {}


def generate_inputs(d: int, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    W = (torch.randn(d, d, device="cuda", dtype=torch.float32) / math.sqrt(d)).contiguous()
    X = (torch.randn(d, d, device="cuda", dtype=torch.float32) / math.sqrt(d)).contiguous()
    A = (torch.randn(d, 16, device="cuda", dtype=torch.float32) / math.sqrt(16)).contiguous()
    B = (torch.randn(d, 16, device="cuda", dtype=torch.float32) / math.sqrt(16)).contiguous()
    return W, X, A, B


def reference_impl(W, X, A, B):
    return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def compile_candidate(cu_path: Path, name: str, build_root: Path) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.cpp_extension import load
    except Exception as exc:
        return {"ok": False, "error": f"torch import failed: {summarize_exception(exc, 1200)}"}

    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA is not available"}

    build_directory = build_root / name
    build_directory.mkdir(parents=True, exist_ok=True)
    try:
        module = load(
            name=name,
            sources=[str(cu_path)],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublas"],
            with_cuda=True,
            verbose=False,
            build_directory=str(build_directory),
        )
        return {"ok": True, "module": module}
    except Exception as exc:
        return {
            "ok": False,
            "error": summarize_exception(exc, 1800),
            "traceback": traceback.format_exc(limit=5),
        }


def check_correctness(
    module,
    d: int,
    seed: int = 123,
    mode: str = "official",
) -> dict[str, Any]:
    import torch

    try:
        with torch.no_grad():
            W, X, A, B = generate_inputs(d, seed=seed)
            y_student = module.forward(W, X, A, B)
            y_ref = reference_impl(W, X, A, B)
            if list(y_student.shape) != list(y_ref.shape):
                return {"d": d, "passed": False, "error": "shape mismatch"}
            if y_student.dtype != y_ref.dtype or y_student.device != y_ref.device:
                return {"d": d, "passed": False, "error": "dtype/device mismatch"}
            if not torch.isfinite(y_student).all():
                return {"d": d, "passed": False, "error": "NaN or Inf in output"}
            diff = y_student - y_ref
            max_abs_err = float(diff.abs().max().item())
            rel_l2_err = float(diff.norm().item() / max(y_ref.norm().item(), 1e-12))
            strict_passed = bool(torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4))
            if mode == "strict":
                passed = strict_passed
            else:
                passed = bool((rel_l2_err <= 1e-5 or strict_passed) and max_abs_err <= 0.25)
        return {
            "d": d,
            "passed": passed,
            "strict_passed": strict_passed,
            "max_abs_err": max_abs_err,
            "rel_l2_err": rel_l2_err,
        }
    except Exception as exc:
        oom = "out of memory" in repr(exc).lower()
        if oom:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return {
            "d": d,
            "passed": False,
            "oom": oom,
            "error": summarize_exception(exc, 1200),
            "traceback": traceback.format_exc(limit=5),
        }


def summarize_correctness(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"passed": False, "max_abs_err": None, "rel_l2_err": None}
    return {
        "passed": all(item.get("passed") for item in items),
        "max_abs_err": max(float(item.get("max_abs_err", 0.0)) for item in items if item.get("max_abs_err") is not None)
        if any(item.get("max_abs_err") is not None for item in items)
        else None,
        "rel_l2_err": max(float(item.get("rel_l2_err", 0.0)) for item in items if item.get("rel_l2_err") is not None)
        if any(item.get("rel_l2_err") is not None for item in items)
        else None,
        "failed_d": [item.get("d") for item in items if not item.get("passed")],
    }


def _time_callable(fn, args: tuple, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args)
        end.record()
        torch.cuda.synchronize()
        _ = out
        times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def benchmark(module, d: int, seed: int = 456, warmup: int = 5, iters: int = 20) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"d": d, "error": "CUDA is not available"}
    try:
        with torch.no_grad():
            W, X, A, B = generate_inputs(d, seed=seed)
            student_ms = _time_callable(module.forward, (W, X, A, B), warmup=warmup, iters=iters)
            cache_key = (int(d), int(seed), int(warmup), int(iters))
            torch_ms = _REFERENCE_TIME_CACHE.get(cache_key)
            if torch_ms is None:
                torch_ms = _time_callable(reference_impl, (W, X, A, B), warmup=warmup, iters=iters)
                _REFERENCE_TIME_CACHE[cache_key] = torch_ms
        return {
            "d": d,
            "student_ms": student_ms,
            "torch_ms": torch_ms,
            "speedup": torch_ms / student_ms if student_ms > 0 else 0.0,
        }
    except Exception as exc:
        oom = "out of memory" in repr(exc).lower()
        if oom:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return {"d": d, "error": summarize_exception(exc, 1200), "oom": oom}


def median_speedup(benchmarks: list[dict[str, Any]]) -> float:
    speeds = [float(item["speedup"]) for item in benchmarks if item.get("speedup", 0) > 0 and not item.get("error")]
    return float(statistics.median(speeds)) if speeds else 0.0
