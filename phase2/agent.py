#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from src.candidates import Candidate, CandidateGenerator
from src.harness import benchmark, check_correctness, compile_candidate, median_speedup, summarize_correctness
from src.llm_client import llm_candidate_hint
from src.ncu_probe import ncu_version, profile_selected
from src.reporter import build_report_payload, write_output_json, write_report2_md
from src.utils import ROOT, ensure_dirs, environment_diagnostics, write_text_atomic


GENERATED_DIR = ROOT / "generated"
BUILD_CACHE_DIR = ROOT / "build_cache"
OUTPUT_JSON = ROOT / "output.json"
REPORT_MD = ROOT / "report2.md"
OPTIMIZED_PATH = ROOT / "optimized_lora.cu"

SCREEN_SIZES = [3584, 4096, 4608]
FINAL_CORRECTNESS_SIZES = [3584, 3585, 4096, 4607, 4608]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact Phase 2 LoRA optimizer")
    parser.add_argument("--smoke", action="store_true", help="Use tiny sizes for compile/interface validation")
    parser.add_argument("--emit-baseline-only", action="store_true", help="Emit safe ATen fallback and exit")
    return parser.parse_args()


def load_dotenv_if_present() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def candidate_path(candidate: Candidate) -> Path:
    return GENERATED_DIR / f"{candidate.name}.cu"


def write_candidate(candidate: Candidate, path: Path) -> None:
    write_text_atomic(path, candidate.source)


def compact_error(text: Any, limit: int = 500) -> str:
    value = "" if text is None else str(text)
    return value if len(value) <= limit else value[:limit] + "...<truncated>"


def speedup_by_d(benchmarks: list[dict[str, Any]]) -> dict[int, float]:
    return {
        int(item["d"]): float(item["speedup"])
        for item in benchmarks
        if item.get("speedup", 0) > 0 and not item.get("error")
    }


def maybe_reorder_from_llm(candidates: list[Candidate], args: argparse.Namespace) -> tuple[list[Candidate], dict[str, Any]]:
    if args.smoke:
        return candidates, {"enabled": _env_flag("ENABLE_LLM", True), "available": False, "used": False, "reason": "smoke mode"}
    hint = llm_candidate_hint([candidate.name for candidate in candidates])
    choice = hint.get("try")
    if not choice:
        return candidates, hint
    preferred = [candidate for candidate in candidates if candidate.name == choice]
    rest = [candidate for candidate in candidates if candidate.name != choice]
    baseline = [candidate for candidate in rest if candidate.name == "safe_aten_basic"]
    rest = [candidate for candidate in rest if candidate.name != "safe_aten_basic"]
    return baseline + preferred + rest, hint


def evaluate_candidate(
    candidate: Candidate,
    sizes: list[int],
    warmup: int,
    iters: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any | None]:
    path = candidate_path(candidate)
    write_candidate(candidate, path)
    result: dict[str, Any] = {
        "name": candidate.name,
        "description": candidate.description,
        "compile_ok": False,
        "correct": False,
        "speedup": 0.0,
        "speedup_by_d": {},
        "correctness_summary": {"passed": False, "max_abs_err": None, "rel_l2_err": None},
        "reason": "not evaluated",
    }
    if args.emit_baseline_only:
        result.update({"compile_ok": False, "correct": True, "reason": "baseline emitted without evaluation"})
        return result, None

    compile_result = compile_candidate(path, f"lora_{candidate.name}", BUILD_CACHE_DIR)
    module = compile_result.pop("module", None)
    result["compile_ok"] = bool(compile_result.get("ok"))
    if not result["compile_ok"] or module is None:
        result["reason"] = "compile failed: " + compact_error(compile_result.get("error"))
        return result, None

    correctness = [check_correctness(module, d=d, seed=1000 + idx, mode="official") for idx, d in enumerate(sizes)]
    summary = summarize_correctness(correctness)
    result["correctness_summary"] = summary
    result["correct"] = bool(summary.get("passed"))
    if not result["correct"]:
        result["reason"] = "correctness failed"
        if summary.get("failed_d"):
            result["reason"] += f" at d={summary.get('failed_d')}"
        return result, module

    benches = [benchmark(module, d=d, seed=2000 + idx, warmup=warmup, iters=iters) for idx, d in enumerate(sizes)]
    result["speedup_by_d"] = speedup_by_d(benches)
    result["speedup"] = median_speedup(benches)
    result["reason"] = "correct benchmarked" if result["speedup"] > 0 else "benchmark failed"
    return result, module


def select_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in results if item.get("compile_ok") and item.get("correct") and item.get("speedup", 0) > 0]
    if not eligible:
        return next((item for item in results if item.get("name") == "safe_aten_basic"), None)
    return max(eligible, key=lambda item: float(item.get("speedup", 0.0)))


def final_validate(
    ranked_results: list[dict[str, Any]],
    modules: dict[str, Any],
    candidate_map: dict[str, Candidate],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.smoke:
        selected = ranked_results[0]
        selected["reason"] = "selected"
        write_candidate(candidate_map[str(selected["name"])], OPTIMIZED_PATH)
        return selected
    for result in ranked_results:
        module = modules.get(str(result.get("name")))
        if module is None:
            continue
        checks = [
            check_correctness(module, d=d, seed=3000 + idx, mode="official")
            for idx, d in enumerate(FINAL_CORRECTNESS_SIZES)
        ]
        summary = summarize_correctness(checks)
        result["correctness_summary"] = summary
        result["correct"] = bool(summary.get("passed"))
        if result["correct"]:
            result["reason"] = "selected"
            write_candidate(candidate_map[str(result["name"])], OPTIMIZED_PATH)
            return result
        result["reason"] = "rejected by final correctness"
    fallback = next(item for item in ranked_results if item.get("name") == "safe_aten_basic")
    write_candidate(candidate_map["safe_aten_basic"], OPTIMIZED_PATH)
    fallback["reason"] = "selected fallback"
    return fallback


def main() -> int:
    start = time.monotonic()
    load_dotenv_if_present()
    args = parse_args()
    ensure_dirs(GENERATED_DIR, BUILD_CACHE_DIR)

    generator = CandidateGenerator()
    candidates = generator.generate()
    candidates, llm_status = maybe_reorder_from_llm(candidates, args)
    candidate_map = {candidate.name: candidate for candidate in candidates}
    baseline = generator.baseline()
    if not OPTIMIZED_PATH.exists():
        write_candidate(baseline, OPTIMIZED_PATH)

    env = environment_diagnostics()
    notes: list[str] = [
        "Full compact GEMM/cuBLAS run with LLM hinting and NCU probe enabled by default.",
        f"CUDA available: {env.get('cuda_available')}",
    ]
    if llm_status.get("try"):
        notes.append(f"LLM suggested trying {llm_status.get('try')} first.")
    ncu_status: dict[str, Any] = (
        {"enabled": _env_flag("ENABLE_NCU", True), "available": False, "attempted": False, "reason": "smoke mode"}
        if args.smoke
        else ncu_version()
    )

    if args.emit_baseline_only:
        write_candidate(baseline, OPTIMIZED_PATH)
        result = {
            "name": baseline.name,
            "compile_ok": False,
            "correct": True,
            "speedup": 0.0,
            "speedup_by_d": {},
            "correctness_summary": {"passed": True, "max_abs_err": None, "rel_l2_err": None},
            "reason": "baseline-only mode",
        }
        payload = build_report_payload(best=result, candidates=[result], notes=notes, elapsed_sec=time.monotonic() - start, llm=llm_status, ncu=ncu_status)
        write_output_json(OUTPUT_JSON, payload)
        write_report2_md(REPORT_MD, payload)
        return 0

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if not cuda_available:
        write_candidate(baseline, OPTIMIZED_PATH)
        result = {
            "name": baseline.name,
            "compile_ok": False,
            "correct": True,
            "speedup": 0.0,
            "speedup_by_d": {},
            "correctness_summary": {"passed": True, "max_abs_err": None, "rel_l2_err": None},
            "reason": "CUDA unavailable; emitted safe baseline",
        }
        notes.append("CUDA unavailable, so compile/benchmark were skipped.")
        ncu_status.update({"attempted": False, "ok": False, "reason": "CUDA unavailable"})
        payload = build_report_payload(best=result, candidates=[result], notes=notes, elapsed_sec=time.monotonic() - start, llm=llm_status, ncu=ncu_status)
        write_output_json(OUTPUT_JSON, payload)
        write_report2_md(REPORT_MD, payload)
        return 0

    sizes = [64, 128] if args.smoke else SCREEN_SIZES
    warmup = _env_int("BENCHMARK_WARMUP", 5)
    iters = _env_int("BENCHMARK_ITERS", 20)

    results: list[dict[str, Any]] = []
    modules: dict[str, Any] = {}
    for candidate in candidates:
        print(f"[agent] evaluating {candidate.name}", flush=True)
        result, module = evaluate_candidate(candidate, sizes, warmup, iters, args)
        results.append(result)
        if module is not None:
            modules[candidate.name] = module
        print(
            f"[agent] {candidate.name}: compile={result['compile_ok']} "
            f"correct={result['correct']} speedup={result['speedup']:.4f} {result['reason']}",
            flush=True,
        )

    best = select_best(results)
    if best is None:
        best = next(item for item in results if item["name"] == "safe_aten_basic")
    ranked = sorted(
        [item for item in results if item.get("compile_ok") and item.get("correct")],
        key=lambda item: float(item.get("speedup", 0.0)),
        reverse=True,
    )
    if not any(item.get("name") == "safe_aten_basic" for item in ranked):
        ranked.append(next(item for item in results if item["name"] == "safe_aten_basic"))
    if best not in ranked:
        ranked.insert(0, best)
    best = final_validate(ranked, modules, candidate_map, args)
    if not args.smoke:
        ncu_status = profile_selected(
            source_path=OPTIMIZED_PATH,
            candidate_name=str(best.get("name")),
            work_dir=ROOT,
            d=_env_int("NCU_PROFILE_D", 4096),
            timeout_s=_env_int("NCU_TIMEOUT_S", 180),
        )
    for item in results:
        if item.get("name") != best.get("name") and item.get("reason") == "correct benchmarked":
            item["reason"] = "rejected"

    payload = build_report_payload(best=best, candidates=results, notes=notes, elapsed_sec=time.monotonic() - start, llm=llm_status, ncu=ncu_status)
    write_output_json(OUTPUT_JSON, payload)
    write_report2_md(REPORT_MD, payload)
    print(f"[agent] selected_candidate={best.get('name')} speedup={best.get('speedup'):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
