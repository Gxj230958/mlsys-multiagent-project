from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import write_json_atomic, write_text_atomic


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def compact_candidate(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": result.get("name"),
        "compile_ok": bool(result.get("compile_ok")),
        "correct": bool(result.get("correct")),
        "speedup": _round(result.get("speedup", 0.0)),
        "reason": result.get("reason"),
    }


def write_output_json(path: Path, data: dict[str, Any]) -> None:
    write_json_atomic(path, data)


def build_report_payload(
    *,
    best: dict[str, Any],
    candidates: list[dict[str, Any]],
    notes: list[str],
    elapsed_sec: float,
    llm: dict[str, Any] | None = None,
    ncu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact = [compact_candidate(item) for item in candidates]
    compile_ok = sum(1 for item in candidates if item.get("compile_ok"))
    correct = sum(1 for item in candidates if item.get("correct"))
    best_correctness = best.get("correctness_summary", {})
    best_payload = {
        "name": best.get("name"),
        "speedup_by_d": {str(k): _round(v) for k, v in best.get("speedup_by_d", {}).items()},
        "median_speedup": _round(best.get("speedup", 0.0)),
        "correctness": {
            "max_abs_err": _round(best_correctness.get("max_abs_err")),
            "rel_l2_err": _round(best_correctness.get("rel_l2_err")),
        },
    }
    payload = {
        "summary": {
            "best_candidate": best.get("name"),
            "best_speedup": _round(best.get("speedup", 0.0)),
            "correct": bool(best.get("correct")),
            "elapsed_sec": round(elapsed_sec, 3),
            "num_candidates_tested": len(candidates),
            "num_compile_ok": compile_ok,
            "num_correct": correct,
        },
        "best": best_payload,
        "candidates": compact,
        "notes": notes,
    }
    if llm is not None:
        payload["llm"] = llm
    if ncu is not None:
        payload["ncu"] = ncu
    return payload


def write_report2_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    best = payload.get("best", {})
    lines = [
        "# Phase 2 Compact Run",
        "",
        f"- Best candidate: `{summary.get('best_candidate')}`",
        f"- Median speedup: `{summary.get('best_speedup')}`",
        f"- Correct: `{summary.get('correct')}`",
        f"- Candidates tested: `{summary.get('num_candidates_tested')}`",
        f"- Compile ok / correct: `{summary.get('num_compile_ok')}` / `{summary.get('num_correct')}`",
        f"- Speedup by d: `{best.get('speedup_by_d')}`",
        f"- Correctness: `{best.get('correctness')}`",
        "",
        "## Candidates",
    ]
    for item in payload.get("candidates", []):
        lines.append(
            f"- `{item.get('name')}` compile={item.get('compile_ok')} "
            f"correct={item.get('correct')} speedup={item.get('speedup')} reason={item.get('reason')}"
        )
    notes = payload.get("notes", [])
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)
    if payload.get("llm"):
        lines.extend(["", "## LLM", f"- `{payload.get('llm')}`"])
    if payload.get("ncu"):
        lines.extend(["", "## NCU", f"- `{payload.get('ncu')}`"])
    write_text_atomic(path, "\n".join(lines) + "\n")
