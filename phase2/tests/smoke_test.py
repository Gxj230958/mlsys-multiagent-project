from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agent_smoke_generates_outputs() -> None:
    proc = subprocess.run(
        [sys.executable, "agent.py", "--smoke"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    optimized = ROOT / "optimized_lora.cu"
    output_json = ROOT / "output.json"
    assert optimized.exists()
    assert output_json.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["summary"]["best_candidate"]
    assert "best" in data
    assert len(output_json.read_text(encoding="utf-8")) < 20_000
    if data["summary"].get("num_compile_ok", 0):
        assert any(item.get("compile_ok") and item.get("correct") for item in data.get("candidates", []))
