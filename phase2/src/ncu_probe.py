from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def ncu_version() -> dict[str, Any]:
    path = shutil.which("ncu")
    if not path:
        return {"enabled": _env_flag("ENABLE_NCU", True), "available": False, "path": None, "version": None}
    try:
        proc = subprocess.run([path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15, check=False)
        first = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
        return {"enabled": _env_flag("ENABLE_NCU", True), "available": proc.returncode == 0, "path": path, "version": first[:160]}
    except Exception as exc:
        return {"enabled": _env_flag("ENABLE_NCU", True), "available": False, "path": path, "version": None, "error": repr(exc)[:300]}


def profile_selected(source_path: Path, candidate_name: str, work_dir: Path, d: int = 4096, timeout_s: int = 180) -> dict[str, Any]:
    if not _env_flag("ENABLE_NCU", True):
        return {"enabled": False, "available": False, "attempted": False, "reason": "ENABLE_NCU=0"}
    status = ncu_version()
    if not status.get("available"):
        status.update({"attempted": False, "ok": False, "reason": "ncu unavailable"})
        return status
    if not _env_flag("NCU_PROFILE", True):
        status.update({"attempted": False, "ok": False, "reason": "NCU_PROFILE=0"})
        return status

    out_dir = work_dir / "generated" / "ncu_compact"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = out_dir / "profile_selected.py"
    log_path = out_dir / "profile_selected.log"
    runner.write_text(
        f"""
import math
import torch
from torch.utils.cpp_extension import load

torch.manual_seed(9090)
d = {int(d)}
module = load(name='ncu_compact_selected', sources=['{source_path}'], extra_cuda_cflags=['-O3'], extra_ldflags=['-lcublas'], with_cuda=True, verbose=False)
W = (torch.randn(d, d, device='cuda') / math.sqrt(d)).contiguous()
X = (torch.randn(d, d, device='cuda') / math.sqrt(d)).contiguous()
A = (torch.randn(d, 16, device='cuda') / 4.0).contiguous()
B = (torch.randn(d, 16, device='cuda') / 4.0).contiguous()
for _ in range(3):
    y = module.forward(W, X, A, B)
torch.cuda.synchronize()
print(float(y[0,0].item()))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cmd = [
        str(status["path"]),
        "--target-processes",
        "all",
        "--set",
        "default",
        "--launch-count",
        "1",
        "python3",
        str(runner),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(work_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
        output = proc.stdout or ""
        log_path.write_text(output[-8000:], encoding="utf-8")
        ok = proc.returncode == 0 and "==ERROR==" not in output
        status.update(
            {
                "attempted": True,
                "ok": ok,
                "candidate": candidate_name,
                "d": int(d),
                "log": str(log_path.relative_to(work_dir)),
                "returncode": proc.returncode,
            }
        )
        if not ok:
            status["error"] = output[-500:]
        return status
    except Exception as exc:
        status.update({"attempted": True, "ok": False, "candidate": candidate_name, "d": int(d), "error": repr(exc)[:500]})
        return status
