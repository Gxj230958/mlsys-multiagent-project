from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    tmp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: list[str], timeout: int = 10) -> str | None:
    try:
        proc = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def environment_diagnostics() -> dict[str, Any]:
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"torch_import_error": repr(exc), "cuda_available": False}

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "deepseek_api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DeepSeek_API_KEY")),
        "generic_api_key_present": bool(os.environ.get("API_KEY")),
        "enable_llm": os.environ.get("ENABLE_LLM", "1"),
        "enable_ncu": os.environ.get("ENABLE_NCU", "1"),
        "nvcc_path": shutil.which("nvcc"),
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "nvidia_smi": command_output(["nvidia-smi", "-L"], timeout=10)
        if shutil.which("nvidia-smi")
        else None,
        **torch_info,
    }


def summarize_exception(exc: BaseException, max_chars: int = 4000) -> str:
    text = repr(exc)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def selected_sizes(smoke: bool, emit_baseline_only: bool) -> list[int]:
    if smoke:
        return [64, 128]
    if emit_baseline_only:
        return []
    if os.environ.get("FULL_SEARCH") == "1":
        return [3584, 3840, 4096, 4352, 4608]
    return [3584, 4096]
