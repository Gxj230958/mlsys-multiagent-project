from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _compact_error(exc: BaseException, limit: int = 500) -> str:
    text = repr(exc)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _load_dotenv_if_present() -> str | None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    seen: set[Path] = set()
    loaded_from = None
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        loaded_from = str(path)
        try:
            for raw in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            continue
    return loaded_from


def _key_status() -> dict[str, bool]:
    return {
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "DeepSeek_API_KEY": bool(os.environ.get("DeepSeek_API_KEY")),
        "API_KEY": bool(os.environ.get("API_KEY")),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
    }


def llm_candidate_hint(candidate_names: list[str]) -> dict[str, Any]:
    env_loaded_from = _load_dotenv_if_present()
    if not _env_flag("ENABLE_LLM", True):
        return {"enabled": False, "available": False, "used": False, "reason": "ENABLE_LLM=0", "env_loaded_from": env_loaded_from}
    provider_hint = os.environ.get("LLM_PROVIDER", "").strip().lower()
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DeepSeek_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    generic_key = os.environ.get("API_KEY")
    if provider_hint == "openai" and openai_key:
        provider = "openai"
    elif provider_hint == "deepseek" and (deepseek_key or generic_key):
        provider = "deepseek"
    elif deepseek_key or generic_key:
        provider = "deepseek"
    elif openai_key:
        provider = "openai"
    else:
        return {
            "enabled": True,
            "available": False,
            "used": False,
            "reason": "no API key",
            "env_loaded_from": env_loaded_from,
            "key_present": _key_status(),
        }

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "used": False,
            "reason": "openai import failed",
            "error": _compact_error(exc),
            "env_loaded_from": env_loaded_from,
            "key_present": _key_status(),
        }

    if provider == "openai":
        api_key = openai_key
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("OPENAI_BASE_MODEL") or os.environ.get("BASE_MODEL") or "gpt-4.1-mini"
    else:
        api_key = deepseek_key or generic_key
        base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DeepSeek_BASE_URL") or os.environ.get("BASE_URL")
        model = os.environ.get("DEEPSEEK_MODEL") or os.environ.get("DEEPSEEK_BASE_MODEL") or os.environ.get("DeepSeek_BASE_MODEL") or os.environ.get("BASE_MODEL") or "deepseek-chat"
        if not base_url:
            return {"enabled": True, "available": False, "used": False, "provider": provider, "reason": "missing base_url", "env_loaded_from": env_loaded_from, "key_present": _key_status()}
    if not api_key:
        return {"enabled": True, "available": False, "used": False, "provider": provider, "reason": "missing api_key", "env_loaded_from": env_loaded_from, "key_present": _key_status()}

    try:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        prompt = {
            "task": "Choose the most promising predefined LoRA candidate family to try first.",
            "candidates": candidate_names,
            "constraints": "Return JSON only, e.g. {\"try\":\"cublas_gemmex_tf32\",\"reason\":\"short\"}. Do not generate code.",
        }
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise CUDA GEMM strategy selector."},
                {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
            ],
            temperature=0.1,
            max_tokens=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "200")),
            timeout=int(os.environ.get("LLM_TIMEOUT_S", "30")),
        )
        text = response.choices[0].message.content if response.choices else ""
        parsed = json.loads(text or "{}")
        choice = str(parsed.get("try", ""))
        if choice not in candidate_names:
            choice = ""
        return {
            "enabled": True,
            "available": True,
            "used": bool(choice),
            "provider": provider,
            "model": model,
            "try": choice or None,
            "reason": str(parsed.get("reason", ""))[:300],
            "env_loaded_from": env_loaded_from,
            "key_present": _key_status(),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": True,
            "used": False,
            "provider": provider,
            "model": model,
            "reason": "LLM call failed",
            "error": _compact_error(exc),
            "env_loaded_from": env_loaded_from,
            "key_present": _key_status(),
        }
