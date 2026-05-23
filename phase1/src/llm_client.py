from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class CompatibleLLMClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("API_KEY")
        self.base_model = os.environ.get("BASE_MODEL")
        self.base_url = os.environ.get("BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def available(self) -> bool:
        return bool(self.api_key and self.base_model)

    def chat_json(self, system_prompt: str, user_payload: dict, timeout_sec: int = 20) -> dict | None:
        if not self.available():
            return None
        payload = {
            "model": self.base_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, indent=2, sort_keys=True)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        response_text = self._post(payload, timeout_sec=timeout_sec)
        if not response_text:
            return None
        return _extract_json_object(response_text)

    def summarize(self, prompt: str, timeout_sec: int = 10) -> str | None:
        if not self.available():
            return None
        payload = {
            "model": self.base_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are summarizing GPU microbenchmark findings for an MLSYS course report. Be concise and factual.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        return self._post(payload, timeout_sec=timeout_sec)

    def _post(self, payload: dict, timeout_sec: int) -> str | None:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None

        try:
            return parsed["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            return None


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
