from __future__ import annotations

from pathlib import Path


PROMPT_FILES = {
    "SpecInterpreterAgent": "SpecInterpreter Agent.txt",
    "ProbePlannerAgent": "ProbePlanner Agent.txt",
    "BenchmarkGeneratorAgent": "BenchmarkGenerator Agent.txt",
    "NcuAnalystAgent": "NcuAnalyst Agent.txt",
    "ExecutionTriageAgent": "ExecutionTriage Agent.txt",
    "ResultAggregatorAgent": "ResultAggregator Agent.txt",
    "ProbeRunnerAgent": "ProbeRunner Agent.txt",
}


class PromptManager:
    def __init__(self, prompts_dir: Path, agent_logs: list[str]) -> None:
        self.prompts_dir = prompts_dir
        self.agent_logs = agent_logs
        self._cache: dict[str, str] = {}

    def validate_required_prompts(self) -> None:
        for agent_name, filename in PROMPT_FILES.items():
            prompt_path = self.prompts_dir / filename
            if not prompt_path.exists():
                self.agent_logs.append(f"Missing prompt file for {agent_name}: {prompt_path}. Deterministic fallback will be used.")

    def get_prompt(self, agent_name: str) -> str:
        if agent_name in self._cache:
            return self._cache[agent_name]
        filename = PROMPT_FILES.get(agent_name)
        if not filename:
            fallback = _fallback_prompt(agent_name)
            self.agent_logs.append(f"No prompt file mapping registered for {agent_name}; using fallback prompt.")
            self._cache[agent_name] = fallback
            return fallback
        prompt_path = self.prompts_dir / filename
        try:
            text = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            text = _fallback_prompt(agent_name)
            self.agent_logs.append(f"Failed to load prompt for {agent_name}: {exc}. Using deterministic fallback prompt.")
        if not text:
            text = _fallback_prompt(agent_name)
            self.agent_logs.append(f"Prompt for {agent_name} was empty. Using deterministic fallback prompt.")
        self._cache[agent_name] = text
        return text


def _fallback_prompt(agent_name: str) -> str:
    return (
        f"You are {agent_name}. Output JSON only. "
        "Preserve the provided schema exactly, avoid static GPU specs, express uncertainty explicitly, "
        "and prefer microbenchmark-derived reasoning over API-reported properties."
    )
