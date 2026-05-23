from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from pathlib import Path

from agents import (
    BenchmarkGeneratorAgent,
    ExecutionTriageAgent,
    NcuAnalystAgent,
    ProbePlannerAgent,
    ResultAggregatorAgent,
    SpecInterpreterAgent,
)
from analyzer import build_parsed_measurements, summarize_execution_for_triage
from llm_client import CompatibleLLMClient
from output_schema import SUPPORTED_METRICS, base_output, canonicalize_metric_name, failed_metric
from probe_builder import build_required_probes
from probe_runner import compile_and_run_probes
from prompt_manager import PromptManager

SUBMISSION_TARGET_SPEC_PATH = Path("/target/target_spec.json")
MISSING_TARGET_SPEC_ERROR = "No target specification found at /target/target_spec.json."


class TargetSpecError(Exception):
    pass


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = base_output()
    requested_metric_names: list[str] = []
    unsupported_metric_results: dict[str, dict] = {}
    output_path = resolve_output_path(project_root)
    logs_dir = project_root / "logs"
    generated_dir = project_root / "benchmarks" / "generated"
    logs_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    clean_runtime_artifacts(logs_dir, generated_dir)

    llm_client = CompatibleLLMClient()
    prompt_manager = PromptManager(project_root / "prompts", output["agent_logs"])
    prompt_manager.validate_required_prompts()

    try:
        spec_path = resolve_target_spec_path(output["agent_logs"])
        output["target_spec_path"] = str(spec_path) if spec_path else None
        if spec_path is None:
            populate_target_spec_failure(output, requested_metric_names, MISSING_TARGET_SPEC_ERROR)
            output["agent_logs"].append(f"Output path: {output_path}")
            return

        raw_spec = load_target_spec(spec_path, output["agent_logs"])
        requested_metric_names = list(raw_spec.get("requested_metrics", []))
        normalized_payload = SpecInterpreterAgent(llm_client, prompt_manager, output["agent_logs"]).run(
            {"target_spec": raw_spec, "supported_metrics": SUPPORTED_METRICS}
        )
        output["normalized_targets"] = normalized_payload["normalized_targets"]
        for unsupported in normalized_payload.get("unsupported_targets", []):
            output["agent_logs"].append(f"Unsupported metric request ignored: {unsupported}")
        unsupported_metric_results = build_unsupported_metric_results(normalized_payload.get("unsupported_targets", []))
        if not output["normalized_targets"]:
            output["results"] = order_results_by_request(requested_metric_names, {}, unsupported_metric_results)
            output["total_metrics"] = len(requested_metric_names)
            output["successful_analyses"] = 0
            output["agent_logs"].append("No supported evaluation targets were derived from /target/target_spec.json.")
            output["agent_logs"].append(f"Output path: {output_path}")
            return

        planner_payload = ProbePlannerAgent(llm_client, prompt_manager, output["agent_logs"]).run(
            {
                "normalized_targets": output["normalized_targets"],
                "environment_constraints": {
                    "avoid_static_specs": True,
                    "api_properties_untrusted": True,
                    "possible_frequency_locking": True,
                    "possible_sm_masking": True,
                },
            }
        )
        output["probe_plan"] = planner_payload.get("probe_plan", [])

        benchmark_payload = BenchmarkGeneratorAgent(llm_client, prompt_manager, output["agent_logs"]).run(
            {
                "probe_plan": output["probe_plan"],
                "existing_probe_templates": [
                    "environment_probe",
                    "pointer_chase_latency",
                    "global_bandwidth",
                    "shared_bandwidth",
                    "core_clock",
                    "fp32_throughput",
                    "bank_conflict",
                ],
            }
        )
        probes = build_required_probes(project_root, benchmark_payload.get("benchmarks", []))
        executions = compile_and_run_probes(probes, logs_dir)
        output["generated_benchmarks"] = [execution.benchmark_record() for execution in executions.values()]
        for execution in executions.values():
            output["agent_logs"].append(
                f"Probe {execution.probe.name}: compile={execution.compile_status}, run={execution.run_status}, ncu={execution.ncu_summary}"
            )

        parsed_measurements = build_parsed_measurements(executions)
        output["environment_notes"] = augment_environment_notes(parsed_measurements.get("environment_notes", {}), output["agent_logs"])

        execution_results = summarize_execution_for_triage(executions)
        ncu_payload = NcuAnalystAgent(llm_client, prompt_manager, output["agent_logs"]).run(
            {
                "ncu_raw_outputs": [
                    {"probe_name": execution.probe.name, "output": execution.ncu_output}
                    for execution in executions.values()
                    if execution.ncu_output or execution.ncu_log
                ],
                "execution_results": execution_results,
            }
        )
        output["ncu_analysis"] = ncu_payload.get("ncu_analysis", [])

        triage_payload = ExecutionTriageAgent(llm_client, prompt_manager, output["agent_logs"]).run(
            {
                "normalized_targets": output["normalized_targets"],
                "execution_results": execution_results,
                "parsed_measurements": parsed_measurements,
                "ncu_analysis": output["ncu_analysis"],
                "environment_notes": output["environment_notes"],
            }
        )
        output["triage"] = triage_payload.get("triage", [])

        aggregated = ResultAggregatorAgent(llm_client, prompt_manager, output["agent_logs"]).run(
            {
                "normalized_targets": output["normalized_targets"],
                "probe_plan": output["probe_plan"],
                "execution_results": execution_results,
                "parsed_measurements": parsed_measurements,
                "triage": output["triage"],
                "environment_notes": output["environment_notes"],
                "ncu_analysis": output["ncu_analysis"],
            }
        )
        output["results"] = order_results_by_request(
            requested_metric_names,
            aggregated.get("results", {}),
            unsupported_metric_results,
        )
        output["agent_logs"].append(aggregated.get("summary", ""))
        output["total_metrics"] = len(requested_metric_names)
        output["successful_analyses"] = sum(1 for result in output["results"].values() if result.get("status") == "success")
        output["agent_logs"].append(f"Output path: {output_path}")
    except TargetSpecError as exc:
        output["agent_logs"].append(str(exc))
        populate_target_spec_failure(output, requested_metric_names, str(exc))
        output["agent_logs"].append(f"Output path: {output_path}")
    except Exception as exc:
        output["agent_logs"].append(f"Unhandled agent exception: {exc}")
        output["agent_logs"].append(traceback.format_exc())
        if not output["results"]:
            requested_names = requested_metric_names or [item["requested_name"] for item in output.get("normalized_targets", [])]
            for requested_name in requested_names:
                canonical = canonicalize_metric_name(requested_name) or requested_name
                output["results"][requested_name] = failed_metric(
                    canonical,
                    f"Agent crashed before collecting the metric: {exc}",
                    [],
                )
            for requested_name, result in unsupported_metric_results.items():
                output["results"].setdefault(requested_name, result)
            output["results"] = order_results_by_request(requested_names, output["results"], unsupported_metric_results)
            output["total_metrics"] = len(requested_names)
            output["successful_analyses"] = 0
    finally:
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def resolve_output_path(project_root: Path) -> Path:
    workspace_path = Path("/workspace")
    if workspace_path.exists() and workspace_path.is_dir():
        return workspace_path / "output.json"
    return project_root / "output.json"


def resolve_target_spec_path(agent_logs: list[str]) -> Path | None:
    if SUBMISSION_TARGET_SPEC_PATH.exists():
        agent_logs.append(f"Selected target specification path: {SUBMISSION_TARGET_SPEC_PATH}")
        return SUBMISSION_TARGET_SPEC_PATH
    agent_logs.append(MISSING_TARGET_SPEC_ERROR)
    return None


def load_target_spec(spec_path: Path, agent_logs: list[str]) -> dict:
    agent_logs.append(f"Loading target specification from {spec_path}")
    try:
        parsed = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TargetSpecError(f"Failed to parse /target/target_spec.json: {exc}") from exc
    requested_metrics = extract_metric_names(parsed)
    if not requested_metrics:
        raise TargetSpecError("No requested metrics found in /target/target_spec.json.")
    return {"requested_metrics": requested_metrics, "raw_spec": parsed}


def populate_target_spec_failure(output: dict, requested_metric_names: list[str], error_message: str) -> None:
    output["normalized_targets"] = []
    output["results"] = {
        metric_name: failed_metric(canonicalize_metric_name(metric_name) or metric_name, error_message, [])
        for metric_name in requested_metric_names
    }
    output["total_metrics"] = len(requested_metric_names)
    output["successful_analyses"] = 0


def extract_metric_names(spec) -> list[str]:
    if isinstance(spec, list):
        return dedupe_metric_names(item for item in spec if isinstance(item, str))
    if isinstance(spec, dict):
        for key in ("requested_metrics", "metrics", "targets"):
            value = spec.get(key)
            if isinstance(value, list):
                return dedupe_metric_names(item for item in value if isinstance(item, str))
        truthy = dedupe_metric_names(key for key, value in spec.items() if value)
        if any(canonicalize_metric_name(key) for key in truthy):
            return [key for key in truthy if canonicalize_metric_name(key)]
    return []


def augment_environment_notes(environment_notes: dict, agent_logs: list[str]) -> dict:
    notes = dict(environment_notes)
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        agent_logs.append("nvidia-smi not found; auxiliary clock evidence unavailable.")
        return notes
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,clocks.current.graphics,clocks.max.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        cleaned = proc.stdout.strip()
        if cleaned:
            agent_logs.append(f"nvidia-smi auxiliary evidence: {cleaned}")
    except Exception as exc:
        agent_logs.append(f"nvidia-smi query failed: {exc}")
    return notes


def clean_runtime_artifacts(logs_dir: Path, generated_dir: Path) -> None:
    for directory in (logs_dir, generated_dir):
        for child in directory.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def build_unsupported_metric_results(metric_names: list[str]) -> dict[str, dict]:
    results = {}
    for metric_name in dedupe_metric_names(metric_names):
        result = failed_metric(
            metric_name,
            f"Requested metric `{metric_name}` is not supported by this evaluator.",
            [],
        )
        result["status"] = "unsupported"
        results[metric_name] = result
    return results


def order_results_by_request(
    requested_metric_names: list[str],
    produced_results: dict[str, dict],
    unsupported_results: dict[str, dict],
) -> dict[str, dict]:
    ordered_results: dict[str, dict] = {}
    for requested_name in requested_metric_names:
        if requested_name in produced_results:
            ordered_results[requested_name] = produced_results[requested_name]
        elif requested_name in unsupported_results:
            ordered_results[requested_name] = unsupported_results[requested_name]
    for requested_name, result in produced_results.items():
        ordered_results.setdefault(requested_name, result)
    for requested_name, result in unsupported_results.items():
        ordered_results.setdefault(requested_name, result)
    return ordered_results


def dedupe_metric_names(metric_names) -> list[str]:
    seen = set()
    ordered_names = []
    for metric_name in metric_names:
        if metric_name in seen:
            continue
        seen.add(metric_name)
        ordered_names.append(metric_name)
    return ordered_names


if __name__ == "__main__":
    main()
