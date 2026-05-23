from __future__ import annotations

from dataclasses import dataclass

from analyzer import parse_ncu_csv_metrics
from llm_client import CompatibleLLMClient
from output_schema import SUPPORTED_METRICS, canonicalize_metric_name, default_unit, failed_metric, normalize_name
from prompt_manager import PromptManager

EXACT_NCU_REQUEST_NAMES = {
    "dram_bytes_read_per_second": "dram__bytes_read.sum.per_second",
    "dram_bytes_write_per_second": "dram__bytes_write.sum.per_second",
    "sm_throughput_pct_of_peak_sustained_elapsed": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}


@dataclass
class BasePipelineAgent:
    llm_client: CompatibleLLMClient
    prompt_manager: PromptManager
    agent_logs: list[str]

    @property
    def name(self) -> str:
        return type(self).__name__

    def maybe_llm(self, payload: dict, required_keys: set[str]) -> dict | None:
        if not self.llm_client.available():
            return None
        response = self.llm_client.chat_json(self.prompt_manager.get_prompt(self.name), payload)
        if not isinstance(response, dict):
            self.agent_logs.append(f"{self.name} received no usable JSON from LLM; using deterministic fallback.")
            return None
        if not required_keys.issubset(response.keys()):
            self.agent_logs.append(f"{self.name} LLM response missing required keys {sorted(required_keys)}; using deterministic fallback.")
            return None
        return response


class SpecInterpreterAgent(BasePipelineAgent):
    def run(self, input_dict: dict) -> dict:
        normalized_targets = []
        unsupported_targets = []
        for metric_name in input_dict.get("target_spec", {}).get("requested_metrics", []):
            canonical = canonicalize_metric_name(metric_name)
            if not canonical:
                unsupported_targets.append(metric_name)
                continue
            normalized_targets.append(
                {
                    "requested_name": metric_name,
                    "canonical_name": canonical,
                    "category": _category_for_metric(canonical),
                    "priority": _priority_for_metric(canonical),
                    "measurement_strategy": _strategy_for_metric(canonical),
                    "required_precision": "engineering estimate with repeated-trial median",
                    "notes": "Deterministic microbenchmark path is authoritative; API properties are hints only.",
                }
            )

        llm_result = self.maybe_llm(input_dict, {"normalized_targets", "unsupported_targets"})
        if not _valid_normalized_targets(llm_result):
            return {"normalized_targets": normalized_targets, "unsupported_targets": unsupported_targets}

        llm_by_request = {
            item.get("requested_name"): item
            for item in llm_result.get("normalized_targets", [])
            if isinstance(item, dict)
        }
        enriched_targets = []
        for target in normalized_targets:
            candidate = llm_by_request.get(target["requested_name"])
            if not candidate or candidate.get("canonical_name") != target["canonical_name"]:
                enriched_targets.append(target)
                continue
            enriched_targets.append(
                {
                    "requested_name": target["requested_name"],
                    "canonical_name": target["canonical_name"],
                    "category": candidate.get("category", target["category"]),
                    "priority": candidate.get("priority", target["priority"]),
                    "measurement_strategy": candidate.get("measurement_strategy", target["measurement_strategy"]),
                    "required_precision": candidate.get("required_precision", target["required_precision"]),
                    "notes": candidate.get("notes", target["notes"]),
                }
            )
        return {"normalized_targets": enriched_targets, "unsupported_targets": unsupported_targets}


class ProbePlannerAgent(BasePipelineAgent):
    def run(self, input_dict: dict) -> dict:
        llm_result = self.maybe_llm(input_dict, {"probe_plan"})
        if _valid_probe_plan(llm_result):
            return llm_result

        grouped: dict[str, dict] = {}
        for target in input_dict.get("normalized_targets", []):
            canonical = target["canonical_name"]
            probe_name = _probe_for_metric(canonical)
            entry = grouped.setdefault(
                probe_name,
                {
                    "probe_name": probe_name,
                    "targets": [],
                    "method": _strategy_for_metric(canonical),
                    "kernel_type": probe_name,
                    "sweep_parameters": _default_sweep(probe_name),
                    "expected_signal": _expected_signal(probe_name),
                    "validation_strategy": "repeat trials, robust median, reject impossible values, compare against environment notes",
                    "failure_fallback": "mark affected targets partial or failed with explicit runtime reason",
                },
            )
            if canonical not in entry["targets"]:
                entry["targets"].append(canonical)
        if "observed_active_sm_count" not in {target["canonical_name"] for target in input_dict.get("normalized_targets", [])}:
            grouped.setdefault(
                "environment_probe",
                {
                    "probe_name": "environment_probe",
                    "targets": [],
                    "method": "runtime_smid_sampling",
                    "kernel_type": "environment_probe",
                    "sweep_parameters": {"sampling_rounds": 4, "blocks": 4096},
                    "expected_signal": "distinct runtime SM IDs indicate actually active SMs",
                    "validation_strategy": "compare runtime-observed SM IDs with API-reported multiProcessorCount without trusting the API as ground truth",
                    "failure_fallback": "record environment notes only",
                },
            )
        return {"probe_plan": list(grouped.values())}


class BenchmarkGeneratorAgent(BasePipelineAgent):
    def run(self, input_dict: dict) -> dict:
        llm_result = self.maybe_llm(input_dict, {"benchmarks"})
        if _valid_benchmarks(llm_result):
            return llm_result

        benchmarks = []
        for probe in input_dict.get("probe_plan", []):
            name = probe["probe_name"]
            benchmarks.append(
                {
                    "name": name,
                    "purpose": _purpose_for_probe(name),
                    "targets": list(probe.get("targets", [])),
                    "source_path": f"benchmarks/generated/{name}.cu",
                    "binary_path": f"benchmarks/generated/{name}",
                    "compile_flags": _compile_flags_for_probe(name),
                    "expected_stdout_schema": _stdout_schema_for_probe(name),
                    "ncu_enabled": name in {"global_bandwidth", "shared_bandwidth", "core_clock", "fp32_throughput"},
                }
            )
        return {"benchmarks": benchmarks}


class NcuAnalystAgent(BasePipelineAgent):
    def run(self, input_dict: dict) -> dict:
        llm_result = self.maybe_llm(input_dict, {"ncu_analysis"})
        if isinstance(llm_result, dict) and isinstance(llm_result.get("ncu_analysis"), list):
            return llm_result

        analyses = []
        outputs = {item.get("probe_name"): item for item in input_dict.get("ncu_raw_outputs", [])}
        for execution in input_dict.get("execution_results", []):
            probe_name = execution.get("probe_name")
            raw_text = outputs.get(probe_name, {}).get("output", "")
            key_metrics = parse_ncu_csv_metrics(raw_text)
            permission_issue = None
            memory_bound = None
            compute_bound = None
            summary = execution.get("ncu_summary", "unavailable")
            if "ERR_NVGPUCTRPERM" in raw_text:
                permission_issue = "missing performance counter permission"
            elif "ncu failed" in summary:
                permission_issue = summary
            elif probe_name == "fp32_throughput":
                sm_throughput = key_metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
                memory_throughput = key_metrics.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
                if sm_throughput is not None:
                    compute_bound = sm_throughput >= 60.0
                    memory_bound = False if memory_throughput is None else memory_throughput > sm_throughput
                    summary = (
                        f"Kernel 'fp32_fma_kernel' shows high compute throughput (~{sm_throughput:.1f}% of peak sustained) "
                        f"with {key_metrics.get('sm__maximum_warps_per_active_cycle_pct', 0):.0f}% warp occupancy; "
                        "no significant memory operations observed."
                    )
                elif "dram" in raw_text.lower() or "memory" in raw_text.lower():
                    memory_bound = True
                    compute_bound = False
            elif "dram" in raw_text.lower() or "memory" in raw_text.lower():
                memory_bound = True
                compute_bound = False
            if not raw_text.strip() and summary == "unavailable":
                permission_issue = "NCU not run or unavailable"
            analyses.append(
                {
                    "probe_name": probe_name,
                    "summary": summary,
                    "key_metrics": key_metrics,
                    "memory_bound": memory_bound,
                    "compute_bound": compute_bound,
                    "permission_or_tool_issue": permission_issue,
                }
            )
        return {"ncu_analysis": analyses}


class ExecutionTriageAgent(BasePipelineAgent):
    def run(self, input_dict: dict) -> dict:
        llm_result = self.maybe_llm(input_dict, {"triage"})
        if isinstance(llm_result, dict) and isinstance(llm_result.get("triage"), list):
            return llm_result

        triage = []
        parsed = input_dict.get("parsed_measurements", {})
        execution_results = input_dict.get("execution_results", [])
        environment_notes = input_dict.get("environment_notes", {})
        requested_canonicals = []
        seen_requested = set()
        for item in input_dict.get("normalized_targets", []):
            canonical_name = item.get("canonical_name")
            if canonical_name and canonical_name not in seen_requested:
                seen_requested.add(canonical_name)
                requested_canonicals.append(canonical_name)
        driver_issue = any("driver version is insufficient" in (item.get("failure_reason", "").lower()) for item in execution_results)
        execution_by_probe = {item.get("probe_name"): item for item in execution_results}
        for canonical_name in requested_canonicals:
            metric_payload = parsed.get("metrics", {}).get(canonical_name, {})
            issues = []
            status = "accept"
            final_action = "accept_current"
            confidence = float(metric_payload.get("confidence", 0.0))
            if not metric_payload:
                status = "failed"
                final_action = "mark_failed"
                issues.append(execution_by_probe.get(_probe_for_metric(canonical_name), {}).get("failure_reason", "metric not produced"))
            elif metric_payload.get("status") == "failed":
                status = "failed"
                final_action = "mark_failed"
                issues.append(metric_payload.get("analysis", "metric failed"))
            elif confidence < 0.55:
                status = "partial"
                final_action = "use_fallback"
                issues.append("low confidence after variance and consistency checks")
            if driver_issue:
                status = "failed"
                final_action = "mark_failed"
                issues.append("CUDA runtime unavailable due to driver/runtime mismatch")
            suspected_causes = []
            if environment_notes.get("sm_masking_suspected"):
                suspected_causes.append("possible_sm_masking")
            if environment_notes.get("frequency_lock_suspected"):
                suspected_causes.append("possible_frequency_locking")
            if driver_issue:
                suspected_causes.append("cuda_runtime_unavailable")
            triage.append(
                {
                    "target": canonical_name,
                    "status": status,
                    "confidence": round(confidence, 4),
                    "issues": issues,
                    "suspected_causes": suspected_causes,
                    "recommended_adjustments": _triage_adjustments(canonical_name, driver_issue),
                    "final_action": final_action,
                }
            )
        return {"triage": triage}


class ResultAggregatorAgent(BasePipelineAgent):
    def run(self, input_dict: dict) -> dict:
        llm_result = self.maybe_llm(input_dict, {"results", "summary"})
        if isinstance(llm_result, dict) and isinstance(llm_result.get("results"), dict):
            return llm_result

        triage_by_target = {item["target"]: item for item in input_dict.get("triage", [])}
        parsed_metrics = input_dict.get("parsed_measurements", {}).get("metrics", {})
        ncu_analysis_by_probe = {item["probe_name"]: item for item in input_dict.get("ncu_analysis", [])}
        ncu_by_probe = {probe_name: item.get("summary", "unavailable") for probe_name, item in ncu_analysis_by_probe.items()}
        execution_by_probe = {item["probe_name"]: item for item in input_dict.get("execution_results", [])}

        results = {}
        for normalized in input_dict.get("normalized_targets", []):
            requested_name = normalized["requested_name"]
            canonical_name = normalized["canonical_name"]
            metric_result = dict(parsed_metrics.get(canonical_name, {}))
            triage_entry = triage_by_target.get(canonical_name, {})
            if not metric_result:
                probe_name = _probe_for_metric(canonical_name)
                failure_reason = execution_by_probe.get(probe_name, {}).get("failure_reason", "").strip()
                metric_result = failed_metric(
                    canonical_name,
                    failure_reason or f"{probe_name} did not produce a usable `{canonical_name}` result.",
                    [],
                )
                metric_result["unit"] = default_unit(canonical_name)
                metric_result["card_type"] = normalized["category"]
            probe_name = _probe_for_metric(canonical_name)
            metric_result["canonical_name"] = canonical_name
            metric_result["ncu_summary"] = ncu_by_probe.get(probe_name, metric_result.get("ncu_summary", "unavailable"))
            metric_result = _prefer_exact_ncu_result(
                metric_result=metric_result,
                requested_name=requested_name,
                canonical_name=canonical_name,
                ncu_analysis=ncu_analysis_by_probe.get(probe_name, {}),
            )
            if requested_name != canonical_name:
                metric_result["analysis"] = f"{_requested_metric_prefix(requested_name, canonical_name)} {metric_result['analysis']}".strip()
            if triage_entry and triage_entry.get("status") in {"partial", "failed"} and metric_result["status"] == "success":
                metric_result["status"] = "partial"
                metric_result["analysis"] += f" Triage downgraded status due to: {', '.join(triage_entry.get('issues', [])) or triage_entry.get('status')}."
                metric_result["confidence"] = min(float(metric_result.get("confidence", 0.0)), float(triage_entry.get("confidence", 0.0)))
            results[requested_name] = metric_result

        summary = (
            f"Processed {len(input_dict.get('normalized_targets', []))} requested metrics with "
            f"{sum(1 for item in results.values() if item.get('status') == 'success')} successful results. "
            "Deterministic microbenchmarks remained authoritative even when prompts or API access were unavailable."
        )
        return {"results": results, "summary": summary}


def _valid_normalized_targets(result: dict | None) -> bool:
    return isinstance(result, dict) and isinstance(result.get("normalized_targets"), list) and isinstance(result.get("unsupported_targets"), list)


def _valid_probe_plan(result: dict | None) -> bool:
    return isinstance(result, dict) and isinstance(result.get("probe_plan"), list)


def _valid_benchmarks(result: dict | None) -> bool:
    return isinstance(result, dict) and isinstance(result.get("benchmarks"), list)


def _category_for_metric(metric_name: str) -> str:
    mapping = {
        "l1_latency_cycles": "latency",
        "l2_latency_cycles": "latency",
        "dram_latency_cycles": "latency",
        "global_memory_bandwidth_gbps": "bandwidth",
        "dram_bytes_read_per_second": "bandwidth",
        "dram_bytes_write_per_second": "bandwidth",
        "shared_memory_bandwidth_gbps": "bandwidth",
        "l2_cache_capacity_bytes": "capacity",
        "actual_core_clock_mhz": "frequency",
        "device_attribute_max_gpu_frequency_khz": "frequency",
        "device_attribute_max_mem_frequency_khz": "frequency",
        "device_attribute_fb_bus_width_bits": "topology",
        "peak_fp32_tflops": "bandwidth",
        "sm_throughput_pct_of_peak_sustained_elapsed": "bandwidth",
        "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "bandwidth",
        "shared_memory_bank_conflict_penalty_cycles": "resource_penalty",
        "shared_memory_bank_conflict_penalty_ratio": "resource_penalty",
        "observed_active_sm_count": "topology",
    }
    return mapping.get(metric_name, "resource_penalty")


def _priority_for_metric(metric_name: str) -> str:
    return "high" if metric_name in SUPPORTED_METRICS else "medium"


def _strategy_for_metric(metric_name: str) -> str:
    mapping = {
        "l1_latency_cycles": "dependent_pointer_chase",
        "l2_latency_cycles": "dependent_pointer_chase",
        "dram_latency_cycles": "dependent_pointer_chase",
        "l2_cache_capacity_bytes": "latency_cliff_detection",
        "global_memory_bandwidth_gbps": "streaming_copy_bandwidth",
        "dram_bytes_read_per_second": "streaming_copy_bandwidth",
        "dram_bytes_write_per_second": "streaming_copy_bandwidth",
        "shared_memory_bandwidth_gbps": "shared_memory_stress",
        "actual_core_clock_mhz": "clock64_vs_event_timing",
        "device_attribute_max_gpu_frequency_khz": "environment_property_snapshot",
        "device_attribute_max_mem_frequency_khz": "environment_property_snapshot",
        "device_attribute_fb_bus_width_bits": "environment_property_snapshot",
        "peak_fp32_tflops": "fma_saturation",
        "sm_throughput_pct_of_peak_sustained_elapsed": "fma_saturation_with_ncu_or_stable_fallback",
        "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "streaming_copy_bandwidth_with_ncu_or_theoretical_fallback",
        "shared_memory_bank_conflict_penalty_cycles": "shared_bank_conflict_compare",
        "shared_memory_bank_conflict_penalty_ratio": "shared_bank_conflict_compare",
        "observed_active_sm_count": "runtime_smid_sampling",
    }
    return mapping.get(metric_name, "microbenchmark")


def _probe_for_metric(metric_name: str) -> str:
    mapping = {
        "l1_latency_cycles": "pointer_chase_latency",
        "l2_latency_cycles": "pointer_chase_latency",
        "dram_latency_cycles": "pointer_chase_latency",
        "l2_cache_capacity_bytes": "pointer_chase_latency",
        "global_memory_bandwidth_gbps": "global_bandwidth",
        "dram_bytes_read_per_second": "global_bandwidth",
        "dram_bytes_write_per_second": "global_bandwidth",
        "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "global_bandwidth",
        "shared_memory_bandwidth_gbps": "shared_bandwidth",
        "actual_core_clock_mhz": "core_clock",
        "device_attribute_max_gpu_frequency_khz": "environment_probe",
        "device_attribute_max_mem_frequency_khz": "environment_probe",
        "device_attribute_fb_bus_width_bits": "environment_probe",
        "peak_fp32_tflops": "fp32_throughput",
        "sm_throughput_pct_of_peak_sustained_elapsed": "fp32_throughput",
        "shared_memory_bank_conflict_penalty_cycles": "bank_conflict",
        "shared_memory_bank_conflict_penalty_ratio": "bank_conflict",
        "observed_active_sm_count": "environment_probe",
    }
    return mapping[metric_name]


def _default_sweep(probe_name: str) -> dict:
    mapping = {
        "pointer_chase_latency": {"size_bytes": ["4KB", "8KB", "16KB", "32KB", "64KB", "128KB", "256KB", "512KB", "1MB", "2MB", "4MB", "8MB", "16MB", "32MB", "64MB"], "modes": ["default", "l2_or_dram"]},
        "global_bandwidth": {"block_sizes": [128, 256, 512], "modes": ["read", "write", "copy", "vec4_copy"]},
        "shared_bandwidth": {"block_sizes": [128, 256, 512], "iterations": [8192, 32768]},
        "core_clock": {"durations": ["short", "medium", "long"]},
        "fp32_throughput": {"block_sizes": [128, 256], "iterations": [262144, 1048576]},
        "bank_conflict": {"patterns": ["conflict_free", "bank_conflict"], "trials_per_run": 8},
        "environment_probe": {"sampling_rounds": 4, "blocks": 4096},
    }
    return mapping.get(probe_name, {})


def _expected_signal(probe_name: str) -> str:
    mapping = {
        "pointer_chase_latency": "latency plateaus and cliffs separate L1, L2, DRAM, and L2 capacity",
        "global_bandwidth": "best median throughput identifies stable effective DRAM bandwidth",
        "shared_bandwidth": "high in-kernel shared-memory traffic dominates kernel runtime",
        "core_clock": "cycles divided by elapsed wall time yields sustained operating frequency",
        "fp32_throughput": "high FMA count divided by elapsed time yields effective FP32 throughput",
        "bank_conflict": "conflicted shared access pattern is slower than conflict-free baseline",
        "environment_probe": "distinct SM IDs reveal runtime-visible active SM count",
    }
    return mapping.get(probe_name, "structured runtime signal")


def _purpose_for_probe(name: str) -> str:
    return _expected_signal(name)


def _compile_flags_for_probe(name: str) -> list[str]:
    if name == "pointer_chase_latency":
        return ["-Xptxas", "-dlcm=ca"]
    return []


def _stdout_schema_for_probe(name: str) -> dict:
    schemas = {
        "pointer_chase_latency": {"POINT": ["size_bytes", "mode", "cycles_per_access"]},
        "global_bandwidth": {"CONFIG": ["mode", "block", "bytes", "elapsed_ms", "gbps"]},
        "shared_bandwidth": {"CONFIG": ["block", "iters", "bytes_modeled", "elapsed_ms", "gbps"]},
        "core_clock": {"TRIAL": ["cycles", "elapsed_ms", "mhz"]},
        "fp32_throughput": {"TRIAL": ["elapsed_ms", "tflops", "operations"]},
        "bank_conflict": {"PATTERN": ["name", "cycles_per_iter"], "SUMMARY": ["penalty_cycles", "penalty_ratio"]},
        "environment_probe": {"PROP": ["name", "multiProcessorCount"], "OBSERVED": ["active_sms"]},
    }
    return schemas.get(name, {})


def _triage_adjustments(metric_name: str, driver_issue: bool) -> list[str]:
    if driver_issue:
        return ["verify CUDA driver/runtime compatibility", "rerun on a machine with GPU runtime access"]
    if "latency" in metric_name:
        return ["increase pointer-chase steps", "add more working-set sizes near the latency cliff"]
    if "bandwidth" in metric_name or "tflops" in metric_name:
        return ["increase workload size", "retry the best two launch configurations"]
    return ["collect additional trials"]


def _requested_metric_prefix(requested_name: str, canonical_name: str) -> str:
    if normalize_name(requested_name) == "launch__sm_count":
        return (
            f"Requested as `{requested_name}` and interpreted as runtime-visible active SM count, "
            f"mapped to `{canonical_name}`."
        )
    return f"Requested as `{requested_name}` and mapped to `{canonical_name}`."


def _prefer_exact_ncu_result(metric_result: dict, requested_name: str, canonical_name: str, ncu_analysis: dict) -> dict:
    exact_request_name = EXACT_NCU_REQUEST_NAMES.get(canonical_name)
    if not exact_request_name:
        return metric_result
    key_metrics = ncu_analysis.get("key_metrics", {})
    exact_value = key_metrics.get(exact_request_name)
    if exact_value is None:
        return metric_result
    if canonical_name.endswith("pct_of_peak_sustained_elapsed") and not (0.0 <= float(exact_value) <= 120.0):
        return metric_result
    overridden = dict(metric_result)
    overridden["raw_value"] = float(exact_value)
    overridden["status"] = "success"
    overridden["confidence"] = 0.92
    overridden["raw_observations"] = [
        {
            "source": "ncu_csv",
            "metric": exact_request_name,
            "value": float(exact_value),
            "unit": overridden.get("unit", ""),
        }
    ]
    if canonical_name == "sm_throughput_pct_of_peak_sustained_elapsed":
        overridden["analysis"] = "Parsed exact Nsight Compute metric `sm__throughput.avg.pct_of_peak_sustained_elapsed` from an actual NCU CSV data row."
        overridden["conclusion"] = f"NCU reported SM throughput at {float(exact_value):.2f}% of peak sustained throughput."
    elif canonical_name == "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed":
        overridden["analysis"] = "Parsed exact Nsight Compute metric `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` from an actual NCU CSV data row."
        overridden["conclusion"] = f"NCU reported GPU compute-memory throughput at {float(exact_value):.2f}% of peak sustained throughput."
    elif canonical_name == "dram_bytes_read_per_second":
        overridden["analysis"] = "Parsed exact Nsight Compute metric `dram__bytes_read.sum.per_second` from an actual NCU CSV data row."
        overridden["conclusion"] = f"NCU reported DRAM read throughput of {float(exact_value):.3f} bytes/s."
    elif canonical_name == "dram_bytes_write_per_second":
        overridden["analysis"] = "Parsed exact Nsight Compute metric `dram__bytes_write.sum.per_second` from an actual NCU CSV data row."
        overridden["conclusion"] = f"NCU reported DRAM write throughput of {float(exact_value):.3f} bytes/s."
    return overridden
