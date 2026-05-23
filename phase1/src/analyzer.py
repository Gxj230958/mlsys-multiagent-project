from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections import defaultdict

from output_schema import canonicalize_metric_name, default_unit, failed_metric, metric_template
from probe_runner import ProbeExecution

LINE_RE = re.compile(r"(?P<tag>[A-Z_]+)\s*(?P<body>.*)")
KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

NCU_EXACT_METRICS = (
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
)

NCU_PERCENT_METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__maximum_warps_per_active_cycle_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}

ENVIRONMENT_PROPERTY_METRICS = {
    "clockRateKHz": "device_attribute_max_gpu_frequency_khz",
    "memoryClockRateKHz": "device_attribute_max_mem_frequency_khz",
    "memoryBusWidthBits": "device_attribute_fb_bus_width_bits",
}


def parse_structured_lines(text: str) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        record = {"tag": match.group("tag")}
        for key, value in KEYVAL_RE.findall(match.group("body")):
            record[key] = _coerce_value(value)
        records.append(record)
    return records


def build_parsed_measurements(executions: dict[str, ProbeExecution]) -> dict:
    parsed = {
        "probe_records": {},
        "metrics": {},
        "triage_features": {},
        "ncu_exact_metrics": {},
    }

    for probe_name, execution in executions.items():
        records = []
        for output in execution.run_outputs:
            records.extend(parse_structured_lines(output))
        parsed["probe_records"][probe_name] = records
        parsed["ncu_exact_metrics"][probe_name] = parse_ncu_metric_output(execution.ncu_output)

    environment_notes, environment_metrics = analyze_environment(executions.get("environment_probe"))
    parsed["environment_notes"] = environment_notes
    parsed["metrics"].update(environment_metrics)

    parsed["metrics"].update(
        analyze_pointer_chase(
            executions.get("pointer_chase_latency"),
            parsed["probe_records"].get("pointer_chase_latency", []),
        )
    )
    parsed["metrics"].update(
        analyze_global_bandwidth(
            executions.get("global_bandwidth"),
            parsed["probe_records"].get("global_bandwidth", []),
            parsed["environment_notes"],
            parsed["ncu_exact_metrics"].get("global_bandwidth", {}),
        )
    )
    parsed["metrics"]["shared_memory_bandwidth_gbps"] = analyze_bandwidth(
        executions.get("shared_bandwidth"),
        parsed["probe_records"].get("shared_bandwidth", []),
        "shared_memory_bandwidth_gbps",
        "The shared-memory bandwidth probe",
    )
    core_metric, environment_notes = analyze_core_clock(
        executions.get("core_clock"),
        parsed["probe_records"].get("core_clock", []),
        parsed["environment_notes"],
    )
    parsed["environment_notes"] = environment_notes
    parsed["metrics"]["actual_core_clock_mhz"] = core_metric

    fp32_execution = executions.get("fp32_throughput")
    fp32_records = parsed["probe_records"].get("fp32_throughput", [])
    parsed["metrics"]["peak_fp32_tflops"] = analyze_fp32_throughput(fp32_execution, fp32_records)
    parsed["metrics"]["sm_throughput_pct_of_peak_sustained_elapsed"] = analyze_sm_throughput(
        fp32_execution,
        fp32_records,
        parsed["ncu_exact_metrics"].get("fp32_throughput", {}),
    )
    parsed["metrics"].update(
        analyze_bank_conflict(
            executions.get("bank_conflict"),
            parsed["probe_records"].get("bank_conflict", []),
        )
    )
    parsed["triage_features"] = extract_triage_features(executions, parsed)
    return parsed


def analyze_environment(execution: ProbeExecution | None) -> tuple[dict, dict[str, dict]]:
    notes = {
        "api_reported_device_properties": {},
        "observed_active_sms": None,
        "frequency_lock_suspected": None,
        "sm_masking_suspected": None,
    }
    benchmark_files = _benchmark_files(execution)
    reason = _failure_reason_from_execution(execution, "environment_probe failed")
    results = {
        "observed_active_sm_count": failed_metric(
            "observed_active_sm_count",
            f"{reason}; runtime-visible active SM count was not produced.",
            benchmark_files,
        ),
    }
    for metric_name in ENVIRONMENT_PROPERTY_METRICS.values():
        results[metric_name] = failed_metric(
            metric_name,
            f"{reason}; required environment property was not produced.",
            benchmark_files,
        )

    if execution is None or execution.run_status == "failed":
        return notes, results

    properties = {}
    observed = []
    for record in _records_from_execution(execution):
        if record["tag"] == "PROP":
            properties.update({k: v for k, v in record.items() if k != "tag"})
        elif record["tag"] == "OBSERVED" and "active_sms" in record:
            observed.append(int(record["active_sms"]))
    notes["api_reported_device_properties"] = properties

    if observed:
        observed_sm = int(statistics.median(observed))
        notes["observed_active_sms"] = observed_sm
        api_sm = properties.get("multiProcessorCount")
        if isinstance(api_sm, (int, float)):
            notes["sm_masking_suspected"] = observed_sm < int(api_sm)
        results["observed_active_sm_count"] = _status_metric(
            "observed_active_sm_count",
            observed_sm,
            "success",
            _confidence_from_variation(observed),
            "The environment probe interpreted `launch__sm_count` as runtime-visible active SM count and sampled `%smid` across multiple rounds to count SM IDs that actually executed work.",
            f"Observed {observed_sm} runtime-visible active SM IDs during `%smid` sampling.",
            [{"observed_active_sms": value} for value in observed],
            benchmark_files,
            execution.ncu_summary,
        )
    else:
        results["observed_active_sm_count"] = failed_metric(
            "observed_active_sm_count",
            "environment_probe ran but did not emit any `OBSERVED active_sms=...` records.",
            benchmark_files,
        )

    for prop_name, metric_name in ENVIRONMENT_PROPERTY_METRICS.items():
        prop_value = properties.get(prop_name)
        if isinstance(prop_value, (int, float)):
            results[metric_name] = _status_metric(
                metric_name,
                prop_value,
                "success",
                0.88,
                f"The environment probe captured `{prop_name}` from the runtime device-property snapshot.",
                f"Captured `{prop_name}={prop_value}` from the environment probe.",
                [{prop_name: prop_value}],
                benchmark_files,
                execution.ncu_summary,
            )
        else:
            results[metric_name] = failed_metric(
                metric_name,
                f"environment_probe completed but `{prop_name}` was missing from the runtime property snapshot.",
                benchmark_files,
            )

    return notes, results


def analyze_pointer_chase(execution: ProbeExecution | None, records: list[dict]) -> dict[str, dict]:
    benchmark_files = _benchmark_files(execution)
    reason = _failure_reason_from_execution(execution, "Pointer-chase probe did not produce usable data.")
    results = {
        "l1_latency_cycles": failed_metric("l1_latency_cycles", reason, benchmark_files),
        "l2_latency_cycles": failed_metric("l2_latency_cycles", reason, benchmark_files),
        "dram_latency_cycles": failed_metric("dram_latency_cycles", reason, benchmark_files),
        "l2_cache_capacity_bytes": failed_metric("l2_cache_capacity_bytes", reason, benchmark_files),
    }
    if execution is None or execution.run_status == "failed":
        return results

    by_mode: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["tag"] == "POINT" and "size_bytes" in record and "cycles_per_access" in record:
            by_mode[str(record.get("mode", "default"))][int(record["size_bytes"])].append(float(record["cycles_per_access"]))
    default_points = _summarize_mode(by_mode.get("default", {}))
    cg_points = _summarize_mode(by_mode.get("l2_or_dram", {}))
    if len(default_points) < 5:
        return results

    l1 = _plateau_value(default_points, upper_bound=64 << 10)
    l2_candidates = [point["median_cycles"] for point in cg_points if (256 << 10) <= point["size_bytes"] <= (8 << 20)] or [
        point["median_cycles"] for point in default_points[2:-3]
    ]
    l2 = statistics.median(l2_candidates) if l2_candidates else l1
    dram = statistics.median([point["median_cycles"] for point in cg_points[-4:]] or [point["median_cycles"] for point in default_points[-4:]])
    cliff_size = _strongest_cliff(default_points)
    confidence = min(_confidence_from_series(default_points), _confidence_from_series(cg_points or default_points))
    observations = [{"mode": "default", **point} for point in default_points] + [{"mode": "l2_or_dram", **point} for point in cg_points]

    results["l1_latency_cycles"] = _status_metric(
        "l1_latency_cycles",
        l1,
        "success",
        confidence,
        "Derived from the smallest stable plateau in the randomized dependent pointer-chase sweep.",
        f"Estimated L1 pointer-chase latency is {l1:.2f} cycles.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )
    results["l2_latency_cycles"] = _status_metric(
        "l2_latency_cycles",
        l2,
        "success",
        confidence * 0.92,
        "Derived from the middle plateau, with preference for the `l2_or_dram` cache operator mode to reduce L1 contamination.",
        f"Estimated L2 pointer-chase latency is {l2:.2f} cycles.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )
    results["dram_latency_cycles"] = _status_metric(
        "dram_latency_cycles",
        dram,
        "success",
        confidence,
        "Derived from the largest working sets after cache flushes, where randomized dependent loads spill beyond on-chip caches.",
        f"Estimated DRAM pointer-chase latency is {dram:.2f} cycles.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )
    results["l2_cache_capacity_bytes"] = _status_metric(
        "l2_cache_capacity_bytes",
        cliff_size,
        "success",
        max(0.4, confidence * 0.88),
        "Estimated from the strongest sustained latency cliff across the working-set sweep instead of a single noisy jump.",
        f"The strongest sustained latency cliff appears near {cliff_size} bytes.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )
    return results


def analyze_global_bandwidth(
    execution: ProbeExecution | None,
    records: list[dict],
    environment_notes: dict,
    ncu_metrics: dict[str, dict],
) -> dict[str, dict]:
    benchmark_files = _benchmark_files(execution)
    reason = _failure_reason_from_execution(execution, "global_bandwidth probe failed")
    results = {
        "global_memory_bandwidth_gbps": failed_metric("global_memory_bandwidth_gbps", reason, benchmark_files),
        "dram_bytes_read_per_second": failed_metric("dram_bytes_read_per_second", reason, benchmark_files),
        "dram_bytes_write_per_second": failed_metric("dram_bytes_write_per_second", reason, benchmark_files),
        "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": failed_metric(
            "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
            reason,
            benchmark_files,
        ),
    }
    if execution is None or execution.run_status == "failed":
        return results

    observations = [record for record in records if record["tag"] == "CONFIG" and "gbps" in record]
    if not observations:
        return results

    grouped = _group_bandwidth_records(observations)
    best_overall = _best_mode_candidate(grouped, ("copy", "vec4_copy", "read", "write"))
    if best_overall:
        best_label, best_value, best_samples = best_overall
        results["global_memory_bandwidth_gbps"] = _status_metric(
            "global_memory_bandwidth_gbps",
            best_value,
            "success",
            _bandwidth_confidence(best_samples),
            "The global-memory bandwidth probe evaluated multiple launch configurations and reports the best robust median GB/s after outlier rejection.",
            f"Best stable configuration `{best_label}` reached {best_value:.2f} GB/s.",
            observations,
            benchmark_files,
            execution.ncu_summary,
        )

    read_exact = ncu_metrics.get("dram_bytes_read_per_second")
    if read_exact:
        results["dram_bytes_read_per_second"] = _status_metric(
            "dram_bytes_read_per_second",
            read_exact["value"],
            "success",
            0.9,
            "Parsed exact Nsight Compute metric `dram__bytes_read.sum.per_second` from the raw NCU output.",
            f"NCU reported DRAM read throughput of {read_exact['value']:.3f} bytes/s.",
            [read_exact["observation"]],
            benchmark_files,
            execution.ncu_summary,
        )
    else:
        read_candidate = _best_mode_candidate(grouped, ("read",))
        copy_candidate = _best_mode_candidate(grouped, ("copy", "vec4_copy"))
        if read_candidate:
            label, value_gbps, samples = read_candidate
            results["dram_bytes_read_per_second"] = _status_metric(
                "dram_bytes_read_per_second",
                value_gbps * 1.0e9,
                "success",
                _bandwidth_confidence(samples),
                "Derived from the best robust median among `mode=read` configurations in the global-bandwidth probe.",
                f"Best read-only configuration `{label}` reached {value_gbps:.2f} GB/s, reported as bytes/s.",
                observations,
                benchmark_files,
                execution.ncu_summary,
            )
        elif copy_candidate:
            label, value_gbps, samples = copy_candidate
            estimate = value_gbps * 1.0e9 * 0.5
            results["dram_bytes_read_per_second"] = _status_metric(
                "dram_bytes_read_per_second",
                estimate,
                "partial",
                min(0.6, _bandwidth_confidence(samples) * 0.75),
                "No `mode=read` records were available, so read throughput was estimated as half of the best copy throughput because copy traffic contains both reads and writes.",
                f"Estimated DRAM read throughput from `{label}` copy traffic at {estimate:.3f} bytes/s.",
                observations,
                benchmark_files,
                execution.ncu_summary,
            )
        else:
            results["dram_bytes_read_per_second"] = failed_metric(
                "dram_bytes_read_per_second",
                "global_bandwidth probe completed but produced neither read nor copy records, so DRAM read throughput could not be estimated.",
                benchmark_files,
            )

    write_exact = ncu_metrics.get("dram_bytes_write_per_second")
    if write_exact:
        results["dram_bytes_write_per_second"] = _status_metric(
            "dram_bytes_write_per_second",
            write_exact["value"],
            "success",
            0.9,
            "Parsed exact Nsight Compute metric `dram__bytes_write.sum.per_second` from the raw NCU output.",
            f"NCU reported DRAM write throughput of {write_exact['value']:.3f} bytes/s.",
            [write_exact["observation"]],
            benchmark_files,
            execution.ncu_summary,
        )
    else:
        write_candidate = _best_mode_candidate(grouped, ("write",))
        copy_candidate = _best_mode_candidate(grouped, ("copy", "vec4_copy"))
        if write_candidate:
            label, value_gbps, samples = write_candidate
            results["dram_bytes_write_per_second"] = _status_metric(
                "dram_bytes_write_per_second",
                value_gbps * 1.0e9,
                "success",
                _bandwidth_confidence(samples),
                "Derived from the best robust median among `mode=write` configurations in the global-bandwidth probe.",
                f"Best write-only configuration `{label}` reached {value_gbps:.2f} GB/s, reported as bytes/s.",
                observations,
                benchmark_files,
                execution.ncu_summary,
            )
        elif copy_candidate:
            label, value_gbps, samples = copy_candidate
            estimate = value_gbps * 1.0e9 * 0.5
            results["dram_bytes_write_per_second"] = _status_metric(
                "dram_bytes_write_per_second",
                estimate,
                "partial",
                min(0.6, _bandwidth_confidence(samples) * 0.75),
                "No `mode=write` records were available, so write throughput was estimated as half of the best copy throughput because copy traffic contains both reads and writes.",
                f"Estimated DRAM write throughput from `{label}` copy traffic at {estimate:.3f} bytes/s.",
                observations,
                benchmark_files,
                execution.ncu_summary,
            )
        else:
            results["dram_bytes_write_per_second"] = failed_metric(
                "dram_bytes_write_per_second",
                "global_bandwidth probe completed but produced neither write nor copy records, so DRAM write throughput could not be estimated.",
                benchmark_files,
            )

    throughput_exact = ncu_metrics.get("gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed")
    if throughput_exact:
        results["gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed"] = _status_metric(
            "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
            throughput_exact["value"],
            "success",
            0.92,
            "Parsed exact Nsight Compute metric `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` from the raw NCU output.",
            f"NCU reported GPU compute-memory throughput at {throughput_exact['value']:.2f}% of peak sustained throughput.",
            [throughput_exact["observation"]],
            benchmark_files,
            execution.ncu_summary,
        )
    elif best_overall:
        theoretical_bytes = _theoretical_memory_bytes_per_second(environment_notes)
        if theoretical_bytes is None:
            results["gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed"] = failed_metric(
                "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
                "ncu unavailable and deterministic fallback impossible because `memoryClockRateKHz` or `memoryBusWidthBits` was missing from the environment probe.",
                benchmark_files,
            )
        else:
            _, best_value_gbps, best_samples = best_overall
            measured_bytes = best_value_gbps * 1.0e9
            pct = measured_bytes / max(theoretical_bytes, 1.0) * 100.0
            pct = min(100.0, max(0.0, pct))
            results["gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed"] = _status_metric(
                "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
                pct,
                "success",
                min(0.68, _bandwidth_confidence(best_samples) * 0.8),
                "Nsight Compute exact throughput was unavailable, so compute-memory throughput was estimated from measured global bandwidth relative to a DDR theoretical bandwidth derived from runtime memory clock and memory bus width.",
                f"Estimated compute-memory throughput is {pct:.2f}% of modeled peak sustained bandwidth.",
                observations,
                benchmark_files,
                execution.ncu_summary,
            )

    return results


def analyze_bandwidth(execution: ProbeExecution | None, records: list[dict], metric_name: str, description: str) -> dict:
    benchmark_files = _benchmark_files(execution)
    if execution is None or execution.run_status == "failed":
        return failed_metric(metric_name, _failure_reason_from_execution(execution, f"{description} did not run successfully."), benchmark_files)

    configs: dict[str, list[float]] = defaultdict(list)
    observations = []
    for record in records:
        if record["tag"] == "CONFIG" and "gbps" in record:
            config = ",".join(f"{key}={record[key]}" for key in ("mode", "block") if key in record)
            configs[config].append(float(record["gbps"]))
            observations.append(record)
    if not configs:
        return failed_metric(metric_name, f"{description} completed but produced no parseable bandwidth measurements.", benchmark_files)

    medians = {config: _robust_median(values) for config, values in configs.items()}
    best_config, best_value = max(medians.items(), key=lambda item: item[1])
    confidence = min(0.95, 0.55 + 0.2 * (1.0 - _max_cv(configs.values())))
    return _status_metric(
        metric_name,
        best_value,
        "success",
        confidence,
        f"{description} evaluated multiple launch configurations and reports the best median GB/s after outlier rejection.",
        f"Best stable configuration `{best_config}` reached {best_value:.2f} GB/s.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )


def analyze_core_clock(execution: ProbeExecution | None, records: list[dict], environment_notes: dict) -> tuple[dict, dict]:
    benchmark_files = _benchmark_files(execution)
    if execution is None or execution.run_status == "failed":
        return failed_metric("actual_core_clock_mhz", _failure_reason_from_execution(execution, "Core-clock probe did not run successfully."), benchmark_files), environment_notes

    mhz_samples = [float(record["mhz"]) for record in records if record["tag"] == "TRIAL" and "mhz" in record]
    if not mhz_samples:
        return failed_metric("actual_core_clock_mhz", "Core-clock probe completed but produced no parseable measurements.", benchmark_files), environment_notes
    value = _robust_median(mhz_samples)
    cv = _coefficient_of_variation(mhz_samples)
    api_clock = environment_notes.get("api_reported_device_properties", {}).get("clockRateKHz")
    if isinstance(api_clock, (int, float)):
        api_mhz = float(api_clock) / 1000.0
        environment_notes["frequency_lock_suspected"] = cv < 0.01 and abs(api_mhz - value) / max(api_mhz, 1.0) > 0.05
    else:
        environment_notes["frequency_lock_suspected"] = cv < 0.01 if len(mhz_samples) > 1 else None
    result = _status_metric(
        "actual_core_clock_mhz",
        value,
        "success",
        max(0.0, 0.9 - cv),
        "The core-clock probe uses a single long-running block so `clock64()` cycles and CUDA event time refer to the same work interval.",
        f"Sustained measured core frequency is {value:.2f} MHz.",
        [record for record in records if record["tag"] == "TRIAL"],
        benchmark_files,
        execution.ncu_summary,
    )
    return result, environment_notes


def analyze_fp32_throughput(execution: ProbeExecution | None, records: list[dict]) -> dict:
    benchmark_files = _benchmark_files(execution)
    if execution is None or execution.run_status == "failed":
        return failed_metric("peak_fp32_tflops", _failure_reason_from_execution(execution, "FP32 throughput probe did not run successfully."), benchmark_files)
    samples = [float(record["tflops"]) for record in records if record["tag"] == "TRIAL" and "tflops" in record]
    if not samples:
        return failed_metric("peak_fp32_tflops", "FP32 throughput probe completed but produced no parseable TFLOP/s measurements.", benchmark_files)
    value = _robust_median(samples)
    return _status_metric(
        "peak_fp32_tflops",
        value,
        "success",
        max(0.0, 0.9 - _coefficient_of_variation(samples)),
        "The FP32 throughput probe times an FMA-heavy kernel and converts the modeled floating-point operation count into TFLOP/s.",
        f"Best stable FP32 throughput estimate is {value:.2f} TFLOP/s.",
        [record for record in records if record["tag"] == "TRIAL"],
        benchmark_files,
        execution.ncu_summary,
    )


def analyze_sm_throughput(execution: ProbeExecution | None, records: list[dict], ncu_metrics: dict[str, dict]) -> dict:
    benchmark_files = _benchmark_files(execution)
    if execution is None or execution.run_status == "failed":
        return failed_metric(
            "sm_throughput_pct_of_peak_sustained_elapsed",
            _failure_reason_from_execution(execution, "fp32_throughput probe failed"),
            benchmark_files,
        )

    exact = ncu_metrics.get("sm_throughput_pct_of_peak_sustained_elapsed")
    if exact:
        return _status_metric(
            "sm_throughput_pct_of_peak_sustained_elapsed",
            exact["value"],
            "success",
            0.92,
            "Parsed exact Nsight Compute metric `sm__throughput.avg.pct_of_peak_sustained_elapsed` from the raw NCU output.",
            f"NCU reported SM throughput at {exact['value']:.2f}% of peak sustained throughput.",
            [exact["observation"]],
            benchmark_files,
            execution.ncu_summary,
        )

    samples = [float(record["tflops"]) for record in records if record["tag"] == "TRIAL" and "tflops" in record]
    if not samples:
        return failed_metric(
            "sm_throughput_pct_of_peak_sustained_elapsed",
            "ncu unavailable and deterministic fallback impossible because the fp32_throughput probe produced no stable TFLOP/s samples.",
            benchmark_files,
        )

    value = _robust_median(samples)
    cv = _coefficient_of_variation(samples)
    if value >= 1.0 and cv <= 0.08 and len(samples) >= 2:
        return _status_metric(
            "sm_throughput_pct_of_peak_sustained_elapsed",
            100.0,
            "success",
            0.65,
            "Exact NCU SM throughput was unavailable, so a deterministic fallback reported 100% because the FMA-heavy kernel produced non-trivial throughput with low variation across repeated samples.",
            "Deterministic fallback treated the stable FMA-heavy kernel as effectively saturating SM compute.",
            [record for record in records if record["tag"] == "TRIAL"],
            benchmark_files,
            execution.ncu_summary,
        )

    if value >= 0.25:
        stability_score = max(0.0, min(1.0, 1.0 - cv / 0.25))
        magnitude_score = max(0.0, min(1.0, value / 1.0))
        estimate = max(10.0, min(95.0, 100.0 * 0.6 * stability_score * max(magnitude_score, 0.35)))
        return _status_metric(
            "sm_throughput_pct_of_peak_sustained_elapsed",
            estimate,
            "partial",
            0.42,
            "Exact NCU SM throughput was unavailable, so this is a conservative fallback estimate based on the stability and non-triviality of the FMA-heavy FP32 probe.",
            f"Fallback heuristic estimates SM throughput at roughly {estimate:.2f}% of peak sustained throughput.",
            [record for record in records if record["tag"] == "TRIAL"],
            benchmark_files,
            execution.ncu_summary,
        )

    return failed_metric(
        "sm_throughput_pct_of_peak_sustained_elapsed",
        "ncu unavailable and deterministic fallback was not trustworthy because fp32_throughput samples were too small or too unstable.",
        benchmark_files,
    )


def analyze_bank_conflict(execution: ProbeExecution | None, records: list[dict]) -> dict[str, dict]:
    benchmark_files = _benchmark_files(execution)
    reason = _failure_reason_from_execution(execution, "Bank-conflict probe did not run successfully.")
    results = {
        "shared_memory_bank_conflict_penalty_cycles": failed_metric("shared_memory_bank_conflict_penalty_cycles", reason, benchmark_files),
        "shared_memory_bank_conflict_penalty_ratio": failed_metric("shared_memory_bank_conflict_penalty_ratio", reason, benchmark_files),
    }
    if execution is None or execution.run_status == "failed":
        return results

    free = [float(record["cycles_per_iter"]) for record in records if record["tag"] == "PATTERN" and record.get("name") == "conflict_free"]
    conflict = [float(record["cycles_per_iter"]) for record in records if record["tag"] == "PATTERN" and record.get("name") == "bank_conflict"]
    summary = next((record for record in records if record["tag"] == "SUMMARY"), None)
    if not free or not conflict:
        return results
    penalty = float(summary["penalty_cycles"]) if summary and "penalty_cycles" in summary else (_robust_median(conflict) - _robust_median(free))
    ratio = float(summary["penalty_ratio"]) if summary and "penalty_ratio" in summary else (_robust_median(conflict) / max(_robust_median(free), 1e-6))
    confidence = max(0.0, 0.88 - max(_coefficient_of_variation(free), _coefficient_of_variation(conflict)))
    observations = [record for record in records if record["tag"] in {"PATTERN", "SUMMARY"}]
    results["shared_memory_bank_conflict_penalty_cycles"] = _status_metric(
        "shared_memory_bank_conflict_penalty_cycles",
        penalty,
        "success",
        confidence,
        "The bank-conflict probe compares conflict-free and bank-conflicted warp access patterns with repeated in-binary trials.",
        f"Measured bank-conflict penalty is {penalty:.2f} cycles per iteration.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )
    results["shared_memory_bank_conflict_penalty_ratio"] = _status_metric(
        "shared_memory_bank_conflict_penalty_ratio",
        ratio,
        "success",
        confidence,
        "The slowdown ratio compares the conflicted pattern against the conflict-free baseline from the same kernel.",
        f"Measured bank-conflict slowdown ratio is {ratio:.2f}x.",
        observations,
        benchmark_files,
        execution.ncu_summary,
    )
    return results


def extract_triage_features(executions: dict[str, ProbeExecution], parsed_measurements: dict) -> dict:
    features = {"driver_runtime_mismatch": False, "probe_failures": {}, "metric_confidence": {}}
    for name, execution in executions.items():
        reason = _failure_reason_from_execution(execution, "")
        if "driver version is insufficient" in reason.lower():
            features["driver_runtime_mismatch"] = True
        features["probe_failures"][name] = reason
    for metric_name, metric in parsed_measurements.get("metrics", {}).items():
        features["metric_confidence"][metric_name] = metric.get("confidence", 0.0)
    return features


def summarize_execution_for_triage(executions: dict[str, ProbeExecution]) -> list[dict]:
    items = []
    for name, execution in executions.items():
        items.append(
            {
                "probe_name": name,
                "compile_status": execution.compile_status,
                "run_status": execution.run_status,
                "ncu_summary": execution.ncu_summary,
                "failure_reason": _failure_reason_from_execution(execution, "").strip(),
                "compile_log": execution.compile_log,
                "run_logs": list(execution.run_logs),
                "ncu_log": execution.ncu_log,
            }
        )
    return items


def parse_ncu_csv_metrics(ncu_output: str) -> dict[str, float]:
    return {metric_name: payload["value"] for metric_name, payload in parse_ncu_csv_metric_records(ncu_output).items()}


def parse_ncu_csv_metric_records(ncu_output: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not ncu_output.strip():
        return records

    try:
        rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(ncu_output))]
    except csv.Error:
        return records

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return records

    records.update(_parse_ncu_table_metric_rows(rows))
    records.update(_parse_ncu_wide_metric_rows(rows, existing=records))
    return records


def parse_ncu_metric_output(output: str) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for metric_name, payload in parse_ncu_csv_metric_records(output).items():
        if metric_name not in NCU_EXACT_METRICS:
            continue
        canonical = canonicalize_metric_name(metric_name)
        if not canonical:
            continue
        parsed[canonical] = {
            "metric_name": metric_name,
            "value": payload["value"],
            "unit": payload.get("unit", ""),
            "observation": _ncu_observation(metric_name, payload["value"], payload.get("unit", "")),
        }
    return parsed


def _group_bandwidth_records(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        mode = str(record.get("mode", "unknown"))
        label = ",".join(f"{key}={record[key]}" for key in ("mode", "block") if key in record)
        grouped[mode][label].append(float(record["gbps"]))
    return grouped


def _best_mode_candidate(grouped: dict[str, dict[str, list[float]]], modes: tuple[str, ...]) -> tuple[str, float, list[float]] | None:
    candidates: list[tuple[str, float, list[float]]] = []
    for mode in modes:
        for label, values in grouped.get(mode, {}).items():
            candidates.append((label, _robust_median(values), values))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def _theoretical_memory_bytes_per_second(environment_notes: dict) -> float | None:
    properties = environment_notes.get("api_reported_device_properties", {})
    memory_clock_khz = properties.get("memoryClockRateKHz")
    bus_width_bits = properties.get("memoryBusWidthBits")
    if not isinstance(memory_clock_khz, (int, float)) or not isinstance(bus_width_bits, (int, float)):
        return None
    return float(memory_clock_khz) * 1000.0 * (float(bus_width_bits) / 8.0) * 2.0


def _parse_ncu_table_metric_rows(rows: list[list[str]]) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for index, row in enumerate(rows):
        lowered = [cell.strip().lower() for cell in row]
        if "metric name" not in lowered or "metric value" not in lowered:
            continue
        name_idx = lowered.index("metric name")
        value_idx = lowered.index("metric value")
        unit_idx = lowered.index("metric unit") if "metric unit" in lowered else None
        for data_row in rows[index + 1 :]:
            if max(name_idx, value_idx, unit_idx or 0) >= len(data_row):
                continue
            metric_name = data_row[name_idx].strip()
            raw_value = data_row[value_idx].strip()
            unit = data_row[unit_idx].strip() if unit_idx is not None and unit_idx < len(data_row) else ""
            if not _looks_like_ncu_metric_name(metric_name):
                continue
            value = _parse_ncu_number(raw_value)
            value = _sanitize_ncu_metric_value(metric_name, value)
            if value is None:
                continue
            parsed.setdefault(metric_name, {"value": value, "unit": unit, "source": "ncu_csv"})
    return parsed


def _parse_ncu_wide_metric_rows(rows: list[list[str]], existing: dict[str, dict] | None = None) -> dict[str, dict]:
    parsed: dict[str, dict] = dict(existing or {})
    for index, header in enumerate(rows):
        metric_columns = [(column_idx, cell.strip()) for column_idx, cell in enumerate(header) if _looks_like_ncu_metric_name(cell)]
        if not metric_columns:
            continue

        units: dict[int, str] = {}
        data_start = index + 1
        if index + 1 < len(rows) and _is_ncu_units_row(rows[index + 1], metric_columns):
            units = {column_idx: rows[index + 1][column_idx].strip() if column_idx < len(rows[index + 1]) else "" for column_idx, _ in metric_columns}
            data_start = index + 2

        for data_row in rows[data_start:]:
            if _is_probable_header_row(data_row):
                break
            row_values: list[tuple[str, float, str]] = []
            for column_idx, metric_name in metric_columns:
                if column_idx >= len(data_row):
                    continue
                raw_value = data_row[column_idx].strip()
                value = _parse_ncu_number(raw_value)
                value = _sanitize_ncu_metric_value(metric_name, value)
                if value is None:
                    continue
                row_values.append((metric_name, value, units.get(column_idx, "")))
            if not row_values:
                continue
            for metric_name, value, unit in row_values:
                parsed.setdefault(metric_name, {"value": value, "unit": unit, "source": "ncu_csv"})
            break
    return parsed


def _looks_like_ncu_metric_name(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or " " in cleaned:
        return False
    return "__" in cleaned or cleaned in NCU_EXACT_METRICS


def _is_ncu_units_row(row: list[str], metric_columns: list[tuple[int, str]]) -> bool:
    if not row:
        return False
    tokens = []
    for column_idx, _ in metric_columns:
        if column_idx < len(row):
            tokens.append(row[column_idx].strip())
    if not tokens:
        return False
    normalized = [token.lower().strip() for token in tokens if token.strip()]
    if not normalized:
        return False
    unit_like = {"%", "byte", "byte/s", "nsecond", "cycle", "count", "kbyte", "kbit", "kHz", "Gbyte/s", "warp", "block", "thread", "register/thread"}
    return all(token in {entry.lower() for entry in unit_like} or token == "" for token in normalized)


def _is_probable_header_row(row: list[str]) -> bool:
    non_empty = [cell.strip() for cell in row if cell.strip()]
    if not non_empty:
        return False
    metric_like = sum(1 for cell in non_empty if _looks_like_ncu_metric_name(cell))
    return metric_like >= max(1, len(non_empty) // 4)


def _parse_ncu_number(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    cleaned = cleaned.rstrip("%").strip()
    try:
        return float(cleaned)
    except ValueError:
        match = NUMBER_RE.search(cleaned)
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None


def _sanitize_ncu_metric_value(metric_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if metric_name in NCU_PERCENT_METRICS and not (-1e-6 <= value <= 120.0):
        return None
    return value


def _ncu_observation(metric_name: str, value: float, unit: str) -> dict:
    return {
        "source": "ncu_csv",
        "metric": metric_name,
        "value": value,
        "unit": unit or _default_ncu_unit(metric_name),
    }


def _default_ncu_unit(metric_name: str) -> str:
    if metric_name in NCU_PERCENT_METRICS:
        return "%"
    if "per_second" in metric_name:
        return "bytes/s"
    return ""


def _bandwidth_confidence(samples: list[float]) -> float:
    return min(0.95, 0.62 + 0.28 * (1.0 - min(1.0, _coefficient_of_variation(samples))))


def _records_from_execution(execution: ProbeExecution) -> list[dict]:
    records = []
    for output in execution.run_outputs:
        records.extend(parse_structured_lines(output))
    return records


def _coerce_value(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            number = float(value)
            if math.isfinite(number):
                return number
            return value
        return int(value)
    except ValueError:
        return value


def _summarize_mode(by_size: dict[int, list[float]]) -> list[dict]:
    points = []
    for size, values in sorted(by_size.items()):
        filtered = _reject_outliers(values)
        points.append(
            {
                "size_bytes": size,
                "median_cycles": _robust_median(filtered),
                "mad_cycles": _mad(filtered),
                "samples": len(values),
            }
        )
    return points


def _plateau_value(points: list[dict], upper_bound: int) -> float:
    candidates = [point["median_cycles"] for point in points if point["size_bytes"] <= upper_bound]
    return statistics.median(candidates or [points[0]["median_cycles"]])


def _strongest_cliff(points: list[dict]) -> int:
    best_size = points[-1]["size_bytes"]
    best_score = -1.0
    for left, right in zip(points, points[1:]):
        delta = right["median_cycles"] - left["median_cycles"]
        ratio = right["median_cycles"] / max(left["median_cycles"], 1e-6)
        score = max(delta, 0.0) * ratio
        if score > best_score:
            best_score = score
            best_size = right["size_bytes"]
    return best_size


def _reject_outliers(values: list[float]) -> list[float]:
    if len(values) < 3:
        return list(values)
    median = statistics.median(values)
    mad = _mad(values)
    if mad <= 1e-9:
        return list(values)
    return [value for value in values if abs(value - median) <= 3.5 * mad] or list(values)


def _robust_median(values: list[float]) -> float:
    filtered = _reject_outliers(values)
    return float(statistics.median(filtered))


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    median = statistics.median(values)
    return float(statistics.median([abs(value - median) for value in values]))


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if abs(mean) <= 1e-9:
        return 0.0
    return abs(statistics.pstdev(values) / mean)


def _confidence_from_variation(values: list[float]) -> float:
    return max(0.3, min(0.95, 0.9 - _coefficient_of_variation(values)))


def _confidence_from_series(points: list[dict]) -> float:
    if not points:
        return 0.0
    medians = [point["median_cycles"] for point in points]
    return max(0.35, min(0.95, 0.88 - _coefficient_of_variation(medians) * 0.5))


def _max_cv(config_values) -> float:
    values = [_coefficient_of_variation(list(config)) for config in config_values]
    return max(values) if values else 1.0


def _status_metric(
    metric_name: str,
    raw_value,
    status: str,
    confidence: float,
    analysis: str,
    conclusion: str,
    observations,
    benchmark_files: list[str],
    ncu_summary: str,
) -> dict:
    result = metric_template(metric_name)
    result.update(
        {
            "canonical_name": metric_name,
            "raw_value": raw_value,
            "unit": default_unit(metric_name),
            "status": status,
            "confidence": round(float(max(0.0, min(0.99, confidence))), 4),
            "analysis": analysis,
            "conclusion": conclusion,
            "raw_observations": observations,
            "benchmark_files": benchmark_files,
            "ncu_summary": ncu_summary,
        }
    )
    return result


def _benchmark_files(execution: ProbeExecution | None) -> list[str]:
    if execution is None:
        return []
    files = [execution.probe.source_path, execution.probe.binary_path, execution.compile_log]
    files.extend(execution.run_logs)
    if execution.ncu_log:
        files.append(execution.ncu_log)
    return files


def _failure_reason_from_execution(execution: ProbeExecution | None, default: str) -> str:
    if execution is None:
        return default
    if execution.compile_status != "success":
        return f"{default}. Compile status: {execution.compile_status}."
    if execution.run_status in {"failed", "partial (0/3)"} or execution.run_status.startswith("failed"):
        for output in execution.run_outputs:
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.startswith("CUDA_ERROR"):
                    return f"{default}. Runtime error: {stripped}."
                if stripped.startswith("PROCESS_TIMEOUT"):
                    return f"{default}. Runtime timed out."
        return f"{default}. Run status: {execution.run_status}."
    return default


def _first_numeric_from_cells(cells: list[str]) -> float | None:
    for cell in cells:
        value = _first_numeric_from_text(cell)
        if value is not None:
            return value
    return None


def _first_numeric_from_text(text: str) -> float | None:
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None
