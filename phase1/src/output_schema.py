from __future__ import annotations

from copy import deepcopy

SUPPORTED_METRICS = [
    "l1_latency_cycles",
    "l2_latency_cycles",
    "dram_latency_cycles",
    "global_memory_bandwidth_gbps",
    "dram_bytes_read_per_second",
    "dram_bytes_write_per_second",
    "shared_memory_bandwidth_gbps",
    "l2_cache_capacity_bytes",
    "actual_core_clock_mhz",
    "device_attribute_max_gpu_frequency_khz",
    "device_attribute_max_mem_frequency_khz",
    "device_attribute_fb_bus_width_bits",
    "peak_fp32_tflops",
    "sm_throughput_pct_of_peak_sustained_elapsed",
    "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
    "shared_memory_bank_conflict_penalty_cycles",
    "shared_memory_bank_conflict_penalty_ratio",
    "observed_active_sm_count",
]

_ALIASES = {
    "l1_latency_cycles": "l1_latency_cycles",
    "l1_cache_latency_cycles": "l1_latency_cycles",
    "l1_cache_access_latency_in_cycles": "l1_latency_cycles",
    "l1_cache_latency": "l1_latency_cycles",
    "l2_latency_cycles": "l2_latency_cycles",
    "l2_cache_latency_cycles": "l2_latency_cycles",
    "l2_cache_access_latency_in_cycles": "l2_latency_cycles",
    "l2_cache_latency": "l2_latency_cycles",
    "dram_latency_cycles": "dram_latency_cycles",
    "dram_access_latency_in_cycles": "dram_latency_cycles",
    "dram_latency": "dram_latency_cycles",
    "global_memory_bandwidth_gbps": "global_memory_bandwidth_gbps",
    "dram_bandwidth_gbps": "global_memory_bandwidth_gbps",
    "vram_bandwidth_gbps": "global_memory_bandwidth_gbps",
    "effective_peak_global_memory_vram_bandwidth": "global_memory_bandwidth_gbps",
    "vram_bandwidth": "global_memory_bandwidth_gbps",
    "dram_bytes_read_per_second": "dram_bytes_read_per_second",
    "dram__bytes_read_sum_per_second": "dram_bytes_read_per_second",
    "dram_bytes_write_per_second": "dram_bytes_write_per_second",
    "dram__bytes_write_sum_per_second": "dram_bytes_write_per_second",
    "shared_memory_bandwidth_gbps": "shared_memory_bandwidth_gbps",
    "effective_peak_shared_memory_bandwidth": "shared_memory_bandwidth_gbps",
    "shared_bandwidth": "shared_memory_bandwidth_gbps",
    "l2_cache_capacity_bytes": "l2_cache_capacity_bytes",
    "l2_cache_size_bytes": "l2_cache_capacity_bytes",
    "l2_capacity": "l2_cache_capacity_bytes",
    "actual_stable_core_clock_frequency_in_mhz_under_sustained_compute_load": "actual_core_clock_mhz",
    "actual_core_clock_mhz": "actual_core_clock_mhz",
    "boost_frequency_mhz": "actual_core_clock_mhz",
    "stable_core_clock_mhz": "actual_core_clock_mhz",
    "core_clock_mhz": "actual_core_clock_mhz",
    "device_attribute_max_gpu_frequency_khz": "device_attribute_max_gpu_frequency_khz",
    "device__attribute_max_gpu_frequency_khz": "device_attribute_max_gpu_frequency_khz",
    "device_attribute_max_mem_frequency_khz": "device_attribute_max_mem_frequency_khz",
    "device__attribute_max_mem_frequency_khz": "device_attribute_max_mem_frequency_khz",
    "device_attribute_fb_bus_width_bits": "device_attribute_fb_bus_width_bits",
    "device__attribute_fb_bus_width": "device_attribute_fb_bus_width_bits",
    "peak_fp32_tflops": "peak_fp32_tflops",
    "fp32_throughput_tflops": "peak_fp32_tflops",
    "sm_throughput_pct_of_peak_sustained_elapsed": "sm_throughput_pct_of_peak_sustained_elapsed",
    "sm__throughput_avg_pct_of_peak_sustained_elapsed": "sm_throughput_pct_of_peak_sustained_elapsed",
    "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput_avg_pct_of_peak_sustained_elapsed": "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed",
    "shared_memory_bank_conflict_latency_penalty_compared_with_conflict_free_access": "shared_memory_bank_conflict_penalty_cycles",
    "shared_memory_bank_conflict_penalty_cycles": "shared_memory_bank_conflict_penalty_cycles",
    "bank_conflict_penalty_cycles": "shared_memory_bank_conflict_penalty_cycles",
    "shared_memory_bank_conflict_penalty_ratio": "shared_memory_bank_conflict_penalty_ratio",
    "bank_conflict_penalty_ratio": "shared_memory_bank_conflict_penalty_ratio",
    "observed_active_sm_count": "observed_active_sm_count",
    "launch__sm_count": "observed_active_sm_count",
    "physical_sm_count": "observed_active_sm_count",
    "active_sm_count": "observed_active_sm_count",
    "sm_count": "observed_active_sm_count",
}

_DEFAULT_UNITS = {
    "l1_latency_cycles": "cycles",
    "l2_latency_cycles": "cycles",
    "dram_latency_cycles": "cycles",
    "global_memory_bandwidth_gbps": "GB/s",
    "dram_bytes_read_per_second": "bytes/s",
    "dram_bytes_write_per_second": "bytes/s",
    "shared_memory_bandwidth_gbps": "GB/s",
    "l2_cache_capacity_bytes": "bytes",
    "actual_core_clock_mhz": "MHz",
    "device_attribute_max_gpu_frequency_khz": "kHz",
    "device_attribute_max_mem_frequency_khz": "kHz",
    "device_attribute_fb_bus_width_bits": "bits",
    "peak_fp32_tflops": "TFLOP/s",
    "sm_throughput_pct_of_peak_sustained_elapsed": "%",
    "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "%",
    "shared_memory_bank_conflict_penalty_cycles": "cycles",
    "shared_memory_bank_conflict_penalty_ratio": "ratio",
    "observed_active_sm_count": "count",
}

_CARD_TYPES = {
    "l1_latency_cycles": "latency_hierarchy",
    "l2_latency_cycles": "latency_hierarchy",
    "dram_latency_cycles": "latency_hierarchy",
    "global_memory_bandwidth_gbps": "bandwidth",
    "dram_bytes_read_per_second": "dynamic_traffic",
    "dram_bytes_write_per_second": "dynamic_traffic",
    "shared_memory_bandwidth_gbps": "bandwidth",
    "l2_cache_capacity_bytes": "capacity",
    "actual_core_clock_mhz": "frequency",
    "device_attribute_max_gpu_frequency_khz": "hardware_attribute_snapshot",
    "device_attribute_max_mem_frequency_khz": "hardware_attribute_snapshot",
    "device_attribute_fb_bus_width_bits": "hardware_attribute_snapshot",
    "peak_fp32_tflops": "throughput",
    "sm_throughput_pct_of_peak_sustained_elapsed": "core_utilization",
    "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "core_utilization",
    "shared_memory_bank_conflict_penalty_cycles": "resource_penalty",
    "shared_memory_bank_conflict_penalty_ratio": "resource_penalty",
    "observed_active_sm_count": "topology",
}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def canonicalize_metric_name(name: str) -> str | None:
    normalized = normalize_name(name)
    return _ALIASES.get(normalized)


def metric_template(metric_name: str) -> dict:
    return {
        "canonical_name": metric_name,
        "raw_value": None,
        "unit": _DEFAULT_UNITS.get(metric_name, ""),
        "analysis": "",
        "conclusion": "",
        "bottleneck": "",
        "card_type": _CARD_TYPES.get(metric_name, "analysis"),
        "status": "failed",
        "confidence": 0.0,
        "raw_observations": [],
        "benchmark_files": [],
        "ncu_summary": "unavailable",
    }


def failed_metric(metric_name: str, reason: str, benchmark_files: list[str] | None = None) -> dict:
    result = metric_template(metric_name)
    result["analysis"] = reason
    result["conclusion"] = reason
    result["benchmark_files"] = benchmark_files or []
    return result


def base_output() -> dict:
    return {
        "target_spec_path": None,
        "total_metrics": 0,
        "successful_analyses": 0,
        "results": {},
        "agent_logs": [],
        "generated_benchmarks": [],
        "probe_plan": [],
        "normalized_targets": [],
        "triage": [],
        "ncu_analysis": [],
        "environment_notes": {
            "api_reported_device_properties": {},
            "observed_active_sms": None,
            "frequency_lock_suspected": None,
            "sm_masking_suspected": None,
        },
    }


def clone_metric(metric_result: dict) -> dict:
    return deepcopy(metric_result)


def default_unit(metric_name: str) -> str:
    return _DEFAULT_UNITS.get(metric_name, "")
