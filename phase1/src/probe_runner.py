from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from probe_builder import ProbeSpec

NCU_METRIC_NAME_BY_CANONICAL = {
    "dram_bytes_read_per_second": "dram__bytes_read.sum.per_second",
    "dram_bytes_write_per_second": "dram__bytes_write.sum.per_second",
    "sm_throughput_pct_of_peak_sustained_elapsed": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu_compute_memory_throughput_pct_of_peak_sustained_elapsed": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}


@dataclass
class ProbeExecution:
    probe: ProbeSpec
    compile_status: str = "pending"
    compile_log: str = ""
    run_status: str = "pending"
    run_logs: list[str] = field(default_factory=list)
    run_outputs: list[str] = field(default_factory=list)
    ncu_summary: str = "unavailable"
    ncu_log: str = ""
    ncu_output: str = ""

    def benchmark_record(self) -> dict:
        return {
            "name": self.probe.name,
            "source_path": self.probe.source_path,
            "binary_path": self.probe.binary_path,
            "purpose": self.probe.purpose,
            "compile_status": self.compile_status,
            "run_status": self.run_status,
            "metrics": list(self.probe.metrics),
        }


def compile_and_run_probes(
    probes: list[ProbeSpec],
    logs_dir: Path,
    trials: int = 3,
    timeout_sec: int = 180,
    enable_ncu: bool = True,
) -> dict[str, ProbeExecution]:
    results: dict[str, ProbeExecution] = {}
    nvcc_path = shutil.which("nvcc")
    ncu_path = shutil.which("ncu") if enable_ncu else None

    for probe in probes:
        execution = ProbeExecution(probe=probe)
        results[probe.name] = execution

        compile_log_path = logs_dir / f"{probe.name}_compile.log"
        if not nvcc_path:
            execution.compile_status = "failed: nvcc not found"
            compile_log_path.write_text("nvcc not found in PATH\n", encoding="utf-8")
            execution.compile_log = str(compile_log_path)
            execution.run_status = "skipped: compile failed"
            continue

        compile_cmd = [
            nvcc_path,
            probe.source_path,
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-o",
            probe.binary_path,
            *probe.compile_args,
        ]
        compile_proc = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        compile_log_path.write_text(compile_proc.stdout, encoding="utf-8")
        execution.compile_log = str(compile_log_path)
        execution.compile_status = "success" if compile_proc.returncode == 0 else f"failed ({compile_proc.returncode})"
        if compile_proc.returncode != 0:
            execution.run_status = "skipped: compile failed"
            continue

        run_failures = 0
        for trial in range(trials):
            run_log_path = logs_dir / f"{probe.name}_run_{trial + 1}.log"
            try:
                run_proc = subprocess.run(
                    [probe.binary_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                run_output = run_proc.stdout
                if run_proc.returncode != 0:
                    run_failures += 1
                    run_output += f"\nPROCESS_EXIT_CODE returncode={run_proc.returncode}\n"
            except subprocess.TimeoutExpired as exc:
                run_failures += 1
                run_output = (exc.stdout or "") + "\nPROCESS_TIMEOUT timeout=1\n"
            run_log_path.write_text(run_output, encoding="utf-8")
            execution.run_logs.append(str(run_log_path))
            execution.run_outputs.append(run_output)

        if run_failures == 0:
            execution.run_status = "success"
        elif run_failures < trials:
            execution.run_status = f"partial ({trials - run_failures}/{trials})"
        else:
            execution.run_status = "failed"

        if ncu_path and probe.enable_ncu and execution.run_status != "failed":
            ncu_log_path = logs_dir / f"{probe.name}_ncu.log"
            ncu_attempts = _ncu_command_attempts(ncu_path, probe)
            try:
                combined_outputs: list[str] = []
                chosen_output = ""
                chosen_return_code = 1
                for attempt_name, ncu_cmd in ncu_attempts:
                    ncu_proc = subprocess.run(
                        ncu_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout_sec,
                        check=False,
                    )
                    attempt_output = ncu_proc.stdout
                    combined_outputs.append(f"[attempt:{attempt_name}]\n{attempt_output}")
                    if ncu_proc.returncode == 0:
                        chosen_output = attempt_output
                        chosen_return_code = 0
                        break
                    chosen_output = attempt_output
                    chosen_return_code = ncu_proc.returncode
                ncu_output = "\n\n".join(combined_outputs)
                ncu_log_path.write_text(ncu_output, encoding="utf-8")
                execution.ncu_log = str(ncu_log_path)
                execution.ncu_output = chosen_output
                execution.ncu_summary = summarize_ncu_output(chosen_output, chosen_return_code)
            except subprocess.TimeoutExpired as exc:
                ncu_output = (exc.stdout or "") + "\nPROCESS_TIMEOUT timeout=1\n"
                ncu_log_path.write_text(ncu_output, encoding="utf-8")
                execution.ncu_log = str(ncu_log_path)
                execution.ncu_output = ncu_output
                execution.ncu_summary = "ncu timed out"

    return results


def summarize_ncu_output(output: str, return_code: int) -> str:
    if return_code != 0:
        if "ERR_NVGPUCTRPERM" in output:
            return "ncu unavailable due to missing GPU performance counter permissions"
        return f"ncu failed with return code {return_code}"

    interesting_lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in ("SM", "DRAM", "L2", "Memory", "Duration", "Throughput")):
            interesting_lines.append(stripped)
        if len(interesting_lines) >= 4:
            break
    if interesting_lines:
        return " | ".join(interesting_lines)
    return "ncu completed but did not yield a compact summary"


def _ncu_metric_names_for_probe(probe: ProbeSpec) -> list[str]:
    names = []
    for metric_name in probe.metrics:
        ncu_name = NCU_METRIC_NAME_BY_CANONICAL.get(metric_name)
        if ncu_name and ncu_name not in names:
            names.append(ncu_name)
    if probe.name == "fp32_throughput":
        for ncu_name in (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__maximum_warps_per_active_cycle_pct",
            "smsp__maximum_warps_avg_per_active_cycle",
        ):
            if ncu_name not in names:
                names.append(ncu_name)
    return names


def _ncu_command_attempts(ncu_path: str, probe: ProbeSpec) -> list[tuple[str, list[str]]]:
    requested_ncu_metrics = _ncu_metric_names_for_probe(probe)
    attempts = []
    if requested_ncu_metrics:
        attempts.append(
            (
                "explicit_metrics",
                [
                    ncu_path,
                    "--target-processes",
                    "all",
                    "--csv",
                    "--page",
                    "raw",
                    "--metrics",
                    ",".join(requested_ncu_metrics),
                    probe.binary_path,
                ],
            )
        )
    attempts.append(
        (
            "basic_set",
            [
                ncu_path,
                "--target-processes",
                "all",
                "--set",
                "basic",
                probe.binary_path,
            ],
        )
    )
    return attempts
