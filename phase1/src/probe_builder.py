from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProbeSpec:
    name: str
    purpose: str
    source_path: str
    binary_path: str
    compile_args: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    enable_ncu: bool = False
    expected_stdout_schema: dict = field(default_factory=dict)


def build_required_probes(project_root: Path, benchmark_plan: list[dict]) -> list[ProbeSpec]:
    generated_dir = project_root / "benchmarks" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    probes: list[ProbeSpec] = []

    for benchmark in benchmark_plan:
        name = benchmark["name"]
        source_path = generated_dir / f"{name}.cu"
        binary_path = generated_dir / name
        source_path.write_text(_source_for_probe(name), encoding="utf-8")
        probes.append(
            ProbeSpec(
                name=name,
                purpose=benchmark["purpose"],
                source_path=str(source_path),
                binary_path=str(binary_path),
                compile_args=list(benchmark.get("compile_flags", [])),
                metrics=list(benchmark.get("targets", [])),
                enable_ncu=bool(benchmark.get("ncu_enabled", False)),
                expected_stdout_schema=dict(benchmark.get("expected_stdout_schema", {})),
            )
        )
    return probes


def _source_for_probe(name: str) -> str:
    sources = {
        "environment_probe": _environment_probe_source(),
        "pointer_chase_latency": _pointer_chase_source(),
        "global_bandwidth": _global_bandwidth_source(),
        "shared_bandwidth": _shared_bandwidth_source(),
        "core_clock": _core_clock_source(),
        "fp32_throughput": _fp32_throughput_source(),
        "bank_conflict": _bank_conflict_source(),
    }
    if name not in sources:
        raise ValueError(f"Unsupported probe template requested: {name}")
    return sources[name]


def _environment_probe_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <cstdio>
#include <set>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void sample_smid_kernel(unsigned int* smids) {
    if (threadIdx.x == 0) {
        unsigned int smid = 0;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smids[blockIdx.x] = smid;
    }
}

int main() {
    int device_count = 0;
    cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "CUDA_ERROR no_device %s\n", cudaGetErrorString(count_status));
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    CHECK_CUDA(cudaSetDevice(device));

    const int blocks = 4096;
    const int rounds = 4;
    unsigned int* d_smids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_smids, blocks * sizeof(unsigned int)));

    std::printf("PROP name=%s\n", prop.name);
    std::printf("PROP multiProcessorCount=%d\n", prop.multiProcessorCount);
    std::printf("PROP clockRateKHz=%d\n", prop.clockRate);
    std::printf("PROP memoryClockRateKHz=%d\n", prop.memoryClockRate);
    std::printf("PROP memoryBusWidthBits=%d\n", prop.memoryBusWidth);
    std::printf("PROP l2CacheSizeBytes=%d\n", prop.l2CacheSize);

    for (int round = 0; round < rounds; ++round) {
        sample_smid_kernel<<<blocks, 64>>>(d_smids);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<unsigned int> h_smids(blocks, 0);
        CHECK_CUDA(cudaMemcpy(h_smids.data(), d_smids, blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::set<unsigned int> unique_smids(h_smids.begin(), h_smids.end());
        std::printf("OBSERVED round=%d active_sms=%zu blocks=%d\n", round, unique_smids.size(), blocks);
    }

    CHECK_CUDA(cudaFree(d_smids));
    return 0;
}
"""


def _pointer_chase_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__device__ __forceinline__ unsigned int load_default(const unsigned int* ptr, unsigned int idx) {
    return ptr[idx];
}

__device__ __forceinline__ unsigned int load_l2_or_dram(const unsigned int* ptr, unsigned int idx) {
#if __CUDA_ARCH__ >= 350
    unsigned int value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr + idx));
    return value;
#else
    return ptr[idx];
#endif
}

template <bool CG_MODE>
__global__ void pointer_chase_kernel(const unsigned int* next_idx, int steps, unsigned long long* cycles_out, unsigned int* sink_out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int idx = 0;
        for (int i = 0; i < 1024; ++i) {
            idx = CG_MODE ? load_l2_or_dram(next_idx, idx) : load_default(next_idx, idx);
        }
        unsigned long long begin = clock64();
        for (int i = 0; i < steps; ++i) {
            idx = CG_MODE ? load_l2_or_dram(next_idx, idx) : load_default(next_idx, idx);
        }
        unsigned long long end = clock64();
        cycles_out[0] = end - begin;
        sink_out[0] = idx;
    }
}

__global__ void flush_cache_kernel(unsigned int* buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        buffer[i] = buffer[i] + 1U;
    }
}

int main() {
    const std::vector<size_t> sizes = {
        4ULL << 10, 8ULL << 10, 16ULL << 10, 32ULL << 10, 64ULL << 10, 128ULL << 10,
        256ULL << 10, 512ULL << 10, 1ULL << 20, 2ULL << 20, 4ULL << 20, 8ULL << 20,
        16ULL << 20, 32ULL << 20, 64ULL << 20
    };
    const size_t flush_elems = (32ULL << 20) / sizeof(unsigned int);
    unsigned int* d_flush = nullptr;
    CHECK_CUDA(cudaMalloc(&d_flush, flush_elems * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_flush, 0, flush_elems * sizeof(unsigned int)));

    unsigned long long* d_cycles = nullptr;
    unsigned int* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(unsigned int)));

    for (size_t size_bytes : sizes) {
        size_t count = std::max<size_t>(1024, size_bytes / sizeof(unsigned int));
        std::vector<unsigned int> perm(count);
        std::iota(perm.begin(), perm.end(), 0U);
        std::mt19937 rng(static_cast<unsigned int>(count ^ 0x9E3779B9U));
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<unsigned int> next_idx(count);
        for (size_t i = 0; i < count; ++i) {
            next_idx[perm[i]] = perm[(i + 1) % count];
        }

        unsigned int* d_next = nullptr;
        CHECK_CUDA(cudaMalloc(&d_next, count * sizeof(unsigned int)));
        CHECK_CUDA(cudaMemcpy(d_next, next_idx.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        int steps = static_cast<int>(std::min<size_t>(1U << 22, std::max<size_t>(1U << 18, count * 8)));

        flush_cache_kernel<<<256, 256>>>(d_flush, static_cast<int>(flush_elems));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        pointer_chase_kernel<false><<<1, 1>>>(d_next, steps, d_cycles, d_sink);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned long long cycles_default = 0;
        CHECK_CUDA(cudaMemcpy(&cycles_default, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double per_default = static_cast<double>(cycles_default) / static_cast<double>(steps);
        std::printf("POINT size_bytes=%zu mode=default cycles_per_access=%.6f steps=%d\n", size_bytes, per_default, steps);

        flush_cache_kernel<<<256, 256>>>(d_flush, static_cast<int>(flush_elems));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        pointer_chase_kernel<true><<<1, 1>>>(d_next, steps, d_cycles, d_sink);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned long long cycles_cg = 0;
        CHECK_CUDA(cudaMemcpy(&cycles_cg, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double per_cg = static_cast<double>(cycles_cg) / static_cast<double>(steps);
        std::printf("POINT size_bytes=%zu mode=l2_or_dram cycles_per_access=%.6f steps=%d\n", size_bytes, per_cg, steps);

        CHECK_CUDA(cudaFree(d_next));
    }

    CHECK_CUDA(cudaFree(d_flush));
    CHECK_CUDA(cudaFree(d_cycles));
    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
"""


def _global_bandwidth_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void copy_scalar_kernel(const float* src, float* dst, size_t n, int repeats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int r = 0; r < repeats; ++r) {
        for (size_t i = idx; i < n; i += stride) {
            dst[i] = src[i];
        }
    }
}

__global__ void read_scalar_kernel(const float* src, float* dst, size_t n, int repeats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (int r = 0; r < repeats; ++r) {
        for (size_t i = idx; i < n; i += stride) {
            acc += src[i];
        }
    }
    if (idx < n) {
        dst[idx] = acc;
    }
}

__global__ void write_scalar_kernel(float* dst, size_t n, int repeats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int r = 0; r < repeats; ++r) {
        for (size_t i = idx; i < n; i += stride) {
            dst[i] = static_cast<float>(i + r);
        }
    }
}

__global__ void copy_vec4_kernel(const uint4* src, uint4* dst, size_t n, int repeats) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int r = 0; r < repeats; ++r) {
        for (size_t i = idx; i < n; i += stride) {
            dst[i] = src[i];
        }
    }
}

int main() {
    size_t free_mem = 0;
    size_t total_mem = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    size_t target_bytes = free_mem / 4;
    const size_t min_bytes = 512ULL << 20;
    const size_t max_bytes = 768ULL << 20;
    if (target_bytes < min_bytes) {
        target_bytes = free_mem / 8;
    }
    target_bytes = target_bytes < min_bytes ? target_bytes : min_bytes + (target_bytes - min_bytes);
    if (target_bytes > max_bytes) {
        target_bytes = max_bytes;
    }
    if (target_bytes < (64ULL << 20)) {
        target_bytes = 64ULL << 20;
    }

    float* d_src = nullptr;
    float* d_dst = nullptr;
    while (target_bytes >= (64ULL << 20)) {
        if (cudaMalloc(&d_src, target_bytes) == cudaSuccess && cudaMalloc(&d_dst, target_bytes) == cudaSuccess) {
            break;
        }
        if (d_src) cudaFree(d_src);
        if (d_dst) cudaFree(d_dst);
        d_src = nullptr;
        d_dst = nullptr;
        target_bytes /= 2;
    }
    if (!d_src || !d_dst) {
        std::fprintf(stderr, "CUDA_ERROR allocation_failed unable_to_allocate_streaming_buffers\n");
        return 3;
    }
    CHECK_CUDA(cudaMemset(d_src, 0, target_bytes));
    CHECK_CUDA(cudaMemset(d_dst, 0, target_bytes));

    size_t elements = target_bytes / sizeof(float);
    size_t vec4_count = target_bytes / sizeof(uint4);
    const int repeats = 8;
    const int blocks = 256;
    const std::vector<int> block_sizes = {128, 256, 512};

    for (int block_size : block_sizes) {
        const char* modes[] = {"copy", "read", "write"};
        for (int mode = 0; mode < 3; ++mode) {
            cudaEvent_t start{}, stop{};
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(start));
            if (mode == 0) {
                copy_scalar_kernel<<<blocks, block_size>>>(d_src, d_dst, elements, repeats);
            } else if (mode == 1) {
                read_scalar_kernel<<<blocks, block_size>>>(d_src, d_dst, elements, repeats);
            } else {
                write_scalar_kernel<<<blocks, block_size>>>(d_dst, elements, repeats);
            }
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            double modeled_bytes = static_cast<double>(target_bytes) * static_cast<double>(repeats) * (mode == 0 ? 2.0 : 1.0);
            double gbps = modeled_bytes / (static_cast<double>(elapsed_ms) / 1.0e3) / 1.0e9;
            std::printf("CONFIG mode=%s block=%d bytes=%zu elapsed_ms=%.6f gbps=%.6f\n", modes[mode], block_size, target_bytes, elapsed_ms, gbps);
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }

        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        copy_vec4_kernel<<<blocks, block_size>>>(reinterpret_cast<const uint4*>(d_src), reinterpret_cast<uint4*>(d_dst), vec4_count, repeats);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double modeled_bytes = static_cast<double>(target_bytes) * 2.0 * static_cast<double>(repeats);
        double gbps = modeled_bytes / (static_cast<double>(elapsed_ms) / 1.0e3) / 1.0e9;
        std::printf("CONFIG mode=vec4_copy block=%d bytes=%zu elapsed_ms=%.6f gbps=%.6f\n", block_size, target_bytes, elapsed_ms, gbps);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    return 0;
}
"""


def _shared_bandwidth_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void shared_bandwidth_kernel(float* out, int iters) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    tile[tid] = static_cast<float>(tid);
    __syncthreads();

    float acc = 0.0f;
    int idx = tid;
    for (int i = 0; i < iters; ++i) {
        idx = (idx + 33) & (blockDim.x - 1);
        float a = tile[idx];
        float b = tile[(idx + 17) & (blockDim.x - 1)];
        tile[idx] = a + b + 1.0f;
        acc += a + b;
    }
    out[blockIdx.x * blockDim.x + tid] = acc;
}

int main() {
    const std::vector<int> block_sizes = {128, 256, 512};
    const int blocks = 256;
    const int iters = 1 << 15;
    float* d_out = nullptr;

    for (int block_size : block_sizes) {
        size_t out_elems = static_cast<size_t>(blocks) * static_cast<size_t>(block_size);
        CHECK_CUDA(cudaMalloc(&d_out, out_elems * sizeof(float)));
        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        shared_bandwidth_kernel<<<blocks, block_size, block_size * sizeof(float)>>>(d_out, 1024);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        shared_bandwidth_kernel<<<blocks, block_size, block_size * sizeof(float)>>>(d_out, iters);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double bytes_modeled = static_cast<double>(blocks) * static_cast<double>(block_size) * static_cast<double>(iters) * sizeof(float) * 4.0;
        double gbps = bytes_modeled / (static_cast<double>(elapsed_ms) / 1.0e3) / 1.0e9;
        std::printf("CONFIG block=%d iters=%d bytes_modeled=%.0f elapsed_ms=%.6f gbps=%.6f\n", block_size, iters, bytes_modeled, elapsed_ms, gbps);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_out));
        d_out = nullptr;
    }
    return 0;
}
"""


def _core_clock_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void sustained_compute_kernel(unsigned long long* cycles_out, float* sink, int iters) {
    float x = 1.0f + static_cast<float>(threadIdx.x);
    unsigned long long begin = clock64();
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 0.000001f);
        x = fmaf(x, 0.999999f, 0.000002f);
        x = fmaf(x, 1.000003f, 0.000003f);
        x = fmaf(x, 0.999997f, 0.000004f);
    }
    unsigned long long end = clock64();
    if (threadIdx.x == 0) {
        cycles_out[blockIdx.x] = end - begin;
    }
    sink[threadIdx.x] = x;
}

int main() {
    const int blocks = 1;
    const int threads = 256;
    const int durations[3] = {1 << 18, 1 << 20, 1 << 22};
    unsigned long long* d_cycles = nullptr;
    float* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cycles, blocks * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_sink, threads * sizeof(float)));

    for (int i = 0; i < 3; ++i) {
        int iters = durations[i];
        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        sustained_compute_kernel<<<blocks, threads>>>(d_cycles, d_sink, 1 << 16);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        sustained_compute_kernel<<<blocks, threads>>>(d_cycles, d_sink, iters);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        unsigned long long cycles = 0;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CHECK_CUDA(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double mhz = static_cast<double>(cycles) / static_cast<double>(elapsed_ms) / 1000.0;
        std::printf("TRIAL duration=%d cycles=%llu elapsed_ms=%.6f mhz=%.6f\n", iters, static_cast<unsigned long long>(cycles), elapsed_ms, mhz);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_cycles));
    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
"""


def _fp32_throughput_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void fp32_fma_kernel(float* sink, int iters) {
    float x = 1.0f + static_cast<float>(threadIdx.x + blockIdx.x);
    float y = 0.5f + static_cast<float>(threadIdx.x);
    #pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, y, 1.0f);
        y = fmaf(y, x, 2.0f);
        x = fmaf(x, y, 3.0f);
        y = fmaf(y, x, 4.0f);
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

int main() {
    const int blocks = 512;
    const int threads = 256;
    const int iteration_counts[2] = {1 << 18, 1 << 20};
    const double fmas_per_iter = 4.0;
    float* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sink, static_cast<size_t>(blocks) * threads * sizeof(float)));

    for (int k = 0; k < 2; ++k) {
        int iters = iteration_counts[k];
        cudaEvent_t start{}, stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        fp32_fma_kernel<<<blocks, threads>>>(d_sink, 1 << 14);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        fp32_fma_kernel<<<blocks, threads>>>(d_sink, iters);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double operations = static_cast<double>(blocks) * static_cast<double>(threads) * static_cast<double>(iters) * fmas_per_iter * 2.0;
        double tflops = operations / (static_cast<double>(elapsed_ms) / 1.0e3) / 1.0e12;
        std::printf("TRIAL iters=%d elapsed_ms=%.6f operations=%.0f tflops=%.6f\n", iters, elapsed_ms, operations, tflops);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
"""


def _bank_conflict_source() -> str:
    return r"""
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void bank_conflict_kernel(unsigned long long* cycles_out, int stride, int iters) {
    __shared__ volatile int tile[32 * 32];
    int tid = threadIdx.x;
    tile[tid] = tid;
    __syncthreads();
    int acc = 0;
    unsigned long long begin = clock64();
    for (int i = 0; i < iters; ++i) {
        acc += tile[(tid * stride) & ((32 * 32) - 1)];
    }
    unsigned long long end = clock64();
    if (tid == 0) {
        cycles_out[0] = end - begin + static_cast<unsigned long long>(acc & 1);
    }
}

int main() {
    const int inner_trials = 8;
    const int iters = 1 << 16;
    unsigned long long* d_cycles = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

    const int strides[2] = {1, 32};
    const char* names[2] = {"conflict_free", "bank_conflict"};
    double medians[2] = {0.0, 0.0};

    for (int mode = 0; mode < 2; ++mode) {
        double total = 0.0;
        for (int trial = 0; trial < inner_trials; ++trial) {
            bank_conflict_kernel<<<1, 32>>>(d_cycles, strides[mode], iters);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            unsigned long long cycles = 0;
            CHECK_CUDA(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            total += static_cast<double>(cycles) / static_cast<double>(iters);
        }
        medians[mode] = total / static_cast<double>(inner_trials);
        std::printf("PATTERN name=%s cycles_per_iter=%.6f iters=%d inner_trials=%d\n", names[mode], medians[mode], iters, inner_trials);
    }

    double penalty_cycles = medians[1] - medians[0];
    double ratio = medians[1] / (medians[0] > 0.0 ? medians[0] : 1.0);
    std::printf("SUMMARY penalty_cycles=%.6f penalty_ratio=%.6f\n", penalty_cycles, ratio);
    CHECK_CUDA(cudaFree(d_cycles));
    return 0;
}
"""
