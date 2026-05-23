
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
