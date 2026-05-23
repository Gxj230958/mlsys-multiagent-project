
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
