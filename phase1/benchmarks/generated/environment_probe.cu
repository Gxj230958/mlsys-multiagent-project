
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
