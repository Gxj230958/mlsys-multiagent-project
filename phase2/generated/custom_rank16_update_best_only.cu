#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(W.scalar_type() == at::kFloat && X.scalar_type() == at::kFloat &&
                A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "all inputs must be float32");
    TORCH_CHECK(W.is_contiguous() && X.is_contiguous() && A.is_contiguous() && B.is_contiguous(),
                "all inputs must be contiguous");
    TORCH_CHECK(W.dim() == 2 && X.dim() == 2 && A.dim() == 2 && B.dim() == 2,
                "all inputs must be 2D");
    const auto d = W.size(0);
    TORCH_CHECK(d > 0, "d must be positive");
    TORCH_CHECK(W.size(1) == d && X.size(0) == d && X.size(1) == d,
                "W and X must be [d, d]");
    TORCH_CHECK(A.size(0) == d && A.size(1) == 16 && B.size(0) == d && B.size(1) == 16,
                "A and B must be [d, 16]");
}

}  // namespace

namespace {

__global__ void rank16_update_kernel(float* __restrict__ Y,
                                     const float* __restrict__ A,
                                     const float* __restrict__ T,
                                     int64_t d) {
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= d || col >= d) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        acc += A[row * 16 + k] * T[static_cast<int64_t>(k) * d + col];
    }
    Y[row * d + col] += acc;
}

}  // namespace

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    check_inputs(W, X, A, B);
    const auto d = W.size(0);
    auto T = at::matmul(B.transpose(0, 1).contiguous(), X).contiguous();
    auto Y = at::matmul(W, X);
    const dim3 block(32, 8);
    const dim3 grid((static_cast<unsigned int>(d) + 32 - 1) / 32,
                    (static_cast<unsigned int>(d) + 8 - 1) / 8);
    rank16_update_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Y.data_ptr<float>(), A.data_ptr<float>(), T.data_ptr<float>(), d);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rank16_update_kernel failed: ", cudaGetErrorString(err));
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized LoRA forward");
}
