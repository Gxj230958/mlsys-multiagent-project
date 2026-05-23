#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
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

void check_cublas(cublasStatus_t status, const char* what) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublas status ", int(status));
}

struct Handle {
    cublasHandle_t h = nullptr;
    Handle() { check_cublas(cublasCreate(&h), "cublasCreate"); }
    ~Handle() { if (h) cublasDestroy(h); }
};

cublasHandle_t get_handle() {
    static thread_local Handle holder;
    return holder.h;
}

void checked_sgemm(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m, int n, int k,
                   const float* alpha,
                   const float* A, int lda,
                   const float* B, int ldb,
                   const float* beta,
                   float* C, int ldc) {
    check_cublas(cublasSgemm(handle, transa, transb, m, n, k,
                             alpha, A, lda, B, ldb, beta, C, ldc),
                 "cublasSgemm");
}

#ifndef CUBLAS_COMPUTE_32F_FAST_TF32
#define CUBLAS_COMPUTE_32F_FAST_TF32 CUBLAS_COMPUTE_32F
#endif

void gemmex_or_sgemm(cublasHandle_t handle,
                     cublasOperation_t transa,
                     cublasOperation_t transb,
                     int m, int n, int k,
                     const float* alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     const float* beta,
                     float* C, int ldc) {
    cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k,
                                         alpha,
                                         A, CUDA_R_32F, lda,
                                         B, CUDA_R_32F, ldb,
                                         beta,
                                         C, CUDA_R_32F, ldc,
                                         CUBLAS_COMPUTE_32F_FAST_TF32,
                                         CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        checked_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

}  // namespace

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
                    check_inputs(W, X, A, B);
                    const int d = static_cast<int>(W.size(0));
                    auto Y = torch::empty_like(W);
                    auto T = torch::empty({16, d}, W.options());
                    cublasHandle_t handle = get_handle();
                    check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()), "cublasSetStream");
                    const float alpha = 1.0f;
                    const float beta_zero = 0.0f;

                    // Row-major W@X as column-major X^T * W^T -> Y^T.
                    checked_sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                d, d, d,
                                &alpha, X.data_ptr<float>(), d,
                                W.data_ptr<float>(), d,
                                &beta_zero, Y.data_ptr<float>(), d);

                    // T = B^T@X. T row-major [16,d] is column-major [d,16].
                    checked_sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                d, 16, d,
                                &alpha, X.data_ptr<float>(), d,
                                B.data_ptr<float>(), 16,
                                &beta_zero, T.data_ptr<float>(), d);

                    auto U = torch::empty_like(Y);
checked_sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            d, d, 16,
            &alpha, T.data_ptr<float>(), d,
            A.data_ptr<float>(), 16,
            &beta_zero, U.data_ptr<float>(), d);
return Y + U;
                }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized LoRA forward");
}
