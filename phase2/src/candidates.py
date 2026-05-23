from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    description: str


class CandidateGenerator:
    """Small, submission-oriented candidate set focused on GEMM orchestration."""

    def generate(self) -> list[Candidate]:
        return [
            Candidate("safe_aten_basic", self._aten_basic_source(), "Conservative ATen matmul fallback."),
            Candidate("aten_addmm_accumulate", self._aten_addmm_source(), "ATen addmm accumulation path."),
            Candidate(
                "custom_rank16_update_best_only",
                self._custom_rank16_source(block_x=32, block_y=8),
                "One known-good custom rank-16 update representative.",
            ),
            Candidate(
                "cublas_sgemm_sequential",
                self._cublas_source(use_gemmex=False, beta_accumulate=False),
                "Single-stream cuBLAS SGEMM path with a separate low-rank update tensor.",
            ),
            Candidate(
                "cublas_sgemm_single_stream_beta_accumulate",
                self._cublas_source(use_gemmex=False, beta_accumulate=True),
                "Single-stream cuBLAS SGEMM path using beta=1 for A*T accumulation into Y.",
            ),
            Candidate(
                "cublas_gemmex_tf32",
                self._cublas_source(use_gemmex=True, beta_accumulate=True),
                "cuBLAS GemmEx path allowing TF32 tensor-core math with SGEMM fallback.",
            ),
        ]

    def baseline(self) -> Candidate:
        return self.generate()[0]

    def _common_checks(self) -> str:
        return dedent(
            r"""
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
            """
        ).strip()

    def _module_def(self) -> str:
        return dedent(
            r"""
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("forward", &forward, "Optimized LoRA forward");
            }
            """
        ).strip()

    def _aten_basic_source(self) -> str:
        return (
            dedent(
                r"""
                #include <torch/extension.h>
                #include <ATen/ATen.h>
                """
            ).strip()
            + "\n\n"
            + self._common_checks()
            + "\n\n"
            + dedent(
                r"""
                torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
                    check_inputs(W, X, A, B);
                    auto T = at::matmul(B.transpose(0, 1).contiguous(), X);
                    return at::matmul(W, X) + at::matmul(A, T);
                }
                """
            ).strip()
            + "\n\n"
            + self._module_def()
            + "\n"
        )

    def _aten_addmm_source(self) -> str:
        return (
            dedent(
                r"""
                #include <torch/extension.h>
                #include <ATen/ATen.h>
                """
            ).strip()
            + "\n\n"
            + self._common_checks()
            + "\n\n"
            + dedent(
                r"""
                torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
                    check_inputs(W, X, A, B);
                    auto T = at::matmul(B.transpose(0, 1).contiguous(), X);
                    auto Y = at::matmul(W, X);
                    return at::addmm(Y, A, T, 1.0, 1.0);
                }
                """
            ).strip()
            + "\n\n"
            + self._module_def()
            + "\n"
        )

    def _custom_rank16_source(self, block_x: int, block_y: int) -> str:
        return (
            dedent(
                r"""
                #include <torch/extension.h>
                #include <ATen/ATen.h>
                #include <ATen/cuda/CUDAContext.h>
                #include <cuda_runtime.h>
                """
            ).strip()
            + "\n\n"
            + self._common_checks()
            + "\n\n"
            + dedent(
                f"""
                namespace {{

                __global__ void rank16_update_kernel(float* __restrict__ Y,
                                                     const float* __restrict__ A,
                                                     const float* __restrict__ T,
                                                     int64_t d) {{
                    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                    const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
                    if (row >= d || col >= d) return;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) {{
                        acc += A[row * 16 + k] * T[static_cast<int64_t>(k) * d + col];
                    }}
                    Y[row * d + col] += acc;
                }}

                }}  // namespace

                torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {{
                    check_inputs(W, X, A, B);
                    const auto d = W.size(0);
                    auto T = at::matmul(B.transpose(0, 1).contiguous(), X).contiguous();
                    auto Y = at::matmul(W, X);
                    const dim3 block({block_x}, {block_y});
                    const dim3 grid((static_cast<unsigned int>(d) + {block_x} - 1) / {block_x},
                                    (static_cast<unsigned int>(d) + {block_y} - 1) / {block_y});
                    rank16_update_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                        Y.data_ptr<float>(), A.data_ptr<float>(), T.data_ptr<float>(), d);
                    const cudaError_t err = cudaGetLastError();
                    TORCH_CHECK(err == cudaSuccess, "rank16_update_kernel failed: ", cudaGetErrorString(err));
                    return Y;
                }}
                """
            ).strip()
            + "\n\n"
            + self._module_def()
            + "\n"
        )

    def _cublas_source(self, use_gemmex: bool, beta_accumulate: bool) -> str:
        gemm_call = "gemmex_or_sgemm" if use_gemmex else "checked_sgemm"
        accumulate = (
            dedent(
                f"""
                    const float beta_one = 1.0f;
                    {gemm_call}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                d, d, 16,
                                &alpha, T.data_ptr<float>(), d,
                                A.data_ptr<float>(), 16,
                                &beta_one, Y.data_ptr<float>(), d);
                    return Y;
                """
            ).strip()
            if beta_accumulate
            else dedent(
                f"""
                    auto U = torch::empty_like(Y);
                    {gemm_call}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                d, d, 16,
                                &alpha, T.data_ptr<float>(), d,
                                A.data_ptr<float>(), 16,
                                &beta_zero, U.data_ptr<float>(), d);
                    return Y + U;
                """
            ).strip()
        )
        math_mode = (
            "    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);\n"
            if use_gemmex
            else ""
        )
        return (
            dedent(
                r"""
                #include <torch/extension.h>
                #include <ATen/ATen.h>
                #include <ATen/cuda/CUDAContext.h>
                #include <cublas_v2.h>
                #include <cuda_runtime.h>
                """
            ).strip()
            + "\n\n"
            + self._common_checks()
            + "\n\n"
            + dedent(
                r"""
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
                """
            ).strip()
            + "\n\n"
            + dedent(
                f"""
                torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {{
                    check_inputs(W, X, A, B);
                    const int d = static_cast<int>(W.size(0));
                    auto Y = torch::empty_like(W);
                    auto T = torch::empty({{16, d}}, W.options());
                    cublasHandle_t handle = get_handle();
                    check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()), "cublasSetStream");
{math_mode}                    const float alpha = 1.0f;
                    const float beta_zero = 0.0f;

                    // Row-major W@X as column-major X^T * W^T -> Y^T.
                    {gemm_call}(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                d, d, d,
                                &alpha, X.data_ptr<float>(), d,
                                W.data_ptr<float>(), d,
                                &beta_zero, Y.data_ptr<float>(), d);

                    // T = B^T@X. T row-major [16,d] is column-major [d,16].
                    {gemm_call}(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                d, 16, d,
                                &alpha, X.data_ptr<float>(), d,
                                B.data_ptr<float>(), 16,
                                &beta_zero, T.data_ptr<float>(), d);

                    {accumulate}
                }}
                """
            ).strip()
            + "\n\n"
            + self._module_def()
            + "\n"
        )
