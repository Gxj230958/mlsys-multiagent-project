#include <torch/extension.h>
#include <ATen/ATen.h>

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

torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    check_inputs(W, X, A, B);
    auto T = at::matmul(B.transpose(0, 1).contiguous(), X);
    return at::matmul(W, X) + at::matmul(A, T);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized LoRA forward");
}
