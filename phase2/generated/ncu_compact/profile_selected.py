import math
import torch
from torch.utils.cpp_extension import load

torch.manual_seed(9090)
d = 4096
module = load(name='ncu_compact_selected', sources=['/workspace/optimized_lora.cu'], extra_cuda_cflags=['-O3'], extra_ldflags=['-lcublas'], with_cuda=True, verbose=False)
W = (torch.randn(d, d, device='cuda') / math.sqrt(d)).contiguous()
X = (torch.randn(d, d, device='cuda') / math.sqrt(d)).contiguous()
A = (torch.randn(d, 16, device='cuda') / 4.0).contiguous()
B = (torch.randn(d, 16, device='cuda') / 4.0).contiguous()
for _ in range(3):
    y = module.forward(W, X, A, B)
torch.cuda.synchronize()
print(float(y[0,0].item()))
