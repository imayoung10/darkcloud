import torch
import umalloc

class CopyUnifiedMemory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return umalloc.to_unified(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

def copy_to_unified(input: torch.Tensor) -> torch.Tensor:
    """
    CPU 텐서를 CUDA Unified Memory 영역으로 복사.
    (pinned memory가 아니라 managed memory 사용)
    """
    return CopyUnifiedMemory.apply(input)
