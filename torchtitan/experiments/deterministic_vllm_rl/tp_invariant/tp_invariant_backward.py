import torch
from torch.autograd import Function
from torchtitan.experiments.deterministic_vllm_rl.tp_invariant.tbik_matmul import matmul_tp_persistent

class TPInvariantLinearFunctionWithBackward(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        output = matmul_tp_persistent(x, weight.t(), bias)
        ctx.save_for_backward(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t() @ x
        if ctx.needs_input_grad[2] and bias is not None:
            reduce_dims = tuple(range(grad_output.dim() - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)
        return grad_x, grad_weight, grad_bias



