import torch
from torch import nn
from torchtitan.experiments.deterministic_vllm_rl.batch_invariant.batch_invariant_backward import (
    RMSNormFunction,
    SiluAndMulFunction,
)
from torchtitan.experiments.deterministic_vllm_rl.env_utils import vllm_is_tp_invariant
from torchtitan.experiments.deterministic_vllm_rl.tp_invariant.module import TPInvariantLinearLayer

class VLLMRMSNorm(nn.Module):
    """
    RMSNorm using vLLM's exact Triton kernel for bitwise determinism.
    Compatible with PyTorch's nn.RMSNorm interface but uses vLLM's implementation.

    Supports gradients through a custom autograd function that uses vLLM's
    kernel for forward and batch-invariant PyTorch ops for backward.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use vLLM's RMSNorm with gradient support for training
        return rms_norm_with_gradients(x, self.weight, self.eps)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class FeedForwardVLLMCompat(nn.Module):
    """
    FeedForward module compatible with vLLM implementation.
    Uses merged gate_up projection like vLLM.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        # Merged gate and up projections (like vLLM's gate_up_proj)
        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)

        # Down projection (like vLLM's down_proj)
        self.down_proj = TPInvariantLinearLayer(hidden_dim, dim, bias=False) if vllm_is_tp_invariant() else nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # Project to gate and up in one go
        gate_up = self.gate_up_proj(x)
        # Apply SiluAndMul activation with gradient support
        activated = silu_and_mul_with_gradients(gate_up)
        # Project down
        output = self.down_proj(activated)
        return output

    def init_weights(self, init_std: float):
        # Initialize like vLLM
        nn.init.trunc_normal_(self.gate_up_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=init_std)

# ============================================================================
# Public API for gradient-enabled vLLM operations
# ============================================================================


def rms_norm_with_gradients(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    RMS normalization with gradient support.

    Uses vLLM's Triton kernel for forward pass (deterministic) and
    batch-invariant PyTorch operations for backward pass.

    Args:
        input: Input tensor [*, hidden_size]
        weight: Weight tensor [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        output: Normalized and scaled tensor [*, hidden_size]
    """
    return RMSNormFunction.apply(input, weight, eps)


def silu_and_mul_with_gradients(x: torch.Tensor) -> torch.Tensor:
    """
    SiluAndMul activation with gradient support.

    Uses vLLM's implementation for forward pass (deterministic) and
    implements proper backward pass for training.

    Args:
        x: Input tensor [..., hidden_dim * 2] where first half is gate, second half is up

    Returns:
        output: silu(gate) * up, shape [..., hidden_dim]
    """
    return SiluAndMulFunction.apply(x)