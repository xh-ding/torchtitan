import torch
from torch.nn import Parameter, init
import math
from torchtitan.experiments.deterministic_vllm_rl.tp_invariant.tp_invariant_backward import (
    TPInvariantLinearFunctionWithBackward,
)

class TPInvariantLinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            b, t, d = input.shape
            input_2d = input.reshape(b * t, d)
        else:
            input_2d = input

        output_2d = TPInvariantLinearFunctionWithBackward.apply(input_2d, self.weight, self.bias)
        return output_2d.reshape(b, t, self.out_features) if is_3d else output_2d

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"