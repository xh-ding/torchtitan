# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test the correctness of TPInvariantLinearLayer and TPInvariantLinearFunctionWithBackward.
"""

import torch
import sys
from torchtitan.experiments.deterministic_vllm_rl.tp_invariant.tp_invariant_backward import (
    TPInvariantLinearLayer,
    TPInvariantLinearFunctionWithBackward,
)


def test_forward_basic():
    """Test the basic forward propagation functionality."""
    print("\n" + "=" * 80)
    print("Test 1: Basic forward propagation")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    in_features = 64
    out_features = 128

    # Create model and input
    layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

    # Forward propagation
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {layer.weight.shape}")

    assert output.shape == (batch_size, out_features), f"Output shape error: {output.shape}"
    assert output.dtype == dtype, f"Output dtype error: {output.dtype}"
    print("✅ Basic forward propagation test passed!")


def test_backward_basic():
    """Test the basic backward propagation functionality."""
    print("\n" + "=" * 80)
    print("Test 2: Basic backward propagation")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    in_features = 64
    out_features = 128

    # Create model and input
    layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)

    # Forward propagation
    output = layer(x)
    loss = output.sum()

    # Backward propagation
    loss.backward()

    print(f"Input gradient shape: {x.grad.shape if x.grad is not None else None}")
    print(f"Weight gradient shape: {layer.weight.grad.shape if layer.weight.grad is not None else None}")
    print(f"Bias gradient shape: {layer.bias.grad.shape if layer.bias.grad is not None else None}")

    assert x.grad is not None, "Input gradient should not be None"
    assert layer.weight.grad is not None, "Weight gradient should not be None"
    assert layer.bias.grad is not None, "Bias gradient should not be None"
    assert x.grad.shape == x.shape, f"Input gradient shape error: {x.grad.shape}"
    assert layer.weight.grad.shape == layer.weight.shape, f"Weight gradient shape error: {layer.weight.grad.shape}"
    assert layer.bias.grad.shape == layer.bias.shape, f"Bias gradient shape error: {layer.bias.grad.shape}"
    print("✅ Basic backward propagation test passed!")


def test_compare_with_standard_linear():
    """Compare with standard PyTorch Linear layer."""
    print("\n" + "=" * 80)
    print("Test 3: Compare with standard PyTorch Linear layer")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 8
    in_features = 128
    out_features = 256

    # Create same weights and biases
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

    # Create custom layer and standard layer
    custom_layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    standard_layer = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )

    # Set same weights and biases
    with torch.no_grad():
        custom_layer.weight.copy_(weight)
        custom_layer.bias.copy_(bias)
        standard_layer.weight.copy_(weight)
        standard_layer.bias.copy_(bias)

    # Forward propagation
    custom_output = custom_layer(x)
    standard_output = standard_layer(x)

    print(f"Custom layer output shape: {custom_output.shape}")
    print(f"Standard layer output shape: {standard_output.shape}")
    print(f"Output difference (max): {(custom_output - standard_output).abs().max().item():.6f}")
    print(f"Output difference (mean): {(custom_output - standard_output).abs().mean().item():.6f}")

    # Note: Since Triton kernel is used, there may be numerical differences, but they should be within a reasonable range
    max_diff = (custom_output - standard_output).abs().max().item()
    # For bfloat16, allow some numerical differences
    assert max_diff < 1.0, f"Output difference too large: {max_diff}"
    print("✅ Compare with standard Linear layer test passed!")


def test_gradient_accuracy():
    """Test the accuracy of gradient calculation."""
    print("\n" + "=" * 80)
    print("Test 4: Gradient calculation accuracy")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    in_features = 64
    out_features = 128

    # Create model and input
    custom_layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    standard_layer = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )

    # Set same weights and biases
    with torch.no_grad():
        weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
        bias = torch.randn(out_features, device=device, dtype=dtype)
        custom_layer.weight.copy_(weight)
        custom_layer.bias.copy_(bias)
        standard_layer.weight.copy_(weight)
        standard_layer.bias.copy_(bias)

    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    x_standard = x.clone().detach().requires_grad_(True)

    # Forward propagation
    custom_output = custom_layer(x)
    standard_output = standard_layer(x_standard)

    # Backward propagation
    custom_loss = custom_output.sum()
    standard_loss = standard_output.sum()

    custom_loss.backward()
    standard_loss.backward()

    # Compare gradients
    print(f"Input gradient difference (max): {(x.grad - x_standard.grad).abs().max().item():.6f}")
    print(f"Weight gradient difference (max): {(custom_layer.weight.grad - standard_layer.weight.grad).abs().max().item():.6f}")
    print(f"Bias gradient difference (max): {(custom_layer.bias.grad - standard_layer.bias.grad).abs().max().item():.6f}")

    # Check gradient shapes
    assert x.grad.shape == x_standard.grad.shape, "Input gradient shape mismatch"
    assert custom_layer.weight.grad.shape == standard_layer.weight.grad.shape, "Weight gradient shape mismatch"
    assert custom_layer.bias.grad.shape == standard_layer.bias.grad.shape, "Bias gradient shape mismatch"

    # For bfloat16, allow some numerical differences
    max_grad_diff = max(
        (x.grad - x_standard.grad).abs().max().item(),
        (custom_layer.weight.grad - standard_layer.weight.grad).abs().max().item(),
        (custom_layer.bias.grad - standard_layer.bias.grad).abs().max().item(),
    )
    assert max_grad_diff < 1.0, f"Gradient difference too large: {max_grad_diff}"
    print("✅ Gradient calculation accuracy test passed!")


def test_without_bias():
    """Test without bias."""
    print("\n" + "=" * 80)
    print("Test 5: Without bias")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    in_features = 64
    out_features = 128

    # Create without bias model
    layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)

    # Forward propagation
    output = layer(x)
    loss = output.sum()

    # Backward propagation
    loss.backward()

    assert layer.bias is None, "Bias should be None"
    assert x.grad is not None, "Input gradient should not be None"
    assert layer.weight.grad is not None, "Weight gradient should not be None"
    print("✅ Without bias test passed!")


def test_different_dtypes():
    """Test different data types."""
    print("\n" + "=" * 80)
    print("Test 6: Different data types")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    batch_size = 4
    in_features = 64
    out_features = 128

    dtypes = [torch.bfloat16, torch.float16]
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        # Only GPUs with Tensor Core support float32
        dtypes.append(torch.float32)

    for dtype in dtypes:
        print(f"\nTest dtype: {dtype}")
        layer = TPInvariantLinearLayer(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            device=device,
            dtype=dtype,
        )
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)

        # Forward propagation
        output = layer(x)
        assert output.dtype == dtype, f"输出 dtype 错误: {output.dtype}"

        # Backward propagation
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, f"Input gradient should not be None (dtype={dtype})"
        assert layer.weight.grad is not None, f"Weight gradient should not be None (dtype={dtype})"
        assert layer.bias.grad is not None, f"Bias gradient should not be None (dtype={dtype})"
        print(f"✅ dtype {dtype} test passed!")


def test_different_shapes():
    """Test different input shapes."""
    print("\n" + "=" * 80)
    print("Test 7: Different input shapes")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    in_features = 128
    out_features = 256

    # Test different batch sizes
    test_shapes = [
        (1, in_features),  # Single sample
        (4, in_features),  # Small batch
        (32, in_features),  # Medium batch
        (128, in_features),  # Large batch
    ]

    layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )

    for batch_size, in_feat in test_shapes:
        print(f"\nTest shape: ({batch_size}, {in_feat})")
        x = torch.randn(batch_size, in_feat, device=device, dtype=dtype, requires_grad=True)

        # Forward propagation
        output = layer(x)
        assert output.shape == (batch_size, out_features), f"Output shape error: {output.shape}"

        # Backward propagation
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, f"Input gradient should not be None (shape={x.shape})"
        assert x.grad.shape == x.shape, f"Input gradient shape error: {x.grad.shape}"
        print(f"✅ Shape ({batch_size}, {in_feat}) test passed!")


def test_function_directly():
    """Test Function class directly."""
    print("\n" + "=" * 80)
    print("Test 8: Test Function directly")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4
    in_features = 64
    out_features = 128

    x = torch.randn(batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(out_features, device=device, dtype=dtype, requires_grad=True)

    # Use Function directly
    output = TPInvariantLinearFunctionWithBackward.apply(x, weight, bias)

    assert output.shape == (batch_size, out_features), f"Output shape error: {output.shape}"

    # Backward propagation
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient should not be None"
    assert weight.grad is not None, "Weight gradient should not be None"
    assert bias.grad is not None, "Bias gradient should not be None"
    print("✅ Function directly test passed!")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Test TPInvariantLinearLayer and TPInvariantLinearFunctionWithBackward")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping all tests")
        return

    try:
        test_forward_basic()
        test_backward_basic()
        test_compare_with_standard_linear()
        test_gradient_accuracy()
        test_without_bias()
        test_different_dtypes()
        test_different_shapes()
        test_function_directly()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed, error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

