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


def test_tp_invariant_output():
    """Test that TPInvariantLinearLayer produces identical outputs across different TP configurations (simulated with matrix chunking)."""
    print("\n" + "=" * 80)
    print("Test 9: TP Invariant Output (Different TP configurations)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 8
    in_features = 256
    out_features = 512

    # Create reference weight and bias
    torch.manual_seed(42)  # For reproducibility
    weight_full = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias_full = torch.randn(out_features, device=device, dtype=dtype)
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

    # Create reference layer with full weight
    ref_layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        ref_layer.weight.copy_(weight_full)
        ref_layer.bias.copy_(bias_full)

    # Get reference output
    ref_output = ref_layer(x)

    # Test different TP degrees (simulating column-wise parallelism)
    tp_degrees = [1, 2, 4, 8]
    
    for tp_degree in tp_degrees:
        if out_features % tp_degree != 0:
            print(f"⚠️  Skipping TP={tp_degree} (out_features={out_features} not divisible by {tp_degree})")
            continue

        print(f"\nTesting TP={tp_degree} (Column-wise parallelism simulation)")

        # Split weight and bias along output features (column-wise parallelism)
        # Each TP rank gets a chunk of out_features
        out_features_per_rank = out_features // tp_degree
        tp_outputs = []

        for rank in range(tp_degree):
            start_idx = rank * out_features_per_rank
            end_idx = (rank + 1) * out_features_per_rank

            # Get weight chunk for this rank
            weight_chunk = weight_full[start_idx:end_idx, :]  # Shape: [out_features_per_rank, in_features]
            bias_chunk = bias_full[start_idx:end_idx]  # Shape: [out_features_per_rank]

            # Create layer for this TP rank
            tp_layer = TPInvariantLinearLayer(
                in_features=in_features,
                out_features=out_features_per_rank,
                bias=True,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                tp_layer.weight.copy_(weight_chunk)
                tp_layer.bias.copy_(bias_chunk)

            # Forward pass for this rank
            tp_output = tp_layer(x)  # Shape: [batch_size, out_features_per_rank]
            tp_outputs.append(tp_output)

        # Concatenate outputs from all TP ranks (simulating all-gather)
        tp_combined_output = torch.cat(tp_outputs, dim=1)  # Shape: [batch_size, out_features]

        # Compare with reference output
        max_diff = (tp_combined_output - ref_output).abs().max().item()
        mean_diff = (tp_combined_output - ref_output).abs().mean().item()

        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Mean difference: {mean_diff:.10f}")

        # For TP invariant layer, outputs should be strictly identical (or very close due to numerical precision)
        # Since we're using the same computation, the results should match exactly
        # However, due to floating point arithmetic order, there might be tiny differences
        # We use a very strict tolerance (1e-5 for bfloat16)
        tolerance = 1e-5
        assert max_diff < tolerance, (
            f"TP={tp_degree}: Output mismatch! Max diff: {max_diff}, "
            f"Expected < {tolerance}. This indicates TPInvariantLinearLayer is not TP invariant."
        )

        # Also verify they are numerically very close (within bfloat16 precision)
        assert torch.allclose(tp_combined_output, ref_output, atol=tolerance, rtol=tolerance), (
            f"TP={tp_degree}: Outputs are not close enough. "
            f"This indicates TPInvariantLinearLayer is not TP invariant."
        )

        print(f"  ✅ TP={tp_degree} test passed!")

    print("\n✅ TP Invariant Output test passed!")


def binary_tree_sum(tensors):
    """
    Sum tensors in binary tree order (simulating all-reduce operation).
    
    This function performs reduction in a binary tree fashion:
    - Pairs adjacent tensors and sums them
    - Repeats until only one tensor remains
    
    Example for 4 tensors [a, b, c, d]:
        Level 1: a+b -> ab, c+d -> cd
        Level 2: ab+cd -> abcd
        Result: abcd
    
    This simulates the binary tree reduction pattern used in real all-reduce
    operations, which may have different numerical precision compared to
    sequential summation.
    
    Args:
        tensors: List of tensors to sum
        
    Returns:
        Sum of all tensors computed in binary tree order
    """
    if len(tensors) == 0:
        raise ValueError("Cannot sum empty list of tensors")
    if len(tensors) == 1:
        return tensors[0]
    
    # Make a copy to avoid modifying the original list
    current = list(tensors)
    
    # Perform binary tree reduction
    while len(current) > 1:
        next_level = []
        # Pair adjacent tensors and sum them
        for i in range(0, len(current) - 1, 2):
            next_level.append(current[i] + current[i + 1])
        # If odd number of tensors, keep the last one
        if len(current) % 2 == 1:
            next_level.append(current[-1])
        current = next_level
    
    return current[0]


def test_tp_invariant_output_rowwise():
    """Test that TPInvariantLinearLayer produces identical outputs with row-wise parallelism simulation."""
    print("\n" + "=" * 80)
    print("Test 10: TP Invariant Output (Row-wise parallelism simulation)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 8
    in_features = 512
    out_features = 256

    # Create reference weight and bias
    torch.manual_seed(43)  # Different seed for this test
    weight_full = torch.randn(out_features, in_features, device=device, dtype=dtype)
    bias_full = torch.randn(out_features, device=device, dtype=dtype)
    x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

    # Create reference layer with full weight
    ref_layer = TPInvariantLinearLayer(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        ref_layer.weight.copy_(weight_full)
        ref_layer.bias.copy_(bias_full)

    # Get reference output
    ref_output = ref_layer(x)

    # Test different TP degrees (simulating row-wise parallelism)
    # In row-wise parallelism, we split the input features (K dimension)
    tp_degrees = [1, 2, 4, 8]

    for tp_degree in tp_degrees:
        if in_features % tp_degree != 0:
            print(f"⚠️  Skipping TP={tp_degree} (in_features={in_features} not divisible by {tp_degree})")
            continue

        print(f"\nTesting TP={tp_degree} (Row-wise parallelism simulation)")

        # Split weight along input features (row-wise parallelism)
        # Each TP rank processes a chunk of input features
        in_features_per_rank = in_features // tp_degree
        tp_partial_outputs = []

        for rank in range(tp_degree):
            start_idx = rank * in_features_per_rank
            end_idx = (rank + 1) * in_features_per_rank

            # Get weight chunk for this rank (all output features, but chunk of input features)
            weight_chunk = weight_full[:, start_idx:end_idx]  # Shape: [out_features, in_features_per_rank]
            x_chunk = x[:, start_idx:end_idx]  # Shape: [batch_size, in_features_per_rank]

            # Create layer for this TP rank
            tp_layer = TPInvariantLinearLayer(
                in_features=in_features_per_rank,
                out_features=out_features,
                bias=False,  # No bias for partial computation in row-wise parallelism
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                tp_layer.weight.copy_(weight_chunk)

            # Forward pass for this rank (partial computation)
            tp_partial_output = tp_layer(x_chunk)  # Shape: [batch_size, out_features]
            tp_partial_outputs.append(tp_partial_output)

        # Sum partial outputs from all TP ranks (simulating all-reduce)
        # Use binary tree order summation to simulate realistic all-reduce operation
        tp_combined_output = binary_tree_sum(tp_partial_outputs)  # Shape: [batch_size, out_features]

        # Add bias (bias is replicated, so we add it once)
        tp_combined_output = tp_combined_output + bias_full

        # Compare with reference output
        max_diff = (tp_combined_output - ref_output).abs().max().item()
        mean_diff = (tp_combined_output - ref_output).abs().mean().item()

        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Mean difference: {mean_diff:.10f}")

        # Verify outputs are strictly identical (or very close)
        tolerance = 1e-5
        assert max_diff < tolerance, (
            f"TP={tp_degree}: Output mismatch! Max diff: {max_diff}, "
            f"Expected < {tolerance}. This indicates TPInvariantLinearLayer is not TP invariant."
        )

        assert torch.allclose(tp_combined_output, ref_output, atol=tolerance, rtol=tolerance), (
            f"TP={tp_degree}: Outputs are not close enough. "
            f"This indicates TPInvariantLinearLayer is not TP invariant."
        )

        print(f"  ✅ TP={tp_degree} test passed!")

    print("\n✅ TP Invariant Output (Row-wise) test passed!")


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
        test_tp_invariant_output()
        test_tp_invariant_output_rowwise()

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

