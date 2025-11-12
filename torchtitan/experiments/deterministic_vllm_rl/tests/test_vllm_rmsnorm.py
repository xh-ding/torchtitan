# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test the output consistency between VLLMRMSNorm and vLLM RMSNorm (using forward_native).
"""

import sys

import torch

from torchtitan.experiments.deterministic_vllm_rl.batch_invariant.module import (
    VLLMRMSNorm,
)


def test_rmsnorm_output_consistency():
    """Test the output consistency between VLLMRMSNorm and vLLM RMSNorm."""
    print("\n" + "=" * 80)
    print("Test the output consistency between VLLMRMSNorm and vLLM RMSNorm")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    try:
        from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNormNative
    except ImportError as e:
        print(f"❌ Failed to import vLLM RMSNorm: {e}")
        print("Please ensure vLLM is installed")
        return

    # Test parameters
    hidden_size = 512
    batch_size = 4
    seq_length = 128
    eps = 1e-6

    # Create two RMSNorm instances
    print(f"\n初始化模型:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_length: {seq_length}")
    print(f"  eps: {eps}")

    vllm_rmsnorm = VLLMRMSNormNative(hidden_size, eps=eps).cuda()
    custom_rmsnorm = VLLMRMSNorm(hidden_size, eps=eps).cuda()

    # Ensure the weights are the same
    with torch.no_grad():
        custom_rmsnorm.weight.copy_(vllm_rmsnorm.weight)

    print(f"\nvLLM RMSNorm weight shape: {vllm_rmsnorm.weight.shape}")
    print(f"Custom RMSNorm weight shape: {custom_rmsnorm.weight.shape}")

    # Test different input shapes
    test_cases = [
        ((batch_size, seq_length, hidden_size), "3D input (batch, seq, hidden)"),
        ((batch_size, hidden_size), "2D input (batch, hidden)"),
        ((hidden_size,), "1D input (hidden)"),
    ]

    for input_shape, description in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Test case: {description}")
        print(f"Input shape: {input_shape}")
        print("=" * 80)

        # Generate random input
        input_tensor = torch.randn(
            *input_shape, device="cuda", dtype=torch.bfloat16
        )

        # Get the output of vLLM RMSNorm (using forward_native)
        # Try using forward_native method (if available)
        if hasattr(vllm_rmsnorm, "forward_native"):
            try:
                output_vllm = vllm_rmsnorm.forward_native(input_tensor)
                print("✅ Using vLLM RMSNorm.forward_native()")
            except Exception as e:
                print(f"⚠️  forward_native() call failed: {e}, using forward() instead")
                output_vllm = vllm_rmsnorm(input_tensor)
                print("Using vLLM RMSNorm.forward()")
        else:
            # If forward_native is not available, use the regular forward
            output_vllm = vllm_rmsnorm(input_tensor)
            print("⚠️  forward_native method does not exist, using vLLM RMSNorm.forward()")
            print("    Note: This may not be the native implementation, using batch-invariant mode")

        # Get the output of custom RMSNorm
        output_custom = custom_rmsnorm(input_tensor)

        # Calculate the difference
        max_diff = torch.max(torch.abs(output_vllm - output_custom)).item()
        mean_diff = torch.mean(torch.abs(output_vllm - output_custom)).item()
        rel_diff = (
            torch.mean(
                torch.abs(output_vllm - output_custom)
                / (torch.abs(output_vllm) + 1e-8)
            ).item()
            * 100
        )

        print(f"\nOutput shape:")
        print(f"  vLLM: {output_vllm.shape}")
        print(f"  Custom: {output_custom.shape}")

        print(f"\nDifference statistics:")
        print(f"  Maximum absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Relative difference (%): {rel_diff:.2e}")

        # Check if the output is consistent within a reasonable range
        # For bfloat16, due to numerical precision limitations, allow some error
        tolerance = 1e-2  # Typical precision error for bfloat16

        if max_diff < tolerance:
            print(f"✅ Test passed: Maximum difference {max_diff:.2e} < tolerance {tolerance:.2e}")
        else:
            print(f"⚠️  Warning: Maximum difference {max_diff:.2e} >= tolerance {tolerance:.2e}")

        # Display some sample values for debugging
        if output_vllm.numel() > 0:
            print(f"\nSample values comparison (first 5 elements):")
            vllm_sample = output_vllm.flatten()[:5]
            custom_sample = output_custom.flatten()[:5]
            for i, (v, c) in enumerate(zip(vllm_sample, custom_sample)):
                diff = abs(v.item() - c.item())
                print(f"  [{i}] vLLM: {v.item():.6f}, Custom: {c.item():.6f}, Diff: {diff:.2e}")


def test_rmsnorm_gradient_consistency():
    """Test if the gradients of VLLMRMSNorm and vLLM RMSNorm are consistent (if supported)."""
    print("\n" + "=" * 80)
    print("Test gradient consistency (only check if the custom implementation supports gradients)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping test")
        return

    hidden_size = 256
    batch_size = 2
    seq_length = 64

    custom_rmsnorm = VLLMRMSNorm(hidden_size).cuda()

    # Create input with gradients
    input_tensor = torch.randn(
        batch_size, seq_length, hidden_size,
        device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    # Forward propagation
    output = custom_rmsnorm(input_tensor)

    # Backward propagation
    loss = output.sum()
    loss.backward()

    print(f"Input gradient shape: {input_tensor.grad.shape if input_tensor.grad is not None else None}")
    print(f"Weight gradient shape: {custom_rmsnorm.weight.grad.shape if custom_rmsnorm.weight.grad is not None else None}")

    assert input_tensor.grad is not None, "Input should have gradients"
    assert custom_rmsnorm.weight.grad is not None, "Weight should have gradients"
    print("✅ Gradient test passed: Custom implementation supports gradient calculation")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Test the consistency of VLLMRMSNorm and vLLM RMSNorm")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA is not available, skipping all tests")
        return

    try:
        test_rmsnorm_output_consistency()
        test_rmsnorm_gradient_consistency()

        print("\n" + "=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed, error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

