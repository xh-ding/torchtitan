from torchtitan.experiments.deterministic_vllm_rl.batch_invariant.batch_invariant_backward import (
    matmul_backward_impl,
    linear_backward_impl,
)

# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_mode = False
_batch_invariant_backward_lib = None


def enable_batch_invariant_backward_mode():
    """Enable batch invariant backward mode to support gradients.

    This function adds backward pass support to vLLM's existing batch_invariant
    implementations by registering the backward operations. vLLM handles all the
    forward passes, we just add gradient support.
    """
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_mode:
        return

    # Get vLLM's batch_invariant library (already created by init_batch_invariance)
    from vllm.model_executor.layers import batch_invariant as vllm_bi

    if (
        not hasattr(vllm_bi, "_batch_invariant_LIB")
        or vllm_bi._batch_invariant_LIB is None
    ):
        raise RuntimeError(
            "vLLM's batch_invariant mode is not initialized. "
            "Call init_batch_invariance() first."
        )

    # Use vLLM's existing library - don't destroy it!
    _batch_invariant_backward_lib = vllm_bi._batch_invariant_LIB

    # Just add the backward operations - everything else is already handled by vLLM
    _batch_invariant_backward_lib.impl(
        "aten::matmul_backward", matmul_backward_impl, "CUDA"
    )
    _batch_invariant_backward_lib.impl(
        "aten::linear_backward", linear_backward_impl, "CUDA"
    )

    _batch_invariant_backward_mode = True


def disable_batch_invariant_backward_mode():
    """Disable batch invariant backward mode."""
    global _batch_invariant_backward_mode, _batch_invariant_backward_lib

    if _batch_invariant_backward_lib is not None:
        _batch_invariant_backward_lib._destroy()

    _batch_invariant_backward_mode = False
    _batch_invariant_backward_lib = None


def is_batch_invariant_backward_mode_enabled():
    """Check if batch invariant backward mode is enabled."""
    return _batch_invariant_backward_mode