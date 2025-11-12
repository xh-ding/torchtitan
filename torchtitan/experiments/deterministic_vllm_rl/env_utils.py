import os


def vllm_is_batch_invariant():
    env_key = "VLLM_BATCH_INVARIANT"
    is_overridden = False
    val = os.getenv(env_key, "0")
    try:
        is_overridden = int(val) != 0
    except ValueError:
        is_overridden = False
    return is_overridden

def vllm_is_tp_invariant():
    env_key = "VLLM_TP_INVARIANT"
    is_overridden = False
    val = os.getenv(env_key, "0")
    try:
        is_overridden = int(val) != 0
    except ValueError:
        is_overridden = False
    if is_overridden:
        assert vllm_is_batch_invariant(), "Batch invariance must be enabled when TP invariance is enabled"
    return is_overridden

def vllm_check_invariant_compat():
    batch_key = "VLLM_BATCH_INVARIANT"
    tp_key = "VLLM_TP_INVARIANT"

    batch_val = os.getenv(batch_key, "0")
    tp_val = os.getenv(tp_key, "0")

    try:
        batch_flag = int(batch_val) != 0
    except ValueError:
        batch_flag = False

    try:
        tp_flag = int(tp_val) != 0
    except ValueError:
        tp_flag = False

    if tp_flag and batch_flag:
        flag = True
    else:
        flag = False

    return flag

def vllm_compatible_mode():
    key = "ALIGN_TRAIN_INFERENCE"
    is_overridden = False
    val = os.getenv(key, "0")
    try:
        is_overridden = int(val) != 0
    except ValueError:
        is_overridden = False
    if is_overridden:
        assert vllm_check_invariant_compat(), "TP invariance and batch invariance must be enabled under train-inference alignment mode"
    return is_overridden