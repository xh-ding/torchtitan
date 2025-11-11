from typing import Callable, Dict, Any

import torch
import triton
import triton.language as tl
import math


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret

# ---- kernel ----
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tp_persistent(
    A_ptr, B_ptr, C_ptr,
    Scratch_ptr, Table_ptr, Count_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sg, stride_si, stride_sm, stride_sn,   # scratch strides
    stride_tg, stride_ti,
    stride_cg, stride_ci,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LEVEL_K: tl.constexpr,                        # = log2(TILE_K)
    TILE_K: tl.constexpr,                         # = K//BLOCK_K
    GRID_N: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if A_LARGE:
        offs_m = offs_m.to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if B_LARGE:
        offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)

    if A_LARGE or B_LARGE:
        row = row.to(tl.int64)
        col = col.to(tl.int64)

    scratch_base = Scratch_ptr + (pid_m*GRID_N+pid_n)*stride_sg
    table_base = Table_ptr + (pid_m*GRID_N +pid_n)*stride_tg
    count_base = Count_ptr + (pid_m*GRID_N +pid_n)*stride_cg

    mask_tile = (row[:, None] < BLOCK_M) & (col[None, :] < BLOCK_N)

    for s_tile_idx in range(0, TILE_K):
        k0 = s_tile_idx * BLOCK_K
        # 这里如果用和Scratch不匹配的精度就 还会有误差 不知道为什么
        acc = tl.zeros((BLOCK_M,BLOCK_N), dtype=ACC_DTYPE)
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + ((k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc = acc + tl.dot(a, b).to(ACC_DTYPE)

        break_flag = 0
        level = 0
        while level < LEVEL_K and break_flag == 0:
            # count最低位+1
            table_level_ptr = table_base+level*stride_ti
            count_level_ptr = count_base+level*stride_ci

            count_value = tl.load(count_level_ptr)

            table_value = tl.load(table_level_ptr)

            carry_over = (table_value == (count_value+1)).to(tl.int1)

            tmp_acc_ptr = scratch_base + level * stride_si + (
                row[:, None] * stride_sm + col[None, :] * stride_sn
            )

            tmp_acc = tl.load(tmp_acc_ptr).to(ACC_DTYPE)
            acc = tmp_acc + acc

            # 进位了
            if carry_over:
                count_value = 0
                tl.store(tmp_acc_ptr, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=mask_tile)
            else:
                count_value += 1
                break_flag = 1
                tl.store(tmp_acc_ptr, acc, mask=mask_tile)
            tl.store(count_level_ptr, count_value)

            level += 1
        # 完成reduce了 此时的break_flag应该为0 同时level == LEVEL_K
        if s_tile_idx == TILE_K - 1:
            c_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
            tl.store(c_ptr, acc.to(OUT_DTYPE), mask=mask_c)

def _get_tl_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16

def matmul_tp_persistent(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor=None):
    assert A.shape[-1] == B.shape[-2], "Dim doesn't match"

    out_dtype = A.dtype
    acc_dtype = A.dtype

    configs = {
        torch.bfloat16: {
            # "BLOCK_SIZE_M": 64,
            # "BLOCK_SIZE_N": 64,
            # "BLOCK_SIZE_K": 128,
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 2,
            "num_warps": 4,
        },
        torch.float16: {
            # "BLOCK_SIZE_M": 64,
            # "BLOCK_SIZE_N": 64,
            # "BLOCK_SIZE_K": 128,
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 2,
            "num_warps": 4,
        },
        torch.float32: {
            # "BLOCK_SIZE_M": 32,
            # "BLOCK_SIZE_N": 32,
            # "BLOCK_SIZE_K": 128,
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_stages": 2,
            "num_warps": 4,
        },
    }

    BLOCK_M=configs[out_dtype]["BLOCK_SIZE_M"]
    BLOCK_N=configs[out_dtype]["BLOCK_SIZE_N"]
    BLOCK_K=configs[out_dtype]["BLOCK_SIZE_K"]
    num_stages=configs[out_dtype]["num_stages"]
    num_warps=configs[out_dtype]["num_warps"]

    M, K = A.shape
    _, N = B.shape
    assert K % BLOCK_K == 0
    T = K // BLOCK_K
    FIRST_LEVEL_BLOCK = T

    LEVEL_K = 1
    while FIRST_LEVEL_BLOCK>2 and FIRST_LEVEL_BLOCK %2 == 0:
        FIRST_LEVEL_BLOCK //= 2
        LEVEL_K += 1

    C = torch.empty((M, N), device=A.device, dtype=out_dtype)

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)

    # Scratch 用累加精度存储
    Scratch = torch.zeros(grid_m * grid_n, LEVEL_K, BLOCK_M, BLOCK_N,
                        device=A.device, dtype=acc_dtype)

    Table = torch.full((grid_m * grid_n, LEVEL_K), 2, device=A.device, dtype=torch.int32)
    Table[:, 0] = FIRST_LEVEL_BLOCK
    Count = torch.zeros((grid_m * grid_n, LEVEL_K), device=A.device, dtype=torch.int32)

    assert Table[0, :].prod() == T

    matmul_kernel_tp_persistent[grid](
        A, B, C,
        Scratch, Table, Count,
        M, N, K,
        *A.stride(), *B.stride(), *C.stride(),
        *Scratch.stride(), *Table.stride(), *Count.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        LEVEL_K=LEVEL_K, TILE_K=T,GRID_N=grid_n,
        ACC_DTYPE=_get_tl_dtype(acc_dtype),
        OUT_DTYPE=_get_tl_dtype(out_dtype),
        A_LARGE=A.numel()>2**31,
        B_LARGE=B.numel()>2**31,
        num_warps=num_warps, num_stages=num_stages
    )
    if bias is not None:
        C += bias
    return C