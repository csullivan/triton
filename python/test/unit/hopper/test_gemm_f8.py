# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import itertools
import os
import re

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # matmul_kernel_param_0
    b_ptr,  # matmul_kernel_param_1
    z_ptr,  # matmul_kernel_param_2
    stride_am,  # matmul_kernel_param_3
    stride_ak,  # matmul_kernel_param_4
    stride_bk,  # matmul_kernel_param_5
    stride_bn,  # matmul_kernel_param_6
    stride_zm,  # matmul_kernel_param_7
    stride_zn,  # matmul_kernel_param_8
    M,
    N,
    K,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  #
    out_dtype: tl.constexpr,
    A_ORDER_0: tl.constexpr,
    A_ORDER_1: tl.constexpr,  #
    B_ORDER_0: tl.constexpr,
    B_ORDER_1: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    block_offset_m = pid_m * BLOCK_M
    block_offset_n = pid_n * BLOCK_N

    a_tile_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(A_ORDER_0, A_ORDER_1),
    )
    b_tile_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, block_offset_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(B_ORDER_0, B_ORDER_1),
    )
    z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = block_offset_m + tl.arange(0, BLOCK_M)
    offs_n = block_offset_n + tl.arange(0, BLOCK_N)
    z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile_ptr, boundary_check=(0, 1))
        b = tl.load(b_tile_ptr, boundary_check=(0, 1))
        z += tl.dot(a, b)
        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])

    z = z.to(out_dtype)

    tl.store(z_ptrs, z, mask=mask)


@pytest.mark.parametrize(
    "BLOCK_M,BLOCK_N,BLOCK_K,NUM_WARPS,NUM_CTAS,M,N,K,TRANS_A,TRANS_B,TRANS_OUTPUT,epilogue,out_dtype,NUM_STAGES",
    [
        # loop over tile shapes and transpose combinations
        (
            *shape_w_c,
            trans_a,
            trans_b,
            trans_output,
            "none",
            out_dtype,
            num_stages,
        )
        for shape_w_c in [
            [64, 16, 32, 4, 1, 128, 256, 64],
        ]
        for out_dtype in ["float32"]
        for trans_a in [False]
        # for trans_b in [True]
        for trans_b in [False]
        for trans_output in [False]
        for num_stages in [3]
    ],
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9"
)
def test_gemm(
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    NUM_WARPS,
    NUM_CTAS,
    M,
    N,
    K,
    TRANS_A,
    TRANS_B,
    TRANS_OUTPUT,
    epilogue,
    out_dtype,
    NUM_STAGES,
):
    input_dtype = torch.float32
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K if K is None else K

    if TRANS_A:
        # a = torch.randn((K, M), device="cuda", dtype=input_dtype).T
        a = torch.full((K, M), 1.0, dtype=input_dtype, device="cuda").T
        a_order = [0, 1]
    else:
        # a = torch.randn((M, K), device="cuda", dtype=input_dtype)
        a = torch.full((M, K), 1.0, dtype=input_dtype, device="cuda")
        a_order = [1, 0]

    if TRANS_B:
        # b = torch.randn((N, K), device="cuda", dtype=input_dtype).T
        b = torch.full((N, K), 1.0, dtype=input_dtype, device="cuda").T
        b_order = [0, 1]
    else:
        # b = torch.randn((K, N), device="cuda", dtype=input_dtype)
        b = torch.full((K, N), 1.0, dtype=input_dtype, device="cuda")
        b_order = [1, 0]

    assert out_dtype == "float32"
    out_dtype = tl.float32
    torch_out_dtype = torch.float32

    if TRANS_OUTPUT:
        z = torch.full((N, M), 1.0, device="cuda", dtype=torch_out_dtype).T
        z_order = [0, 1]
    else:
        z = torch.full((M, N), 1.0, device="cuda", dtype=torch_out_dtype)
        z_order = [1, 0]

    # torch result
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    a_f32 = a.to(torch.float8_e5m2).to(torch.float16)
    b_f32 = b.to(torch.float8_e5m2).to(torch.float16)
    golden = torch.matmul(a_f32, b_f32).to(torch.float32)

    def grid(META):
        grid = (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        print("grid = ", grid)
        return grid

    # # TODO(csullivan): fp8
    # a = a.to(torch.float8_e5m2)
    # b = b.to(torch.float8_e5m2)
    print("stride_am=", a.stride(0))
    print("stride_ak=", a.stride(1))
    print("stride_bk=", b.stride(0))
    print("stride_bn=", b.stride(1))
    print("stride_zm=", z.stride(0))
    print("stride_zn=", z.stride(1))

    pgm = matmul_kernel[grid](
        a_ptr=a,  # 0
        b_ptr=b,  # 1
        z_ptr=z,  # 2
        stride_am=a.stride(0),  # 3
        stride_ak=a.stride(1),  #
        stride_bk=b.stride(0),  # 5
        stride_bn=b.stride(1),  #
        stride_zm=z.stride(0),  # 7
        stride_zn=z.stride(1),  #
        M=M,  # 9
        N=N,  # 10
        K=K,  # 11
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
        out_dtype=out_dtype,
        A_ORDER_0=a_order[0],
        A_ORDER_1=a_order[1],
        B_ORDER_0=b_order[0],
        B_ORDER_1=b_order[1],
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
    )

    with open("matmul.ptx", "w") as a:
        print(pgm.asm["ptx"], file=a)
    with open("matmul.ttir", "w") as a:
        print(pgm.asm["ttir"], file=a)
    with open("matmul.ttgir", "w") as a:
        print(pgm.asm["ttgir"], file=a)
    with open("matmul.llir", "w") as a:
        print(pgm.asm["llir"], file=a)
    torch.set_printoptions(profile="full")
    golden = torch.nn.functional.normalize(golden)
    z = torch.nn.functional.normalize(z)

    assert_close(z, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
