"""
Block Scaled Matrix Multiplication
==================================
This tutorial demonstrates a Triton implementation of block scaled matrix multiplication
which is generic over FP4 and FP8 formats. The formats supported in the tutorial are the OCP microscaling
formats, including mxfp4 and mxfp8, as well as NVIDIA's nvfp4 format. These matrix multiplications
are accelerated by fifth generation tensor core instructions on CUDA devices with compute capability 10.

Users can run the tutorial with each of the supported formats by passing the `--format`
argument and can benchmark the performance of each by specifying matrix dimensions
and iteration steps.

.. code-block:: bash

    # FP4
    python 10-block-scaled-matmul.py --format nvfp4
    python 10-block-scaled-matmul.py --format mxfp4 --K_range 512 8192 --bench

    # FP8
    python 10-block-scaled-matmul.py --format mxfp8 --K_range 8192 16384 --K_step 2048 --bench

Future updates to this tutorial which support mixed precision block scaled matmul are planned.
"""

# %%
# Background
# ----------
#
# CUDA devices that support PTX 8.7 and later can utlize block scaled matrix multiply
# instructions. In order for low latency access to these scale factors in the fast
# inner loop over tensor core MMAs, it is important to ensure that the blocked
# scale factors are stored in a contiguous memory layout according to their access
# pattern.
#
# The block scaled matmul tensor core instructions compute the following product:
#
#     C = (A * scale_a) @ (B * scale_b)
#
# where scale_a and scale_b are the blocked scale factors for the A and B matrices.
# Under block scaled matmul, each scale factor is broadcast and multiplied across a
# vector of elements from the A and B matrices, usually along their respective K axes.
# The number of elements of A and B over which each scale factor is broadcast is herein
# refered to as the vector size (VEC_SIZE).
#
# In a linear row-major layout, the scale factors would take the shape
#
#     (M, K // VEC_SIZE) and (N, K // VEC_SIZE)   [1]
#
# in global memory. However, to avoid non-contiguous memory access, it is beneficial to
# instead store the scale factors in a packed block layout. For the LHS matrix this layout
# is given by
#
#     (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4)   [2].
#
# In this way, each tensor core MMA in the fast inner loop over K blocks can achieve contiguous
# access of a block of 128 rows of scale factors along the M axis, for each BLOCK_M x BLOCK_K
# subtile of the matrix A.
#
# In order to conform with Triton's language semantics for dot_scaled, the scale factors
# are prepared in the above 5D layout [2], but are then logically transposed and reshaped into
# the 2D layout [1] expected by tl.dot_scaled.
#
# For more detailed information on the scale factor layout, see
#  1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
#  2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
#

import argparse
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.tools.tensor_descriptor as td
import triton.profiler as proton
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

# import triton.runtime.driver as driver
# driver.set_tma_debug_enabled(1)  # Enable debugging

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2. * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_ptr, a_scale,  #
        b_ptr, b_scale,  #
        output_ptr,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_sk: tl.constexpr, stride_sb: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        stride_cm, stride_cn,
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
        ):  #

    if ELEM_PER_BYTE == 1:
        dtype = tl.float8e4nv
    elif ELEM_PER_BYTE == 2:
        dtype = tl.dtype("uint8")

    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K // ELEM_PER_BYTE], strides=[K // ELEM_PER_BYTE, 1], block_shape=[BLOCK_M, BLOCK_K // ELEM_PER_BYTE])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K // ELEM_PER_BYTE], strides=[K // ELEM_PER_BYTE, 1], block_shape=[BLOCK_N, BLOCK_K // ELEM_PER_BYTE])
    # c_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])

    # a_scale_desc = tl.make_tensor_descriptor(a_scale, shape=[M // 128, K // VEC_SIZE // 4, 2, 256], strides=[stride_sk, stride_sb, 256, 1], block_shape=[rep_m, rep_k, 2, 256])
    # b_scale_desc = tl.make_tensor_descriptor(b_scale, shape=[N // 128, K // VEC_SIZE // 4, 2, 256], strides=[stride_sk, stride_sb, 256, 1], block_shape=[rep_n, rep_k, 2, 256])
    # a_scale_desc = tl.make_tensor_descriptor(a_scale, shape=[M // 128, K // VEC_SIZE // 4, 4, 128], strides=[stride_sk, stride_sb, 128, 1], block_shape=[rep_m, rep_k, 4, 128])
    # b_scale_desc = tl.make_tensor_descriptor(b_scale, shape=[N // 128, K // VEC_SIZE // 4, 4, 128], strides=[stride_sk, stride_sb, 128, 1], block_shape=[rep_n, rep_k, 4, 128])
    a_scale_desc = tl.make_tensor_descriptor(a_scale, shape=[M // 128, K // VEC_SIZE // 4, 32, 16], strides=[stride_sk, stride_sb, 16, 1], block_shape=[rep_m, rep_k, 32, 16])
    b_scale_desc = tl.make_tensor_descriptor(b_scale, shape=[N // 128, K // VEC_SIZE // 4, 32, 16], strides=[stride_sk, stride_sb, 16, 1], block_shape=[rep_n, rep_k, 32, 16])

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0


    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0


    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):


        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])



        scale_a = a_scale_desc.load([offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([offs_scale_n, offs_scale_k, 0, 0])


        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4)

        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)


        if ELEM_PER_BYTE == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)
        offs_k += BLOCK_K // ELEM_PER_BYTE
        offs_scale_k += rep_k

    # c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator)


def block_scaled_matmul(a, a_scale, b, b_scale, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")


    grid = (triton.cdiv(M, configs["BLOCK_SIZE_M"]) * triton.cdiv(N, configs["BLOCK_SIZE_N"]), 1)

    scale_strides = [a_scale.stride(0), a_scale.stride(1), a_scale.stride(2), a_scale.stride(3)]

    block_scaled_matmul_kernel[grid](a, a_scale, b, b_scale, output, M, N, K, *scale_strides, output.stride(0), output.stride(1), dtype_dst,
                                     configs["ELEM_PER_BYTE"], configs["VEC_SIZE"], configs["BLOCK_SIZE_M"],
                                     configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
                                     rep_m, rep_n, rep_k, configs["num_stages"],
    )
                                    #  enable_warp_specialization=False,
                                    #  use_ttg_ws=False)
    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    # BLOCK_M = 128
    # BLOCK_N = 256
    # BLOCK_K = 256 if "fp4" in block_scale_type else 128
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE = 2 if "fp4" in block_scale_type else 1

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    # Similar to Hopper's wgmma symmetric fp8 instruction, the RHS is expected
    # to be in col-major layout for Blackwell's tcgen05.mma when using fp4 operands.
    # To conform to the expected semantics of tl.dot_scaled, (M, K) x (K, N),
    # the data is generated in col-major layout, packed along K for fp4, and then
    # logically transposed. Note that if one operand is of fp8 precision, unlike Hopper,
    # Blackwell supports both row-major and col-major layouts for the RHS matrix.
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type == "mxfp8":
        a_ref = a_ref.to(torch.float32)
        b_ref = b_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        # Pack two fp4 elements per byte along K
        a = a_ref.to_packed_tensor(dim=1)
        b = b_ref.to_packed_tensor(dim=1)
    b_ref = b_ref.to(torch.float32).T


    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)


    reference = None
    if compute_reference:
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)

            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE": ELEM_PER_BYTE,
        "VEC_SIZE": VEC_SIZE,
    }
    return a, a_scale, b, b_scale, rep_m, rep_n, rep_k, configs, reference


def validate_block_scaled(M, N, K, block_scale_type="nvfp4"):
    a, a_scale, b, b_scale, rep_m, rep_n, rep_k, configs, reference = initialize_block_scaled(M, N, K, block_scale_type,
                                                                                   compute_reference=True)
    output = block_scaled_matmul(a, a_scale, b, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type})")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    a, a_scale, b, b_scale, rep_m, rep_n, rep_k, configs, _ = initialize_block_scaled(M, N, K, block_scale_type,
                                                                                      compute_reference=False)
    _ = block_scaled_matmul(a, a_scale, b, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a, a_scale, b, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    proton.deactivate(0)
    print("Done benchmarking")


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8"], default="nvfp4")
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        torch.manual_seed(42)


    # validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format)
    # apic
    # validate_block_scaled(2048+128, 2048+256, 256*6,  block_scale_type=args.format)
    validate_block_scaled(2048+128, 2048+256, 8192,  block_scale_type=args.format)

    if args.bench:
        proton.start("block_scaled_matmul", hook="triton")
        # proton.start("test", hook="triton")
        for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
            bench_block_scaled(K, reps=10000, block_scale_type="nvfp4")
            bench_block_scaled(K, reps=10000, block_scale_type="mxfp4")
            bench_block_scaled(K, reps=10000, block_scale_type="mxfp8")
        proton.finalize()
        show_profile("block_scaled_matmul")
