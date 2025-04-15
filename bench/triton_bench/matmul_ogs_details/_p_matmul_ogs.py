import torch
import torch
import triton
import triton.language as tl
from triton_bench.meta import get_scaled_dot_format_string, inline_function, cuda_capability_geq
from triton_bench.mxfp import _unswizzle_mx_block
from triton_bench.numerics import float_to_flex, load_scale, nan_propagating_absmax_reduce, compute_scale
from ._common import make_matmul_repr, matmul_launch_metadata, swizzle2d, xcd_swizzle

# fmt: off
@triton.jit
def _make_tensor_desc(ptr, shape, strides, block_shape, transpose: tl.constexpr = False):
    tl.static_assert(len(shape) == len(strides))
    tl.static_assert(len(strides) == len(block_shape))
    if transpose:
        # Pass constexpr(1) to workaround torchflow tracer changing values of 1 to 2 during compile.
        # We check that the stride is actually 1 before launching the kernel.
        return tl.make_tensor_descriptor(
            ptr,
            shape=shape[:-2] + [shape[-1], shape[-2]],
            strides=strides[:-2] + [strides[-1], tl.constexpr(1)],
            block_shape=block_shape[:-2] + [block_shape[-1], block_shape[-2]],
        )
    else:
        # Pass constexpr(1) to workaround torchflow tracer changing values of 1 to 2 during compile.
        # We check that the stride is actually 1 before launching the kernel.
        return tl.make_tensor_descriptor(
            ptr,
            shape=shape,
            strides=strides[:-1] + [tl.constexpr(1)],
            block_shape=block_shape,
        )

@inline_function
def _load_tensor_desc(desc, offs, transpose: tl.constexpr = False, _builder=None):
    if transpose:
        offs = offs[:-2] + [offs[-1], offs[-2]]
    res = desc.load(offs, _builder=_builder)
    res = tl.reshape(res, desc.block_shape[-2:], _builder=_builder)
    if transpose:
        res = tl.trans(res, _builder=_builder)
    return res


# Helper function to recreate a TMA desc with the same fields, but with a new pointer and optional new shape.
@inline_function
def _update_tensor_desc(desc, ptr, shape=None, _builder=None):
    return tl.make_tensor_descriptor(
        ptr,
        shape=shape or desc.shape,
        # last dim must be constexpr 1; reflecting the old descriptor drops the constexpr
        strides=desc.strides[:-1] + [tl.constexpr(1)],
        block_shape=desc.block_shape,
        _builder=_builder,
    )

@triton.jit
def _load_tile_attrs(
    tile_id, num_tiles, grid_m, grid_n, padding_m,
    M, ExptData, ExptHist, ExptOffs,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr):
    # unpack and swizzle program ids
    pid_emnk = tile_id
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, num_tiles // SPLIT_K, XCD_SWIZZLE)
    pid_e = pid_emnk // ((grid_m - padding_m) * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % ((grid_m - padding_m) * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: tl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, (grid_m - padding_m), grid_n, GROUP_M)

    # unpack expert data
    if ExptData is None:
        tl.static_assert(M is not None)
        expt_id, start_z, start_m, block_id, eM = pid_e, pid_e, 0, pid_m, -1
    else:
        tl.static_assert(M is None)
        expt_data = tl.load(ExptData + pid_m)
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        eM = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        start_z = 0

    off_m = BLOCK_M * block_id
    off_n = BLOCK_N * pid_n

    return expt_id, start_z, start_m, eM, off_m, off_n, pid_k


@triton.jit
def _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, offs, mask):
    mask = mask & (offs < writeback_size)
    offs = tl.load(WriteBackIndx + offs, mask=mask, other=-1)
    mask = offs != -1
    return (offs, mask)


_matmul_ogs_repr = make_matmul_repr("_p_matmul_ogs", [0, 1, 2])
@triton.jit(repr=_matmul_ogs_repr, launch_metadata=matmul_launch_metadata)
def _p_matmul_ogs(
             Y, Out, stride_y_k, stride_y_z, stride_y_m, stride_y_n,
             YExpectedScale, YActualScale, YChecksumScale,
             X, stride_x_z, stride_x_m, stride_x_k,
             XScale,
             W, stride_w_e, stride_w_k, stride_w_n, W_TRANSPOSE: tl.constexpr,
             WScale,
             MxScale, stride_mx_e, stride_mx_k, stride_mx_n, MX_TRANSPOSE: tl.constexpr,
             B, stride_b_e, # Bias
             NRows, M, N, K, # shapes
             # expt data
             Betas, Gammas,
             GatherIndx,
             ScatterSrcIndx, num_idxs,
             WriteBackIndx, writeback_size,
             ExptHist, ExptOffs, ExptOffsSum, ExptData,
             # true grid size
             batch_size, grid_m, grid_n,
             # Out scale
             out_alpha,
             # MoE config
             N_EXPTS_TOT: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
             # precision config
             MAX_NUM_IMPRECISE_ACC: tl.constexpr, ALLOW_TF32: tl.constexpr,
             FLEXPOINT_SATURATE_INF: tl.constexpr,
             PER_BATCH_SCALE: tl.constexpr,
             # optimization config
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr, SWIZZLE_MX: tl.constexpr,
             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
             W_CACHE_MODIFIER: tl.constexpr,
             NUM_SMS: tl.constexpr,
             TOKENS_PER_EXPT_FOR_ANNOTATION=None,
             UPCAST_INDICES:tl.constexpr=False,
             DISABLE_Y_TMA: tl.constexpr=False,
             SWAP_XW: tl.constexpr = False):
    Y = Out  # Y is passed for the purposes of annotation; replace it with Out

    VEC_SIZE: tl.constexpr = 32
    w_type: tl.constexpr = W.dtype.element_ty
    tl.static_assert(w_type == tl.uint8 or (w_type == tl.float8e4nv or w_type == tl.float8e5),
                        "mx_weight_ptr must be uint8")
    tl.static_assert(MxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
    tl.static_assert(BLOCK_K % VEC_SIZE == 0, "BLOCK_K must be a multiple of VEC_SIZE")

    # We have pack 2 fp4 values in a byte
    W_PACK_DIVISOR: tl.constexpr = 2 if W.dtype.element_ty == tl.uint8 else 1
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_PACK_DIVISOR


    padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64

    x_desc = tl.make_tensor_descriptor(
        X,
        # When M is ragged, we don't mask the input rows, but mask the accumulator result in the epilogue.
        # So shape[0] here is the global number of rows in the X matrix, which allows using an invariant descriptor.
        shape=[NRows, K],
        strides=[stride_x_m, stride_x_k],
        block_shape=[BLOCK_M, BLOCK_K]
    )


    w_desc = _make_tensor_desc(W,
        shape=[N_EXPTS_TOT if ExptData is not None else batch_size,
            (K + W_PACK_DIVISOR - 1) // W_PACK_DIVISOR, N],
        strides=[stride_w_e, stride_w_k, stride_w_n],
        block_shape=[1, PACKED_BLOCK_K_W, BLOCK_N],
        transpose=W_TRANSPOSE)


    PackedK = (K + VEC_SIZE - 1) // VEC_SIZE
    mx_desc = tl.make_tensor_descriptor(
        MxScale,
        shape=[
            N_EXPTS_TOT if ExptData is not None else batch_size,
            (N + 127) // 128, (PackedK + 3) // 4, 32, 4 * 4,
        ],
        strides=[stride_mx_e, stride_mx_n, stride_mx_k, 4 * 4, 1],
        block_shape=[1, BLOCK_N // 128, BLOCK_K // VEC_SIZE // 4, 32, 4 * 4]
    )

    k_tiles = tl.cdiv(K, BLOCK_K * SPLIT_K)
    num_tiles = batch_size * (grid_m - padding_m) * grid_n * SPLIT_K


    tile_id1 = tl.program_id(0) - NUM_SMS

    THREADS_PER_BLOCK: tl.constexpr = tl.extra.cuda.num_threads()
    local_absmax = tl.full([THREADS_PER_BLOCK], 0.0, tl.uint32)

    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr = False
    WARP_SPECIALIZE: tl.constexpr = True


    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER, warp_specialize=WARP_SPECIALIZE):
        expt_id, start_z, start_m, eM, off_m, off_n, pid_k = _load_tile_attrs(
            tile_id, num_tiles, grid_m, grid_n, padding_m,
            M, ExptData, ExptHist, ExptOffs,
            BLOCK_M, BLOCK_N, SPLIT_K,
            GROUP_M, XCD_SWIZZLE)

        # Base pointers and offsets. These will be DCE'ed if unused in the TMA path.
        XBase = X + start_z.to(index_type) * stride_x_z
        offs_x_k = tl.arange(0, BLOCK_K)[None, :] * stride_x_k
        if SPLIT_K > 1:
            offs_x_k += pid_k.to(index_type) * BLOCK_K * stride_x_k
        offs_w_n = off_n + tl.arange(0, BLOCK_N)
        offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)


        x_desc = _update_tensor_desc(x_desc, XBase)

        acc = tl.zeros((BLOCK_N, BLOCK_M) if SWAP_XW else (BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in tl.range(k_tiles, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER):
            off_k = pid_k * BLOCK_K + ki * BLOCK_K * SPLIT_K
            off_k_w = pid_k * PACKED_BLOCK_K_W + ki * PACKED_BLOCK_K_W * SPLIT_K

            x = x_desc.load([start_m + off_m, off_k])

            w = _load_tensor_desc(w_desc, [expt_id, off_k_w, off_n], transpose=W_TRANSPOSE)

            x_format: tl.constexpr = get_scaled_dot_format_string(x.dtype)
            mx_format: tl.constexpr = get_scaled_dot_format_string(w.dtype)
            x_scales = tl.full((BLOCK_M, BLOCK_K // VEC_SIZE), 127, dtype=tl.uint8)
            w_scales = mx_desc.load([expt_id, off_n // 128, ki * (BLOCK_K // VEC_SIZE // 4 * SPLIT_K), 0, 0])
            w_scales = w_scales.reshape((w_scales.shape[1], w_scales.shape[2] * 32 * 4 * 4))
            w_scales = w_scales.reshape((BLOCK_N // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4))
            w_scales = w_scales.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
            acc = tl.dot_scaled(x, x_scales, x_format, w, w_scales, mx_format, acc=acc, fast_math=True)


        tile_id1 += NUM_SMS
        expt_id1, start_z1, start_m1, eM1, off_m1, off_n1, pid_k1 = _load_tile_attrs(
            tile_id1, num_tiles, grid_m, grid_n, padding_m,
            M, ExptData, ExptHist, ExptOffs,
            BLOCK_M, BLOCK_N, SPLIT_K,
            GROUP_M, XCD_SWIZZLE)

        # Determine output row offsets and mask
        offs_m = off_m1 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M if M is not None else offs_m < eM1


        YBase = Y + start_z1.to(index_type) * stride_y_z + start_m1.to(index_type) * stride_y_m
        y_desc = tl.make_tensor_descriptor(
            YBase + pid_k1.to(index_type) * stride_y_k,
            shape=[M if M is not None else eM1, N],
            strides=[stride_y_m, stride_y_n],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        # bias + scale
        offs_y_n = off_n1 + tl.arange(0, BLOCK_N)
        mask_n = offs_y_n < N
        BPtrs = B + expt_id1 * stride_b_e + offs_y_n
        if pid_k1 == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)

        betas = tl.full([BLOCK_M], 1, dtype=tl.float32)

        gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        x_scale = load_scale(XScale)
        w_scale = load_scale(WScale)
        acc *= x_scale * w_scale
        acc = acc + bias[None, :] * betas[:, None]
        acc *= gammas[:, None]
        if out_alpha is not None:
            acc *= out_alpha

        acc_view = tl.reshape(
            acc, [acc.numel // THREADS_PER_BLOCK, THREADS_PER_BLOCK], can_reorder=True)
        local_absmax = tl.maximum(local_absmax, nan_propagating_absmax_reduce(acc_view, axis=0))
        acc = float_to_flex(acc, YExpectedScale,
                            None, # ActualScale: local absmax is tracked and updated after the loop
                            YChecksumScale,
                            None, # mask: acc is manually masked to 0
                            Y, FLEXPOINT_SATURATE_INF)



        y_desc.store([off_m1, off_n1], acc.to(Y.dtype.element_ty))
    if YActualScale is not None:
        tl.atomic_max(YActualScale, compute_scale(local_absmax.to(tl.float32, bitcast=True), Y), sem="relaxed")


_per_device_alloc_fns = {}
def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}
        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if stream not in _per_stream_tensors or _per_stream_tensors[stream].numel() < size:
                _per_stream_tensors[stream] = torch.empty(size, device=device, dtype=torch.int8)
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]
