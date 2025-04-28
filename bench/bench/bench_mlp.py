from pathlib import Path
import json
import triton.profiler as proton
import torch
import triton_bench.swiglu
from triton_bench.numerics_details.mxfp import downcast_to_mxfp
from triton_bench.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_bench.numerics import InFlexData
from triton_bench.routing import routing
from triton_bench.target_info import is_hip, get_cdna_version
import argparse

if torch.cuda.is_available() and not is_hip():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def _query_gpu_specs():
    import subprocess
    if is_hip():
        cmd = ["rocm-smi", "--showproductname", "-d=0", "--csv"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        model = output.splitlines()[1].split(",")[2]
        if model in ["0x74a9", "0x74a1"]:
            name = "AMD Instinct MI300X"
        elif model == "0x74a5":
            name = "AMD Instinct MI325X"
        else:
            name = "AMD"
    else:
        cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i=0"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        name = output.splitlines()[0]

    gpu_specs = {
        "NVIDIA H100 80GB HBM3": {"MAX_TFLOPS8": 1979, "MAX_TFLOPS16": 989, "MAX_TBPS": 3.35},
        "HGX GB200": {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0},
        "NVIDIA B200": {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0},
        "AMD Instinct MI300X": {"MAX_TFLOPS8": 2615, "MAX_TFLOPS16": 1307, "MAX_TBPS": 5.3},
        "AMD Instinct MI325X": {"MAX_TFLOPS8": 2615, "MAX_TFLOPS16": 1307, "MAX_TBPS": 6.0},
    }
    return gpu_specs.get(name)


SPECS = _query_gpu_specs()


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt["swizzle_mx_scale"]
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis)
        return w, InFlexData(), MicroscalingCtx(weight_scale=mx_scales, swizzle_mx=swizzle_mx_scale,
                                                actual_weight_scale_shape=weight_scale_shape)


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype,
              # tensor / expert parallelism
              TP=1, EP=1, name="",
              # flags
              only_first_matmul=False, only_second_matmul=False,
              num_iterations=100, no_proton=False):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    dev = "cuda"

    # input
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot, ), device=dev)
    b1 = torch.randn((dim2 // TP, ), device=dev)
    b2 = torch.randn((dim1, ), device=dev)

    # -- numerics --
    optg = dict()
    opt1 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    opt2 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt2)
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    pcs = triton_bench.swiglu.PrecisionConfig(limit=1.0)
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))

    # -- benchmark --
    if no_proton:
        proton_active = False
        fpath = None
    else:
        fpath = Path(f"logs/{name}/{batch}-{dim1}-{dim2}-{n_expts_tot}-{n_expts_act}-{x_dtype}-{w_dtype}.hatchet")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        proton.start(str(fpath.with_suffix('')), hook="triton")
        proton_active = True

    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    x = torch.randn((batch, dim1), device=dev)
    xg = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
    x = x.to(x_dtype)
    # run layer
    if proton_active:
        proton.start(str(fpath.with_suffix('')), hook="triton")
    for i in range(num_iterations):
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg, role_tag=0)
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, simulated_ep=EP)
        else:
            rdata = gather_indx = scatter_indx = None
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, role_tag=1)
        x = triton_bench.swiglu.swiglu(x, 1.0, pcs)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2, role_tag=2)

    if proton_active:
        proton.finalize()

    # -- analyze --
    if not Path(f"{fpath}").exists():
        print(f"Proton file {fpath} not found. Skipping analysis.")
        return 0.0, 0.0, 0.0

    with open(f"{fpath}") as fd:
        data = json.load(fd)
        # TODO: this will be broken if kernels use scopes themselves
        # compute useful (a.k.a. matmul) bytes and flops
        matmuls = [
            x for x in data[0]["children"] if "_matmul" in x["frame"]["name"] and "metadata" not in x["frame"]["name"]
        ]
        tot_bytes = sum([x["metrics"]["bytes"] for x in matmuls])
        tot_flops = {w: sum([x["metrics"].get(f"flops{w}", 0) for x in matmuls]) for w in [8, 16]}
        # compute total time (incl. "not useful" work)
        # TODO: proton should really be recording that in the json instead of
        # relying on the user to aggregate
        tot_time = sum(x["metrics"].get("time (ns)", 0) for x in data[0]["children"])
        min_time_flops = min_time_bytes = 0
        if SPECS is not None:
            min_time_flops = sum([tot_flops[w] / SPECS[f"MAX_TFLOPS{w}"] for w in [8, 16]]) * 1e-3
            min_time_bytes = tot_bytes / SPECS["MAX_TBPS"] * 1e-3
            min_time = max(min_time_flops, min_time_bytes)
            util = min_time / tot_time
        else:
            util = 0.0
        tflops = sum([tot_flops[w] for w in [8, 16]]) / tot_time * 1e-3
        tbps = tot_bytes / tot_time * 1e-3
        print(f"Utilization: {util:.0%}; {tflops:>6.1f} TFLOPs, {tbps:.1f} TB/s")

    return util, tflops, tbps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark MLP Layer')

    # Configuration parameters
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--dim1', type=int, default=None, help='Input dimension (Hidden size)')
    parser.add_argument('--dim2', type=int, default=None, help='Intermediate dimension (FFN size)')
    parser.add_argument('--n_expts_tot', type=int, default=None, help='Total number of experts')
    parser.add_argument('--n_expts_act', type=int, default=None, help='Number of active experts')
    parser.add_argument('--x_dtype', type=str, default=None, choices=['fp16', 'bf16', 'fp8'], help='Activation data type')
    parser.add_argument('--w_dtype', type=str, default=None, choices=['bf16', 'fp8', 'mx4'], help='Weight data type')
    parser.add_argument('--TP', type=int, default=None, help='Tensor Parallelism degree')
    parser.add_argument('--EP', type=int, default=None, help='Expert Parallelism degree')

    # Configuration flags
    parser.add_argument('--dense', action='store_true', help='Use dense configuration (8192×8192×8192, 1 expert)')
    parser.add_argument('--llama4', action='store_true', help='Use llama4 configuration (2048×5120×8192, 128 experts, 4 active)')
    parser.add_argument('--fp8xmxfp4', action='store_true', help='Use fp8 for activations and mx4 for weights')
    parser.add_argument('--fp8xfp8', action='store_true', help='Use fp8 for activations and weights')

    # Benchmark control flags
    parser.add_argument('--only_first_matmul', action='store_true', help='Run only the first matmul operation')
    parser.add_argument('--only_second_matmul', action='store_true', help='Run only the second matmul operation')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations for the benchmark loop')
    parser.add_argument('--no_proton', action='store_true', help='Disable proton profiling')
    parser.add_argument('--num_sms', type=int, default=None, help='Number of SMs (informational)')

    parser.add_argument('--name', type=str, default="benchmark", help='Name for the benchmark run')

    args = parser.parse_args()

    # --- Validate mutually exclusive flags ---
    if args.only_first_matmul and args.only_second_matmul:
        raise ValueError("Flags --only_first_matmul and --only_second_matmul are mutually exclusive.")

    if args.dense and args.llama4:
        raise ValueError("Flags --dense and --llama4 are mutually exclusive.")

    # --- Set configuration based on flags ---
    # Default values depend on which model is selected
    if args.dense:
        # Dense configuration defaults
        batch_size = 8192 if args.batch_size is None else args.batch_size
        dim1 = args.dim1 if args.dim1 is not None else 8192
        dim2 = args.dim2 if args.dim2 is not None else 8192
        n_expts_tot = args.n_expts_tot if args.n_expts_tot is not None else 1
        n_expts_act = args.n_expts_act if args.n_expts_act is not None else 1
        tp = args.TP if args.TP is not None else 1
        ep = args.EP if args.EP is not None else 1
        run_name = "dense"
    elif args.llama4:
        # LLaMa 4 configuration defaults
        batch_size = 2048 if args.batch_size is None else args.batch_size
        dim1 = args.dim1 if args.dim1 is not None else 5120
        dim2 = args.dim2 if args.dim2 is not None else 8192
        n_expts_tot = args.n_expts_tot if args.n_expts_tot is not None else 128
        n_expts_act = args.n_expts_act if args.n_expts_act is not None else 4
        tp = args.TP if args.TP is not None else 4
        ep = args.EP if args.EP is not None else 1
        run_name = "llama4"
    else:
        raise ValueError("No configuration selected. Please use --dense or --llama4.")

    # Handle data type settings
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    qxdtype = "fp8" if has_native_mx4 else "bf16"  # Default quantized x type

    # Set datatypes based on flags
    if args.fp8xmxfp4:
        x_dtype = "fp8"
        w_dtype = "mx4"
    elif args.fp8xfp8:
        x_dtype = "fp8"
        w_dtype = "fp8"
    else:
        x_dtype = args.x_dtype if args.x_dtype is not None else "fp8"
        w_dtype = args.w_dtype if args.w_dtype is not None else "fp8"

    # --- Run Benchmark ---
    print(f"Running benchmark: {run_name}")
    print(f"Batch: {args.batch_size}, Dim1: {dim1}, Dim2: {dim2}")
    print(f"Experts: {n_expts_tot} total, {n_expts_act} active")
    print(f"Dtypes: x={x_dtype}, w={w_dtype}")
    print(f"Parallelism: TP={tp}, EP={ep}")
    print(f"Iterations: {args.num_iterations}, Only first matmul: {args.only_first_matmul}, Only second matmul: {args.only_second_matmul}")
    print(f"No proton: {args.no_proton}")

    util, tflops, tbps = bench_mlp(
        batch=batch_size,
        dim1=dim1,
        dim2=dim2,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        TP=tp,
        EP=ep,
        name=run_name,
        only_first_matmul=args.only_first_matmul,
        only_second_matmul=args.only_second_matmul,
        num_iterations=args.num_iterations,
        no_proton=args.no_proton
    )

    print(f"Result: Util={util}, TFLOPS={tflops}, TBps={tbps}")
