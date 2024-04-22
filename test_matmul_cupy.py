import torch
from torch.testing import assert_close
import cupy as cp
import ml_dtypes


def ceildiv(a, b):
    return -(a // -b)


# Define the path to the PTX file containing the matmul_kernel
ptx_code_path = "./matmul.ptx"

# Create a RawModule object to load the PTX code
raw_module = cp.RawModule(path=ptx_code_path)

# Retrieve the matmul kernel function from the PTX code
matmul_kernel = raw_module.get_function("matmul_kernel")

# Define the dimensions of the matrices
M, N, K = 128, 256, 64

BLOCK_M = 64
BLOCK_N = 16
BLOCK_K = 32

# Create random matrices A and B and an output matrix C

A = torch.full((M, K), 1.0, dtype=torch.float32, device="cuda")
B = torch.full((K, N), 1.0, dtype=torch.float32, device="cuda")
C = torch.full((M, N), 0.0, dtype=torch.float32, device="cuda")
# B = cp.ones((K, N)).astype(cp.float32)
# C = cp.zeros((M, N), dtype=cp.float32)
# A = cp.ones((M, K)).astype("float8_e5m2")
# B = cp.ones((K, N)).astype("float8_e5m2")
# C = cp.zeros((M, N), dtype="float8_e5m2")
# import ipdb

# ipdb.set_trace()


# Set the strides for each matrix
stride_am, stride_ak = A.stride()
stride_bk, stride_bn = B.stride()
stride_zm, stride_zn = C.stride()

# Set grid and block sizes for the kernel launch
block = (128, 1, 1)
blocks_per_grid_x = int(ceildiv(M, BLOCK_M) * ceildiv(N, BLOCK_N))
# grid = (blocks_per_grid_x, 1, 1)
grid = (32, 1, 1)

shared_memory_size = 32 * 1024


# Launch the kernel
matmul_kernel(
    grid,
    block,
    (
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        int(stride_am),
        int(stride_bk),
        int(stride_zm),
        M,
        N,
        K,
    ),
    shared_mem=shared_memory_size,
)

# Print the output matrix C
print("Output Matrix C:")
print(C)


A_torch = torch.from_numpy(cp.asnumpy(A))
B_torch = torch.from_numpy(cp.asnumpy(B))
golden = torch.matmul(A_torch, B_torch)

print("Ref: ")
print(golden)

golden = torch.nn.functional.normalize(golden)
res = torch.from_numpy(cp.asnumpy(C))
res = torch.nn.functional.normalize(res)
assert_close(res, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
