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

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32

# Create random matrices A and B and an output matrix C

A = torch.full((M, K), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
B = torch.full((K, N), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
C = torch.full((M, N), 0.0, dtype=torch.float32, device="cuda")
A_f32 = A.to(torch.float32)
B_f32 = B.to(torch.float32)
golden = torch.matmul(A_f32, B_f32)


# Set the strides for each matrix
stride_am, stride_ak = A.stride()
stride_bk, stride_bn = B.stride()
stride_zm, stride_zn = C.stride()

# Set grid and block sizes for the kernel launch
block = (128, 1, 1)
blocks_per_grid_x = int(ceildiv(M, BLOCK_M) * ceildiv(N, BLOCK_N))
grid = (blocks_per_grid_x, 1, 1)
shared_memory_size = 3200


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


print("Ref: ")
print(golden)

golden = torch.nn.functional.normalize(golden)
C = torch.nn.functional.normalize(C)
assert_close(C, golden, rtol=1e-2, atol=1e-3, check_dtype=False)
