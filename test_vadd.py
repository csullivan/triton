import torch
import triton
import triton.language as tl
from torch.testing import assert_close

# Define the kernel for vector addition
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Compute the index of the element to process
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Only operate on indices that are within bounds
    mask = idx < N
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)

    # Perform element-wise addition
    c = a + b

    # Store the result
    tl.store(c_ptr + idx, c, mask=mask)


# Parameters
N = 1024  # Size of the vectors
BLOCK_SIZE = 128  # Number of elements each program should process

# Create random input vectors
a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")
golden = a + b

# Allocate output vector
c = torch.empty_like(a)

# Calculate grid size based on the total number of elements and block size
grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

# Launch the kernel with the appropriate grid configuration
pgm = add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)

with open("vadd.ptx", "w") as a:
    print(pgm.asm["ptx"], file=a)

assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)

# Print the result
print(c)
