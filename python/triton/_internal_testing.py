import os
import re
import numpy as np
import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import _path_to_binary
import pytest

from numpy.random import RandomState
from typing import Optional, Union
from triton.runtime.jit import TensorWrapper, reinterpret, type_canonicalisation_dict

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
integral_dtypes = int_dtypes + uint_dtypes
float_dtypes = ['float16', 'float32', 'float64']
float_dtypes_with_bfloat16 = float_dtypes + ['bfloat16']
dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_float8_dtypes = ['float8_e4m3fn', 'float8_e5m2']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def get_current_target():
    if is_interpreter():
        return None
    return triton.runtime.driver.active.get_current_target()


def is_cuda():
    target = get_current_target()
    return False if target is None else target.backend == "cuda"


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hip():
    target = get_current_target()
    return False if target is None else target.backend == "hip"


def is_hip_mi200():
    target = get_current_target()
    if target is None or target.backend != 'hip':
        return False
    return target.arch == 'gfx90a'


def is_hip_mi300():
    target = get_current_target()
    if target is None or target.backend != 'hip':
        return False
    return target.arch in ('gfx940', 'gfx941', 'gfx942')


def is_hip_cdna():
    return is_hip_mi200() or is_hip_mi300()


def is_xpu():
    target = get_current_target()
    return False if target is None else target.backend == "xpu"


def get_arch():
    target = get_current_target()
    return "" if target is None else str(target.arch)


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Workaround. Never return zero so tests of division don't error out.
        return x
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32') & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device, dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def str_to_triton_dtype(x: str) -> tl.dtype:
    return tl.str_to_ty(type_canonicalisation_dict[x])


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def supports_tma(byval_only=False):
    if is_interpreter():
        return True
    if not is_cuda():
        return False
    _, cuda_version = _path_to_binary("ptxas")
    min_cuda_version = (12, 0) if byval_only else (12, 3)
    cuda_version_tuple = tuple(map(int, cuda_version.split(".")))
    assert len(cuda_version_tuple) == 2, cuda_version_tuple
    return torch.cuda.get_device_capability()[0] >= 9 and cuda_version_tuple >= min_cuda_version


def tma_skip_msg(byval_only=False):
    if byval_only:
        return "Requires __grid_constant__ TMA support (NVIDIA Hopper or higher, CUDA 12.0 or higher)"
    else:
        return "Requires advanced TMA support (NVIDIA Hopper or higher, CUDA 12.3 or higher)"


requires_tma = pytest.mark.skipif(not supports_tma(), reason=tma_skip_msg())


class MXFP4Tensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with four bit E2M1 floating point data as defined by the
        opencompute microscaling specification. c.f.
        https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp4e2m1 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def random(self):
        S = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)
        E = torch.randint(0, 4, size=self.size, dtype=torch.uint8, device=self.device)
        M = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)

        self.data = ((S << 3) | (E << 1) | M).type(torch.uint8)
        return self

    def to(self, dtype):
        """
        Convert fp4e2m1 data to float32.

        Returns:
        - A torch tensor of type dtype representing the fp4e2m1 data.
        """
        assert dtype == torch.float32, "Currently only float32 is supported for fp4e2m1 to float conversion"

        data = self.data
        S = ((data >> 3) & 0x1).type(dtype)
        E = ((data >> 1) & 0x3).type(dtype)
        M = (data & 0x1).type(dtype)

        # The MXF4 E2M1 spec defines 0bS000 as zero
        value = torch.zeros_like(S)
        is_zero = (E == 0) & (M == 0)
        non_zero_mask = ~is_zero
        if non_zero_mask.any():
            S_nz = S[non_zero_mask]
            E_nz = E[non_zero_mask]
            M_nz = M[non_zero_mask]

            sign = torch.pow(-1, S_nz)
            # Normal and subnormal handling for the exponent and mantissa
            exponent = torch.where(E_nz == 0, E_nz, E_nz - 1)
            mantissa = torch.where(E_nz == 0, M_nz * 0.5, 1.0 + M_nz * 0.5)
            value_nz = sign * torch.pow(2, exponent) * mantissa

            value[non_zero_mask] = value_nz

        # For zeros, the values must remain zero with the correct sign
        value[is_zero & (S == 1)] *= -1
        return value.type(torch.float32)

    def _from_float(self, values):
        """
        Convert float32 numbers to mxf4 e2m1 format.
        * No encodings are reserved for Inf or NaN in mxf4.
        * Conversion from float supports roundTiesToEven rounding mode.
        * If a value exceeds the mxf4 representable range after rounding,
          clamps to the maximum mxf4 magnitude, preserving the sign.
        * If a value has magnitude less than the minimum subnormal magnitude
          in mxf4 after rounding, converts to zero.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to fp4 format.
        """
        S = torch.signbit(values).type(torch.uint8)
        abs_values = torch.abs(values)

        is_zero = (abs_values == 0)
        is_invalid = torch.isnan(values) | torch.isinf(values)

        # Enumerate all possible E2M1 exponent and mantissa values. We will
        # use these to compare the distance between float32 and all possible
        # E2M1 floats to find the nearest E2M1 representable value
        E_bits = torch.tensor([0, 1, 2, 3], dtype=torch.uint8, device=self.device)
        M_bits = torch.tensor([0, 1], dtype=torch.uint8, device=self.device)

        candidate_values = []
        candidate_E = []
        candidate_M = []

        for E in E_bits:
            if E == 0:
                # Subnormals
                exponent = 0
                for M in M_bits:
                    significand = M * 0.5
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)
            else:
                # Normals
                exponent = E.item() - 1
                for M in M_bits:
                    significand = 1.0 + M * 0.5
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)

        candidates = torch.tensor(candidate_values, dtype=torch.float32, device=self.device)
        candidate_E = torch.tensor(candidate_E, dtype=torch.uint8, device=self.device)
        candidate_M = torch.tensor(candidate_M, dtype=torch.uint8, device=self.device)

        abs_values_flat = abs_values.view(-1)
        N = abs_values_flat.shape[0]
        abs_values_expanded = abs_values_flat.unsqueeze(1)

        # Clamp invalid values to the max e2m1 representable value
        max_candidate_value = candidates.max().item()
        abs_values_flat[is_invalid.view(-1)] = max_candidate_value

        # Compute distance between all abs_values and candidate e2m1 values
        errors = torch.abs(abs_values_expanded - candidates.unsqueeze(0))

        # To implement roundTiesToEven, we need to break ties by preferring
        # even mantissas (M == 0). We do so by adding an epsilon bias to shift
        # the closest candidate with an even mantissa closer to the float value
        min_errors, _ = torch.min(errors, dim=1, keepdim=True)
        is_tie = (errors == min_errors)
        # More than one candidate has the min error for some float value
        if is_tie.sum() > 1:
            M_bits_expanded = candidate_M.unsqueeze(0).expand(N, -1)
            tie_breaker = (M_bits_expanded == 0).type(torch.int32)

            errors = errors - (tie_breaker * 1e-6)

        best_indices = torch.argmin(errors, dim=1)

        E_selected = candidate_E[best_indices]
        M_selected = candidate_M[best_indices]
        E = E_selected.view(abs_values.shape)
        M = M_selected.view(abs_values.shape)

        E[is_zero] = 0
        M[is_zero] = 0

        return ((S << 3) | (E << 1) | M).type(torch.uint8)

    def to_packed_tensor(self, dim):
        """
        Packs two e2m1 elements into a single uint8 along the specified dimension.

        Parameters:
        - dim: The dimension along which to pack the elements.

        Returns:
        - A torch tensor of dtype uint8 with two e2m1 elements packed into one uint8.
        """
        data = self.data
        assert 0 <= dim < data.ndim, \
            "The dimension to pack along is not within the range of tensor dimensions"

        size_along_dim = data.size(dim)
        new_size_along_dim = (size_along_dim + 1) // 2

        # If the size is odd, we pad the data along dim with zeros at the end
        if size_along_dim % 2 != 0:
            pad_sizes = [0] * (2 * data.ndim)
            pad_index = (data.ndim - dim - 1) * 2 + 1
            pad_sizes[pad_index] = 1
            data = torch.nn.functional.pad(data, pad_sizes, mode='constant', value=0)

        new_shape = list(data.shape)
        new_shape[dim] = new_size_along_dim
        new_shape.insert(dim + 1, 2)  # packed dimension of length 2
        data = data.reshape(*new_shape)

        low = data.select(dim + 1, 0)
        high = data.select(dim + 1, 1)
        packed = (high << 4) | low

        return packed

    def unpack_packed_tensor(self, packed_tensor, dim, original_shape):
        """
        Unpacks a tensor where two fp4 elements are packed into a single uint8.

        Parameters:
        - packed_tensor: The packed tensor
        - dim: The dimension along which the tensor was packed.
        - original_shape: The shape of the original tensor before packing.

        Returns:
        - A tensor with the original data unpacked into uint8 elements containing one
          fp4e2m1 element in the least significant bits.
        """
        high = (packed_tensor >> 4) & 0xF
        low = packed_tensor & 0xF

        stacked = torch.stack((low, high), dim=dim + 1)

        # Flatten along dim and dim+1 and then merge
        shape = list(stacked.shape)
        new_shape = shape[:dim] + [shape[dim] * 2] + shape[dim + 2:]
        data = stacked.reshape(*new_shape)

        # Remove any padding
        if original_shape[dim] % 2 != 0:
            indices = [slice(None)] * data.ndim
            indices[dim] = slice(0, original_shape[dim])
            data = data[tuple(indices)]

        return data.type(torch.uint8)


class MXScaleTensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with microscaling E8M0 block scale factors.

        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp8e8m0 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def random(self, low=None, high=None):
        """
        Generate random E8M0 data within a specified range.
        * Excludes the NaN encoding (255).
        """
        bias = 127

        min_exponent = 0 if low is None else max(0, int(torch.log2(torch.tensor(low))) + bias)
        max_exponent = 254 if high is None else min(254, max(0, int(torch.log2(torch.tensor(high))) + bias))
        assert min_exponent <= max_exponent, "Low must be less than or equal to high"

        E = torch.randint(min_exponent, max_exponent + 1, size=self.size, dtype=torch.uint8, device=self.device)
        self.data = E
        return self

    def to(self, dtype):
        assert dtype == torch.float32, "Currently only float32 is supported for f8e8m0 to float conversion"
        data = self.data.type(dtype)
        is_nan = (data == 255)
        e_biased = data.clone()
        e_biased[is_nan] = 0
        e = e_biased - 127
        value = torch.pow(2.0, e)
        value[is_nan] = torch.nan
        return value.type(dtype)

    def _from_float(self, values):
        """
        Convert float32 numbers to E8M0 format.
        * Values <= 0, NaNs, and Infs are converted to the NaN encoding (255).
        * Positive values are converted by computing the floor of log2(value) to get the exponent.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to E8M0 format.
        """
        result = torch.empty_like(values, dtype=torch.uint8, device=self.device)

        is_invalid = torch.isnan(values) | torch.isinf(values) | (values <= 0)
        result[is_invalid] = 255

        valid_values = values[~is_invalid]
        e = torch.floor(torch.log2(valid_values))
        e_biased = e + 127
        e_biased_int = e_biased.type(torch.int32)
        e_biased_clamped = torch.clamp(e_biased_int, 0, 254)
        result[~is_invalid] = e_biased_clamped.type(torch.uint8)

        return result
