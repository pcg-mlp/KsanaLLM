# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import torch

os.environ["PATH"] = "/usr/local/nvidia/lib64:" + os.environ["PATH"]

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(n_elements, meta['block_size']),  # noqa: E731
    )  # noqa: E731
    kernel = add_kernel[grid](x, y, output, n_elements, block_size=128)
    return output, kernel


if __name__ == "__main__":
    torch.manual_seed(0)
    SIZE = 98432
    x = torch.rand(SIZE, device='cuda')
    y = torch.rand(SIZE, device='cuda')
    output_triton, kernel = add(x, y)
    with open("triton_wrapper_test_kernel.cubin", "wb") as _f:
        _f.write(kernel.asm['cubin'])
    with open("triton_wrapper_test_kernel.ptx", "w") as _f:
        SHM_SIZE = 0
        try:
            SHM_SIZE = kernel.metadata["shared"]
        except TypeError:
            SHM_SIZE = kernel.metadata.shared
        KERNEL_NAME = "default"
        try:
            KERNEL_NAME = kernel.metadata["name"]
        except TypeError:
            KERNEL_NAME = kernel.metadata.name
        print("//shared_memory:", SHM_SIZE, end=", ", file=_f)
        print("kernel_name:", KERNEL_NAME, file=_f)
        print(kernel.asm['ptx'], file=_f)
    exit(0)
