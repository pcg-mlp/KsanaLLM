# Copyright 2024 Tencent Inc.  All rights reserved.

import ctypes
import json
from typing import Any, Dict, List
from warnings import warn

# TODO define decorator to share the RuntimeError/CUDA_SUCCESS logic among different library functions

# One of the following libraries must be available to load
libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
for libname in libnames:
    try:
        cuda = ctypes.CDLL(libname)
    except OSError:
        continue
    else:
        break
else:
    raise ImportError(f'Could not load any of: {", ".join(libnames)}')

# Constants from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

# Conversions from semantic version numbers
# Borrowed from original gist and updated from the "GPUs supported" section of this Wikipedia article
# https://en.wikipedia.org/wiki/CUDA
SEMVER_TO_CORES = {
    (1, 0): 8,    # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,   # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,   # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,   # Volta
    (7, 2): 64,
    (7, 5): 64,   # Turing
    (8, 0): 64,   # Ampere
    (8, 6): 64,
}
SEMVER_TO_ARCH = {
    (1, 0): 'tesla',
    (1, 1): 'tesla',
    (1, 2): 'tesla',
    (1, 3): 'tesla',

    (2, 0): 'fermi',
    (2, 1): 'fermi',

    (3, 0): 'kepler',
    (3, 2): 'kepler',
    (3, 5): 'kepler',
    (3, 7): 'kepler',

    (5, 0): 'maxwell',
    (5, 2): 'maxwell',
    (5, 3): 'maxwell',

    (6, 0): 'pascal',
    (6, 1): 'pascal',
    (6, 2): 'pascal',

    (7, 0): 'volta',
    (7, 2): 'volta',

    (7, 5): 'turing',

    (8, 0): 'ampere',
    (8, 6): 'ampere',
}


def get_cuda_device_specs() -> List[Dict[str, Any]]:
    """Generate spec for each GPU device with format

    {
        'name': str,
        'compute_capability': (major: int32_t, minor: int32_t),
        'cores': int32_t,
        'cuda_cores': int32_t,
        'concurrent_threads': int32_t,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float
    }
    """

    # Type-binding definitions for ctypes
    num_gpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    free_mem = ctypes.c_size_t()
    total_mem = ctypes.c_size_t()
    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    # Check expected initialization
    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(f'cuInit failed with error code {result}: {error_str.value.decode()}')
    result = cuda.cuDeviceGetCount(ctypes.byref(num_gpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(f'cuDeviceGetCount failed with error code {result}: {error_str.value.decode()}')

    # Iterate through available devices
    device_specs = []
    for i in range(num_gpus.value):
        spec = {}
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError(f'cuDeviceGet failed with error code {result}: {error_str.value.decode()}')

        # Parse specs for each device
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            spec.update(name=name.split(b'\0', 1)[0].decode())
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            spec.update(compute_capability=(cc_major.value, cc_minor.value))
            spec.update(architecture=SEMVER_TO_ARCH.get((cc_major.value, cc_minor.value), 'unknown'))
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            spec.update(
                cores=cores.value,
                cuda_cores=cores.value * SEMVER_TO_CORES.get((cc_major.value, cc_minor.value), 'unknown'))
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                spec.update(concurrent_threads=cores.value * threads_per_core.value)
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
            spec.update(gpu_clock_mhz=clockrate.value / 1000.)
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
            spec.update(mem_clock_mhz=clockrate.value / 1000.)

        # Attempt to determine available vs. free memory
        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            warn(f'cuCtxCreate failed with error code {result}: {error_str.value.decode()}')
        else:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(free_mem), ctypes.byref(total_mem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
            if result == CUDA_SUCCESS:
                spec.update(
                    total_mem_mb=total_mem.value / 1024**2,
                    free_mem_mb=free_mem.value / 1024**2)
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                warn(f'cuMemGetInfo failed with error code {result}: {error_str.value.decode()}')
            cuda.cuCtxDetach(context)
        device_specs.append(spec)
    return device_specs


if __name__ == '__main__':
    gpu_properties = get_cuda_device_specs()
    if len(gpu_properties) > 0:
        print("".join([str(i) for i in gpu_properties[0]["compute_capability"]]))
    else:
        print("UNKNOWN")
