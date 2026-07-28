#!/usr/bin/env bash
# Build the ksana_llm wheel for AWS H200 (SM90a only).
#
# Env vars (all optional):
#   CMAKE_GENERATOR  CMake generator, defaults to Ninja
#   CMAKE_ARGS       Extra -D options forwarded to cmake
#
# SM archs are pinned to 90a via setup.py's --with-sm flag (H200 only).
set -euo pipefail

export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
export CMAKE_ARGS="${CMAKE_ARGS:--Wno-dev \
  -DNCCL_INCLUDE_DIR=/usr/local/cuda/include \
  -DNCCL_LIB_DIR=/usr/local/cuda/lib64 \
  -DWITH_FLASH_ATTN_CACHE=ON \
  -DWITH_VLLM_FLASH_ATTN=ON \
  -DWITH_TRPC_ENDPOINT=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12}"
export BUILD_JOBS=48

# CUSTOM_KSANA_VERSION=0.1.0

numactl --membind=0 --cpunodebind=0 python setup.py bdist_wheel --with-sm 90a

echo
echo "Wheel built:"
ls -lh dist/*.whl
