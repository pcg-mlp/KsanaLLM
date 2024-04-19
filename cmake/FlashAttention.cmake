# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule update to fetch FlashAttention")
find_package(Git QUIET)

set(FLAS_ATTN_PYTHON_SO, "")
set(FLAS_ATTN_VERSION, "")
set(FLAS_ATTN_MINOR_VERSION, "")
execute_process(COMMAND python -c "import torch,flash_attn_2_cuda;print(flash_attn_2_cuda.__file__)" OUTPUT_VARIABLE FLAS_ATTN_PYTHON_SO OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import flash_attn;print(flash_attn.__version__)" OUTPUT_VARIABLE FLAS_ATTN_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import flash_attn;print(flash_attn.__version__.split('.')[1])" OUTPUT_VARIABLE FLAS_ATTN_MINOR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "FLAS_ATTN_PYTHON_SO: ${FLAS_ATTN_PYTHON_SO}")

add_library(flash_attn_kernels UNKNOWN IMPORTED)
set_property(TARGET flash_attn_kernels PROPERTY IMPORTED_LOCATION "${FLAS_ATTN_PYTHON_SO}")
add_definitions("-DENABLE_FLASH_ATTN_2")
set(ENABLE_FLASH_ATTN_2 TRUE)
add_definitions("-DENABLE_FLASH_ATTN_MINOR_${FLAS_ATTN_MINOR_VERSION}")
message(STATUS "using flash attention ${FLAS_ATTN_VERSION} from python")