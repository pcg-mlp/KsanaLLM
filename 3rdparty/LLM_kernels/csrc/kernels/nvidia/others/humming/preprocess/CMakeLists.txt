# Copyright 2026 Tencent Inc. All rights reserved.
#
# Preprocessing kernels for humming weight format conversion
file(GLOB HUMMING_PREPROCESS_SRCS *.cu)
if(HUMMING_PREPROCESS_SRCS)
  add_library(llm_kernels_nvidia_kernel_humming_preprocess STATIC ${HUMMING_PREPROCESS_SRCS})
  set_property(TARGET llm_kernels_nvidia_kernel_humming_preprocess PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET llm_kernels_nvidia_kernel_humming_preprocess PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  # Resolve Python3 interpreter, prefer find_package result
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_FOUND)
    set(_HUMMING_PYTHON "${Python3_EXECUTABLE}")
  else()
    find_program(_HUMMING_PYTHON python3)
  endif()

  if(NOT _HUMMING_PYTHON)
    message(FATAL_ERROR "python3 interpreter not found, required for humming-kernels installation")
  endif()

  if(DEFINED ENV{DISABLE_FORCE_INSTALL_HUMMING})
    message(STATUS "DISABLE_FORCE_INSTALL_HUMMING is set, skipping auto-install of humming and triton")
  else()
    message(STATUS "Installing humming from git+https://github.com/inclusionAI/humming.git@d596b9f ...")
    execute_process(
      COMMAND ${_HUMMING_PYTHON} -m pip install git+https://github.com/inclusionAI/humming.git@d596b9f
      RESULT_VARIABLE HUMMING_INSTALL_RESULT
    )
    if(NOT HUMMING_INSTALL_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to install humming-kernels (exit code: ${HUMMING_INSTALL_RESULT}). "
                          "Please install manually: pip install git+https://github.com/inclusionAI/humming.git@d596b9f")
    endif()

    message(STATUS "Installing triton==3.2.0 ...")
    execute_process(
      COMMAND ${_HUMMING_PYTHON} -m pip install triton==3.2.0
      RESULT_VARIABLE TRITON_INSTALL_RESULT
    )
    if(NOT TRITON_INSTALL_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to install triton==3.2.0 (exit code: ${TRITON_INSTALL_RESULT}). "
                          "Please install manually: pip install triton==3.2.0")
    endif()
  endif()

  execute_process(
    COMMAND ${_HUMMING_PYTHON} -c "import humming; import os; print(os.path.join(os.path.dirname(humming.__file__), 'include'))"
    OUTPUT_VARIABLE HUMMING_PIP_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE HUMMING_PIP_RESULT
    ERROR_VARIABLE HUMMING_PIP_ERROR
  )

  if(HUMMING_PIP_RESULT EQUAL 0 AND HUMMING_PIP_INCLUDE_DIR)
    set(HUMMING_INCLUDE_DIR "${HUMMING_PIP_INCLUDE_DIR}")
    message(STATUS "Using humming pip package include: ${HUMMING_INCLUDE_DIR}")
  elseif(DEFINED ENV{DISABLE_FORCE_INSTALL_HUMMING})
    message(FATAL_ERROR "humming-kernels not found and DISABLE_FORCE_INSTALL_HUMMING is set. "
                        "Please install humming manually or unset DISABLE_FORCE_INSTALL_HUMMING:\n"
                        "  pip install git+https://github.com/inclusionAI/humming.git@d596b9f\n"
                        "  pip install triton==3.2.0")
  else()
    message(FATAL_ERROR "humming-kernels import failed after installation (${HUMMING_PIP_ERROR}). "
                        "Please install manually:\n"
                        "  pip install git+https://github.com/inclusionAI/humming.git@d596b9f\n"
                        "  pip install triton==3.2.0")
  endif()

  target_include_directories(llm_kernels_nvidia_kernel_humming_preprocess PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${HUMMING_INCLUDE_DIR}
  )
  target_link_libraries(llm_kernels_nvidia_kernel_humming_preprocess PUBLIC
    -lcudart llm_kernels_nvidia_utils ${TORCH_LIBRARIES})
endif()
