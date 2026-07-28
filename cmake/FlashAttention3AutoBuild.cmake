# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Flash Attention 3 Auto Build Module
# This module automatically builds and installs Flash Attention 3 from source

message(STATUS "Setting up Flash Attention 3 Auto Build")

# FA3 install source: git tag of Dao-AILab/flash-attention, hopper subdirectory.
set(FA3_GIT_URL "git+https://github.com/Dao-AILab/flash-attention.git@v2.8.2#subdirectory=hopper")

# Function to check if flash_attn_3 is available in Python
# Verifies both the namespace package and the compiled _C extension load
# (matching the runtime lookup in flash_attention_backend.cpp, which needs
# `flash_attn_3._C.__file__`). A zipped .egg produced by the deprecated
# `python setup.py install` passes the namespace check but cannot load the
# .so from inside the zip, so we must probe _C directly.
function(check_flash_attn_3_available result_var)
    execute_process(
        COMMAND python -c "import torch, flash_attn_3._C; print('available')"
        OUTPUT_VARIABLE FA3_CHECK_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if("${FA3_CHECK_OUTPUT}" STREQUAL "available")
        set(${result_var} TRUE PARENT_SCOPE)
        message(STATUS "Flash Attention 3 is already available in Python environment")
    else()
        set(${result_var} FALSE PARENT_SCOPE)
        message(STATUS "Flash Attention 3 is not available in Python environment")
    endif()
endfunction()

# Function to auto build Flash Attention 3
function(auto_build_flash_attn_3)
    message(STATUS "Starting Flash Attention 3 auto build process...")

    # pip delegates the git clone to the git CLI, so git must be on PATH
    find_package(Git REQUIRED)

    # Check if Python is available
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    message(STATUS "Building and installing Flash Attention 3 from ${FA3_GIT_URL} ...")
    # Install directly from the git tag: pip handles clone, checkout and
    # submodule initialization (hopper needs csrc/cutlass) in one step,
    # replacing the previous manual clone + checkout + local install flow.
    # Known trade-offs (accepted, since this function only runs when
    # flash_attn_3._C is not importable):
    #   - pip clones into a temporary directory and discards it after the
    #     build, so a failed build retries with a full recompile (no
    #     incremental rebuild from a persistent source tree).
    #   - pip recursively initializes all submodules, including the ROCm-only
    #     composable_kernel (~160MB extra download that NVIDIA builds don't
    #     need); there is no pip flag to skip it.
    # Flag rationale:
    #   - `pip install` (not the deprecated `python setup.py install`):
    #     setup.py install produces a zipped .egg containing the compiled
    #     _C.abi3.so; Python cannot dlopen a .so from inside a zip, and
    #     zipimport also fails to resolve the namespace package (no
    #     __init__.py), so `import flash_attn_3` ends up failing even though
    #     pip reports the package as installed. pip install produces a flat
    #     layout with the .so extracted to disk.
    #   --no-build-isolation: FA3's setup.py imports torch at configure time,
    #     so the build must reuse the current environment rather than a fresh
    #     isolated venv (which would lack torch).
    #   --no-deps: einops/ninja/packaging/torch are already present; skip
    #     reinstalling them to save time and avoid version churn.
    #   --force-reinstall: overwrite any prior broken (zipped-egg) install so
    #     the auto-build is self-healing when invoked from a dirty state.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip install
                --no-build-isolation --no-deps --force-reinstall
                "${FA3_GIT_URL}"
        RESULT_VARIABLE FA3_BUILD_RESULT
        OUTPUT_VARIABLE FA3_BUILD_OUTPUT
        ERROR_VARIABLE FA3_BUILD_ERROR
    )

    if(NOT FA3_BUILD_RESULT EQUAL 0)
        message(WARNING "Flash Attention 3 build failed: ${FA3_BUILD_ERROR}")
        message(STATUS "Build output: ${FA3_BUILD_OUTPUT}")
        message(STATUS "You may need to install Flash Attention 3 manually")
    else()
        message(STATUS "Flash Attention 3 built and installed successfully")
    endif()

    message(STATUS "Flash Attention 3 auto build process completed")
endfunction()
