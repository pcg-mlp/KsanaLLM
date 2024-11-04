# Copyright 2024 Tencent Inc.  All rights reserved.

add_subdirectory(gemm_wrapper)
add_subdirectory(embedding)
add_subdirectory(layernorm)
add_subdirectory(paged_attention)
add_subdirectory(rotary_embedding)
add_subdirectory(samplers)
add_subdirectory(add)
add_subdirectory(activation)
add_subdirectory(assemble_last_token)
add_subdirectory(cast)
# TODO(karlluo): will be support in multi-lora
# add_subdirectory(bgmv)
# TODO(karlluo): will be support in quantization
add_subdirectory(asymmetric_gemm)
add_subdirectory(weight_only_batched_gemv)
add_subdirectory(all_reduce)
add_subdirectory(permute)
add_subdirectory(alibi)
add_subdirectory(gptq_marlin)
