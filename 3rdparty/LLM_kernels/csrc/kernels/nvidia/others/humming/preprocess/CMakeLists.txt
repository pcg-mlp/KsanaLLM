# Preprocessing kernels for humming weight format conversion
file(GLOB HUMMING_PREPROCESS_SRCS *.cu)
if(HUMMING_PREPROCESS_SRCS)
  add_library(llm_kernels_nvidia_kernel_humming_preprocess STATIC ${HUMMING_PREPROCESS_SRCS})
  set_property(TARGET llm_kernels_nvidia_kernel_humming_preprocess PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET llm_kernels_nvidia_kernel_humming_preprocess PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(llm_kernels_nvidia_kernel_humming_preprocess PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
  )
  target_link_libraries(llm_kernels_nvidia_kernel_humming_preprocess PUBLIC
    -lcudart llm_kernels_nvidia_utils ${TORCH_LIBRARIES})
endif()
