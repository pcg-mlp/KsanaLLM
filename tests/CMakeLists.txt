# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/ksana_llm/python/ksana_llm ${CMAKE_CURRENT_BINARY_DIR})
# copy python script for standalone test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/src/ksana_llm/python/ksana_llm ${CMAKE_BINARY_DIR})

py_test(model_llama_integration_test_prefix_cache SRCS ${PROJECT_SOURCE_DIR}/tests/model/llama_integration_test.py
                                     ARGS --config_file ${PROJECT_SOURCE_DIR}/tests/model/llama_test_prefix_cache.yaml 
                                          --tokenizer_dir /model/llama-hf/13B)

py_test(model_llama_integration_test SRCS ${PROJECT_SOURCE_DIR}/tests/model/llama_integration_test.py
                                     ARGS --config_file ${PROJECT_SOURCE_DIR}/tests/model/llama_test.yaml 
                                          --tokenizer_dir /model/llama-hf/13B)