# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE batch_scheduler_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/*.cpp)
list(FILTER batch_scheduler_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "batch_scheduler_SRCS: ${batch_scheduler_SRCS}")

add_library(batch_scheduler STATIC ${batch_scheduler_SRCS})
target_link_libraries(batch_scheduler PUBLIC runtime)

if(WITH_CUDA)
  set(MODELS_TEST_LIBS ${TORCH_LIBRARIES}  ${TORCH_PYTHON_LIBRARY} )
else()
  set(MODELS_TEST_LIBS "")
endif()
set(MODELS_TEST_DEPS batch_scheduler runtime device utils
    ipc executor distributed connector transfer engine
    models model_loader cache_manager samplers configure service supervisor)

cpp_test(batch_scheduler_helper_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/batch_scheduler_test_helper_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(batch_scheduler_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/batch_scheduler_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(event_driven_scheduler_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/event_driven_scheduler_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(event_driven_schedule_processor_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/event_driven_schedule_processor_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(schedule_hook_manager_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/schedule_hook_manager_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(basic_workload_balancer_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(pp_batch_distributor_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/workload_balance/pp_batch_distributor_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(global_cache_put_tracker_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/global_cache_put_tracker_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(global_cache_fetcher_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/global_cache_fetcher_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(global_cache_query_hook_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/global_cache_query_hook_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(planning_workload_estimator_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/planning_workload_estimator_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(event_driven_state_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/event_driven_state_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(model_perf_backend_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/event_driven_scheduler/model_perf/model_perf_backend_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})

cpp_test(continuous_batching_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/strategy/continuous_batching_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
#cpp_test(batch_scheduler_balance_reqs_algo_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler/batch_scheduler_balance_reqs_algo_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
#cpp_test(pp_multibatch_wb_test SRCS ${PROJECT_SOURCE_DIR}/csrc/batch_scheduler//workload_balance/pp_multibatch_balancer_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})

