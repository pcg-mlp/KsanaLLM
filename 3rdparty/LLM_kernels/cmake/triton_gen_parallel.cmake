# Queue Triton/TileLang kernel generation and run with exclusive GPU packing.
# Each job is either single-GPU (default) or MULTI_GPU (uses the whole pool).
# Pool = CUDA_VISIBLE_DEVICES (comma-separated), else nvidia-smi (or 1).

set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_RUNNER "${CMAKE_CURRENT_LIST_DIR}/run_triton_gen_jobs.py")

function(_ksana_json_escape in_var out_var)
  string(REPLACE "\\" "\\\\" _t "${${in_var}}")
  string(REPLACE "\"" "\\\"" _t "${_t}")
  string(REPLACE "\t" "\\t" _t "${_t}")
  string(REPLACE "\n" "\\n" _t "${_t}")
  set(${out_var} "${_t}" PARENT_SCOPE)
endfunction()

# Strip trailing slashes; ABSOLUTE only when the path is already absolute so a
# relative token is not resolved against the caller CMakeLists directory.
function(_ksana_norm_dir_path in_path out_var)
  set(_p "${in_path}")
  string(REGEX REPLACE "/+$" "" _p "${_p}")
  if(_p STREQUAL "")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()
  if(IS_ABSOLUTE "${_p}")
    get_filename_component(_p "${_p}" ABSOLUTE)
    string(REGEX REPLACE "/+$" "" _p "${_p}")
  endif()
  set(${out_var} "${_p}" PARENT_SCOPE)
endfunction()

function(ksana_triton_gen_init)
  set(_job_dir "${CMAKE_BINARY_DIR}/triton_gen_jobs")
  set(_log_dir "${CMAKE_BINARY_DIR}/build_log")
  file(REMOVE_RECURSE "${_job_dir}")
  file(MAKE_DIRECTORY "${_job_dir}")
  file(MAKE_DIRECTORY "${_log_dir}")
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_JOB_DIR "${_job_dir}")
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_LOG_DIR "${_log_dir}")
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_JOBS_FILE "${_job_dir}/jobs.jsonl")
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_ENQUEUED_NAMES "")
  file(WRITE "${_job_dir}/jobs.jsonl" "")

  set(_gpus "")
  if(DEFINED ENV{CUDA_VISIBLE_DEVICES} AND NOT "$ENV{CUDA_VISIBLE_DEVICES}" STREQUAL "")
    set(_gpus "$ENV{CUDA_VISIBLE_DEVICES}")
  else()
    execute_process(
      COMMAND nvidia-smi --query-gpu=index --format=csv,noheader
      OUTPUT_VARIABLE _smi_out
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_smi_out)
      string(REPLACE "\n" "," _gpus "${_smi_out}")
      string(REPLACE " " "" _gpus "${_gpus}")
    endif()
  endif()
  if(_gpus STREQUAL "")
    set(_gpus "0")
  endif()
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_GPUS "${_gpus}")
  string(REPLACE "," ";" _gpu_list "${_gpus}")
  list(LENGTH _gpu_list _ngpu)
  message(STATUS "Triton/TileLang kernel gen parallel=${_ngpu} GPUs (CUDA_VISIBLE_DEVICES=${_gpus})")

  # Skip stamps live under each job subdirectory so wiping one NAME is closed-loop.
  # Layout: ${TRITON_KERNEL_PATH}/<NAME>/md5_sum/.${NAME}.md5sums
  # TRITON_KERNEL_PATH: env ${KSANA_TRITON_KERNEL_PATH}/triton_kernel_files, else
  # ${CMAKE_BINARY_DIR}/../triton_kernel_files.
  if(NOT TRITON_KERNEL_PATH)
    message(FATAL_ERROR "TRITON_KERNEL_PATH is empty; cannot place triton kernel md5 stamps")
  endif()
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_ARTIFACT_DIR "${TRITON_KERNEL_PATH}")
  message(STATUS "Triton/TileLang kernel stamps: ${TRITON_KERNEL_PATH}/<NAME>/md5_sum")
endfunction()

# Skip is decided only from ${TRITON_KERNEL_PATH}/${NAME}/md5_sum. A job is trusted iff
#   1. .${NAME}.md5sums is non-empty, and
#   2. .${NAME}.md5sums.files lists at least one "<content-md5>  <relpath>" line, and
#   3. every listed file exists under ${TRITON_KERNEL_PATH}/${NAME} and still hashes to that md5.
# Missing / empty / malformed stamps mean regenerate. There is no fallback to
# older path-only manifests, root md5_sum/, or build-tree md5 files.
function(_ksana_triton_stamp_trusted stamp_file artifact_dir out_var)
  set(_manifest "${stamp_file}.files")
  if(NOT EXISTS "${stamp_file}" OR NOT EXISTS "${_manifest}" OR "${artifact_dir}" STREQUAL "")
    set(${out_var} FALSE PARENT_SCOPE)
    return()
  endif()

  file(READ "${stamp_file}" _stamp_raw)
  string(STRIP "${_stamp_raw}" _stamp_raw)
  if(_stamp_raw STREQUAL "")
    set(${out_var} FALSE PARENT_SCOPE)
    return()
  endif()

  file(STRINGS "${_manifest}" _entries)
  set(_nlisted 0)
  foreach(_line IN LISTS _entries)
    if(_line STREQUAL "")
      continue()
    endif()
    if(NOT _line MATCHES "^([0-9a-fA-F]+)[ \t]+(.+)$")
      message(STATUS "  malformed kernel manifest line in ${_manifest}: ${_line}")
      set(${out_var} FALSE PARENT_SCOPE)
      return()
    endif()
    set(_want "${CMAKE_MATCH_1}")
    set(_rel "${CMAKE_MATCH_2}")
    string(TOLOWER "${_want}" _want)
    set(_path "${artifact_dir}/${_rel}")
    if(NOT EXISTS "${_path}")
      message(STATUS "  recorded kernel file missing: ${_path}")
      set(${out_var} FALSE PARENT_SCOPE)
      return()
    endif()
    file(MD5 "${_path}" _got)
    if(NOT _got STREQUAL _want)
      message(STATUS "  kernel content md5 mismatch: ${_path}")
      set(${out_var} FALSE PARENT_SCOPE)
      return()
    endif()
    math(EXPR _nlisted "${_nlisted}+1")
  endforeach()
  if(_nlisted EQUAL 0)
    set(${out_var} FALSE PARENT_SCOPE)
    return()
  endif()
  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(ksana_triton_kernel_md5_file NAME out_var)
  get_property(_jobs_file GLOBAL PROPERTY KSANA_TRITON_GEN_JOBS_FILE)
  if(NOT _jobs_file)
    message(FATAL_ERROR "ksana_triton_kernel_md5_file: call ksana_triton_gen_init() first")
  endif()
  if(NOT TRITON_KERNEL_PATH)
    message(FATAL_ERROR "ksana_triton_kernel_md5_file: TRITON_KERNEL_PATH is empty")
  endif()

  set(_job_artifact_dir "${TRITON_KERNEL_PATH}/${NAME}")
  set(_file "${_job_artifact_dir}/md5_sum/.${NAME}.md5sums")
  _ksana_triton_stamp_trusted("${_file}" "${_job_artifact_dir}" _ok)
  if(NOT _ok)
    if(EXISTS "${_file}" OR EXISTS "${_file}.files")
      message(STATUS "md5_sum for ${NAME} is missing, incomplete, or content mismatch; kernels will be regenerated")
      file(REMOVE "${_file}" "${_file}.files")
    endif()
  endif()
  set(${out_var} "${_file}" PARENT_SCOPE)
endfunction()

# ksana_enqueue_triton_gen(
#   NAME <id>
#   WORKING_DIRECTORY <dir>
#   MD5_FILE <path>
#   MD5_VALUE <hex>
#   COMMAND <argv...>
#   [MULTI_GPU]             # exclusive: occupy the whole GPU pool until this job exits
#   [ENV KEY=VAL ...]
# )
function(ksana_enqueue_triton_gen)
  cmake_parse_arguments(ARG "MULTI_GPU" "NAME;WORKING_DIRECTORY;MD5_FILE;MD5_VALUE" "COMMAND;ENV" ${ARGN})
  if(NOT ARG_NAME OR NOT ARG_COMMAND OR NOT ARG_WORKING_DIRECTORY OR NOT ARG_MD5_FILE)
    message(FATAL_ERROR "ksana_enqueue_triton_gen: NAME, COMMAND, WORKING_DIRECTORY, MD5_FILE required")
  endif()
  if(ARG_MULTI_GPU)
    set(_multi_gpu_json "true")
    set(_multi_gpu_msg "multi-gpu (all pool GPUs)")
  else()
    set(_multi_gpu_json "false")
    set(_multi_gpu_msg "single-gpu")
  endif()

  get_property(_jobs_file GLOBAL PROPERTY KSANA_TRITON_GEN_JOBS_FILE)
  if(NOT _jobs_file)
    message(FATAL_ERROR "ksana_enqueue_triton_gen: call ksana_triton_gen_init() first")
  endif()
  if(NOT TRITON_KERNEL_PATH)
    message(FATAL_ERROR "ksana_enqueue_triton_gen: TRITON_KERNEL_PATH is empty")
  endif()

  get_property(_enqueued_names GLOBAL PROPERTY KSANA_TRITON_GEN_ENQUEUED_NAMES)
  if(_enqueued_names AND ARG_NAME IN_LIST _enqueued_names)
    message(FATAL_ERROR
      "ksana_enqueue_triton_gen: duplicate NAME \"${ARG_NAME}\". "
      "Each cmake job NAME must be unique (live dir, stamp, and the Python "
      "runner's running-map all key off NAME).")
  endif()
  list(APPEND _enqueued_names "${ARG_NAME}")
  set_property(GLOBAL PROPERTY KSANA_TRITON_GEN_ENQUEUED_NAMES "${_enqueued_names}")

  set(_live_dir "${TRITON_KERNEL_PATH}/${ARG_NAME}")
  # Generate into a hidden staging dir so a failed creator cannot delete
  # last-known-good cubins. TritonWrapper skips names starting with '.'.
  set(_staging "${TRITON_KERNEL_PATH}/.generating/${ARG_NAME}")
  file(REMOVE_RECURSE "${_staging}")
  file(MAKE_DIRECTORY "${_staging}")

  _ksana_norm_dir_path("${TRITON_KERNEL_PATH}" _root_n)
  set(_cmd_out "")
  set(_replaced FALSE)
  foreach(_a IN LISTS ARG_COMMAND)
    _ksana_norm_dir_path("${_a}" _a_n)
    if(NOT _a_n STREQUAL "" AND _a_n STREQUAL "${_root_n}")
      set(_a "${_staging}")
      set(_replaced TRUE)
    endif()
    list(APPEND _cmd_out "${_a}")
  endforeach()
  if(NOT _replaced)
    message(FATAL_ERROR
      "ksana_enqueue_triton_gen(${ARG_NAME}): COMMAND has no output path equal to "
      "TRITON_KERNEL_PATH=${TRITON_KERNEL_PATH} (after normalizing slashes). "
      "Pass \${TRITON_KERNEL_PATH}; do not hardcode HOME or another cache root.")
  endif()

  _ksana_json_escape(ARG_NAME _name_e)
  _ksana_json_escape(ARG_WORKING_DIRECTORY _cwd_e)
  _ksana_json_escape(ARG_MD5_FILE _md5f_e)
  _ksana_json_escape(ARG_MD5_VALUE _md5v_e)
  _ksana_json_escape(_staging _artdir_e)
  _ksana_json_escape(_live_dir _livedir_e)

  set(_cmd_json "")
  foreach(_a IN LISTS _cmd_out)
    _ksana_json_escape(_a _ae)
    if(_cmd_json STREQUAL "")
      set(_cmd_json "\"${_ae}\"")
    else()
      set(_cmd_json "${_cmd_json}, \"${_ae}\"")
    endif()
  endforeach()

  set(_env_json "")
  foreach(_kv IN LISTS ARG_ENV)
    if(_kv MATCHES "^([^=]+)=(.*)$")
      set(_ek "${CMAKE_MATCH_1}")
      set(_ev "${CMAKE_MATCH_2}")
      _ksana_json_escape(_ek _eke)
      _ksana_json_escape(_ev _eve)
      if(_env_json STREQUAL "")
        set(_env_json "\"${_eke}\": \"${_eve}\"")
      else()
        set(_env_json "${_env_json}, \"${_eke}\": \"${_eve}\"")
      endif()
    endif()
  endforeach()

  file(APPEND "${_jobs_file}"
    "{\"name\": \"${_name_e}\", \"cwd\": \"${_cwd_e}\", \"cmd\": [${_cmd_json}], \"env\": {${_env_json}}, \"md5_file\": \"${_md5f_e}\", \"md5_value\": \"${_md5v_e}\", \"artifact_dir\": \"${_artdir_e}\", \"live_dir\": \"${_livedir_e}\", \"multi_gpu\": ${_multi_gpu_json}}\n")
  message(STATUS "Queued triton/tilelang gen: ${ARG_NAME} (${_multi_gpu_msg})")
endfunction()

function(ksana_triton_gen_run)
  get_property(_jobs_file GLOBAL PROPERTY KSANA_TRITON_GEN_JOBS_FILE)
  get_property(_log_dir GLOBAL PROPERTY KSANA_TRITON_GEN_LOG_DIR)
  get_property(_gpus GLOBAL PROPERTY KSANA_TRITON_GEN_GPUS)
  if(NOT _log_dir)
    set(_log_dir "${CMAKE_BINARY_DIR}/build_log")
    file(MAKE_DIRECTORY "${_log_dir}")
  endif()
  if(NOT _jobs_file OR NOT EXISTS "${_jobs_file}")
    return()
  endif()
  file(READ "${_jobs_file}" _jobs_raw)
  if(_jobs_raw STREQUAL "")
    message(STATUS "No Triton/TileLang kernel gen jobs queued")
    return()
  endif()

  get_property(_runner GLOBAL PROPERTY KSANA_TRITON_GEN_RUNNER)
  find_program(_ksana_python3 NAMES python3 python)
  if(NOT _ksana_python3)
    message(FATAL_ERROR "python3 required to run parallel Triton kernel generation")
  endif()

  message(STATUS "Running queued Triton/TileLang kernel generation on GPUs: ${_gpus}")
  message(STATUS "  kernel gen logs: ${_log_dir}/<job>.log (this step can take tens of minutes)")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
            ${_ksana_python3} "${_runner}"
            --jobs-file "${_jobs_file}"
            --log-dir "${_log_dir}"
            --gpus "${_gpus}"
    RESULT_VARIABLE _gen_rc)
  if(_gen_rc AND NOT _gen_rc EQUAL 0)
    message(FATAL_ERROR "Parallel Triton/TileLang kernel generation failed (rc=${_gen_rc}). See ${_log_dir}/*.log")
  endif()
endfunction()
