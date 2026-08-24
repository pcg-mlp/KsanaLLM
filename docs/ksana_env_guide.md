# KsanaLLM Environment Variables

This document lists all environment variables consumed by the KsanaLLM project (runtime + build-time).

## Distributed / Cluster

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `WORLD_SIZE` | Total node count in distributed cluster | `1` |
| `NODE_RANK` | Rank of current node in distributed cluster | `0` |
| `MASTER_HOST` | Master node host for distributed mode | `""` (required if `WORLD_SIZE`>1) |
| `MASTER_PORT` | Master node port for distributed mode | None (required if `WORLD_SIZE`>1) |
| `EXPERT_MASTER_HOST` | Master host for expert-parallel distributed setup | None |
| `EXPERT_MASTER_PORT` | Master port for expert-parallel distributed setup | None |
| `EXPERT_NODE_RANK` | Node rank for expert-parallel distributed setup | `0` |
| `USE_TCP_DATA_CHANNEL` | Force TCP data channel instead of NCCL/RDMA | None |
| `IFA_NAME` | Network interface name used for socket binding | `bond1` |
| `SAMPLING_NODE_OFFLOAD_LAYER_NUM` | Layers to offload to sampling node in distributed mode | `0` |
| `PD_V2_WARMUP_PEER_WAIT_MS` | PD-v2 warmup peer wait timeout (ms) | None |
| `DECODE_NODE_BENCHMARK` | Force role to decode for benchmarking | `0` |

## Device / GPU

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU visibility (set by launcher/tests) | None |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend NPU visibility (set by launcher/tests) | None |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA allocator config, read by all-reduce layers | None |
| `CUDA_MODULE_LOADING` | Force lazy CUDA module loading in VL plugins | `LAZY` (set by plugin) |

## Ksana Core

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `KSANA_TRITON_KERNEL_PATH` | Path for pre-built Triton kernel files | `$HOME/.cache/KsanaLLM` |
| `KSANA_IPC_NAME` | Shared-memory IPC channel name | Derives from engine PID |
| `KSANA_EXECUTOR_ID` | Executor rank id (set by serving_server) | rank_id |
| `KSANA_HTTP_PORT` | Marks that plugin runs under HTTP server; controls preprocess path | None |
| `KSANA_ENABLE_ENCODER_STREAMING` | Enable encoder streaming in VL plugins | `0` |
| `KSANA_DISABLE_DECOUPLED_PREPROCESS` | Force-disable decoupled multimodal preprocess | `0` |
| `KSANA_DISABLE_DEQUANT_FP4_SPARSE` | Disable FP4 sparse dequant path, fall back to full expert dequant | Unset (sparse enabled) |
| `KSANA_MOE_GATE_BACKEND` | MoE gate topk/softplus backend selector | None |
| `KSANA_TEST_WITH_SERVER` | Enable HTTP router client test with real server | None |

## KLLM (Logging / Scheduling / Kernel)

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `KLLM_LOG_LEVEL` | Log level (DEBUG/INFO/…) | `INFO` |
| `KLLM_LOG_FILE` | Log file basename/path | `ksana_llm` |
| `KLLM_STAT_FILE` | Path for scheduler statistics writer | None |
| `KLLM_SCHED_TRACE_LEVEL` | Scheduler event tracer level | None |
| `KLLM_USE_FP32_COMPRESSOR` | Force compressor state dtype to FP32 (else BF16) | Unset (BF16) |
| `KLLM_MAX_DEVICE_BLOCKS` | Cap on max device blocks in block manager | Computed from memory |
| `KLLM_INDEX_TOPK_PATTERN` | Comma-list of layer indices to skip DSA indexer topk | None |
| `KLLM_INDEX_SKIP_TOPK_OFFSET` | Override indexer skip topk offset | From DSA config |
| `KLLM_INDEX_TOPK_FREQ` | Override indexer skip topk frequency | From DSA config |
| `KLLM_INDEXER_MAX_CHUNKED_Q_SIZE` | Max chunked Q size for indexer memory alloc | `0` (disabled) |
| `KLLM_DISABLE_SGL_ATTN` | Disable SGL attention in Qwen VL plugins | `0` |
| `KLLM_DISABLE_PDL` | Disable PDL feature (SM90+) | Unset (enabled on SM>=90) |
| `KLLM_KERNEL_PERF` | Enable perf timing for kernel tests | Unset |
| `KLLM_DG_JIT_DEBUG` | DeepGEMM JIT runtime debug | None |
| `KLLM_DG_JIT_DUMP_CUBIN` | Dump generated cubin from DeepGEMM JIT | None |
| `KLLM_DG_CACHE_DIR` | DeepGEMM JIT cache directory | `$HOME` / `$APPDATA` fallback |
| `KLLM_DG_NVCC_COMPILER` | DeepGEMM nvcc compiler override | Derived from `CUDA_HOME` |

## Humming JIT

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `KSANA_HUMMING_BASE_DIR` | Humming JIT base source directory override | Derived from `__FILE__` |
| `KSANA_HUMMING_KERNEL_PATH` | Humming JIT kernel cache directory | `$HOME/.cache/KsanaLLM/humming_kernel_files` |
| `HUMMING_INCLUDE_DIR` | Humming JIT include directory | Derived |
| `HUMMING_REGENERATE` | Force regeneration of Humming JIT kernels | Unset |
| `HUMMING_JIT_LOG` | Verbose Humming JIT compile logs | Unset |
| `HUMMING_DISABLE_PARALLEL_BUILD` | Disable parallel Humming build (Python autotool) | `1` |

## Model / Weight / Cache

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `ENABLE_MODEL_PERF` | Enable model perf instrumentation | None |
| `MODEL_PERF_CONFIG` | Path to model perf JSON config | None |
| `ENABLE_MODEL_CACHE` | Enable serialized weight cache load/store | Unset |
| `MODEL_CACHE_PATH` | Base path for cached model weights | `model_config_.path` |
| `ENABLE_OLD_LOADER` | Force old weight loader path | Unset |
| `ENABLE_NEW_ARGMAX` | Enable new argmax kernel path (sampler tests) | Unset |
| `USE_OLD_ARGMAX` | Force old argmax kernel implementation | Unset |
| `ENABLE_LAYER_TRACKER` | Enable per-layer progress tracker | `0` |
| `ENABLE_PROFILE_EVENT` | Enable internal profile event tracing | Unset |
| `ENABLE_GLOBAL_CACHE_HOOK` | Enable global-cache query/put hook (non pd_v2). When `1`, serving also auto-starts local `mooncake_master` if RPC port is free | `0` |
| `DISABLE_WARMUP` | Skip warmup phase during engine init | Unset |
| `MAX_WARMUP_INPUT_LENGTH` | Override warmup input length upper bound | None |

## EPLB (Expert Parallel Load Balancing)

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `ENABLE_DUMP_EPLB` | Enable EPLB expert activation data dump | `false` |
| `DUMP_EPLB_PATH` | EPLB data dump output directory | `$HOME/.cache/KsanaLLM/EPLB/` |
| `EPLB_CONFIG_PATH` | Path to EPLB configuration YAML | None |

## MoE / Quantization / Attention

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `ENABLE_O_PROJ_OUT_OF_DP` | Move o_proj outside DP (requires `attn_tp==1`) | Unset |
| `FORCE_NON_CONTIGUOUS_SWA` | Force all flash batches to use SWA non-contiguous cache | Unset |
| `SHARED_EXPERT_PARALLEL` | Enable shared-expert parallel path (DeepSeek-V4) | `0` |
| `CONTEXT_DSA_THRESHOLD` | DSA vs MHA context-length threshold | `20*1024` or `256*1024` |
| `DISABLE_DEEPGEMM` | Disable DeepGEMM in blockwise matmul (BF16 path) | Unset (DeepGEMM enabled) |
| `DEEPGEMM_INCLUDE_DIR` | DeepGEMM header include directory | None |
| `MATMUL_LAYER_USE_WORKSPACE` | Force matmul layer to use workspace path | Unset |
| `QUANT_PROFILE` | Quantization profiling iteration count | 5/10/15/20 (varies by layer) |
| `W4AFP8_MOE_BACKEND` | Backend selector for W4AFP8 MoE | None |
| `FP4_MOE_BACKEND` | Backend for FP4 MoE (marlin/humming_w4a16/humming_w4a8/triton) | `FP4Triton` |
| `USE_ON_FLY_WFP4AFP8` | Enable on-fly WFP4AFP8 dequant path | Unset (offline path) |
| `ContextDecodeUseFP8Cache` | Enable FP8 KV cache in context decode attention | None |
| `SELECT_ALL_REDUCE_BY_SIZE` | Choose all-reduce impl by tensor size | Unset (also set in run.sh) |

## Mooncake / Router

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `MOONCAKE_MASTER_BIN` | Absolute path to `mooncake_master` (used when `ENABLE_GLOBAL_CACHE_HOOK=1` auto-starts master) | PATH lookup |
| `MC_TCP_BIND_ADDRESS` | Mooncake connector TCP bind address | None |
| `MC_DISABLE_HUGETLB` | Disable HugeTLB for Mooncake buffers | Unset |
| `MC_TENT_CONF` | Path to Mooncake tent config file | None (required if used) |
| `MC_REDIS_PASSWORD` | Redis password for Mooncake alive store | None |
| `ROUTER_CONFIG_PATH` | Simple router config INI path | None |

Serving CLI（与 yaml `master_server_addr` / `service_endpoint` 端口对齐）:

| Flag | Description | Default |
| --- | --- | --- |
| `--mooncake-http-port` | Mooncake HTTP metadata server port | `8080` |
| `--mooncake-rpc-port` | Mooncake master RPC port | `50051` |
| `--mooncake-metrics-port` | Mooncake master metrics port | `9003` |

Yaml 字段样例（含 `put_hotkey_window_sec` / `swa_tail_block_num` / `swa_front_equal_parts` 等）见
`examples/ksana_llm.yaml` 中注释块 `setting.global_cache_connector`。

## Version / Platform Reporter

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `POD_NAME` | K8s pod name for version reporter | `unknow` |
| `SUMERU_POD_NAME` | Sumeru platform pod name | `unknow` |
| `SUMERU_APP` | Sumeru app name for version reporter | `unknow` |
| `SUMERU_SERVER` | Sumeru server name for version reporter | `unknow` |

## Build / Package

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `CUSTOM_KSANA_VERSION` | Override built package version in setup.py | None |
| `CMAKE_ARGS` | Extra CMake args passed to build (setup.py) | `""` |

## Benchmark / Test Gate

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `BENCH_BLOCK_SIZE` | Global-cache benchmark block size | `128` |
| `BENCH_BLOCK_BYTE_SIZE` | Global-cache benchmark block byte size | `1048576` |
| `BENCH_NUM_BLOCKS` | Global-cache benchmark block count | `16` |
| `BENCH_NUM_LAYERS` | Global-cache benchmark layer count | `32` |
| `BENCH_ITERATIONS` | Global-cache benchmark iteration count | `10` |
| `SKIP_TEST_IF_MODEL_NOT_FOUND` | Skip model tests when model files missing | None |
| `SKIP_TEST_IF_INVALID_TRITON_VERSION` | Skip test when Triton version invalid | None |

## NCCL / UCX / SHARP

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `NCCL_IB_GID_INDEX` | NCCL InfiniBand GID index | `3` |
| `NCCL_IB_SL` | NCCL InfiniBand service level | `3` |
| `NCCL_CHECK_DISABLE` | Disable NCCL runtime checks | `1` |
| `NCCL_P2P_DISABLE` | Disable NCCL P2P | `0` |
| `NCCL_IB_DISABLE` | Disable NCCL IB transport | `0` |
| `NCCL_LL_THRESHOLD` | NCCL low-latency threshold | `16384` |
| `NCCL_IB_CUDA_SUPPORT` | Enable CUDA-aware NCCL IB | `1` |
| `NCCL_SOCKET_IFNAME` | NCCL socket interface | `bond1` |
| `NCCL_TOPO_AFFINITY` | NCCL topology affinity level | `0` |
| `NCCL_IB_HCA` | NCCL IB HCA device list | `mlx5_bond_*` list |
| `NCCL_COLLNET_ENABLE` | Enable NCCL CollNet | `0` |
| `NCCL_NET_GDR_LEVEL` | NCCL GDR network level | `2` |
| `NCCL_IB_QPS_PER_CONNECTION` | NCCL QPs per IB connection | `4` |
| `NCCL_IB_TC` | NCCL IB traffic class | `160` |
| `NCCL_PXN_DISABLE` | Disable NCCL PXN routing | `0` |
| `SHARP_COLL_ENABLE_SAT` | Disable SHARP SAT | `0` |
| `UCX_NET_DEVICES` | UCX net device | `bond1` |

## Third-party JIT

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `TILELANG_CACHE_DIR` | Tilelang autotuner cache dir (mhc autotuner) | Project-configured |

## CMake-only (build-time, not runtime)

`CUDNN_ROOT_DIR`, `CUDNN_INCLUDE_DIR`, `CUDNN_LIBRARY`, `NCCL_ROOT_DIR`, `NCCL_INCLUDE_DIR`, `NCCL_LIB_DIR`, `NCCL_VERSION`, `ASCEND_HOME_PATH`, `ATB_HOME_PATH`, `ASCEND_MAIN_PATH` — consumed by `cmake/FindCUDNN.cmake`, `cmake/FindNCCL.cmake`, `cmake/ascend.cmake`.
