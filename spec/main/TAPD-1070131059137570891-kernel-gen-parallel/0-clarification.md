---
spec_version: "1"
spec_type: change
id: "1070131059137570891"
status: draft
subdomain: main
cross_subdomains: []
complexity: L
work_type: perf
---

> TAPD：[kernel算子编译加速](https://tapd.woa.com/tapd_fe/70131059/story/detail/1070131059137570891)  
> 父需求：编译加速（1070131059137476238）。TAPD 描述为空；正文以 commit `0ee1beca` → `40c6ac48` 与会话共识为准。  
> 同父需求另一单 [kernel cache 算子分离子目录](https://tapd.woa.com/tapd_fe/70131059/story/detail/1070131059137483057) 管任务子目录与 stamp 闭环，**不在本单范围**。  
> 复杂度：按「cmake 队列 + Python GPU packing + 10 处 enqueue」定为 **L**。  
> `work_type`：按 TAPD 标题「编译加速」定为 **perf**（未改推理对外语义）。

### A. 工作类型（work_type）【必填，先于目标】
`perf` — 性能优化（缩短 cmake 配置期 Triton/TileLang kernel 生成墙钟；默认不改 cubin 语义与运行时发射）

次要动机：把各算子 `execute_process` 改成可排队的 cmake API（`ksana_enqueue_triton_gen` / `ksana_triton_gen_run`），并支持 `MULTI_GPU` 独占整池。主类型仍是编译加速，不新增推理能力。

### B. 目标（goal）

问题：nvidia 侧约 10 个 Triton/TileLang creator 在 cmake **configure** 阶段用 `execute_process` **串行**跑完（每条可数十分钟），多卡机器上也只吃到当前进程可见的一张卡（或整池被单进程占满却仍排队），configure 墙钟被生成链拉长。

优化对象：配置期 kernel 生成墙钟。GPU 池 = `CUDA_VISIBLE_DEVICES`（逗号分隔），未设则 `nvidia-smi --query-gpu=index`，再不行则 `0`。调度：默认任务 **单卡**，空闲卡上按需入队填满；声明 `MULTI_GPU` 的任务（当前仅 `fused_moe_wfp4afp8`）**独占整池直至退出**，避免与单卡任务抢卡。跳过条件不变：`DISABLE_GENERATE_TRITON_KERNEL`、源文件 md5 stamp 命中仍不入队。失败：任一 job 非 0 → `ksana_triton_gen_run` `FATAL_ERROR`，日志在 `${CMAKE_BINARY_DIR}/build_log/<NAME>.log`。可观测：心跳（默认 60s，`KSANA_TRITON_GEN_HEARTBEAT_SEC=0` 关闭）打印 running/queued、log 增长与末行，结束打 wall / sum-of-jobs。

TAPD 未给数值 SLO；基线 = 改造前串行 `execute_process` 墙钟。不改 creator 脚本内部算法、不改运行时 `TritonWrapper` 加载语义（加载路径由另一单处理）。

**cmake 配置期把 Triton/TileLang kernel 生成改为多卡队列并行（单卡 packing + MULTI_GPU 独占），缩短生成墙钟且失败可定位到 per-job 日志。**
