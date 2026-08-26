### A. 问题分析与方案总述【必填】

- 问题分析：
  1. 改造前各算子 `CMakeLists.txt` 在 `NEED_REGENERATE` 分支直接 `execute_process(COMMAND bash …_creator.sh …)`，configure 线程串行等待；stamp 在 cmake 里 `file(WRITE ${MD5_FILE})`，与进程成功绑定在同一同步调用里。
  2. 多卡机器上串行仍只跑一条 creator；部分 creator（`fused_moe_wfp4afp8`）内部会用多卡，若与其它单卡 job 同时可见同一 `CUDA_VISIBLE_DEVICES` 会抢卡。
  3. 首版 runner（`0ee1beca`）若按 GPU 数静态切队列，卡先空出来也无法接下一个 job（`56832bc0` 改为按需从 pending 取）。
  4. 长任务无输出时难以区分「还在跑」和「挂死」（`40c6ac48` 加心跳与 timing summary）。

- 方案总体思路：configure 收集 JSONL 任务，**全部子目录 add 完后**一次 `ksana_triton_gen_run`。Python `schedule_jobs` 维护 `free` GPU 列表：能塞下 `need` 张卡就启动线程；`need = pool_size if multi_gpu else 1`；pending 按 `need` 降序再按原序，避免 MULTI_GPU 被单卡填满饿死。每个 job 子进程 `CUDA_VISIBLE_DEVICES` 设为分到的卡。成功后 runner 写 stamp（当前树还写 `.files` 清单，清单逻辑与 cache 子目录单共用同一 runner）。不采用 cmake `execute_process` + `&` 后台：无法做 GPU exclusive 与统一 FATAL。不把 packing 写进各 creator。

对照 commit：`0ee1beca` 引入队列；`56832bc0` 按需执行；`244268ca` MULTI_GPU + wfp4afp8；`40c6ac48` 打印。任务子目录/`artifact_dir`/wrapper 查找见 TAPD `1070131059137483057`，本 plan **不要求再改那些路径**。

### B. 代码定位【必填】

- `3rdparty/LLM_kernels/cmake/triton_gen_parallel.cmake` — `ksana_triton_gen_init`：清空并创建 `${CMAKE_BINARY_DIR}/triton_gen_jobs/jobs.jsonl`、`build_log/`，解析 GPU 池写 `KSANA_TRITON_GEN_GPUS`。`ksana_enqueue_triton_gen`：校验 NAME/COMMAND/WORKING_DIRECTORY/MD5_FILE；可选 `MULTI_GPU`、`ENV KEY=VAL`；追加一行 JSON（`name,cwd,cmd,env,md5_file,md5_value,multi_gpu`；后续单另加 `artifact_dir`）。`ksana_triton_gen_run`：`python3 run_triton_gen_jobs.py --jobs-file --log-dir --gpus`，rc≠0 则 `FATAL_ERROR`。
- `3rdparty/LLM_kernels/cmake/run_triton_gen_jobs.py` — `parse_gpus` / `job_num_gpus` / `schedule_jobs`（Condition + 线程 + `try_dispatch`）/ `run_job`（设 `CUDA_VISIBLE_DEVICES`、`KSANA_TRITON_GEN_LOG_DIR`、合并 job.env，stdout→`<log-dir>/<name>.log`）/ `commit_success`（成功写 stamp）/ 心跳 `KSANA_TRITON_GEN_HEARTBEAT_SEC`（默认 60，0 关闭）/ 结束 timing summary 与失败 log 尾 80 行。
- `3rdparty/LLM_kernels/csrc/kernels/nvidia/CMakeLists.txt` — `include(triton_gen_parallel.cmake)` + `ksana_triton_gen_init()` 在子目录之前；全部 `add_subdirectory` 之后 `ksana_triton_gen_run()`。
- 串行 `execute_process` → `ksana_enqueue_triton_gen`（COMMAND 与原先一致）：
  - `.../moe/fused_moe/CMakeLists.txt` — NAME `fused_moe`
  - `.../moe/fused_batched_moe/CMakeLists.txt` — `fused_batched_moe`
  - `.../moe/fused_moe_gptq_int4_fp8_kernel/CMakeLists.txt` — `fused_moe_gptq_int4_fp8_kernel`
  - `.../moe/fused_moe_kernel_gptq_awq/CMakeLists.txt` — `fused_moe_kernel_gptq_awq`
  - `.../moe/fused_moe_wfp4afp8/CMakeLists.txt` — `fused_moe_wfp4afp8` **+ MULTI_GPU**
  - `.../moe/humming_triton_fp8_quant/CMakeLists.txt` — `triton_fp8_quant`
  - `.../moe/humming_triton_moe_fused_mul_sum/CMakeLists.txt` — `triton_moe_fused_mul_sum`
  - `.../attn_res/CMakeLists.txt` — `attn_res`
  - `.../kda/CMakeLists.txt` — `kda`
  - `.../mhc/CMakeLists.txt` — `mhc`（可带 `ENV LD_PRELOAD=…`）
- 不改各 `*_creator.sh` / `mhc_compile.py` 本体。

### C. 调用链影响【必填】

- `ksana_triton_gen_init` / `ksana_triton_gen_run` — 仅 `csrc/kernels/nvidia/CMakeLists.txt`。必须 **init 在所有 enqueue 之前、run 在全部 enqueue 之后**，否则 jobs.jsonl 空或未定义。
- `ksana_enqueue_triton_gen` — 上列 10 个 CMakeLists 的 `NEED_REGENERATE` 分支。接口相对 `execute_process`：stamp 延后到 runner 成功；失败不再在该子目录 cmake 里当场 FATAL（统一在 run 结束 FATAL）。
- `ksana_triton_kernel_md5_file` — skip 检查仍在各子目录 configure 时调用；与本单并行调度正交（路径布局由另一单定义）。
- `run_triton_gen_jobs.py::schedule_jobs` — 仅被 `ksana_triton_gen_run` 的 `execute_process` 调用；无 C++ 调用方。
- 运行时 `TritonWrapper::InvokeTritonKernel` — **本单不改**。

### E. 测试要点【必填】

澄清为 perf 占位、无 §G；本阶段自补（均可二元判断）：

- T01（unit，建议抽 `schedule_jobs` 或等价纯函数）：池 2 卡、2 个单卡 job → 可同时 running（或 mock 线程下 `free` 被两 job 分完）；无 MULTI_GPU 时不会出现一张卡上两个 job。
- T02（unit）：池 2 卡、1 个 `multi_gpu` + 1 个单卡 → MULTI_GPU 先占 2 卡；其结束前单卡不得 start；结束后单卡才获得卡。
- T03（unit）：池 1 卡、`multi_gpu` job `need=1`（`job_num_gpus` 用 `pool_size`）→ 能启动，不死锁。
- T04（cmake/manual）：`DISABLE_GENERATE_TRITON_KERNEL=1` configure → jobs.jsonl 空或 run 打印 queue empty，无 creator 进程。
- T05（cmake/manual）：stamp 命中 → 对应 NAME 不出现在 queued jobs 列表。
- T06（sandbox/GPU）：≥2 卡且需 regen 的单卡 job ≥2 → `build_log` 出现并行 START；墙钟小于串行之和（无固定倍速闸门）。
- T07（sandbox/GPU）：`fused_moe_wfp4afp8` regen 时日志 `GPU=` 为整池列表，同时段无其它 job START。
- T08（sandbox）：故意让一 job COMMAND 失败 → cmake FATAL，stdout 含该 `<name>.log` 尾。

仓库当前 **没有** 针对 `run_triton_gen_jobs.py` 的 pytest；T01–T03 若本轮不加，coding-snapshot 须声明延期。T04–T08 依赖 cmake/GPU，作 verify 集成而非编码机闸门。

### G. 风险待办【必填，可为空】

- `nvidia-smi` 失败时池退化为 `"0"`，多卡机器若未 export `CUDA_VISIBLE_DEVICES` 且 smi 不可用，会退回单卡串行，加速不明显而非报错。
- MULTI_GPU 与单卡混跑时，若还有第二个 MULTI_GPU，二者不能重叠；目前只有 wfp4afp8 一处。
- cmake `execute_process` 跑 runner 时心跳打到 configure 日志；`PYTHONUNBUFFERED=1` 已设。无 GPU 的编码机无法做 T06–T07。
- 与 cache 子目录单共用 `triton_gen_parallel.cmake` / runner：改 packing 时不要拆掉另一单的 `artifact_dir` / stamp `.files`。
- 无数值 SLO：review 不能把未承诺的墙钟倍数当 CODE_GAP。

### Env 执行环境提示【必填，本项目新增】

- 本次改动能否在编码环境内做语法/类型级自检？**部分能**：cmake/python 可审阅；`schedule_jobs` 单测（若落地）不需 GPU。完整 configure 生成不能在无 NVIDIA / 无 cmake 缓存的编码机上跑。
- 是否涉及需要容器内才能验证的行为？**是**：真 creator、多卡 packing、FATAL 路径需要 GPU 与完整 cmake。
- 建议的验证范围（增量单测目标）：若加 pytest，覆盖 T01–T03（`3rdparty/LLM_kernels/cmake/run_triton_gen_jobs.py` 的 packing）；否则本单 verify 以 cmake 日志人工/流水线为主。
- 是否预判需要集成测试：**是**（T04–T08）；自动化程度：需人工触发完整 configure（或 CI 已有 kernel gen 流水）；结果可看 `build_log/*.log` 与 cmake FATAL，无需对 cubin 数值做新断言。
