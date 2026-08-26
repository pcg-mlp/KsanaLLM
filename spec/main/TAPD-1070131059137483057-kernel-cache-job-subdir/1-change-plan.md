### A. 问题分析与方案总述【必填】

- 问题分析：
  1. `ksana_enqueue_triton_gen` 把全局 `KSANA_TRITON_GEN_ARTIFACT_DIR`（即 `TRITON_KERNEL_PATH`）写进每条 job JSON；所有 creator 的 `COMMAND` 最后一个路径参数也是这个根目录。10 个队列任务（`fused_moe` / `fused_moe_wfp4afp8` / `fused_moe_kernel_gptq_awq` / `fused_moe_gptq_int4_fp8_kernel` / `fused_batched_moe` / `kda` / `mhc` / `attn_res` / `triton_fp8_quant` / `triton_moe_fused_mul_sum`）并发写同一平铺目录。
  2. `run_triton_gen_jobs.py::collect_artifacts` 对 `artifact_dir.rglob("*")` 按 mtime 窗口收录；注释已承认会扫到邻居文件。`ksana_triton_kernel_md5_file` 的 skip 检查若仍用**全局**根拼 `.files` relpath，会与子目录布局错位。
  3. 原先 `TritonWrapper::InvokeTritonKernel` 只拼 `{kernels_dir_}/{kernel_name}.{json,cubin}`。只改生成目录、不改查找，运行时会全部 miss。`InitKernelsDir` 曾用 `$HOME/.cache/KsanaLLM/triton_kernel_files`，多份 checkout 会互相覆盖 cubin。
  4. KDA 单测 canary 若仍按根平铺 `GetKernelsDir()+"/l2norm_fwd_kernel_..."` 判断 skip，子目录落地后会在产物已存在时误 skip。
  5. 未采用「打包拍平」：`setup.py` 已 `copytree` + `triton_kernel_files/**/*`，拍平会把同名冲突推迟到装包且丢失隔离。未采用 `kernel_base_name→NAME` 硬编码表。不采用「根平铺 + 子目录双查」：旧布局不兼容。

- 方案总体思路：隔离键 = enqueue 的 `NAME`。入队时输出目录与 job `artifact_dir` 均为 `${TRITON_KERNEL_PATH}/${NAME}`；regen 时 `REMOVE_RECURSE` 该子目录（cubin 与该任务 `md5_sum/` 一起清空）。stamp 写在 `${TRITON_KERNEL_PATH}/${NAME}/md5_sum/`。运行时只选一个 `kernels_dir_`，只在其**直接子目录**中按**文件名**找 `{stem}.json` / `{stem}.cubin`（不按 NAME 查表；0 或 >1 命中 throw）。路径：`KSANA_TRITON_KERNEL_PATH` 非空 → cmake/runtime 用 `{env}/triton_kernel_files`；未设置 → cmake 写 `${CMAKE_BINARY_DIR}/../triton_kernel_files`，运行时用编进二进制的该绝对路径。**不写 HOME cache**。`setup.py` 从 build 目录 cmake 记录拷贝。不加 YAML。

### B. 代码定位【必填】

- `CMakeLists.txt` — `TRITON_KERNEL_PATH`：env 非空则 `ABSOLUTE ${KSANA_TRITON_KERNEL_PATH}/triton_kernel_files`，否则 `${CMAKE_BINARY_DIR}/../triton_kernel_files`；CACHE FORCE。写入 `${CMAKE_BINARY_DIR}/triton_kernel_path.txt` 供打包；`add_compile_definitions(TRITON_KERNEL_PATH=...)` 编进二进制作运行时默认。不再因该 env 报 FATAL。
- `3rdparty/LLM_kernels/cmake/nvidia.cmake` — 不再 `mkdir ${CMAKE_BINARY_DIR}/triton_kernel_files`；改为 `file(MAKE_DIRECTORY "${TRITON_KERNEL_PATH}")`。
- `3rdparty/LLM_kernels/cmake/triton_gen_parallel.cmake` — `ksana_enqueue_triton_gen`：`job_dir = ${TRITON_KERNEL_PATH}/${ARG_NAME}`；`REMOVE_RECURSE` + `MAKE_DIRECTORY`（仅 enqueue / NEED_REGENERATE）。`ARG_COMMAND` 中 `STREQUAL ${TRITON_KERNEL_PATH}` 的参数替换为 `job_dir`；JSON `artifact_dir` 写 `job_dir`。`ksana_triton_kernel_md5_file(NAME)`：stamp 为 `${TRITON_KERNEL_PATH}/${NAME}/md5_sum/.${NAME}.md5sums`；`_ksana_triton_stamp_trusted` 的 `artifact_dir` 为任务子目录。不再在 cache 根创建全局 `md5_sum/`。
- `3rdparty/LLM_kernels/cmake/run_triton_gen_jobs.py` — `collect_artifacts` 跳过 `md5_sum/` 与 stamp 目录；`commit_success` 把 stamp 写到 `md5_file.parent`（即任务 `md5_sum/`）。文档字符串去掉「Jobs share one output dir」。
- 各 creator `CMakeLists.txt` 的 `COMMAND ... ${TRITON_KERNEL_PATH}` **可不手改**（enqueue 替换）。当前 10 处均为字面 `${TRITON_KERNEL_PATH}`。
- `csrc/kernels/nvidia/triton_wrapper.h` / `triton_wrapper.cpp` — `ResolveKernelArtifact(stem, ext)`：只扫 `kernels_dir_` 下一层目录中的 `{stem}{ext}`；跳过名为 `md5_sum` 与隐藏目录；不读根平铺、不递归二层。`InvokeTritonKernel` 用其结果加载 json/cubin。`InitKernelsDir`：非空 `KSANA_TRITON_KERNEL_PATH` → `{env}/triton_kernel_files`；否则用 cmake 编进的绝对 `TRITON_KERNEL_PATH`，**不用** cwd 相对 `../triton_kernel_files`。不读 HOME。
- `csrc/kernels/nvidia/triton_wrapper_test.cpp` — T01–T06；`InitKernelsDir`：env 优先；无 env / 空 env → 编译期绝对路径且与 cwd 无关；再 Init；`SetKernelsDir`。解析层不走 `cuModuleLoad`。
- `csrc/kernels/nvidia/kda_kernel_wrapper_test.cpp`（canary ~L86、~L303）与 `csrc/layers/nvidia/kda_attention_layer_test.cpp`（~L84）— skip 改为 `.../kda/<canary>.json/.cubin`。
- `setup.py` — copytree 源读 cmake 记录（`build_temp/triton_kernel_path.txt`，否则 `CMakeCache.txt` 的 `TRITON_KERNEL_PATH`），不读 `KSANA_TRITON_KERNEL_PATH`；ignore 仅 `.creator_logs` / `*.log`；`package_data` 含 `triton_kernel_files/**/md5_sum/.*`。`python/ksana_llm/__init__.py` — 仅运行时 `setdefault`（wheel）。
- `.gitignore` — `/triton_kernel_files/`。`docs/ksana_env_guide.md` — cmake 与 runtime 共用该变量，env 优先。
- `tools/generate_triton_kernel_cubin.sh` — 本次不修；§G 记债。

### C. 调用链影响【必填】

- `ksana_enqueue_triton_gen` — 调用方：上列 10 个 `CMakeLists.txt`（`NEED_REGENERATE` 分支）。接口仍是 NAME/COMMAND/MD5_*。
- `ksana_triton_kernel_md5_file(NAME)` — 同上 10 处；skip 检查产物根为 `.../${NAME}`。旧 stamp 对不上 → miss、重生。
- `TritonWrapper::InvokeTritonKernel` — 生产调用：`triton_wrapper.cpp` 内 fused_moe / MHC / Humming / AttnRes，以及 `kda_kernel_wrapper.cpp` 多处 `Singleton<TritonWrapper>::GetInstance()`。不改参数列表。
- `TritonWrapper::GetKernelsDir` — 仍返回**根**；KDA canary 自己拼 `kda/`。
- `TritonWrapper::InitKernelsDir` — ctor 与单测再解析；与 cmake 同一 env 优先规则。
- `dump_kernel` / 各 `*_creator.sh` — 仍写 `OUTPUT_DIR`；由 cmake 传入任务子目录。

### D. 数据兼容

- 磁盘：`triton_kernel_files/<NAME>/*.cubin|json`；stamp 在 `<NAME>/md5_sum/`。根平铺与根上旧 `md5_sum/` **不读、不迁、不双写**。HOME cache 不再作为生成或默认加载路径。
- 迁移：无转换工具。下次 cmake 因 stamp 路径对不上而 regen。
- 运行时：只认任务子目录；同名多任务 throw。wheel 含子目录树及各任务 `md5_sum/`。
- 配置：无 YAML。cmake 与运行时均：`KSANA_TRITON_KERNEL_PATH` 优先于 sibling/`../triton_kernel_files`。

### E. 测试要点【必填】

澄清为 refactor 占位、无 §G；本阶段自补（均可二元判断）：

- T01（unit）`triton_wrapper_test`：`SetKernelsDir(tmp)`，仅 `tmp/kda/{stem}.json` → Resolve 命中该路径。
- T02（unit）：仅根平铺 `{stem}.json` → Resolve 失败。
- T03（unit）：平铺与 `tmp/kda/{stem}.json` 同时存在 → 命中子目录。
- T04（unit）：`fused_moe/` 与 `kda/` 同名 → 冲突失败。
- T05（unit）：根 `md5_sum/{stem}.json` 与 `kda/md5_sum/{stem}.json` 均不作为命中。
- T06（unit）：env 优先；无 env / 空 env → cmake 编进的绝对 `TRITON_KERNEL_PATH`（与 cwd 无关，不是 `../triton_kernel_files`）；再 Init；`SetKernelsDir`。
- T07（unit）KDA canary：仅 `.../kda/<canary>.json+.cubin` 存在时才不 skip。
- T08（unit，可选 python）：`collect_artifacts(job_a_dir)` 不含 job_b；未加则延期并在 coding-snapshot 声明。
- 容器：完整 cmake gen + `cuModuleLoad` **不是**本需求闸门；verify 以 T01–T07 编译运行为主。cmake 配置阶段 env 非空时应把 `TRITON_KERNEL_PATH` 指到 `{env}/triton_kernel_files`（可用日志确认，不强制单独用例）。

### G. 风险待办【必填，可为空】

- COMMAND 替换依赖参数值精确 `STREQUAL ${TRITON_KERNEL_PATH}`。写成 HOME 字面量或多余 `PATH/` 则仍写根、stamp 查子目录 → 每次 regen。
- regen 只删任务子目录，不扫删根上旧平铺（只占磁盘、不再加载）。
- `tools/generate_triton_kernel_cubin.sh` 仍指向过时平铺路径。
- cmake 与运行时若 env 不一致，会生成到一处、加载另一处；打包以 build 目录 cmake 记录为准，不依赖打包时的 env。
- 运行时未设 env 时使用编译期绝对路径，与 ctest 的 `CMAKE_CURRENT_BINARY_DIR` cwd 无关。wheel 仍靠 `__init__.py` setdefault env，覆盖编进二进制的本机构建路径。
- `ai_docs` 仍把 cache 写成平铺 / HOME；本次不改 wiki（reflect 再提）。

### Env 执行环境提示【必填，本项目新增】

- 本次改动能否在编码环境内做语法/类型级自检？**部分能**：C++/cmake/python 文本可审阅；本机若无完整 `build/`、gtest、GPU，不能编译 `triton_wrapper_test` 或跑 kernel gen（F4）。
- 是否涉及需要容器内才能验证的行为？**是**：真 cubin 与 `cuModuleLoad` 需 NVIDIA GPU；布局解析单测不需要 GPU。
- 建议的验证范围（增量单测目标）：`triton_wrapper_test`（T01–T06）；能编则加 `kda_kernel_wrapper_test` / `kda_attention_layer_test`（T07）。
- 是否预判需要集成测试：**否**作为本需求闸门；完整 `ksana_triton_gen_run` 走现有编译流水。
