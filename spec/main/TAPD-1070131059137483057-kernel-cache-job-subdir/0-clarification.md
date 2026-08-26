---
spec_version: "1"
spec_type: change
id: "1070131059137483057"
status: draft
subdomain: main
cross_subdomains: []
complexity: L
work_type: refactor
---

> TAPD：[kernel cache的算子分离子目录](https://tapd.woa.com/tapd_fe/70131059/story/detail/1070131059137483057)  
> 父需求：编译加速（1070131059137476238）。TAPD 描述为空，正文以会话共识为准。  
> 复杂度：按「影响面 cmake / runner / wrapper / 单测」定为 **L**；制品落在同一 `TRITON_KERNEL_PATH` 树内，不升 XL。

### A. 工作类型（work_type）【必填，先于目标】
`refactor` — 架构重构（结构/可维护性，默认行为不变）

次要动机：并发生成时 `collect_artifacts` 对整棵 cache `rglob` 会串台、同名文件无命名空间。主类型仍是重构布局，不新增推理能力。

### B. 目标（goal）

问题：Triton/TileLang 生成物全部平铺在 `triton_kernel_files/`，任务之间没有目录命名空间；cmake 队列多任务并发写入同一目录，stamp 采集按 mtime 扫整树，容易把邻居文件记进本任务清单，也挡不住未来同名 cubin/json 覆盖。不同仓库副本若共用 `$HOME/.cache` 还会互相覆盖。

重构范围：按 **cmake 生成队列任务**（`ksana_enqueue_triton_gen` 的 `NAME`，如 `fused_moe`、`kda`、`mhc`）把该任务写出的 kernel 放到 `triton_kernel_files/<NAME>/`。该任务的 stamp 也放在同一子目录的 `md5_sum/`（`triton_kernel_files/<NAME>/md5_sum/`），与 cubin 闭环；cache 根不再放全局 `md5_sum/`。生成、stamp 校验、`collect_artifacts` 的 `artifact_dir` 都必须绑到该任务子目录。运行时 `TritonWrapper` 在仍只选**一个** `kernels_dir_` 的前提下，**只从该根下的任务子目录**加载 `{kernel_name}.json/.cubin`；根上平铺文件不参与解析，子目录内的 `md5_sum/` 不参与解析。打包（`setup.py` copytree）跟随目录树，不在打包时拍平；**`md5_sum/` 一并打进 wheel**，供后期校验；`.creator_logs` / `*.log` 仍 ignore。

`kernels_dir_` 来源（不读 YAML）：cmake 与运行时同一条规则——`KSANA_TRITON_KERNEL_PATH` 非空则用 `{env}/triton_kernel_files`；未设置时 cmake 写 `${CMAKE_BINARY_DIR}/../triton_kernel_files`，并把该**绝对路径**编进二进制，运行时默认用它（不依赖 cwd）。`make test` 的 cwd 是各测试 `CMAKE_CURRENT_BINARY_DIR`，不能再用相对路径 `../triton_kernel_files`。不再写 `$HOME/.cache/KsanaLLM/triton_kernel_files`。不加 YAML 配置项。

行为不变：`ConstructKernelName` 与 kernel 发射语义不变；同一 `kernel_base_name` + key 映射仍应对应同一份已生成 cubin 的内容，只改磁盘路径。不按 creator 内部 GPU 候选（如 wfp4afp8 的 24 组 BLOCK）再拆一层子目录。不改算子数值结果，不改 GPU 调度策略。旧平铺 cache **不兼容、不双写、不读**；下次 configure 因 stamp 对不上而按新布局重生即可。不加 YAML 配置项。

**按 cmake 生成任务 NAME 隔离 kernel cache 子目录，运行时只从任务子目录加载 cubin。**
