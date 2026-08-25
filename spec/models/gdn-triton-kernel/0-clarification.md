---
spec_version: "1"
spec_type: change
id: "gdn-triton-kernel"
status: draft
subdomain: models
cross_subdomains: []
complexity: L
work_type: feature
---

### A. 工作类型（work_type）【必填，先于目标】
`feature` — 功能实现（新增能力/配置/对外行为）

### B. 目标（goal）
现状：Qwen3.5/3.6 混合架构使用 GatedDeltaNet（GDN，线性注意力）作为 backbone，其 Triton kernel（chunk
前向六 kernel 链、decode recurrent 更新、conv1d 等）KsanaLLM 尚无实现；vLLM v0.21.0
（`vllm/model_executor/layers/fla/` + `mamba/ops/`）有成套生产验证的实现。

目标：vendor vLLM v0.21.0 的 GDN kernel 源码（保留上游 license 与源链接）；配套离线 AOT cubin 生成管线
（`gdn/tools/gen_gdn_cubins.py` + `cubin_gen/*.py`，按模型几何产出 cubin/json/ptx；cubin 的 .entry
符号与 `kernel_key_map_` 键、cubin 文件名前缀三者统一——vendored `@triton.jit` 函数按需改名对齐，
其中与 KDA 集合同名 kernel 以 `gdn_` 前缀做命名空间隔离）；并落地 15 用例的 launch 单测守护
kernel ABI（参数数量/顺序/类型与 C++ 调用侧 1:1）。

**为 Qwen3.5/3.6 GDN 建立 vendored Triton kernel 通路（AOT cubin 生成 + C++ 启动 ABI），供后续模型侧 MR 接入。**

### C. 非目标（non_goals）
- 不含模型侧接入（`csrc/models/qwen3_5/` forward / weight loader / GDN 层编排——后续 MR）
- 不含 GDN 状态池 / mamba cache 管理（后续 MR）
- 不做 kernel 数值或性能优化（vendor 源逐字对齐 vLLM v0.21.0，仅 `@triton.jit` 函数名按上条改名对齐；融合 kernel 与未融合链 bitwise 一致）
- 不含 gluon / TMA warp-specialized 路径（上游实验特性，不 vendor）

### D. 影响面（impact_surface）
- `3rdparty/LLM_kernels/csrc/kernels/nvidia/others/vllm/main/gdn/`（新增：vendored ops + cubin 生成器 + tools）
- `csrc/kernels/nvidia/triton_wrapper.h/.cpp`（`kernel_key_map_` 新增 GDN 条目；loader 行为与 master 一致——符号名 == `kernel_base_name`）
- `csrc/kernels/nvidia/gdn_kernel_launch_test.cpp`（新增：15 用例 ABI 守护单测）
- `csrc/kernels/nvidia/CMakeLists.txt`、`csrc/kernels/CMakeLists.txt`（测试目标注册）

本次**不涉及**：模型代码、cache/调度、`moe/common.py`（dead-param 兜底链移除后与 master 一致，不入本 MR）。

### E. 行为不变量（invariants）
- 非 GDN kernel 的 `InvokeTritonKernel` 调用行为不变：既有调用零改动，loader 与 master 逐字节一致
- 与 KDA 集合同名 kernel（`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` / `recompute_w_u_fwd_kernel`）的 cubin 互不串加载：GDN 侧文件名前缀与 `.entry` 符号均带 `gdn_` 前缀（cubin 符号查找限定在本文件模块内，与 KDA 无符号冲突）
- cubin 或 json 缺失时启动显式失败（`KLLM_THROW`），不静默跳过 kernel 发射导致推理结果错乱
- vendored kernel 数值行为与 vLLM v0.21.0 上游一致（`fused_conv_post_c1` 与未融合 conv+post 链 bitwise 一致）
- 跨语言 ABI 契约（python 生成器 ↔ C++ 调用侧，改动任一侧须同步另一侧）：
  - `rmsnorm_gated` 的 gate 缓冲（z）为物化 `[M=L*HV, N=V]` 行主序 fp32（kernel 整行 tile 读取，非 `[L, HV]` 广播布局）
  - `NUM_CACHE_LINES=2049` 双处定义：`cubin_gen/fused_conv_post_c1.py` 与 C++ `kGdnNumCacheLines`（2048 live slots + slot 0）
  - `conv_state` 必须与 x/w 同 dtype（bf16）；`h`（chunk 中间状态）为 model dtype（bf16），`h0/ht` 为 fp32——dtype_contract 表为唯一真值
  - `SISQ = 1 + mtp` 在生成器 `--SISQ`、`gen_gdn_cubins.py` 文件名模板、C++ 测试三处推导（SISQ 进文件名，错配即加载失败 KLLM_THROW）
  - `conv_state_indices` 生成期烘焙 stride=1，C++ 侧必须传连续 `[N]` int32 数组

### F. 功能分支（functional_branches）
- cubin 在而 json 缺的脏状态：`ensure` 预检视为缺失重新生成（json 是运行期加载入口，缺失直接 terminate）
- 并发 TP-rank 同时触发 cubin 生成：文件锁串行化，后到者重检后静默退出
- chunk_h 缺 `H0OFF` 键拼名命中旧 cubin → 参数错位（Xid 31）：文件名键隔离杜绝
- `rmsnorm_gated` 的 dead 参数槽（B/Mean，HAS_BIAS=0/IS_RMS_NORM=1 下不读取）：必须传有效地址而非 nullptr，否则 `cuLaunchKernelEx` invalid-argument
- MTP spec 变体（SL=3+mtp 的 conv update + spec recurrent）仅 mtp>0 配置生成

### G. 建议测试点（suggested_tests）
- id: T01
  scenario: `gdn_kernel_launch_test` 15 用例（fused_post_conv / chunk 族 6 kernel / rmsnorm_gated / fused_conv_post_c1 主+boundary / conv1d_update 含 SL4 spec 变体 / spec recurrent MTP1+2 / packed_decode），真实 launch + sync + error check
  expect: 全部用例 launch 成功且 `cudaGetLastError()==cudaSuccess`；参数槽位数与 C++ 调用侧一致（错位即越界/OOB）
  covers: ABI 不变量、dead 参数槽分支
  check_type: unit
  carrier: sm89/sm90 实机执行（非 CI——A10 非 GDN 目标平台，CMake 构建门禁排除 sm86）；结果回贴 4-reviews
- id: T02
  scenario: `gen_gdn_cubins.py ensure` 幂等（全量存在时静默；删任一 cubin+json 后只补缺失）
  expect: 无重复编译、无脏状态残留；后校验发现模板漂移时非零退出
  covers: 脏状态分支、文件锁分支
  check_type: sandbox
- id: T03
  scenario: 端到端生成正确性（0.8B/27B TP1/TP2、含 MTP spec decode）
  expect: 输出与接入前 baseline 一致；spec 与非 spec 路径 token 级一致
  covers: 数值不变量
  check_type: manual

### H. 资源/硬件约束（resource_constraints）
- cubin 按运行 GPU 的 SM 由 `ensure` 在模型 init 时生成（非 CMake 期），构建无 GPU 依赖；A10(sm86)/H20(sm90) 均可
- `solve_tril` USE_TMA=False（sm_90 以下无 TMA）

### I. 回退方案（rollback）
- revert 本 commit 即完全回退（kernel 通路无存量调用方；模型侧 MR 未合入前无依赖）
- cubin 产物目录（`~/.cache/KsanaLLM/triton_kernel_files/`）可整体删除，`ensure` 幂等重生成

### J. 风险备注（risk_notes）
- vendored 源锁 vLLM v0.21.0：上游修复/演进不会自动跟进，需要时人工比对重 vendor（`@triton.jit` 函数名已改为与文件名前缀一致，比对时注意 def 行的差异）
- 符号改名（符号==文件名前缀）后，旧缓存 cubin 与新命名不符且文件名未变，`ensure` 存在性检查判不出：需删除 `~/.cache/KsanaLLM/triton_kernel_files` 整体重生成（加载失败表现为 `cuModuleGetFunction` NOT_FOUND，显式报错）
- KDA/GDN 重名隔离是命名约定（`gdn_` 前缀 + `kernel_key_map_` 双条目），非类型系统保障；后续任一侧新增同名 kernel 需人工核对两处 map 条目与 cubin 命名
- Triton 版本敏感：dead-param 剥除行为 3.2/3.6 不同，本 MR 按 3.2 镜像固化（num_params 兜底链已随全线统一 3.2 移除）；若镜像升级需复核各 kernel 实参数

### K. 需要人确认的点（待确认清单）
- [ ] [?] TAPD 单号待补——目录暂用 slug `gdn-triton-kernel`，有关联单后可重命名目录
- [ ] [?] 复杂度定档 L（跨 3rdparty/csrc 两代码树 + AOT 生成链）是否符合团队口径
- [ ] [AUTO?] T03 端到端验证挂到后续模型侧 MR 执行（本 MR 无模型 forward 路径，单测 + 生成器自检为本地验证上限）
