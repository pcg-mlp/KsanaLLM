# Remote Cache Loading 优化设计文档

## 1. 背景与动机

### 1.1 现有 Remote Cache Loading 流程

在当前实现中，FA cache 和 SWA cache 均存放在显存中。为了提高 prefix cache 匹配的 token 命中长度，系统支持在外部存储（remote store）中存放这些 cache block，并在请求推理前尝试从外部存储加载。

当前流程（以 input token 对应 10 个 block、SWA 窗口对应 1 个 block 为例）：

```
Step 1. GetRequestPrefixBlockNumber
        → 匹配显存中满足 FA cache 和 SWA cache 的最远 token
        → FA cache 命中更高时，按 SWA cache 命中位置截断 FA cache 命中长度
        → 假定 FA cache 匹配长度=5，SWA cache 匹配长度=3
        → 截断后：请求关联 fa_blocks=3 个，swa_blocks=1 个

Step 2. AllocateRequestBlocks(N=5)
        → 分配 5 个新 block 用于远程加载
        → 请求关联 fa_blocks=8 个，swa_blocks=6 个

Step 3. 远程加载 block index [3,7] 区间的 5 个 fa block 和 5 个 swa block
        → 假设最远加载到 index=5

Step 4. UpdateRequestTokens
        → 确认 shareable_kvcache_token_num = 6 × block_token_num

Step 5. 继续计算剩余 block index [6,9] 的 token
```

### 1.2 现有流程的问题

**问题 1：SWA block 过度分配**

在 Step 2 中，`AllocateRequestBlocks` 需要分配 N 个 SWA block（与 FA block 数量相同）。但在 Step 3 中，远程加载实际只会返回 SWA 窗口大小的结果（即 1 个 SWA block）。这导致大量 SWA block 被无效占用，造成显存浪费。

**问题 2：FA cache 命中被不必要截断**

在 Step 1 中，`GetRequestPrefixBlockNumber` 的当前语义是让请求进入可以直接执行的状态，因此需要将 FA cache 位置和 SWA cache 对齐。然而 Step 3 还会尝试远程加载两种 cache，本来 FA block 可以保留 5 个，只需尝试远程加载 index 区间 [5,7] 的 3 个 cache，而不是从 index=3 开始加载 5 个。

### 1.3 设计目标

1. 新增 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` 接口（对应原 `need_align_swa=false` 模式），允许 FA cache 和 SWA cache 命中位置不对齐，减少不必要的远程加载范围。原 `GetRequestPrefixBlockNumber` 保持 `need_align_swa=true` 语义不变。
2. 新增 `AllocateBlocksForRemoteCache` 接口，仅分配实际需要的 SWA block 数量（SWA 窗口大小），避免过度分配。
3. 新增 `ConfirmRemoteCacheLoadingResult` 接口，根据远程加载结果调整 block 状态，并支持预留部分 block 供后续推理使用。
4. 在 `SWACacheManager` 上新增 `Transfer` 接口，支持 SWA block 在不同 FA block 之间的转移。

---

## 2. 优化后的流程设计

### 2.1 新流程概览

以 input token 对应 10 个 block、SWA 窗口对应 1 个 block 为例：

```
Step 1. GetRequestPrefixBlockNumberBeforeRemoteCacheLoading
        → 匹配显存中 FA cache 和 SWA cache 的最远 token，不强制对齐
        → FA cache 匹配长度=5，SWA cache 匹配长度=3
        → 请求关联 fa_blocks=5 个，swa_blocks=1 个（绑定在 fa_block[2] 上）

Step 2. AllocateBlocksForRemoteCache(fa_block_num=3, swa_block_num=1)
        → 分配 3 个新 fa block（追加在 cached_blocks 上）
        → 分配 1 个新 swa block（记录在 remote_cache_blocks 上）
        → 返回 remote_fa_blocks=[5,6,7]，remote_swa_blocks=[swa_0]

Step 3. 远程加载
        → fa_blocks 需加载 index [5,7]（3 个）
        → swa_blocks 需加载 index [4,7]（4 个，因为 swa 只匹配到 index=3）
        → 假设最远加载到 matched_block_index=5
        → fa block index [0,5] 内容均可用
        → remote_swa_blocks[k] 包含 index=5 的 swa block 内容
        → matched_remote_swa_block_indexs[0]=k

Step 4. ConfirmRemoteCacheLoadingResult(matched_block_index=5,
                                        matched_remote_swa_block_indexs=[k])
        → 调用 SWACacheManager::Transfer 将 remote_swa_blocks[k] 转移到 fa_block[5]
        → 调整 cached_blocks 和 remote_cache_blocks 状态
        → 可选：预留部分 block 供后续推理

Step 5. 继续计算剩余 block index [6,9] 的 token
```

### 2.2 新旧流程对比

| 步骤 | 旧流程 | 新流程 |
|---|---|---|
| Step 1 | `GetRequestPrefixBlockNumber`，FA/SWA 强制对齐，FA 命中截断为 3 | `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading`，FA/SWA 不对齐，FA 命中保留 5 |
| Step 2 | 分配 5 个 FA block + 5 个 SWA block | 分配 3 个 FA block + 1 个 SWA block |
| Step 3 | 远程加载 [3,7] 共 5 个 FA+SWA block | FA 加载 [5,7]，SWA 加载 [4,7] |
| Step 4 | UpdateRequestTokens 直接确认 | ConfirmRemoteCacheLoadingResult 调整 |

---

## 3. 请求状态机

### 3.1 remote_loading 状态

在新流程中，`PrefixCachedRequest` 引入两个远程加载子状态，用于区分 `AllocateBlocksForRemoteCache` 调用前后的不同阶段：

- **`kRemoteLoading`**：调用 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` 后进入，此时 `cached_blocks` 中的 block **尚未**将 `req_id` 注册到引用计数中（仅作为候选）。
- **`kRemoteLoadingBlockAllocated`**：调用 `AllocateBlocksForRemoteCache` 成功后进入，此时新分配的 FA block 和 SWA block 已将 `req_id` 注册到引用计数中，必须通过 `TruncateRequestBlocks` 释放。

```
                           │
          ┌────────────────┴────────────────────────────────┐
          │ GetRequestPrefixBlockNumber                      │ GetRequestPrefixBlockNumberBeforeRemoteCacheLoading
          │ （FA/SWA 强制对齐）                              │ （FA/SWA 不对齐）
          ▼                                                  ▼
    kRunning（直接推理）                              kRemoteLoading
                                                           │
                                                           │  可选：AllocateBlocksForRemoteCache
                                                           │  （最多调用一次）
                                                           │
                                          ┌────────────────┴────────────────────────────────┐
                                          │ 未调用 AllocateBlocksForRemoteCache              │ 调用 AllocateBlocksForRemoteCache 成功
                                          │ （跳过远程加载，直接确认）                        │
                                          ▼                                                  ▼
                                    kRemoteLoading                               kRemoteLoadingBlockAllocated
                                          │                                                  │
                                          └──────────────────┬──────────────────────────────┘
                                                             │
                                                             ▼
                                                   ConfirmRemoteCacheLoadingResult
                                                             │
                                          ┌──────────────────┴──────────────────────────────┐
                                          │ 来自 kRemoteLoading                              │ 来自 kRemoteLoadingBlockAllocated
                                          │ （未分配过 block）                               │ （已分配 block，req_id 已注册）
                                          ▼                                                  ▼
                                    kWaiting（重新等待调度）                          kRunning（进入正常推理）
```

**状态约束：**

| 状态 | 允许的接口调用 | 禁止的接口调用 |
|---|---|---|
| `kRemoteLoading` | `AllocateBlocksForRemoteCache`、`ConfirmRemoteCacheLoadingResult`、`DestroyFinishedRequest`、`DestroySwappedRequest` | `AllocateRequestBlocks`、`UpdateRequestTokens`、`AppendFilledCachedBlock` 等推理相关接口 |
| `kRemoteLoadingBlockAllocated` | `ConfirmRemoteCacheLoadingResult`、`DestroyFinishedRequest`、`DestroySwappedRequest` | `AllocateBlocksForRemoteCache`（已调用过）、`AllocateRequestBlocks`、`UpdateRequestTokens`、`AppendFilledCachedBlock` 等推理相关接口 |
| 其他状态 | 原有接口（不变） | `AllocateBlocksForRemoteCache`、`ConfirmRemoteCacheLoadingResult` |

**状态转换触发条件：**

- `kWaiting/kRunning` → `kRemoteLoading`：调用 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` 后自动进入。
- `kRemoteLoading` → `kRemoteLoadingBlockAllocated`：调用 `AllocateBlocksForRemoteCache` 成功后进入，此时 block 已注册 `req_id`。
- `kRemoteLoading` → `kWaiting`：调用 `ConfirmRemoteCacheLoadingResult` 后，若未曾调用过 `AllocateBlocksForRemoteCache`（即仍处于 `kRemoteLoading`），则请求回到 `kWaiting` 状态，等待调度器重新分配 block 后推理。
- `kRemoteLoadingBlockAllocated` → `kRunning`：调用 `ConfirmRemoteCacheLoadingResult` 后，若已调用过 `AllocateBlocksForRemoteCache`，则请求进入 `kRunning` 状态，可直接推理（FA cache 与 SWA cache 已对齐）。
- `kRemoteLoading` → 终止：调用 `DestroyFinishedRequest` 或 `DestroySwappedRequest` 时，`cached_blocks` 中的 block 尚未注册 `req_id`，直接丢弃数据结构即可，无需调用 `TruncateRequestBlocks`。
- `kRemoteLoadingBlockAllocated` → 终止：调用 `DestroyFinishedRequest` 或 `DestroySwappedRequest` 时，block 已注册 `req_id`，必须调用 `TruncateRequestBlocks` 释放，同时清理 `remote_cache_swa_fa_block_ids` 中所有未处理的 SWA block。

**`PrefixCachedRequest` 状态字段扩展：**

```cpp
// 在 PrefixCachedRequest 内部用两个独立 bool 标记远程加载子状态：
//
// is_remote_loading：
//   true  → 请求处于 kRemoteLoading 或 kRemoteLoadingBlockAllocated 状态。
//   false → 请求处于其他状态（kWaiting / kRunning / kFinished 等）。
//
// is_remote_loading_block_allocated：
//   true  → AllocateBlocksForRemoteCache 已成功调用，block 已注册 req_id。
//            仅在 is_remote_loading=true 时有意义。
//   false → AllocateBlocksForRemoteCache 尚未调用（或已通过 ConfirmRemoteCacheLoadingResult 退出）。
//
// 组合语义：
//   is_remote_loading=true,  is_remote_loading_block_allocated=false → kRemoteLoading
//   is_remote_loading=true,  is_remote_loading_block_allocated=true  → kRemoteLoadingBlockAllocated
//   is_remote_loading=false, is_remote_loading_block_allocated=false → 正常状态
```

---

## 4. 接口设计

### 4.0 GetRequestPrefixBlockNumber / GetRequestPrefixBlockNumberBeforeRemoteCacheLoading 与 AllocateRequestBlocks / AllocateBlocksForRemoteCache 的语义关系

#### 4.0.1 多次调用与延迟分配

`GetRequestPrefixBlockNumber` 和 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` **均可能被多次调用**。调用方在每次查询后，会根据返回的 `shared_block_num`、`unique_block_num` 等信息判断当前显存是否有足够空间执行下一步操作。只有当空间充足时，才会继续调用对应的 Alloc 接口（`AllocateRequestBlocks` 或 `AllocateBlocksForRemoteCache`）。

```
[正常推理场景]                              [远程加载场景]

loop:                                       loop:
    GetRequestPrefixBlockNumber(req_id, …)      GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(req_id, …)
        → 返回 shared_block_num,                    → 返回 shared_block_num（SWA 对齐），
          unique_block_num                            fa_prefix_block_num（FA 实际命中）

    if 空间不足:                                if 空间不足:
        等待或放弃，稍后重试                        等待或放弃，稍后重试
    else:                                       else:
        调用 AllocateRequestBlocks                  调用 AllocateBlocksForRemoteCache
        break                                       break
```

#### 4.0.2 cached_blocks 有效性检查与重新匹配

`GetRequestPrefixBlockNumber` 和 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` 在每次调用时，均会**检查上次查询结果中 `cached_blocks` 和对应 SWA block 的有效性**：

- 若某个 `cached_blocks` 中的 block 或其对应的 SWA block 在上次查询之后已被系统回收（释放），则该 block 视为**失效**。
- 失效的 block 会从 `cached_blocks` 中删除，并触发**重新匹配**：从失效位置开始，重新在 prefix cache 树中查找可复用的 block。
- 重新匹配的结果会更新 `cached_blocks`，并重新计算 `shared_block_num`。

这一机制保证了：即使在两次查询调用之间有其他请求抢占了 block，本次查询仍能返回一个**自洽且最新**的匹配结果。

#### 4.0.3 req_id 登记时机：查询阶段不登记

**`GetRequestPrefixBlockNumber` 和 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` 均不会将 `req_id` 登记（注册引用）到任何 `cached_blocks` 对应的 FA block 或其关联的 SWA block 上。**

这意味着：
- 查询阶段匹配到的 cached FA block（包括重新匹配后的结果）仅作为"候选"，不持有引用计数。
- **SWA block 的登记操作必须与其关联的 FA block 的登记操作保持同步**：FA block 未登记时，对应的 SWA block 也不得登记。若违反此约束（例如 FA block 未登记但 SWA block 已登记），则当 FA block 被系统回收时，SWA block 将因缺少 FA block 的引用而无法被正确释放，导致 SWA block 泄漏。
- 在查询接口返回之后、Alloc 接口调用之前，这些候选 FA block 及其关联的 SWA block 仍可能被其他请求回收。
- 这正是为什么 `AllocateRequestBlocks`/`AllocateBlocksForRemoteCache` 在实际分配前需要**再次验证** cached_blocks 有效性的原因（见 §4.0.2）。

查询阶段会将 SWA window 范围内的 FA block id 预填入 `swa_window_fa_block_ids`（纯追踪，不调用 `ReuseSWABlock`），供 Alloc 函数直接使用，无需重新扫描。

#### 4.0.4 req_id 登记时机：分配阶段完成登记

`AllocateRequestBlocks` 的输入参数 `req_block_ids` 表示**当前已经被登记过的 block**（即在之前某次成功的分配中已注册引用的 block）。

在通过可行性检查（空间充足、block 有效）后，`AllocateRequestBlocks`/`AllocateBlocksForRemoteCache` 会执行实际分配，并将 `req_id` 登记到所有**尚未登记**的 block 上，包括：

| block 类型 | 说明 |
|---|---|
| 重新匹配的 cached FA block | 查询接口重新匹配后、尚未登记的 cached block |
| 重新匹配的 SWA block | 与上述 FA block 对应的 SWA block（直接遍历查询阶段预填的 `swa_window_fa_block_ids`，调用 `ReuseSWABlock`） |
| 新分配的 FA block | `AllocateRequestBlocks`/`AllocateBlocksForRemoteCache` 新分配的 FA block |
| 新分配的 SWA block | 与新 FA block 对应的 SWA block（`AllocateRequestBlocks`）或远程加载用的临时 SWA block（`AllocateBlocksForRemoteCache`） |

**登记完成后，这些 block 的引用计数增加，不会被系统自动回收，直到请求显式释放。**

#### 4.0.5 两对接口的对应关系

| 场景 | 查询接口 | 分配接口 |
|---|---|---|
| 正常推理（显存 prefix cache） | `GetRequestPrefixBlockNumber` | `AllocateRequestBlocks` |
| 远程加载前的预查询 | `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` | `AllocateBlocksForRemoteCache` |

两对接口均遵循上述"查询不登记、分配时登记"的语义，区别仅在于：
- `AllocateRequestBlocks` 为每个新 FA block 立即分配并绑定一个 SWA block。
- `AllocateBlocksForRemoteCache` 仅分配 SWA 窗口大小数量的 SWA block，延迟绑定到实际匹配的 FA block（由 `ConfirmRemoteCacheLoadingResult` 完成）。

---

### 4.1 GetRequestPrefixBlockNumberBeforeRemoteCacheLoading 接口

在 [`CacheManagerInterface`](../../csrc/cache_manager/cache_manager_interface.h) 和 [`PrefixCacheManager`](../../csrc/cache_manager/prefix_cache_manager.cpp) 中新增独立接口 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading`，对应原 `need_align_swa=false` 语义。原 `GetRequestPrefixBlockNumber` 接口保持不变（FA/SWA 强制对齐）。

```cpp
// 远程加载前的 prefix block 查询接口。
// FA cache 和 SWA cache 命中位置允许不对齐，FA cache 保留更长的命中长度，
// 供后续 AllocateBlocksForRemoteCache + ConfirmRemoteCacheLoadingResult 使用。
// 此时 cached_blocks 中 FA block 数量可能多于 SWA window 覆盖范围。
//
// 调用后请求进入 kRemoteLoading 状态，禁止直接调用 AllocateRequestBlocks 等推理接口，
// 必须通过 ConfirmRemoteCacheLoadingResult 退出 kRemoteLoading 后才能进入推理。
virtual Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(
    int64_t req_id, const std::vector<int>& input_token_ids,
    size_t check_token_num, size_t& shared_block_num,
    size_t& unique_block_num, size_t& shared_token_num) = 0;
```

**与 `GetRequestPrefixBlockNumber` 的行为差异：**

- Phase 1 扫描逻辑不变，仍然计算 `last_swa_valid_idx`（SWA 对齐后的最远有效端点）。
- FA cache 扫描继续向前，不在 `last_swa_valid_idx` 处截断，记录 FA cache 实际命中的最远位置 `last_fa_valid_idx`。
- `cached_blocks` 保留到 `last_fa_valid_idx`，而非截断到 `last_swa_valid_idx`。
- `shared_block_num` 返回 `last_swa_valid_idx`（SWA 对齐的有效 block 数，用于确定可直接推理的 token 数）。
- `PrefixCachedRequest` 中新增字段 `fa_prefix_block_num` 记录 `last_fa_valid_idx`，供后续接口使用。
- **Phase 2（SWA window 预填充）**：将 `[win_start, last_swa_valid_idx)` 范围内的 FA block id 填入 `swa_window_fa_block_ids`（纯追踪，不调用 `ReuseSWABlock`）。由于 `cached_blocks` 中的 FA block 尚未将 `req_id` 注册到引用计数中，若此时调用 `ReuseSWABlock`，将违反 FA/SWA 注册同步约束（§4.0.3）：FA block 被系统回收时，SWA block 因缺少 FA block 引用而无法被正确释放，导致 SWA block 泄漏。`ReuseSWABlock` 的调用延迟到 `AllocateBlocksForRemoteCache` 中执行：该接口在将 `req_id` 注册到 FA block 之后，直接遍历 `swa_window_fa_block_ids` 调用 `ReuseSWABlock`，无需重新扫描。
- 调用后自动将请求置为 `is_remote_loading = true`（kRemoteLoading 状态）。

**`PrefixCachedRequest` 扩展：**

```cpp
struct PrefixCachedRequest {
  // ... 已有字段 ...

  // GetRequestPrefixBlockNumberBeforeRemoteCacheLoading 调用后，
  // FA cache 实际命中的最远 block 数（不受 SWA 对齐截断）。
  // 等于 cached_blocks.size()，在 AllocateBlocksForRemoteCache 中用于：
  //   1. 计算需要远程加载的 FA block 范围（新 block 从 fa_prefix_block_num 开始）。
  //   2. 重新扫描 cached_blocks[0..fa_prefix_block_num) 以注册 SWA block
  //      （延迟自查询阶段，见 §4.0.3）。
  size_t fa_prefix_block_num = 0;

  // AllocateBlocksForRemoteCache 分配的远程 SWA block 对应的临时 FA block id 列表。
  // 通过这些 fa_block_id 可经由 SWACacheManager 查询到对应的 SWA memory_block_id。
  // ConfirmRemoteCacheLoadingResult 调用后，匹配成功的 block 通过 Transfer 绑定到实际 FA block，
  // 未匹配的 block 通过 FreeSWABlock 释放，列表随后清空。
  std::vector<size_t> remote_cache_swa_fa_block_ids;
};
```

### 4.2 AllocateBlocksForRemoteCache 接口

新增接口，用于为远程加载分配 block：

```cpp
// 为远程 cache 加载分配 block。
//
// req_block_ids：调用方当前已登记的 block 的 memory_block_id 列表（输入/输出）。
//   - 传入时包含已通过 GetRequestPrefixBlockNumberBeforeRemoteCacheLoading 匹配并已登记的
//     cached_blocks 对应的 memory_block_id（首次调用时通常为空）。
//   - 函数返回后，新分配的 FA block 的 memory_block_id 会追加到此列表中。
//   - 与 §4.0.4 "分配阶段完成登记"语义对应：函数内部会将 req_id 登记到
//     req_block_ids 中尚未登记的 cached_blocks 和新分配的 FA block 上。
//
// fa_block_num：需要远程加载的 FA block 数量（= 目标 FA block 总数 - 已命中 FA block 数）。
// swa_block_num：需要远程加载的 SWA block 数量（= SWA 窗口大小，与 fa_block_num 无关）。
//
// 分配逻辑：
//   - 新的 FA block 追加在 cached_blocks 上（与 AllocateRequestBlocks 类似）。
//   - 新的 SWA block 信息记录在 remote_cache_swa_fa_block_ids 上，默认新FA block的数量< 新SWA block的数量，新的SWA block依次绑定到新的FA block上。
//
// 返回：
//   remote_fa_block_ids：新分配的 FA block 的 memory_block_id 列表（长度=fa_block_num）。
//   remote_swa_block_ids：新分配的 SWA block 的 memory_block_id 列表（长度=swa_block_num）。
//
// 状态转换：
//   调用成功后，请求状态从 kRemoteLoading 转换为 kRemoteLoadingBlockAllocated。
//   此后 DestroyFinishedRequest/DestroySwappedRequest 必须通过 TruncateRequestBlocks 释放已注册的 block。
Status AllocateBlocksForRemoteCache(int64_t req_id, std::vector<int>& req_block_ids,
                                    size_t fa_block_num, size_t swa_block_num,
                                    std::vector<int>& remote_fa_block_ids,
                                    std::vector<int>& remote_swa_block_ids) override;
```

**与 `AllocateRequestBlocks` 的区别：**

| 特性 | AllocateRequestBlocks | AllocateBlocksForRemoteCache |
|---|---|---|
| FA block 处理 | 追加到 cached_blocks | 追加到 cached_blocks（相同） |
| SWA block 处理 | 每个 FA block 分配一个 SWA block | 仅分配 swa_block_num 个，记录在 remote_cache_blocks |
| SWA block 绑定 | 立即绑定到对应 FA block | 延迟绑定，由 ConfirmRemoteCacheLoadingResult 完成 |
| 使用场景 | 请求正常推理前的 block 分配 | 远程加载前的 block 预分配 |

### 4.3 ConfirmRemoteCacheLoadingResult 接口

新增接口，根据远程加载结果调整 block 状态：

```cpp
// 根据远程 cache 加载结果确认 block 状态，将请求从 kRemoteLoading 状态转换为 kRunning。
//
// kvcached_token_ids：远程加载成功的 token id 列表（完整前缀，长度 = shareable_kvcache_token_num）。
//   - 用于调用 UpdateRequestTokens，将新加载的 block 合并进 prefix cache 树。
//   - 仅当 cached_blocks.size() > fa_prefix_block_num 时（即有新 block 被加载）才使用。
//
// shareable_kvcache_token_num：远程加载成功的 token 数量（必须是 block_token_num 的整数倍）。
//   - shareable_block_num = shareable_kvcache_token_num / block_token_num，
//     等价于原设计中的 matched_block_index + 1。
//   - FA block index [0, shareable_block_num - 1] 的内容均可用。
//   - shareable_block_num 之后的 FA block 内容不可用，将被截断释放。
//
// matched_remote_swa_block_indexs：匹配成功的 remote_cache_swa_fa_block_ids 偏移列表。
//   - 长度 <= 1（SWA 窗口大小固定为 1）。
//   - 若为空（size == 0）：无远程 SWA 匹配，FA cache 截断至 SWA 窗口右边界。
//   - 若为 [k]（size == 1）：remote_cache_swa_fa_block_ids[k] 对应的 SWA block 匹配成功，
//     调用 SWACacheManager::Transfer 将其绑定到 cached_blocks[shareable_block_num-1]。
//
// 注意：reserved_block_num（预留 block）功能暂未实现，如需支持请另行扩展。
//
// 处理逻辑（SWA 路径）：
//   Case 1（matched_remote_swa_block_indexs 为空）：
//     释放所有 remote SWA block，截断 cached_blocks 至 SWA 窗口右边界。
//   Case 2（matched_remote_swa_block_indexs = [k]）：
//     释放旧 SWA 窗口（SetReclaimable），释放未匹配的 remote SWA block，
//     将匹配的 SWA block Transfer 到目标 FA block，重建 SWA 窗口，
//     截断 cached_blocks 至 shareable_block_num。
//
// 状态转换（根据调用前的子状态）：
//   来自 kRemoteLoading（未调用过 AllocateBlocksForRemoteCache）：
//     → req_state = kWaiting，is_remote_loading = false，is_remote_loading_block_allocated = false。
//     cached_blocks 中的 block 尚未注册 req_id，直接丢弃数据结构，不调用 TruncateRequestBlocks。
//     请求回到 kWaiting 状态，等待调度器重新分配 block 后推理。
//   来自 kRemoteLoadingBlockAllocated（已调用过 AllocateBlocksForRemoteCache）：
//     → req_state = kRunning，is_remote_loading = false，is_remote_loading_block_allocated = false。
//     若有新 block 则调用 UpdateRequestTokens 合并进 prefix cache 树。
Status ConfirmRemoteCacheLoadingResult(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                       size_t shareable_kvcache_token_num,
                                       const std::vector<size_t>& matched_remote_swa_block_indexs) override;
```

### 4.4 SWACacheManager::Transfer 接口

在 [`SWACacheManager`](../../csrc/cache_manager/swa_cache_manager.h) 上新增 `Transfer` 接口，支持将一个 SWA block 从一个 FA block 转移到另一个 FA block：

```cpp
// 将 src_fa_block_id 对应的 SWA block 转移到 dst_fa_block_id。
//
// 使用场景：ConfirmRemoteCacheLoadingResult 中，remote_swa_blocks 中的 SWA block
// 原本绑定在临时 FA block 上，需要转移到实际匹配的 FA block 上。
//
// 前置条件（违反时返回 false 并打印 WARNING）：
//   1. src_fa_block_id 对应的 SWA block 必须存在。
//   2. src SWA block 的 is_content_fixed 必须为 false
//      （远程加载的内容尚未提交到 prefix cache 树）。
//   3. src SWA block 必须恰好被一个请求持有（active_req_ids.size() == 1），
//      不允许共享，以避免时序上的状态问题。
//   4. dst_fa_block_id 不能已有任何绑定的 SWA block（包括 reclaimable 状态的 block）。
//
// 转移后：
//   - src_fa_block_id 的 SWA block 记录被移除。
//   - dst_fa_block_id 绑定到原 src 的 memory_block_id。
//   - 保留原 SWA block 的 active_req_ids、hit_count 等状态。
//   - fa_block_id 和 prefix_fa_block_num 更新为 dst 对应的值。
//   - 由于 src block 满足 is_content_fixed=false 且恰好一个持有者，
//     其不在任何 reclaimable 结构中，无需额外的 LFU 维护。
//
// 返回：成功返回 true，失败返回 false。
bool Transfer(size_t src_fa_block_id, size_t dst_fa_block_id, size_t dst_prefix_fa_block_num);
```

---

## 5. 数据结构变更

### 5.1 PrefixCachedRequest 扩展

```cpp
struct PrefixCachedRequest {
  int64_t req_id;
  RequestState req_state = RequestState::kWaiting;
  size_t shared_block_num = 0;
  std::vector<PrefixCachedBlock*> cached_blocks;
  std::vector<size_t> swa_window_fa_block_ids;
  size_t swa_window_start_block_idx = 0;

  // 新增：标识请求是否处于 remote_loading 状态（kRemoteLoading 或 kRemoteLoadingBlockAllocated）。
  // 在 GetRequestPrefixBlockNumberBeforeRemoteCacheLoading 后置为 true，
  // 在 ConfirmRemoteCacheLoadingResult 调用后置为 false。
  // 处于 remote_loading 状态时，禁止调用 AllocateRequestBlocks、
  // UpdateRequestTokens、AppendFilledCachedBlock 等推理相关接口。
  bool is_remote_loading = false;

  // 新增：标识 AllocateBlocksForRemoteCache 是否已成功调用。
  // 与 is_remote_loading 组合使用，区分两个远程加载子状态：
  //   is_remote_loading=true,  is_remote_loading_block_allocated=false → kRemoteLoading
  //     cached_blocks 中的 block 尚未注册 req_id，DestroyFinishedRequest 直接丢弃数据结构。
  //   is_remote_loading=true,  is_remote_loading_block_allocated=true  → kRemoteLoadingBlockAllocated
  //     block 已注册 req_id，DestroyFinishedRequest 必须调用 TruncateRequestBlocks 释放。
  // 在 AllocateBlocksForRemoteCache 成功后置为 true，
  // 在 ConfirmRemoteCacheLoadingResult 调用后置为 false。
  bool is_remote_loading_block_allocated = false;

  // 新增：GetRequestPrefixBlockNumberBeforeRemoteCacheLoading 调用后，
  // FA cache 实际命中的最远 block 数（不受 SWA 对齐截断）。
  // 等于 cached_blocks.size()，在 AllocateBlocksForRemoteCache 中用于：
  //   1. 计算需要远程加载的 FA block 范围（新 block 从 fa_prefix_block_num 开始）。
  //   2. 重新扫描 cached_blocks[0..fa_prefix_block_num) 以注册 SWA block
  //      （延迟自查询阶段，见 §4.0.3）。
  size_t fa_prefix_block_num = 0;

  // 新增：AllocateBlocksForRemoteCache 分配的远程 SWA block 对应的临时 FA block id 列表。
  // 通过这些 fa_block_id 可经由 SWACacheManager 查询到对应的 SWA memory_block_id，
  // 无需单独保存 memory_block_id。
  // ConfirmRemoteCacheLoadingResult 调用后，匹配成功的 block 通过 Transfer 绑定到实际 FA block，
  // 未匹配的 block 通过 FreeSWABlock 释放，列表随后清空。
  std::vector<size_t> remote_cache_swa_fa_block_ids;
};
```

### 5.2 SWACacheManager 新增接口

```cpp
class SWACacheManager {
 public:
  // ... 已有接口 ...

  // 将 src_fa_block_id 对应的 SWA block 转移到 dst_fa_block_id。
  // 详见 §4.4。
  bool Transfer(size_t src_fa_block_id, size_t dst_fa_block_id, size_t dst_prefix_fa_block_num);
};
```

---

## 6. 关键场景分析

### 6.1 正常远程加载场景

```
配置：block_token_num=16, SWA 窗口=1 block, input=10 blocks

Step 1. GetRequestPrefixBlockNumberBeforeRemoteCacheLoading
  显存命中：FA block [0,4]（5 个），SWA block 绑定在 fa_block[2]（匹配到 index=3）
  cached_blocks = [block0, block1, block2, block3, block4]
  fa_prefix_block_num = 5
  shared_block_num = 3（SWA 对齐后的有效 block 数）

Step 2. AllocateBlocksForRemoteCache(fa_block_num=3, swa_block_num=1)
  分配 FA block [5,6,7]，追加到 cached_blocks
  分配 1 个 remote SWA block（swa_slot_0），临时 fa_block_id 记录在 remote_cache_swa_fa_block_ids
  remote_fa_block_ids = [mem_5, mem_6, mem_7]
  remote_swa_block_ids = [swa_mem_0]

Step 3. 远程加载
  FA 加载范围：block index [5,7]（3 个）
  SWA 加载范围：block index [4,7]（4 个，因为 SWA 只匹配到 index=3）
  加载结果：matched_block_index=5，remote_swa_blocks[0] 包含 index=5 的 SWA 内容
  matched_remote_swa_block_indexs = [0]

Step 4. ConfirmRemoteCacheLoadingResult(matched_block_index=5,
                                        matched_remote_swa_block_indexs=[0])
  Transfer(remote_swa_fa_block_ids[0], fa_block[5].block_id, prefix_fa_block_num=5)
  释放 FA block [6,7]（未匹配）
  cached_blocks 截断为 [block0..block5]
  remote_cache_swa_fa_block_ids 清空

Step 5. 继续计算 block index [6,9] 的 token
```

### 6.2 远程加载失败场景

```
Step 3. 远程加载失败，matched_block_index=4（未超过显存命中范围）
  matched_remote_swa_block_indexs = []（无 SWA block 匹配）

Step 4. ConfirmRemoteCacheLoadingResult(matched_block_index=4,
                                        matched_remote_swa_block_indexs=[])
  释放所有 remote_swa_blocks（无匹配）
  释放 FA block [5,7]（全部未匹配）
  cached_blocks 截断为 [block0..block4]
  回退到显存命中状态，shared_block_num=3（SWA 对齐）

Step 5. 继续计算 block index [5,9] 的 token（比正常多计算 1 个 block）
```

### 6.3 预留 block 场景

```
Step 4. ConfirmRemoteCacheLoadingResult(matched_block_index=5,
                                        matched_remote_swa_block_indexs=[0],
                                        reserved_block_num=1)
  Transfer(remote_swa_fa_block_ids[0], fa_block[5].block_id, 5)
  预留 FA block[6]（reserved_block_num=1），不释放
  释放 FA block[7]（超出预留范围）
  cached_blocks = [block0..block6]（block6 为预留 block）
  注意：预留的 block[6] 必须有对应的 SWA cache，否则不能预留

Step 5. 继续计算 block index [7,9] 的 token（block[6] 已预留，无需重新分配）
```

### 6.4 SWA block Transfer 操作

```
场景：remote_swa_blocks[0] 原本绑定在临时 fa_block_id=999 上
      需要转移到实际匹配的 fa_block_id=5 上

Transfer(src_fa_block_id=999, dst_fa_block_id=5, dst_prefix_fa_block_num=5)

操作：
  1. 从 swa_blocks_[999] 取出 memory_block_id=42
  2. 删除 swa_blocks_[999] 记录
  3. 创建 swa_blocks_[5]：
       fa_block_id=5, memory_block_id=42
       is_content_fixed=true（远程加载内容已固定）
       hit_count=0, is_reclaimable=false
       prefix_fa_block_num=5
  4. 返回 true
```

---

## 7. 接口调用时序

```
调用方（Engine）
        │
        ├─ GetRequestPrefixBlockNumberBeforeRemoteCacheLoading
        │   └─ 返回：shared_block_num（SWA 对齐），fa_prefix_block_num（FA 实际命中）
        │
        ├─ AllocateBlocksForRemoteCache(fa_block_num, swa_block_num)
        │   └─ 返回：remote_fa_block_ids, remote_swa_block_ids
        │
        ├─ [远程加载操作，由调用方负责]
        │   └─ 输入：remote_fa_block_ids, remote_swa_block_ids
        │   └─ 输出：matched_block_index, matched_remote_swa_block_indexs
        │
        ├─ ConfirmRemoteCacheLoadingResult(matched_block_index,
        │                                  matched_remote_swa_block_indexs,
        │                                  reserved_block_num)
        │   ├─ SWACacheManager::Transfer(...)  [内部调用]
        │   └─ 释放未匹配的 block
        │
        └─ [继续推理，使用更新后的 cached_blocks]
```

---

## 8. 改动范围

### 8.1 修改文件

| 文件 | 修改内容 |
|---|---|
| [`csrc/cache_manager/cache_manager_interface.h`](../../csrc/cache_manager/cache_manager_interface.h) | 新增 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading`、`AllocateBlocksForRemoteCache`、`ConfirmRemoteCacheLoadingResult` 纯虚接口 |
| [`csrc/cache_manager/prefix_cache_manager.h`](../../csrc/cache_manager/prefix_cache_manager.h) | `PrefixCachedRequest` 增加 `is_remote_loading`、`is_remote_loading_block_allocated`、`fa_prefix_block_num`、`remote_cache_swa_fa_block_ids` 字段 |
| [`csrc/cache_manager/prefix_cache_manager.cpp`](../../csrc/cache_manager/prefix_cache_manager.cpp) | 实现 `GetRequestPrefixBlockNumberBeforeRemoteCacheLoading`；实现 `AllocateBlocksForRemoteCache`（成功后置 `is_remote_loading_block_allocated=true`）；实现 `ConfirmRemoteCacheLoadingResult`（根据 `is_remote_loading_block_allocated` 决定转入 `kWaiting` 或 `kRunning`）；更新 `DestroyFinishedRequest`/`DestroySwappedRequest`（根据 `is_remote_loading_block_allocated` 决定是否调用 `TruncateRequestBlocks`） |
| [`csrc/cache_manager/swa_cache_manager.h`](../../csrc/cache_manager/swa_cache_manager.h) | 新增 `Transfer` 接口声明 |
| [`csrc/cache_manager/swa_cache_manager.cpp`](../../csrc/cache_manager/swa_cache_manager.cpp) | 实现 `Transfer` 方法 |
| [`csrc/cache_manager/direct_cache_manager.h`](../../csrc/cache_manager/direct_cache_manager.h) | 新增 `AllocateBlocksForRemoteCache`、`ConfirmRemoteCacheLoadingResult` 的空实现（返回 `Status::NOT_SUPPORTED`） |

### 8.2 新增文件

本次改动不需要新增文件，所有变更均在现有文件上进行扩展。

---

## 9. 注意事项

1. **`GetRequestPrefixBlockNumberBeforeRemoteCacheLoading` 的使用限制**：该接口仅用于远程加载场景。调用后请求进入 `kRemoteLoading` 状态，必须随后调用 `AllocateBlocksForRemoteCache`（可选）和 `ConfirmRemoteCacheLoadingResult`，不能直接进入推理状态（因为 FA cache 和 SWA cache 未对齐）。

2. **预留 block 的 SWA cache 保证**：`ConfirmRemoteCacheLoadingResult` 中，`reserved_block_num` 指定的预留 block 必须每个都有对应的 SWA cache（来自 `matched_remote_swa_block_indexs` 或已有的显存 SWA cache）。若 SWA cache 不足，应减少预留数量。

3. **Transfer 操作的原子性**：`SWACacheManager::Transfer` 在 `PrefixCacheManager` 的锁保护下执行，无需额外加锁。Transfer 操作本身不涉及显存拷贝，仅修改 `swa_blocks_` 映射关系，开销极小。

4. **remote_cache_swa_fa_block_ids 的生命周期**：`remote_cache_swa_fa_block_ids` 中记录的临时 FA block id 在 `ConfirmRemoteCacheLoadingResult` 调用后必须全部处理完毕（Transfer 或 FreeSWABlock），不允许遗留未处理的 remote SWA block。

5. **与现有 SWA cache 的交互**：`AllocateBlocksForRemoteCache` 分配的 remote SWA block 在 `SWACacheManager` 中以临时 FA block id 注册（`is_content_fixed=false, is_reclaimable=false`）。若 `ConfirmRemoteCacheLoadingResult` 未被调用（如请求被取消），需在 `DestroyFinishedRequest`/`DestroySwappedRequest` 中清理这些临时记录。

6. **`AllocateBlocksForRemoteCache` 的预检查**：与 `AllocateRequestBlocks` 类似，在实际分配前需同时检查 FA 侧和 SWA 侧的可用 block 数量，避免半分配状态。SWA 侧只需检查 `swa_block_num` 个（而非 `fa_block_num` 个），这是本优化的核心收益之一。

7. **`matched_block_index` 与 SWA 对齐**：`ConfirmRemoteCacheLoadingResult` 中，`matched_block_index` 之后的 FA block 若要保留（通过 `reserved_block_num`），必须保证这些 block 有对应的 SWA cache。实现时需验证 `matched_remote_swa_block_indexs` 中是否有足够的 SWA block 覆盖预留范围。

8. **`DestroyFinishedRequest`/`DestroySwappedRequest` 中的子状态判断**：在销毁处于远程加载状态的请求时，必须根据 `is_remote_loading_block_allocated` 区分处理方式：
   - `is_remote_loading=true, is_remote_loading_block_allocated=false`（`kRemoteLoading`）：`cached_blocks` 中的 block 尚未注册 `req_id`，**不得**调用 `TruncateRequestBlocks`，直接丢弃数据结构即可。若有 `remote_cache_swa_fa_block_ids`（理论上此时为空，因为 `AllocateBlocksForRemoteCache` 未调用），也无需处理。
   - `is_remote_loading=true, is_remote_loading_block_allocated=true`（`kRemoteLoadingBlockAllocated`）：block 已注册 `req_id`，**必须**先清理 `remote_cache_swa_fa_block_ids` 中的 remote SWA block（`FreeSWABlock`），再调用 `TruncateRequestBlocks` 释放已注册的 FA block。

   若不区分这两种子状态，在 `kRemoteLoading` 时错误调用 `TruncateRequestBlocks` 会导致对未注册 `req_id` 的 block 执行错误的引用计数操作；在 `kRemoteLoadingBlockAllocated` 时跳过 `TruncateRequestBlocks` 则会导致 block 泄漏。
