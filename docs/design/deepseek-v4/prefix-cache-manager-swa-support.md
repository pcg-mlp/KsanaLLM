# PrefixCacheManager SWA Cache 支持设计文档

## 1. 背景与动机

### 1.1 SWA 机制简介

Sliding Window Attention（SWA）是 DeepSeek-V4 等模型中部分 Transformer 层采用的注意力机制。与全量注意力（Full Attention，FA）不同，SWA 层的每个 token 只关注距离自身最近的 `W` 个 token（滑动窗口），因此其 KV cache 只需保留窗口内的内容。

在 KsanaLLM 的 block 管理体系中，FA block 和 SWA block 使用独立的 block allocator group，block 大小也不同（SWA block 通常是 FA block 的数倍，因为 SWA 层数 × 每层 KV cache 大小）。

### 1.2 现有实现的局限

[`DirectCacheManager`](../../csrc/cache_manager/direct_cache_manager.cpp) 已实现基础 SWA 支持，其策略是：当 token 滑出 SWA 窗口时，立即调用 `FreeBlocks` 释放对应的 SWA block。这种"即用即弃"策略对于无 prefix cache 的场景是合理的。

然而，[`PrefixCacheManager`](../../csrc/cache_manager/prefix_cache_manager.cpp) 目前完全不支持 SWA，且 DirectCacheManager 的即用即弃策略也不适用于 prefix cache 场景：

- PrefixCacheManager 将 FA block 持久化在前缀树上，供后续请求复用。
- 若对应的 SWA block 已被释放，命中该 FA block 的新请求无法获得 SWA cache，导致 prefix cache 命中失效或推理结果错误。
- SWA block 尺寸远大于 FA block，无法全量保留所有 FA block 对应的 SWA block，必须设计选择性存储策略。

### 1.3 设计目标

1. 在 PrefixCacheManager 上实现 SWA KV cache 支持，使 SWA 模型能够利用 prefix cache 机制复用历史 SWA cache。
2. 设计独立的 `SWACacheManager` 模块，管理 FA block 与 SWA block 之间的动态绑定关系。
3. 实现基于 LFU（Least Frequently Used）的 SWA block 选择性存储与驱逐策略。
4. 在 prefix cache 匹配阶段增加 SWA window 内 SWA cache 有效性检查，确保命中的 FA block 对应的 SWA cache 仍然存在。

---

## 2. 整体架构

### 2.1 模块关系

```
PrefixCacheManager
├── block_allocator_group_          (FA block 的显存管理)
├── swa_cache_manager_              (新增，SWA block 的逻辑管理)
│   └── swa_block_allocator_group_  (SWA block 的显存管理)
├── root_cached_block_              (前缀树根节点)
├── free_cached_blocks_             (空闲 FA block 队列)
└── reusable_cached_blocks_         (可复用 FA block 集合)
```

`SWACacheManager` 是一个独立模块，通过 `fa_block_id`（即 `PrefixCachedBlock::block_id`）作为外部键进行操作，与 PrefixCacheManager 的前缀树机制正交。

### 2.2 FA block 与 SWA block 的绑定关系

每个 FA block 在其生命周期内，可以与至多一个 SWA block 绑定。绑定关系由 `SWACacheManager` 维护，PrefixCachedBlock 结构本身不感知 SWA block 的存在。

```
PrefixCachedBlock (fa_block_id=5)
        │
        │  fa_block_id 作为键
        ▼
SWACacheManager::swa_blocks_
        │
        └── SWACachedBlock {
              fa_block_id = 5,
              memory_block_id = 42,   ← SWA 显存 block id
              is_content_fixed = true,
              hit_count = 7,
              is_reclaimable = false  ← 由上层 PrefixCacheManager 直接控制
            }
```

---

## 3. SWACacheManager 设计

### 3.1 数据结构

```cpp
// SWA block 的状态描述
struct SWACachedBlock {
  // 对应的 FA block id（PrefixCachedBlock::block_id）
  size_t fa_block_id = 0;

  // 实际的 SWA 显存 block id（来自 swa_block_allocator_group_）
  int memory_block_id = -1;

  // 是否已填充完整 SWA KV cache，内容已固定，可被其他请求复用
  bool is_content_fixed = false;

  // prefix cache 命中次数，用于 LFU 驱逐决策
  size_t hit_count = 0;

  // 是否处于可回收状态，由上层 PrefixCacheManager 直接控制：
  //   - 请求结束或 SWA 窗口滑过该 block 时，上层调用 SetReclaimable(true)
  //   - 新请求 prefix cache 命中该 block（block 进入新请求 SWA 窗口）时，
  //     上层调用 SetReclaimable(false)
  bool is_reclaimable = false;

  // 当前 FA block 前面的前序 FA block 数量（即 block_index，等于该 block 在 FA block 序列中的位置）
  // 用于驱逐策略：相同 hit_count 时，prefix_fa_block_num 越大的 block 复用价值越高（节省更多计算），
  // 应优先保留，即优先驱逐 prefix_fa_block_num 较小的 block
  size_t prefix_fa_block_num = 0;
};

class SWACacheManager {
 public:
  SWACacheManager(std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group,
                  size_t sliding_window_size,
                  size_t block_token_num);

  // 为指定 FA block 分配一个 SWA block（is_content_fixed=false, is_reclaimable=false）
  // prefix_fa_block_num：当前 FA block 前面的前序 FA block 数量（即 block_index），
  //                      用于后续驱逐策略决策
  // 若该 FA block 已有关联的 SWA block：
  //   - 若旧 block 处于 reclaimable 状态（FA block 重用路径），先释放旧 block 再分配新 block
  //   - 若旧 block 处于 non-reclaimable 状态（仍被请求使用），拒绝分配并返回 -1，
  //     以防止覆写正在使用的 SWA cache
  // 返回分配到的 SWA memory_block_id，失败返回 -1
  int AllocateSWABlock(size_t fa_block_id, size_t prefix_fa_block_num);

  // 查询当前可供分配的 SWA block 数量（空闲池 + 可回收池之和）
  // 用于 AllocateRequestBlocks 预检查：在实际分配前确认 SWA 侧有足够的 block 可用，
  // 避免 FA block 已分配而 SWA block 不足导致的半分配状态
  size_t GetAvailableBlockCount() const;

  // 将指定 FA block 对应的 SWA block 标记为 content-fixed
  void SetSWABlockContentFixed(size_t fa_block_id);

  // 设置指定 FA block 对应 SWA block 的可回收状态
  // is_reclaimable=true：请求结束或 SWA 窗口滑过，block 可被驱逐
  // is_reclaimable=false：新请求命中该 block，block 进入活跃使用
  void SetReclaimable(size_t fa_block_id, bool is_reclaimable);

  // 增加 prefix cache 命中次数
  void IncrementHitCount(size_t fa_block_id);

  // 释放指定 FA block 对应的 SWA block（FA block 被驱逐时调用）
  void FreeSWABlock(size_t fa_block_id);

  // 查询指定 FA block 对应的 SWA block 是否存在且有效
  bool IsSWABlockValid(size_t fa_block_id) const;

  // 获取指定 FA block 对应的 SWA memory_block_id，不存在返回 -1
  int GetSWAMemoryBlockId(size_t fa_block_id) const;

 private:
  // 按优先级驱逐一个可回收的 SWA block，返回被驱逐的 fa_block_id
  size_t EvictOneSWABlock();

  std::shared_ptr<BlockAllocatorGroupInterface> swa_block_allocator_group_;
  size_t sliding_window_size_;
  size_t block_token_num_;

  // fa_block_id -> SWACachedBlock
  std::unordered_map<size_t, std::unique_ptr<SWACachedBlock>> swa_blocks_;

  // 可回收的非 content-fixed SWA block（优先驱逐）
  std::unordered_set<size_t> reclaimable_non_content_fixed_;

  // 可回收的 content-fixed SWA block，按 hit_count 升序排列（LFU）
  std::list<size_t> reclaimable_content_fixed_lfu_;
  std::unordered_map<size_t, std::list<size_t>::iterator> lfu_iters_;
};
```

### 3.2 SWA block 状态机

`is_reclaimable` 由上层 `PrefixCacheManager` 直接控制，`SWACacheManager` 不维护引用计数，状态转换简洁清晰：

```
                    AllocateSWABlock()  ◄──────────────────────────────────────┐
                          │                                                    │
                          ▼                                                    │
                    ALLOCATED (is_content_fixed=false, is_reclaimable=false)   │
                    ← 当前请求正在写入，不可回收                                   │
                          │                                                    │
              ┌───────────┴──────────────────────────────┐                    │
              │                                          │                    │
   SetSWABlockContentFixed()                     FreeSWABlock()               │
   FA block 填充完成，挂入前缀树                  以下情况直接释放，不经过 CONTENT_FIXED：
              │                                  · FA block 被 merge 到已有前缀树节点
              ▼                                  · 请求结束但 block 不具备复用条件
        CONTENT_FIXED (is_content_fixed=true,      （如 partial block、请求中止等）
                      is_reclaimable=false)               │
        ← 此时仍有请求在使用，不可回收                    ▼
              │                                        FREED
  ┌───────────┼───────────────────────────────┐  FreeSWABlock() 释放显存
  │           │                               │
SetReclaimable(false)  IncrementHitCount()  SetReclaimable(true)
新请求命中，进入活跃    hit_count++          请求结束/窗口滑过
  │                                               │
  ▼                                               ▼
IN_USE (is_reclaimable=false)          RECLAIMABLE (is_reclaimable=true)
不可被驱逐                              加入可回收集合，等待驱逐
                                                  │
                              ┌───────────────────┴──────────────────────┐
                              │                                          │
                   EvictOneSWABlock() 或 FreeSWABlock()          AllocateSWABlock()
                   （FA block 被 LRU 驱逐时）                    FA block 从 reusable_cached_blocks_
                              │                                  取出重用，内容将被覆写
                              ▼                                          │
                           FREED                                         │
                    FreeSWABlock() 释放显存                              └──► ALLOCATED
                                                                         （旧 SWA block 先被释放，
                                                                          再分配新 SWA block）
```

**关键设计原则**：`is_reclaimable` 的语义是"当前没有请求正在使用此 SWA block"，由上层在请求生命周期事件（分配、命中、结束、窗口推进）中直接设置，无需在 `SWACacheManager` 内部维护引用计数。

**直接释放路径**：`ALLOCATED` 状态的 SWA block（`is_content_fixed=false`）在以下情况下直接调用 `FreeSWABlock()` 释放，不进入 `CONTENT_FIXED` 或 `RECLAIMABLE` 状态：
- FA block 在 `AppendFilledCachedBlock` 中被 merge 到已有前缀树节点
- 请求结束时 FA block 不具备复用条件（如 partial block、请求中止等）

**FA block 重用路径**：当 `AllocateRequestBlocks` 从 `reusable_cached_blocks_` 取出 FA block 时，该 FA block 可能处于 `RECLAIMABLE` 状态（有关联的 SWA block）。此时调用 `AllocateSWABlock` 会先释放旧 SWA block，再分配新 SWA block，使 SWA block 从 `RECLAIMABLE` 直接转换到新的 `ALLOCATED` 状态。

### 3.3 SWA block 分配优先级

当需要为新的 FA block 分配 SWA block 时，按以下优先级依次尝试：

**优先级 a（最高）：从空闲池直接分配**

```
swa_block_allocator_group_->GetDeviceBlockAllocator(0)->AllocateBlocks(1, blocks)
```

若空闲池有剩余，直接分配，无需驱逐任何已有 SWA block。

**优先级 b（最低）：驱逐可回收 + `is_content_fixed=true` 的 block（LFU + prefix_fa_block_num）**

这类 block 有 prefix cache 价值，应尽量保留。驱逐时按以下复合排序选取价值最低的 block：

1. **主排序键**：`hit_count` 升序（命中次数越少，复用价值越低）
2. **次排序键**：`prefix_fa_block_num` 升序（相同命中次数时，prefix_fa_block_num 越小的 block 复用时节省的计算量越少，优先驱逐）

驱逐后，对应 FA block 的 `IsSWABlockValid` 返回 `false`，prefix cache 匹配时该位置的 SWA cache 视为无效。

> **注**：`is_content_fixed=false` 的 SWA block 不会出现在可回收集合中——它们要么正在被当前请求写入（不可回收），要么在请求结束时直接被 `FreeSWABlock()` 释放。因此驱逐候选只有 `is_content_fixed=true` 的 block。

---

## 4. PrefixCacheManager 集成设计

### 4.1 PrefixCachedRequest 扩展

在 [`PrefixCachedRequest`](../../csrc/cache_manager/prefix_cache_manager.h) 中增加 SWA 相关字段：

```cpp
struct PrefixCachedRequest {
  int64_t req_id;
  RequestState req_state = RequestState::kWaiting;
  size_t shared_block_num = 0;
  std::vector<PrefixCachedBlock*> cached_blocks;

  // 新增：当前请求在 SWA window 内命中的 FA block id 列表
  // 用于：
  //   1. 请求结束时批量调用 SetReclaimable(true)，标记 SWA block 可回收; 或者block部分填充，直接调用FreeSWABlock释放
  //   2. UpdateRequestTokens 中 SWA 窗口推进时，对滑出窗口的 block 调用 SetReclaimable(true)
  std::vector<size_t> swa_window_fa_block_ids;
};
```

### 4.2 FA block 操作与 SWACacheManager 操作的对应关系

| PrefixCacheManager 操作 | SWACacheManager 操作 | 说明 |
|---|---|---|
| `GetRequestPrefixBlockNumber`：Phase 2 — 最终端点 SWA window 内的命中 FA block | `SetReclaimable(fa_block_id, false)` + `IncrementHitCount(fa_block_id)` | 对 `[win_start, last_swa_valid_idx)` 范围内的 block 标记不可回收并记录命中次数；Phase 1 纯扫描不产生副作用 |
| `AllocateRequestBlocks`：**分配前预检查** | `GetAvailableBlockCount()` | 在实际分配任何 block 之前，同时检查 FA 侧和 SWA 侧的可用 block 数量，确保两侧均有足够 block 后才开始分配；详见 §4.3 |
| `AllocateRequestBlocks`：从 `free_cached_blocks_` 取出新 FA block | `AllocateSWABlock(fa_block_id)` | 为新 FA block 申请 SWA block，`is_content_fixed=false, is_reclaimable=false` |
| `AllocateRequestBlocks`：从 `reusable_cached_blocks_` 取出 FA block（经由 `FreeCachedBlocks`） | `AllocateSWABlock(fa_block_id)` | FA block 内容将被覆写；若该 FA block 已有 SWA block（`RECLAIMABLE` 状态），先释放旧 SWA block，再分配新 SWA block |
| `AppendFilledCachedBlock`：FA block 填充完成，挂入前缀树 | `SetSWABlockContentFixed(fa_block_id)` | 标记 SWA block 内容固定，可复用 |
| `AppendFilledCachedBlock`：FA block merge 到已有节点 | `FreeSWABlock(new_fa_block_id)` | 新 FA block 被合并，释放其 SWA block |
| `UpdateRequestTokens`：SWA 窗口推进，block 滑出窗口 | `SetReclaimable(fa_block_id, true)` | block 不再被当前请求 SWA 窗口覆盖，标记可回收 |
| `DestroyFinishedRequest` | `SetReclaimable(fa_block_id, true)` for each id in `swa_window_fa_block_ids` | 请求结束，释放所有 SWA window 内 block 的占用 |
| `FreeCachedBlocks`：FA block 被 LRU 驱逐（不再重用） | `FreeSWABlock(fa_block_id)` | FA block 彻底释放时同步释放 SWA block |

### 4.3 AllocateRequestBlocks 分配前预检查

在 `AllocateRequestBlocks` 中，**必须在任何实际分配操作开始之前**，同时检查 FA 侧和 SWA 侧的可用 block 数量，确保两侧均有足够的 block 可供分配，才开始执行分配工作。这一预检查避免了"FA block 已分配而 SWA block 不足"的半分配状态，保证分配的原子性。

#### 可用 block 数量的计算

**FA 侧可用 block 数**：

```
fa_available = free_cached_blocks_.size() + reusable_cached_blocks_.size()
```

其中 `free_cached_blocks_` 是空闲 FA block 队列，`reusable_cached_blocks_` 是可复用（可被驱逐重用）的 FA block 集合。

**SWA 侧可用 block 数**：

```
swa_available = swa_cache_manager_->GetAvailableBlockCount()
```

`GetAvailableBlockCount()` 返回空闲 SWA block 数量与可回收 SWA block 数量之和：

```
swa_available = free_swa_blocks + reclaimable_non_content_fixed_.size() + reclaimable_content_fixed_lfu_.size()
```

其中 `free_swa_blocks` 来自 `swa_block_allocator_group_` 的空闲池查询。

#### 预检查逻辑

```cpp
// 计算本次请求需要分配的新 FA block 数量
const size_t needed_blocks = /* 请求所需总 block 数 */ - request->shared_block_num;

if (needed_blocks == 0) {
  // 无需分配，直接返回
  return Status::OK();
}

// FA 侧可用 block 数检查
const size_t fa_available = free_cached_blocks_.size() + reusable_cached_blocks_.size();
if (fa_available < needed_blocks) {
  return Status::RESOURCE_EXHAUSTED("Not enough FA blocks");
}

// SWA 侧可用 block 数检查（仅当 SWA 启用时）
if (swa_cache_manager_ != nullptr) {
  const size_t swa_available = swa_cache_manager_->GetAvailableBlockCount();
  if (swa_available < needed_blocks) {
    return Status::RESOURCE_EXHAUSTED("Not enough SWA blocks");
  }
}

// 两侧均有足够 block，开始实际分配
// ...
```

#### GetAvailableBlockCount 的实现语义

`SWACacheManager::GetAvailableBlockCount()` 的计数规则：

- **空闲池**：`swa_block_allocator_group_` 中尚未分配的 SWA block 数量（通过 allocator 接口查询）
- **可回收池**：`reclaimable_non_content_fixed_.size() + reclaimable_content_fixed_lfu_.size()`（可被驱逐腾出的 SWA block 数量）

两者之和即为当前可供 `AllocateSWABlock` 成功调用的最大次数上界。注意此处是**乐观估计**：可回收池中的 block 在驱逐时不会失败（驱逐操作本身不依赖外部资源），因此该上界是准确的。

#### 设计要点

1. **预检查不修改任何状态**：`GetAvailableBlockCount()` 是纯查询接口，不触发驱逐，不修改任何 block 状态。
2. **检查与分配之间无并发修改**：`AllocateRequestBlocks` 在持有锁的情况下执行，预检查与后续分配之间不会有其他线程修改 block 状态，因此预检查结果在分配阶段仍然有效。
3. **SWA 禁用时跳过**：当 `swa_cache_manager_ == nullptr` 时，跳过 SWA 侧检查，行为与原 `PrefixCacheManager` 完全一致。

### 4.4 prefix cache 匹配时的 SWA window 检查（GetRequestPrefixBlockNumber）

修改 [`GetRequestPrefixBlockNumber`](../../csrc/cache_manager/prefix_cache_manager.cpp) 中的前缀树遍历逻辑，在 SWA window 内的 FA block 匹配时增加 SWA cache 有效性检查。

实现分为两个阶段：**Phase 1** 纯扫描确定最远有效端点，**Phase 2** 对最终端点的 SWA window 内 block 注册状态。

#### SWA window 大小计算

设 SWA 窗口大小为 `W`（token 数），block 大小为 `B`（token 数）：

```
sliding_window_blocks = ceil(W / B)
```

`sliding_window_blocks` 是 SWA window 覆盖的 block 数量（向上取整，确保边界 block 被包含）。当 `swa_cache_manager_ == nullptr` 或 `W == 0` 时，`sliding_window_blocks = 0`，表示 SWA 禁用，所有 FA 命中均无条件有效。

#### 端点有效性判定：连续窗口条件

SWA 要求每个 token 都能看到其前 `W` 个 token 的 KV cache。因此，**端点 `(i+1)` 有效**当且仅当：位置 `[max(0, i+1 - W_blocks), i]` 内的所有 block 均有有效的 SWA cache（即最后 `W_blocks` 个 block 连续有效）。

实现上使用**连续命中计数器** `swa_valid_in_window`：

- 遇到 SWA 有效的 block：`swa_valid_in_window++`
- 遇到 SWA 无效的 block：`swa_valid_in_window = 0`（连续性中断，重置）
- 当 `swa_valid_in_window >= sliding_window_blocks` 时，当前端点有效，更新 `last_swa_valid_idx = i + 1`

这与原设计中"独立检查每个 block"的方式不同——**单个 block 有效不足以使端点有效，必须保证最后 `W_blocks` 个 block 全部连续有效**。

**示例：**

```
block_token_num=16, W=32 tokens → sliding_window_blocks=2, FA 匹配 block 0~9
SWA cache 状态：block0~3=有效, block4=有效, block5=已驱逐, block6=有效, block7=有效, block8=已驱逐, block9=有效

匹配过程（streak = swa_valid_in_window，从 i=0 开始累积）：
  i=0: SWA 有效 → streak=1, <2 → 不更新
  i=1: SWA 有效 → streak=2, >=2 → last_swa_valid_idx=2
  i=2: SWA 有效 → streak=3, >=2 → last_swa_valid_idx=3
  i=3: SWA 有效 → streak=4, >=2 → last_swa_valid_idx=4
  i=4: SWA 有效 → streak=5, >=2 → last_swa_valid_idx=5
  i=5: SWA 无效 → streak=0（连续性中断）
  i=6: SWA 有效 → streak=1, <2 → 不更新
  i=7: SWA 有效 → streak=2, >=2 → last_swa_valid_idx=8
  i=8: SWA 无效 → streak=0（连续性中断）
  i=9: SWA 有效 → streak=1, <2 → 不更新

最终 last_swa_valid_idx=8，shared_block_num=8（block 0~7）
Phase 2 注册：win_start=8-2=6，注册 block 6~7
```

> **注**：SWA 禁用区域（`sliding_window_blocks=0`）时，`swa_valid_in_window` 不参与判断，所有 FA 命中位置均直接有效，`last_swa_valid_idx` 不被使用（`cached_blocks` 不截断）。

#### Phase 1：FA 树扫描，确定 last_swa_valid_idx

```cpp
// sliding_window_blocks = ceil(sliding_window_size / block_token_num)，0 表示 SWA 禁用
size_t sliding_window_blocks = 0;
if (swa_cache_manager_ != nullptr && cache_manager_config_.sliding_window_size > 0) {
  sliding_window_blocks = (cache_manager_config_.sliding_window_size + block_token_num - 1) / block_token_num;
}

size_t last_swa_valid_idx = 0;
size_t swa_valid_in_window = 0;  // 当前连续 SWA 有效 block 数

// 对已匹配 block 预填充 streak（从尾部向前统计连续有效尾段）
if (sliding_window_blocks > 0 && !request->cached_blocks.empty()) {
  const size_t already = request->cached_blocks.size();
  for (size_t k = already; k > 0; --k) {
    if (swa_cache_manager_->IsSWABlockValid(request->cached_blocks[k - 1]->block_id)) {
      ++swa_valid_in_window;
    } else {
      break;  // 连续性中断，停止
    }
  }
  if (swa_valid_in_window >= sliding_window_blocks) {
    last_swa_valid_idx = already;
  }
}

PrefixCachedBlock* cur_block = request->cached_blocks.empty() ? root_cached_block_ : request->cached_blocks.back();
for (size_t i = request->cached_blocks.size(); i < check_token_num / block_token_num; ++i) {
  cur_block = FindChildCacheBlock(cur_block, tokens + i * B, B);
  if (cur_block == nullptr) break;

  request->cached_blocks.push_back(cur_block);

  if (sliding_window_blocks == 0) {
    // SWA 禁用：所有 FA 命中无条件有效
    continue;
  }

  // 更新连续命中计数器
  if (swa_cache_manager_->IsSWABlockValid(cur_block->block_id)) {
    ++swa_valid_in_window;
  } else {
    swa_valid_in_window = 0;  // 连续性中断，重置
  }

  // 端点有效条件：最后 W_blocks 个 block 全部连续有效
  if (swa_valid_in_window >= sliding_window_blocks) {
    last_swa_valid_idx = i + 1;
  }
  // 连续性不足时继续扫描，不提前终止
}

// 截断到最远有效端点
if (sliding_window_blocks > 0) {
  request->cached_blocks.resize(last_swa_valid_idx);
}
```

#### Phase 2：对最终端点的 SWA window 注册状态

确定 `last_swa_valid_idx` 后，对最终端点的 SWA window 内 block（即 `[last_swa_valid_idx - W_blocks, last_swa_valid_idx)` 范围）执行状态注册：

```cpp
if (sliding_window_blocks > 0 && last_swa_valid_idx > 0) {
  const size_t win_start =
      (last_swa_valid_idx >= sliding_window_blocks) ? last_swa_valid_idx - sliding_window_blocks : 0;
  request->swa_window_start_block_idx = win_start;
  for (size_t i = win_start; i < last_swa_valid_idx; ++i) {
    const size_t fa_block_id = request->cached_blocks[i]->block_id;
    if (swa_cache_manager_->IsSWABlockValid(fa_block_id)) {
      swa_cache_manager_->SetReclaimable(fa_block_id, false);
      swa_cache_manager_->IncrementHitCount(fa_block_id);
      request->swa_window_fa_block_ids.push_back(fa_block_id);
    }
  }
}
```

**关键设计要点：**

1. **两阶段分离**：Phase 1 纯扫描，不产生任何 SWA 状态副作用；Phase 2 仅对最终确定的端点注册状态。避免了扫描过程中对被截断 block 的误操作。

2. **注册范围是最终端点的 window**：只注册 `[win_start, last_swa_valid_idx)` 范围内的 block，而非所有扫描过的 SWA 有效 block。这确保 `swa_window_fa_block_ids` 精确反映当前请求实际使用的 SWA window。

3. **`swa_window_start_block_idx`**：记录最终端点 SWA window 的起始 block 索引，供 `UpdateRequestTokens` 中窗口推进时使用。

### 4.5 swa_window_fa_block_ids 的维护

`swa_window_fa_block_ids` 记录当前请求持有的、在 SWA window 内的所有 FA block id（包括 prefix cache 命中的和新分配的），用于在请求结束或窗口推进时批量调用 `SetReclaimable(true)`。

**填充时机 1 — prefix cache 命中**：在 `GetRequestPrefixBlockNumber` 的 Phase 2 中，对最终端点的 SWA window 内 block（`[win_start, last_swa_valid_idx)` 范围）调用 `SetReclaimable(fa_block_id, false)` 并将其 `fa_block_id` 加入 `swa_window_fa_block_ids`。注意只有通过 `IsSWABlockValid` 检查的 block 才会被加入（Phase 1 已保证此范围内所有 block 均有效，因此实际上全部加入）。

**填充时机 2 — 新 block 分配**：在 `AllocateRequestBlocks` 中，为请求分配新 FA block（来自 `free_cached_blocks_` 或 `reusable_cached_blocks_`）时，这些 block 将被当前请求写入，均属于当前请求的 SWA window，需将其 `fa_block_id` 追加到 `swa_window_fa_block_ids`。注意：此时不需要调用 `SetReclaimable(false)`，因为新分配的 block 处于 `ALLOCATED` 状态（`is_reclaimable=false`），由 `AllocateSWABlock` 保证。

**窗口推进时**：在 `UpdateRequestTokens` 中，当 SWA 窗口左边界推进时，对滑出窗口的 FA block 调用 `SetReclaimable(fa_block_id, true)`，并从 `swa_window_fa_block_ids` 中移除。窗口不会在 `AllocateRequestBlocks` 时自动推进，只有 `UpdateRequestTokens` 才会触发窗口滑动。

**清理时机**：在 `DestroyFinishedRequest` 中，遍历 `swa_window_fa_block_ids`，对每个 `fa_block_id` 调用 `swa_cache_manager_->SetReclaimable(fa_block_id, true)`。

---

## 5. 关键场景分析

### 5.1 正常 prefix cache 命中场景

```
配置：block_token_num=16, W=64 tokens → sliding_window_blocks=4

请求 A 完成后，FA block 0~7 留在前缀树，对应 SWA block 均为 content-fixed。
DestroyFinishedRequest 调用 SetReclaimable(true) for FA block 4~7（请求 A 的 SWA window 内 block）。
is_reclaimable=true，加入 reclaimable_content_fixed_lfu_。

请求 B 到来，token 序列与 A 完全相同（8 个 block），FA 全部命中：

Phase 1 扫描（streak = swa_valid_in_window）：
  i=0: SWA 有效 → streak=1, <4 → 不更新
  i=1: SWA 有效 → streak=2, <4 → 不更新
  i=2: SWA 有效 → streak=3, <4 → 不更新
  i=3: SWA 有效 → streak=4, >=4 → last_swa_valid_idx=4
  i=4: SWA 有效 → streak=5, >=4 → last_swa_valid_idx=5
  i=5: SWA 有效 → streak=6, >=4 → last_swa_valid_idx=6
  i=6: SWA 有效 → streak=7, >=4 → last_swa_valid_idx=7
  i=7: SWA 有效 → streak=8, >=4 → last_swa_valid_idx=8
  → 全部 8 个 FA block 命中，shared_block_num=8

Phase 2 注册（win_start = 8-4 = 4，注册 block 4~7）：
  对 FA block 4~7 调用 SetReclaimable(false) + IncrementHitCount
  将 FA block 4~7 加入请求 B 的 swa_window_fa_block_ids
  swa_window_start_block_idx = 4
```

### 5.2 SWA block 被驱逐后的 prefix cache 截断场景

```
配置：block_token_num=16, W=64 tokens → sliding_window_blocks=4

SWA 显存不足，SWACacheManager 驱逐了 FA block 5、7 对应的 SWA block（hit_count 最低）。
（FA block 5、7 的 is_reclaimable=true，在 reclaimable_content_fixed_lfu_ 中被选中驱逐）

请求 C 到来，token 序列与 A 相同（8 个 block），FA 全部命中：
SWA cache 状态：block0~4=有效, block5=已驱逐, block6=有效, block7=已驱逐

Phase 1 扫描（streak = swa_valid_in_window）：
  i=0: SWA 有效 → streak=1
  i=1: SWA 有效 → streak=2
  i=2: SWA 有效 → streak=3
  i=3: SWA 有效 → streak=4, >=4 → last_swa_valid_idx=4
  i=4: SWA 有效 → streak=5, >=4 → last_swa_valid_idx=5
  i=5: SWA 无效 → streak=0（连续性中断）
  i=6: SWA 有效 → streak=1, <4 → 不更新 last_swa_valid_idx
  i=7: SWA 无效 → streak=0（连续性中断）
  → 遍历结束，last_swa_valid_idx=5，截断到 block 0~4，shared_block_num=5

Phase 2 注册（win_start = 5-4 = 1，注册 block 1~4）：
  对 FA block 1~4 调用 SetReclaimable(false) + IncrementHitCount
  将 FA block 1~4 加入请求 C 的 swa_window_fa_block_ids
  swa_window_start_block_idx = 1

  请求 C 需要重新计算 FA block 5~7 的 KV cache（block 5 SWA 缺失导致连续性中断，
  block 6 虽然 FA 命中但其 SWA window [3,6] 中 block 5 缺失，端点 7 无效）
```

### 5.3 新 FA block 分配时的 SWA block 驱逐场景

```
SWA 显存已满，所有 SWA block 均已分配。

请求 D 分配新 FA block（fa_block_id=100），需要 SWA block：
  1. 尝试从空闲池分配 → 失败（已满）
  2. 尝试从 reclaimable_non_content_fixed_ 驱逐 → 假设有 fa_block_id=50（is_content_fixed=false）
     → 驱逐 fa_block_id=50 的 SWA block，将其显存 block id 分配给 fa_block_id=100
  3. 若 reclaimable_non_content_fixed_ 也为空，从 reclaimable_content_fixed_lfu_ 头部
     选取 hit_count 最低的 fa_block_id=30 驱逐
     → 驱逐后 IsSWABlockValid(30) 返回 false
     → 后续 prefix cache 匹配遇到 FA block 30 时会截断
```

---

## 6. 改动范围

### 6.1 新增文件

| 文件 | 说明 |
|---|---|
| `csrc/cache_manager/swa_cache_manager.h` | `SWACacheManager` 类声明及 `SWACachedBlock` 结构定义 |
| `csrc/cache_manager/swa_cache_manager.cpp` | `SWACacheManager` 实现，包含 LFU 驱逐逻辑 |

### 6.2 修改文件

| 文件 | 修改内容 |
|---|---|
| `csrc/cache_manager/prefix_cache_manager.h` | `PrefixCachedRequest` 增加 `swa_window_fa_block_ids`；`PrefixCacheManager` 增加 `swa_cache_manager_` 成员；构造函数增加 `swa_block_allocator_group` 参数 |
| `csrc/cache_manager/prefix_cache_manager.cpp` | 修改 `GetRequestPrefixBlockNumber`、`AllocateRequestBlocks`、`AppendFilledCachedBlock`、`UpdateRequestTokens`、`DestroyFinishedRequest`、`FreeCachedBlocks` |
| `csrc/cache_manager/cache_manager_factory.h/.cpp` | 创建 `PrefixCacheManager` 时透传 `swa_block_allocator_group` 参数 |

---

## 7. 注意事项

1. **SWA block 预算规划**：SWA block 的大小是 FA block 的数倍，需在配置层面合理规划 `swa_block_num`，避免占用过多显存影响 FA block 的可用数量。建议在 `CacheManagerConfig` 中增加独立的 `swa_block_num` 配置项。

2. **SWA window 边界的 block 对齐**：SWA window 的左边界 token 不一定对齐到 block 边界。当 `swa_left_token % block_token_num != 0` 时，边界 block 内有部分 token 在窗口内、部分在窗口外。采用保守策略：只要 block 内有任意 token 在 SWA window 内，就要求该 block 的 SWA cache 存在。


4. **`is_reclaimable` 与 FA block `active_requests` 的语义差异**：FA block 的 `active_requests` 追踪所有引用该 block 的请求（包括 SWA window 外的），而 SWA block 的 `is_reclaimable` 只反映 SWA window 内是否有请求正在使用。两者由上层分别维护，互不干扰。

5. **`is_reclaimable` 状态的正确性保证**：由于 `is_reclaimable` 由上层直接控制，需确保每次 `SetReclaimable(false)` 都有对应的 `SetReclaimable(true)` 调用（在请求结束或窗口推进时）。`swa_window_fa_block_ids` 列表是追踪这一配对关系的关键数据结构。

6. **无 SWA 配置时的兼容性**：当 `swa_block_allocator_group` 为 `nullptr` 时，`swa_cache_manager_` 不初始化，所有 SWA 相关逻辑跳过，行为与原 PrefixCacheManager 完全一致。

7. **并发安全**：`SWACacheManager` 的内部数据结构在多线程环境下需要加锁保护，特别是 `reclaimable_content_fixed_lfu_` 列表的修改操作。
