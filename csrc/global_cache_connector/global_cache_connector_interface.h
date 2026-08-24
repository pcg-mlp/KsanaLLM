/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "global_cache_connector/global_cache_connector_types.h"
#include "utils/status.h"

namespace ksana_llm {

/**
 * @brief 全局 KV cache 连接器的抽象接口。
 *
 * 该接口由外部动态库实现（编译期链接 .so），用于将 KsanaLLM
 * 连接到全局 KV cache 存储，在本地 prefix cache 查询之后进行补充。
 *
 * 该连接器在 engine 和 executor 两个进程中使用：
 *   - Engine：在本地 prefix cache 查询后查询全局 cache，管理拉取生命周期。
 *   - Executor：执行全局 cache 与 GPU 显存之间的实际 KV 数据传输。
 */
class GlobalCacheConnectorInterface {
 public:
  virtual ~GlobalCacheConnectorInterface() = default;

  /**
   * @brief 使用给定的配置初始化连接器。
   * 必须在调用其他任何 API 之前调用一次。线程安全的初始化由实现方负责。
   */
  virtual Status Initialize(const GlobalCacheConnectorConfig& config) = 0;

  /**
   * @brief 查询全局 cache 并同时拉取匹配的 KV cache 数据。
   *
   * 在本地 PrefixCacheManager 查询到匹配后，该接口会检查全局存储中是否有
   * 额外的缓存 block，并在一次操作中完成查询和拉取。
   *
   * matched_ranges 是一个 vector<int>，大小始终是 2 的倍数。每两个连续
   * 的元素构成一个闭区间 [start, end]，表示从全局 cache 拉取到的 token
   * 位置范围。例如 {0, 127, 256, 383} 表示拉取了 [0,127] 和 [256,383]
   * 两个区间的 token。
   *
   * @param token_ids            完整的输入 token 序列。
   * @param local_cached_len     本地 prefix cache 已匹配的 token 数量。
   * @param tp_size              Tensor parallel 大小。
   * @param tp_rank              当前设备的 tensor parallel rank。
   * @param layer_index          要拉取的 KV cache 的模型层索引。
   * @param block_size           每个 block 包含的 token 数量。
   * @param block_byte_size      单个 KV cache block 的字节大小。
   * @param block_ptrs           设备上 KV cache block 数据的指针，每个 block 对应一个指针。
   * @param matched_ranges       输出：闭区间 [start, end] 的 token 位置范围对。
   *                             大小始终是 2 的倍数。
   * @param retry_until_full_hit 为 true 时（decode 节点），在接口内部持续重试未命中的 block，
   *                             已成功拉取的 block 不会被重复拉取。全部命中或超时后返回。
   * @param timeout_ms           retry_until_full_hit 模式下的超时时间（毫秒）。
   * @param swa_ctx              SWA 侧上下文；非空时先把命中边界对齐到有 SWA 的位置，
   *                             再按对齐后的边界拉取主 KV 与边界 SWA。
   * @param swa_result           SWA 侧输出；swa_ctx 非空时必须一并传入。
   * @return Status 成功返回 OK，否则返回错误。
   */
  virtual Status QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size, int tp_rank,
                               int layer_index, int block_size, size_t block_byte_size,
                               const std::vector<void*>& block_ptrs, std::vector<int>& matched_ranges,
                               bool retry_until_full_hit = false, int timeout_ms = 5000,
                               const GlobalCacheSwaContext* swa_ctx = nullptr,
                               GlobalCacheSwaResult* swa_result = nullptr) = 0;

  /**
   * @brief 将本地计算完成的 KV cache block 上传到全局存储。
   *
   * 在请求完成 prefill 后，可将计算好的 KV cache 上传到全局存储，
   * 供其他实例复用。按 layer 调用。
   *
   * @param token_ids       与该 KV cache 关联的 token 序列。
   * @param tp_size         Tensor parallel 大小。
   * @param tp_rank         当前设备的 tensor parallel rank。
   * @param layer_index     上传的 KV cache 对应的模型层索引。
   * @param block_size      每个 block 包含的 token 数量。
   * @param block_byte_size 单个 KV cache block 的字节大小。
   * @param block_ranges    闭区间 [start, end] 的 token 位置范围对，表示要上传的范围。
   *                        大小始终是 2 的倍数。
   * @param block_ptrs      设备上 KV cache block 数据的指针，每个 block 对应一个指针。
   * @param put_ok_out      可选输出：远端不存在且 BatchPut 成功的 block 个数。
   * @return Status 成功返回 OK，否则返回错误。
   */
  virtual Status PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank, int layer_index,
                                      int block_size, size_t block_byte_size, const std::vector<int>& block_ranges,
                                      const std::vector<void*>& block_ptrs, int* put_ok_out = nullptr) = 0;

  /**
   * @brief 将 SWA block 上传到全局存储，key 用 "skv_" 命名空间。
   *
   * 与主 KV 不同，这里传入的是**显式的稀疏 block 索引列表**而不是连续区间：SWA block 未压缩，
   * 字节数远大于主 KV block（DSv4 实测约 7.6 倍），全量存储会让全局 cache 占用翻倍，
   * 因此调用方按「尾部 N 密存 + 前段等分中位点稀存」选点后再调本接口。
   *
   * @param token_ids           与该 KV cache 关联的 token 序列。
   * @param tp_rank             当前设备的 tensor parallel rank。
   * @param layer_index         上传的 KV cache 对应的模型层索引。
   * @param block_size          每个 block 包含的 token 数量。
   * @param swa_block_byte_size 单个 SWA block 的字节大小。
   * @param block_idxs          要上传的 FA block 全局索引列表，升序。
   * @param block_ptrs          与 block_idxs 一一对应的设备指针。
   * @param put_ok_out          可选输出：远端不存在且 BatchPut 成功的 block 个数。
   * @return Status 成功返回 OK，否则返回错误。
   */
  virtual Status PutSwaBlocks(const std::vector<int>& token_ids, int tp_rank, int layer_index, int block_size,
                              size_t swa_block_byte_size, const std::vector<size_t>& block_idxs,
                              const std::vector<void*>& block_ptrs, int* put_ok_out = nullptr) = 0;

  /**
   * @brief 将 prefill 采样得到的生成 token(s) 保存到全局存储。
   *
   * 在 prefill 完成并采样 + MTP 后调用，保存 generated_tokens + draft_tokens，
   * 以便 decode 节点全命中时可直接获取这些 token，跳过 prefill 计算。
   *
   * @param token_ids  完整的输入 token 序列（不含生成 token）。
   * @param tokens     要保存的 token 列表（generated_tokens + draft_tokens）。
   * @return Status 成功返回 OK，否则返回错误。
   */
  virtual Status PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) = 0;

  /**
   * @brief 查询全局存储中是否缓存了指定输入序列的生成 token(s)。
   *
   * @param token_ids  完整的输入 token 序列。
   * @param tokens     输出：如果命中，存放缓存的 token 列表。
   * @param found      输出：是否命中。
   * @return Status 成功返回 OK，否则返回错误。
   */
  virtual Status QueryFirstTokens(const std::vector<int>& token_ids, std::vector<int>& tokens, bool& found) = 0;

  /**
   * @brief 关闭连接器并释放所有资源。
   * 调用后不应再调用任何其他 API。
   */
  virtual Status Shutdown() = 0;
};

}  // namespace ksana_llm
