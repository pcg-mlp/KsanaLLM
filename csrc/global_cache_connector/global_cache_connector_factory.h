/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "global_cache_connector/global_cache_connector_interface.h"

namespace ksana_llm {

/**
 * @brief 全局 cache 连接器的创建工厂。
 *
 * 根据 config.backend_type 选择并创建具体的后端实现。
 * 目前支持的后端: "mooncake-mock" (MooncakeMockConnector), "mooncake" (MooncakeConnector)。
 * 新增后端只需添加对应的头文件和一个 else if 分支。
 *
 * 每次调用 CreateConnector 返回独立的实例，由调用方通过 shared_ptr 管理生命周期。
 */
class GlobalCacheConnectorFactory {
 public:
  /**
   * @brief 根据 config.backend_type 创建并初始化连接器实例。
   *
   * @param config 连接器配置，其中 backend_type 决定使用哪个后端实现。
   * @return std::shared_ptr<GlobalCacheConnectorInterface> 已初始化的连接器，失败时返回 nullptr。
   */
  static std::shared_ptr<GlobalCacheConnectorInterface> CreateConnector(const GlobalCacheConnectorConfig& config);
};

}  // namespace ksana_llm
