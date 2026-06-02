/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/device_utils.h"

namespace ksana_llm {

// Forward declaration to avoid circular include with layer_workspace_manager.h
class LayerWorkspaceManager;

// Layer execution context: optional overrides for stream and workspace.
// Passed as a default parameter to layer Forward() methods.
// When default-constructed (all nullptrs), the layer uses its default behavior.
struct LayerExecContext {
  Stream* stream = nullptr;                         // Override compute stream (null = use primary)
  LayerWorkspaceManager* workspace_mgr = nullptr;   // Override workspace (null = use default)

  // Default construction: all fields null, layer uses default behavior
  LayerExecContext() = default;

  // Convenience: specify stream only
  explicit LayerExecContext(Stream& s) : stream(&s) {}

  // Convenience: specify stream + workspace
  LayerExecContext(Stream& s, LayerWorkspaceManager& ws) : stream(&s), workspace_mgr(&ws) {}

  bool HasStreamOverride() const { return stream != nullptr; }
  bool HasWorkspaceOverride() const { return workspace_mgr != nullptr; }
};

}  // namespace ksana_llm
