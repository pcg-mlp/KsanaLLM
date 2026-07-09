/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <dlfcn.h>

#include <memory>
#include <string>

#include "pd_v2/public/pd_v2_config.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {

// Signature of the ksana_pd_v2_create_connector factory exported by every
// pd_v2 backend shared library (see pd_v2_connector.h).
using PdV2ConnectorFactoryFn = std::shared_ptr<PdV2ConnectorInterface> (*)(const PdV2ConnectorConfig&);

// Result of loading a pd_v2 backend .so: the raw dlopen handle (owned by the
// caller — dlclose it on teardown), the resolved factory symbol, and the
// resolved library name (kept for logging).
struct PdV2ConnectorLibrary {
  void* handle = nullptr;
  PdV2ConnectorFactoryFn factory = nullptr;
  std::string so_name;
};

// dlopen libpd_v2_connector_<backend_type>.so with RTLD_NOW | RTLD_GLOBAL and
// resolve the ksana_pd_v2_create_connector factory symbol. Both the engine
// and the executor load the same backend .so; the role is selected later via
// PdV2ConnectorConfig at factory time. Throws (KLLM_THROW) on dlopen/dlsym
// failure. `context_hint` (e.g. "engine" / "executor") is embedded in the
// failure message to disambiguate the caller; pass "" to omit it.
inline PdV2ConnectorLibrary LoadPdV2ConnectorLibrary(const std::string& backend_type,
                                                     const std::string& context_hint = "") {
  PdV2ConnectorLibrary lib;
  lib.so_name = "libpd_v2_connector_" + backend_type + ".so";
  const std::string where = context_hint.empty() ? "" : (" in " + context_hint);

  lib.handle = dlopen(lib.so_name.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (lib.handle == nullptr) {
    const char* err = dlerror();
    KLLM_THROW(std::string("pd_v2 dlopen(") + lib.so_name + ") failed" + where + ": " + (err ? err : "?"));
  }

  lib.factory = reinterpret_cast<PdV2ConnectorFactoryFn>(dlsym(lib.handle, "ksana_pd_v2_create_connector"));
  if (lib.factory == nullptr) {
    const char* err = dlerror();
    KLLM_THROW(std::string("pd_v2 dlsym(ksana_pd_v2_create_connector) failed") + where + ": " + (err ? err : "?"));
  }
  return lib;
}

}  // namespace pd_v2
}  // namespace ksana_llm
