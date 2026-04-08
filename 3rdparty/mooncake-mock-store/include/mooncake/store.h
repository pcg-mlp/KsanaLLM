/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// When real Mooncake store is enabled, ReplicateConfig is defined in replica.h.
// Must include BEFORE opening namespace mooncake to avoid pulling transitive
// includes (glog, atomic, ylt...) into the mooncake namespace.
#ifdef ENABLE_MOONCAKE_REAL_STORE
#  include "replica.h"
#endif

namespace mooncake {

#ifndef ENABLE_MOONCAKE_REAL_STORE
struct ReplicateConfig {
  size_t replica_num = 1;
  bool with_soft_pin = false;
  std::string preferred_segment;
};
#endif

/// Mock replacement for MooncakeDistributedStore.
///
/// Uses a shared directory as backend: each key = one file.
/// flock() guarantees cross-process correctness.
///
/// Thread safety: cross-process safe via flock; however flock is per-process,
/// so concurrent threads within the same process are NOT mutually excluded.
/// Callers must add their own mutex if multiple threads access the same key.
///
/// Key charset: keys must only contain [a-zA-Z0-9_\-] characters.
/// Keys containing '/' or '..' will cause path traversal issues.
class MockStore {
 public:
  MockStore();
  ~MockStore();

  /// metadata_server is the shared directory path.
  /// Other params are kept for API compat but ignored.
  int setup(const char* local_hostname, const char* metadata_server, uint64_t global_segment_size = 0,
            uint64_t local_buffer_size = 0, const char* protocol = "", const char* device_name = "",
            const char* master_server_addr = "");

  int close();
  const char* get_hostname() const;

  int put(const char* key, const void* data, size_t size, const ReplicateConfig* config = nullptr);
  int64_t get(const char* key, void** out_data);
  int remove(const char* key);
  long remove_all();
  int is_exist(const char* key);
  int64_t get_size(const char* key);

  int register_buffer(uintptr_t ptr, size_t size);
  int unregister_buffer(uintptr_t ptr);
  int put_from(const char* key, uintptr_t ptr, size_t size, const ReplicateConfig* config = nullptr);
  int64_t get_into(const char* key, uintptr_t ptr, size_t size);

  int put_batch(const char** keys, const void** values, const size_t* sizes, size_t count,
                const ReplicateConfig* config = nullptr);
  int get_batch(const char** keys, size_t count, void** out_datas, int64_t* out_sizes);

 private:
  bool initialized_ = false;
  std::string hostname_;
  std::string dir_;

  std::string path(const char* key) const;
};

}  // namespace mooncake
