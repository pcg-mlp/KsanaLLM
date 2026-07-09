/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "utils/status.h"

// Forward decl to keep hiredis out of public headers.
struct redisContext;

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// Thin synchronous hiredis wrapper used by MooncakeEngineConnector for the
// pd_v2 service-discovery key namespace `pd_v2/decode_alive/<cluster>/*`.
//
// One instance owns one `redisContext`; all calls take an internal mutex
// because hiredis sync clients are not thread-safe. Designed for the
// keepalive (Decode) and discovery-poll (Prefill) loops, both of which
// are single-thread per process.
//
// Connection string: `redis://host:port[/db]` — the same metadata_addr
// pd_v2 hands to Mooncake's TransferEngine. Password is sourced from the
// `MC_REDIS_PASSWORD` env (set at boot from ConnectorConfig.redis_password).
class RedisAliveStore {
 public:
  RedisAliveStore() = default;
  ~RedisAliveStore();

  RedisAliveStore(const RedisAliveStore&) = delete;
  RedisAliveStore& operator=(const RedisAliveStore&) = delete;

  // Returns RET_INVALID_ARGUMENT if `redis_url` is not redis://, or
  // RET_RUNTIME_FAILED if the TCP connect / AUTH / SELECT fails. On
  // success the underlying `redisContext` is owned by this instance
  // until Disconnect() / dtor.
  Status Connect(const std::string& redis_url, const std::string& password);

  // Idempotent.
  void Disconnect();

  bool IsConnected() const;

  // SET key value EX ttl_s — overwrites if already present.
  Status Set(const std::string& key, const std::string& value, int ttl_s);

  // DEL key — best-effort, OK whether or not the key existed.
  Status Del(const std::string& key);

  // KEYS pattern — synchronous, drains all matching keys into `out`.
  // Production Redis discourages KEYS on huge keyspaces; pd_v2 uses
  // the dedicated namespace `pd_v2/decode_alive/<cluster>/*` which has size on
  // the order of (#Decode nodes) — fine for KEYS.
  Status Keys(const std::string& pattern, std::vector<std::string>* out);

  // GET key. `out` is set to "" if the key is missing (returns OK).
  Status Get(const std::string& key, std::string* out);

 private:
  Status ConnectLocked(const std::string& redis_url, const std::string& password);

  // Ensures ctx_ is live before issuing a command. hiredis sync contexts are
  // permanently broken once ctx_->err != 0 (every later redisCommand returns
  // NULL), so a transient Redis blip would otherwise wedge the keepalive /
  // discovery loops until process restart. When ctx_ is null or in an error
  // state, frees any stale handle and reconnects using the url/password
  // captured at Connect() time. Returns the connect failure (leaving ctx_
  // null) when the reconnect itself fails. Caller must hold mu_.
  Status EnsureConnectedLocked();

  mutable std::mutex mu_;
  redisContext* ctx_ = nullptr;

  // Captured at Connect() so EnsureConnectedLocked() can transparently
  // re-establish a dropped connection. Empty stored_url_ means Connect()
  // was never called.
  std::string stored_url_;
  std::string stored_password_;
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
