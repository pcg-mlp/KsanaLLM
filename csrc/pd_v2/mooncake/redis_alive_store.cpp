/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/redis_alive_store.h"

#include <hiredis/hiredis.h>

#include <cstdlib>
#include <cstring>

#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

// Parse "redis://host:port[/db]" into pieces. Returns RET_INVALID_ARGUMENT
// on schemes other than redis:// or on a missing port.
Status ParseRedisUrl(const std::string& url, std::string* host, int* port, int* db) {
  constexpr const char* kPrefix = "redis://";
  if (url.rfind(kPrefix, 0) != 0) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("RedisAliveStore: '{}' is not a redis:// URL", url));
  }
  std::string rest = url.substr(std::strlen(kPrefix));
  std::string hostport;
  *db = 0;
  size_t slash = rest.find('/');
  if (slash == std::string::npos) {
    hostport = rest;
  } else {
    hostport = rest.substr(0, slash);
    try {
      *db = std::stoi(rest.substr(slash + 1));
    } catch (...) {
      *db = 0;
    }
  }
  size_t colon = hostport.find(':');
  if (colon == std::string::npos) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("RedisAliveStore: '{}' has no host:port", url));
  }
  *host = hostport.substr(0, colon);
  try {
    *port = std::stoi(hostport.substr(colon + 1));
  } catch (...) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("RedisAliveStore: '{}' port unparsable", url));
  }
  if (host->empty() || *port <= 0 || *port > 65535) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("RedisAliveStore: '{}' invalid host/port", url));
  }
  return Status();
}

// Free reply on scope exit; null-safe.
struct ReplyGuard {
  redisReply* r = nullptr;
  ~ReplyGuard() {
    if (r != nullptr) freeReplyObject(r);
  }
};

}  // namespace

RedisAliveStore::~RedisAliveStore() { Disconnect(); }

bool RedisAliveStore::IsConnected() const {
  std::lock_guard<std::mutex> lk(mu_);
  return ctx_ != nullptr && ctx_->err == 0;
}

Status RedisAliveStore::Connect(const std::string& redis_url, const std::string& password) {
  std::lock_guard<std::mutex> lk(mu_);
  if (ctx_ != nullptr) {
    return Status(RET_INVALID_ARGUMENT, "RedisAliveStore already connected");
  }
  // Remember the params so a later command can reconnect after a Redis blip.
  stored_url_ = redis_url;
  stored_password_ = password;
  return ConnectLocked(redis_url, password);
}

Status RedisAliveStore::EnsureConnectedLocked() {
  if (ctx_ != nullptr && ctx_->err == 0) {
    return Status();
  }
  // Drop the permanently-broken context — hiredis never recovers it.
  if (ctx_ != nullptr) {
    KLLM_LOG_WARNING << "RedisAliveStore: context in error state (" << ctx_->errstr << "), reconnecting";
    redisFree(ctx_);
    ctx_ = nullptr;
  }
  if (stored_url_.empty()) {
    return Status(RET_INVALID_ARGUMENT, "RedisAliveStore: not configured (Connect never called)");
  }
  return ConnectLocked(stored_url_, stored_password_);
}

Status RedisAliveStore::ConnectLocked(const std::string& redis_url, const std::string& password) {
  std::string host;
  int port = 0;
  int db = 0;
  STATUS_CHECK_RETURN(ParseRedisUrl(redis_url, &host, &port, &db));

  // 2s connect timeout — the Redis backend is on the same datacenter and
  // an unresponsive endpoint indicates a real outage; longer waits stall
  // boot. Read timeout uses the same value to keep KEYS / SET bounded.
  timeval timeout{};
  timeout.tv_sec = 2;
  timeout.tv_usec = 0;
  redisContext* c = redisConnectWithTimeout(host.c_str(), port, timeout);
  if (c == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: redisConnect({},{}) returned null", host, port));
  }
  if (c->err != 0) {
    std::string err = c->errstr;
    redisFree(c);
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: connect to {}:{} failed: {}", host, port, err));
  }
  if (redisSetTimeout(c, timeout) != REDIS_OK) {
    KLLM_LOG_WARNING << "RedisAliveStore: redisSetTimeout failed: " << c->errstr;
  }

  // AUTH (default user) if a password is configured. MC_REDIS_PASSWORD
  // env takes precedence — set by inference_engine bootstrap from
  // ConnectorConfig.redis_password and read here so we don't add another
  // config touchpoint.
  std::string pw = password;
  if (pw.empty()) {
    const char* env = std::getenv("MC_REDIS_PASSWORD");
    if (env != nullptr) pw = env;
  }
  if (!pw.empty()) {
    redisReply* r = static_cast<redisReply*>(redisCommand(c, "AUTH %s", pw.c_str()));
    ReplyGuard g{r};
    if (r == nullptr || r->type == REDIS_REPLY_ERROR) {
      std::string err = (r != nullptr && r->str != nullptr) ? r->str : "no reply";
      redisFree(c);
      return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: AUTH failed: {}", err));
    }
  }

  if (db != 0) {
    redisReply* r = static_cast<redisReply*>(redisCommand(c, "SELECT %d", db));
    ReplyGuard g{r};
    if (r == nullptr || r->type == REDIS_REPLY_ERROR) {
      std::string err = (r != nullptr && r->str != nullptr) ? r->str : "no reply";
      redisFree(c);
      return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: SELECT {} failed: {}", db, err));
    }
  }

  ctx_ = c;
  KLLM_LOG_INFO << "RedisAliveStore connected " << host << ":" << port << " db=" << db
                << " auth=" << (pw.empty() ? "no" : "yes");
  return Status();
}

void RedisAliveStore::Disconnect() {
  std::lock_guard<std::mutex> lk(mu_);
  if (ctx_ != nullptr) {
    redisFree(ctx_);
    ctx_ = nullptr;
  }
  // Explicit teardown is final: forget the params so a stray command after
  // shutdown fails cleanly instead of silently re-establishing the connection.
  // Transient drops keep ctx_ non-null with ctx_->err set and are handled by
  // EnsureConnectedLocked() without going through Disconnect().
  stored_url_.clear();
  stored_password_.clear();
}

Status RedisAliveStore::Set(const std::string& key, const std::string& value, int ttl_s) {
  std::lock_guard<std::mutex> lk(mu_);
  STATUS_CHECK_RETURN(EnsureConnectedLocked());
  redisReply* r = static_cast<redisReply*>(
      redisCommand(ctx_, "SET %b %b EX %d", key.data(), key.size(), value.data(), value.size(), ttl_s));
  ReplyGuard g{r};
  if (r == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: SET reply null err={}", ctx_->errstr));
  }
  if (r->type == REDIS_REPLY_ERROR) {
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: SET error '{}'", r->str ? r->str : ""));
  }
  return Status();
}

Status RedisAliveStore::Del(const std::string& key) {
  std::lock_guard<std::mutex> lk(mu_);
  STATUS_CHECK_RETURN(EnsureConnectedLocked());
  redisReply* r = static_cast<redisReply*>(redisCommand(ctx_, "DEL %b", key.data(), key.size()));
  ReplyGuard g{r};
  if (r == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: DEL reply null err={}", ctx_->errstr));
  }
  return Status();
}

Status RedisAliveStore::Keys(const std::string& pattern, std::vector<std::string>* out) {
  if (out == nullptr) return Status(RET_INVALID_ARGUMENT, "Keys: out null");
  out->clear();
  std::lock_guard<std::mutex> lk(mu_);
  STATUS_CHECK_RETURN(EnsureConnectedLocked());
  redisReply* r = static_cast<redisReply*>(redisCommand(ctx_, "KEYS %b", pattern.data(), pattern.size()));
  ReplyGuard g{r};
  if (r == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: KEYS reply null err={}", ctx_->errstr));
  }
  if (r->type != REDIS_REPLY_ARRAY) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("RedisAliveStore: KEYS unexpected reply type {}", static_cast<int>(r->type)));
  }
  out->reserve(r->elements);
  for (size_t i = 0; i < r->elements; ++i) {
    redisReply* el = r->element[i];
    if (el != nullptr && el->type == REDIS_REPLY_STRING && el->str != nullptr) {
      out->emplace_back(el->str, el->len);
    }
  }
  return Status();
}

Status RedisAliveStore::Get(const std::string& key, std::string* out) {
  if (out == nullptr) return Status(RET_INVALID_ARGUMENT, "Get: out null");
  out->clear();
  std::lock_guard<std::mutex> lk(mu_);
  STATUS_CHECK_RETURN(EnsureConnectedLocked());
  redisReply* r = static_cast<redisReply*>(redisCommand(ctx_, "GET %b", key.data(), key.size()));
  ReplyGuard g{r};
  if (r == nullptr) {
    return Status(RET_RUNTIME_FAILED, fmt::format("RedisAliveStore: GET reply null err={}", ctx_->errstr));
  }
  if (r->type == REDIS_REPLY_NIL) return Status();
  if (r->type != REDIS_REPLY_STRING) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("RedisAliveStore: GET unexpected reply type {}", static_cast<int>(r->type)));
  }
  out->assign(r->str, r->len);
  return Status();
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
