/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/redis_alive_store.h"

#include <gtest/gtest.h>
#include <hiredis/hiredis.h>

#include <chrono>
#include <cstdlib>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

// The test environment exposes a Redis the smoke runbook also uses:
//   redis://29.226.50.29:6379  password Redis@123456
// If the Redis is unreachable (e.g. CI without network), each test
// gracefully exits via GTEST_SKIP() — RedisAliveStore unit tests must
// not block deterministic CI on the presence of an external service.
constexpr const char* kTestRedisUrl = "redis://29.226.50.29:6379";
constexpr const char* kTestRedisPassword = "Redis@123456";
constexpr const char* kTestKeyspace = "pd_v2_test/redis_alive_store/";

// Returns true iff a fresh Connect() succeeds against the smoke Redis.
// Used by tests as a precondition; SKIP if false.
bool TryConnectOnce() {
  RedisAliveStore probe;
  return probe.Connect(kTestRedisUrl, kTestRedisPassword).OK();
}

// Wipe every key under kTestKeyspace before/after each test so reruns
// don't see stale data. Keys() pulls them all, then DEL.
void WipeTestKeyspace(RedisAliveStore& store) {
  std::vector<std::string> keys;
  ASSERT_TRUE(store.Keys(std::string(kTestKeyspace) + "*", &keys).OK());
  for (const auto& k : keys) {
    ASSERT_TRUE(store.Del(k).OK());
  }
}

// Opens a raw hiredis connection used by the reconnect test as an out-of-band
// admin to forcibly drop the store's server-side connection (CLIENT KILL).
// Returns nullptr on any failure so the caller can SKIP. Host/port are parsed
// from kTestRedisUrl (redis://host:port).
redisContext* AdminConnect() {
  std::string rest = std::string(kTestRedisUrl).substr(std::strlen("redis://"));
  size_t colon = rest.find(':');
  if (colon == std::string::npos) return nullptr;
  std::string host = rest.substr(0, colon);
  int port = std::atoi(rest.substr(colon + 1).c_str());
  redisContext* c = redisConnect(host.c_str(), port);
  if (c == nullptr || c->err != 0) {
    if (c != nullptr) redisFree(c);
    return nullptr;
  }
  redisReply* r = static_cast<redisReply*>(redisCommand(c, "AUTH %s", kTestRedisPassword));
  bool ok = (r != nullptr && r->type != REDIS_REPLY_ERROR);
  if (r != nullptr) freeReplyObject(r);
  if (!ok) {
    redisFree(c);
    return nullptr;
  }
  return c;
}

// Snapshot the source addresses (addr=ip:port) of all clients currently
// connected to the server, parsed from `CLIENT LIST`.
std::set<std::string> ClientAddrs(redisContext* admin) {
  std::set<std::string> addrs;
  redisReply* r = static_cast<redisReply*>(redisCommand(admin, "CLIENT LIST"));
  if (r != nullptr && r->type == REDIS_REPLY_STRING && r->str != nullptr) {
    std::string list(r->str, r->len);
    size_t pos = 0;
    while ((pos = list.find("addr=", pos)) != std::string::npos) {
      pos += std::strlen("addr=");
      size_t end = list.find_first_of(" \t\r\n", pos);
      addrs.insert(list.substr(pos, end == std::string::npos ? end : end - pos));
    }
  }
  if (r != nullptr) freeReplyObject(r);
  return addrs;
}

}  // namespace

TEST(RedisAliveStoreTest, ConnectAndDisconnect) {
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  RedisAliveStore store;
  ASSERT_TRUE(store.Connect(kTestRedisUrl, kTestRedisPassword).OK());
  EXPECT_TRUE(store.IsConnected());
  store.Disconnect();
  EXPECT_FALSE(store.IsConnected());
  // Disconnect is idempotent.
  store.Disconnect();
}

TEST(RedisAliveStoreTest, RejectsNonRedisUrl) {
  RedisAliveStore store;
  Status s = store.Connect("http://localhost:8080", "");
  EXPECT_FALSE(s.OK());
  EXPECT_FALSE(store.IsConnected());
}

TEST(RedisAliveStoreTest, RejectsBadAuth) {
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  RedisAliveStore store;
  Status s = store.Connect(kTestRedisUrl, "definitely-wrong-password");
  EXPECT_FALSE(s.OK());
  EXPECT_FALSE(store.IsConnected());
}

TEST(RedisAliveStoreTest, SetGetDelRoundTrip) {
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  RedisAliveStore store;
  ASSERT_TRUE(store.Connect(kTestRedisUrl, kTestRedisPassword).OK());
  WipeTestKeyspace(store);

  const std::string key = std::string(kTestKeyspace) + "round_trip";
  const std::string value = R"({"epoch":1234567890})";
  ASSERT_TRUE(store.Set(key, value, /*ttl_s=*/30).OK());

  std::string got;
  ASSERT_TRUE(store.Get(key, &got).OK());
  EXPECT_EQ(got, value);

  ASSERT_TRUE(store.Del(key).OK());

  // After DEL, GET returns OK with empty string (the missing-key contract).
  std::string after;
  ASSERT_TRUE(store.Get(key, &after).OK());
  EXPECT_TRUE(after.empty());

  WipeTestKeyspace(store);
}

TEST(RedisAliveStoreTest, KeysFiltersByPattern) {
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  RedisAliveStore store;
  ASSERT_TRUE(store.Connect(kTestRedisUrl, kTestRedisPassword).OK());
  WipeTestKeyspace(store);

  const std::string a = std::string(kTestKeyspace) + "alpha";
  const std::string b = std::string(kTestKeyspace) + "beta";
  const std::string c = std::string(kTestKeyspace) + "gamma";
  ASSERT_TRUE(store.Set(a, "1", 30).OK());
  ASSERT_TRUE(store.Set(b, "2", 30).OK());
  ASSERT_TRUE(store.Set(c, "3", 30).OK());

  std::vector<std::string> keys;
  ASSERT_TRUE(store.Keys(std::string(kTestKeyspace) + "*", &keys).OK());
  std::set<std::string> got(keys.begin(), keys.end());
  EXPECT_EQ(got.count(a), 1u);
  EXPECT_EQ(got.count(b), 1u);
  EXPECT_EQ(got.count(c), 1u);
  EXPECT_EQ(got.size(), 3u);

  WipeTestKeyspace(store);
}

TEST(RedisAliveStoreTest, SetTtlExpires) {
  // §3.3 keepalive contract: after TTL elapses without renew, the key
  // disappears. Production cycles every 2s with TTL=5s; this test uses
  // ttl=1s with a 1500ms wait to keep wall-clock time bounded.
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  RedisAliveStore store;
  ASSERT_TRUE(store.Connect(kTestRedisUrl, kTestRedisPassword).OK());
  WipeTestKeyspace(store);

  const std::string key = std::string(kTestKeyspace) + "ttl_expiry";
  ASSERT_TRUE(store.Set(key, R"({"epoch":42})", /*ttl_s=*/1).OK());

  std::string got;
  ASSERT_TRUE(store.Get(key, &got).OK());
  EXPECT_EQ(got, R"({"epoch":42})");

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));

  std::string after;
  ASSERT_TRUE(store.Get(key, &after).OK());
  EXPECT_TRUE(after.empty()) << "key should have expired";
}

TEST(RedisAliveStoreTest, RenewExtendsTtl) {
  // SETting again with a fresh TTL extends the lifetime — this is the
  // continuous-renew model for the Decode keepalive thread (§3.3 SET +
  // 2s renewal). After 700ms a re-SET pushes TTL out, so the key must
  // still be present at 1100ms (which would be past the original 1s).
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  RedisAliveStore store;
  ASSERT_TRUE(store.Connect(kTestRedisUrl, kTestRedisPassword).OK());
  WipeTestKeyspace(store);

  const std::string key = std::string(kTestKeyspace) + "renew";
  const std::string value = R"({"epoch":7})";
  ASSERT_TRUE(store.Set(key, value, /*ttl_s=*/1).OK());

  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  ASSERT_TRUE(store.Set(key, value, /*ttl_s=*/1).OK());  // renew

  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  std::string got;
  ASSERT_TRUE(store.Get(key, &got).OK());
  EXPECT_EQ(got, value) << "renewal should have extended TTL past 1s";

  WipeTestKeyspace(store);
}

TEST(RedisAliveStoreTest, ReconnectsAfterConnectionDropped) {
  // Regression for the lost-keepalive bug: a transient Redis blip leaves the
  // hiredis sync context permanently broken (every later redisCommand returns
  // NULL), so without auto-reconnect the Decode keepalive / discovery loops
  // would stay wedged until process restart. Here we forcibly kill the store's
  // server-side connection out-of-band and assert the store transparently
  // re-establishes it on a subsequent command.
  if (!TryConnectOnce()) GTEST_SKIP() << "Redis unavailable for tests";
  redisContext* admin = AdminConnect();
  if (admin == nullptr) GTEST_SKIP() << "admin Redis connection unavailable";

  // Baseline the set of connected clients BEFORE the store connects so the
  // diff afterwards isolates the store's own connection — avoids killing
  // unrelated tenants on the shared smoke Redis.
  std::set<std::string> before = ClientAddrs(admin);

  RedisAliveStore store;
  ASSERT_TRUE(store.Connect(kTestRedisUrl, kTestRedisPassword).OK());
  WipeTestKeyspace(store);

  const std::string key = std::string(kTestKeyspace) + "reconnect";
  ASSERT_TRUE(store.Set(key, "before", /*ttl_s=*/30).OK());

  std::set<std::string> after = ClientAddrs(admin);
  int killed = 0;
  for (const auto& addr : after) {
    if (before.count(addr) != 0) continue;  // pre-existing client, not ours
    redisReply* r = static_cast<redisReply*>(redisCommand(admin, "CLIENT KILL ADDR %s", addr.c_str()));
    if (r != nullptr) {
      ++killed;
      freeReplyObject(r);
    }
  }
  ASSERT_GT(killed, 0) << "failed to identify/kill the store's connection";

  // The first command after the kill may still fail: hiredis only flips
  // ctx_->err when it next touches the dead socket. EnsureConnectedLocked()
  // then reconnects on the following call, so recovery must happen within a
  // couple of attempts. Pre-fix code would loop here forever (NULL replies).
  bool recovered = false;
  for (int attempt = 0; attempt < 5; ++attempt) {
    if (store.Set(key, "after", /*ttl_s=*/30).OK()) {
      recovered = true;
      break;
    }
  }
  EXPECT_TRUE(recovered) << "store did not auto-reconnect after connection drop";
  EXPECT_TRUE(store.IsConnected());

  std::string got;
  ASSERT_TRUE(store.Get(key, &got).OK());
  EXPECT_EQ(got, "after");

  WipeTestKeyspace(store);
  redisFree(admin);
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
