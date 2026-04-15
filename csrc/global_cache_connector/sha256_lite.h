/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

// Minimal header-only SHA256 implementation.
// Removes the external OpenSSL dependency for the default (mock-only) build.
// Only used by global_cache_key_utils.h for token prefix hashing.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ksana_llm {
namespace sha256_detail {

static constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t Rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t Sigma0(uint32_t x) { return Rotr(x, 2) ^ Rotr(x, 13) ^ Rotr(x, 22); }
inline uint32_t Sigma1(uint32_t x) { return Rotr(x, 6) ^ Rotr(x, 11) ^ Rotr(x, 25); }
inline uint32_t Gamma0(uint32_t x) { return Rotr(x, 7) ^ Rotr(x, 18) ^ (x >> 3); }
inline uint32_t Gamma1(uint32_t x) { return Rotr(x, 17) ^ Rotr(x, 19) ^ (x >> 10); }

inline void TransformBlock(uint32_t state[8], const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) | (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) | static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    w[i] = Gamma1(w[i - 2]) + w[i - 7] + Gamma0(w[i - 15]) + w[i - 16];
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + kK[i] + w[i];
    uint32_t t2 = Sigma0(a) + Maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

// Compute SHA256 digest of `data` (len bytes) into `digest` (32 bytes).
inline void Sha256(const uint8_t* data, size_t len, uint8_t digest[32]) {
  uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

  // Process full 64-byte blocks.
  size_t i = 0;
  for (; i + 64 <= len; i += 64) {
    TransformBlock(state, data + i);
  }

  // Pad the remaining bytes.
  uint8_t buf[128];
  size_t remaining = len - i;
  std::memcpy(buf, data + i, remaining);
  buf[remaining++] = 0x80;

  // Need room for 8-byte length at the end of a 64-byte block.
  size_t pad_len = (remaining <= 56) ? 64 : 128;
  std::memset(buf + remaining, 0, pad_len - remaining);

  // Append bit length (big-endian, 64-bit).
  uint64_t bit_len = static_cast<uint64_t>(len) * 8;
  for (int j = 0; j < 8; ++j) {
    buf[pad_len - 1 - j] = static_cast<uint8_t>(bit_len >> (j * 8));
  }

  for (size_t off = 0; off < pad_len; off += 64) {
    TransformBlock(state, buf + off);
  }

  // Write digest (big-endian).
  for (int j = 0; j < 8; ++j) {
    digest[j * 4] = static_cast<uint8_t>(state[j] >> 24);
    digest[j * 4 + 1] = static_cast<uint8_t>(state[j] >> 16);
    digest[j * 4 + 2] = static_cast<uint8_t>(state[j] >> 8);
    digest[j * 4 + 3] = static_cast<uint8_t>(state[j]);
  }
}

}  // namespace sha256_detail
}  // namespace ksana_llm
