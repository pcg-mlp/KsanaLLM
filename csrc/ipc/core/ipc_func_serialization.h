/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

inline void SerializeInit(size_t& offset) { offset = 0; }

template <typename T>
void SerializePod(const T& value, void* ptr, size_t& offset) {
  uint8_t* dst = reinterpret_cast<uint8_t*>(ptr);
  memcpy(dst + offset, &value, sizeof(T));
  offset += sizeof(T);
}

template <typename T>
T DeserializePod(const void* ptr, size_t& offset) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
  T value = *reinterpret_cast<const T*>(src + offset);
  offset += sizeof(T);
  return value;
}

template <typename T>
void SerializeEnum(const T& value, void* ptr, size_t& offset) {
  using EnumType = std::underlying_type_t<T>;
  EnumType underlying_value = static_cast<EnumType>(value);
  SerializePod<EnumType>(underlying_value, ptr, offset);
}

template <typename T>
T DeserializeEnum(const void* ptr, size_t& offset) {
  using EnumType = std::underlying_type_t<T>;
  EnumType underlying_value = DeserializePod<EnumType>(ptr, offset);
  T value = static_cast<T>(underlying_value);
  return value;
}

inline void SerializeString(const std::string& value, void* ptr, size_t& offset) {
  size_t str_len = value.length();
  SerializePod<size_t>(str_len, ptr, offset);

  uint8_t* dst = reinterpret_cast<uint8_t*>(ptr);
  memcpy(dst + offset, value.c_str(), str_len);
  offset += str_len;
}

inline std::string DeserializeString(const void* ptr, size_t& offset) {
  size_t str_len = DeserializePod<size_t>(ptr, offset);

  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
  std::string value = std::string(reinterpret_cast<const char*>(src + offset), str_len);
  offset += str_len;
  return value;
}

inline void SerializeMemory(const void* value, size_t value_bytes, void* ptr, size_t& offset) {
  SerializePod<size_t>(value_bytes, ptr, offset);

  uint8_t* dst = reinterpret_cast<uint8_t*>(ptr);
  memcpy(dst + offset, value, value_bytes);
  offset += value_bytes;
}

inline void DeserializeMemory(void* value, const void* ptr, size_t& offset) {
  size_t value_bytes = DeserializePod<size_t>(ptr, offset);

  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
  memcpy(value, src + offset, value_bytes);
  offset += value_bytes;
}

template <typename T>
inline void SerializeArray(const T* value, size_t value_len, void* ptr, size_t& offset) {
  SerializePod<size_t>(value_len, ptr, offset);

  uint8_t* dst = reinterpret_cast<uint8_t*>(ptr);
  size_t value_bytes = value_len * sizeof(T);
  memcpy(dst + offset, value, value_bytes);
  offset += value_bytes;
}

template <typename T>
inline void DeserializeArray(T* value, const void* ptr, size_t& offset) {
  size_t value_len = DeserializePod<size_t>(ptr, offset);

  const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
  size_t value_bytes = value_len * sizeof(T);
  memcpy(value, src + offset, value_bytes);
  offset += value_bytes;
}

template <typename T>
void SerializeVector(const std::vector<T>& vec, void* ptr, size_t& offset) {
  size_t vec_size = vec.size();
  SerializePod<size_t>(vec_size, ptr, offset);

  for (const T& v : vec) {
    SerializePod<T>(v, ptr, offset);
  }
}

template <>
inline void SerializeVector<std::string>(const std::vector<std::string>& vec, void* ptr, size_t& offset) {
  size_t vec_size = vec.size();
  SerializePod<size_t>(vec_size, ptr, offset);

  for (const std::string& v : vec) {
    SerializeString(v, ptr, offset);
  }
}

template <typename T>
void DeserializeVector(std::vector<T>& vec, const void* ptr, size_t& offset) {
  size_t vec_size = DeserializePod<size_t>(ptr, offset);

  vec.clear();
  for (size_t i = 0; i < vec_size; ++i) {
    T v = DeserializePod<T>(ptr, offset);
    vec.push_back(v);
  }
}

template <>
inline void DeserializeVector<std::string>(std::vector<std::string>& vec, const void* ptr, size_t& offset) {
  size_t vec_size = DeserializePod<size_t>(ptr, offset);

  vec.clear();
  for (size_t i = 0; i < vec_size; ++i) {
    std::string v = DeserializeString(ptr, offset);
    vec.push_back(v);
  }
}

template <typename K, typename V>
void SerializeKeyToVector(const std::unordered_map<K, std::vector<V>>& dict, void* ptr, size_t& offset) {
  size_t dict_size = dict.size();
  SerializePod<size_t>(dict_size, ptr, offset);

  for (auto it = dict.begin(); it != dict.end(); ++it) {
    K key = it->first;
    std::vector<V> vec = it->second;
    SerializePod<K>(key, ptr, offset);
    SerializeVector<V>(vec, ptr, offset);
  }
}

template <typename K, typename V>
void DeserializeKeyToVector(std::unordered_map<K, std::vector<V>>& dict, const void* ptr, size_t& offset) {
  size_t dict_size = DeserializePod<size_t>(ptr, offset);

  for (size_t i = 0; i < dict_size; ++i) {
    std::vector<V> vec;
    K key = DeserializePod<K>(ptr, offset);
    DeserializeVector<V>(vec, ptr, offset);
    dict[key] = vec;
  }
}

template <typename T>
void SerializeVectorOfVector(const std::vector<std::vector<T>>& vecs, void* ptr, size_t& offset) {
  size_t vec_size = vecs.size();
  SerializePod<size_t>(vec_size, ptr, offset);

  for (const std::vector<T>& vec : vecs) {
    SerializeVector<T>(vec, ptr, offset);
  }
}

template <typename T>
void DeserializeVectorOfVector(std::vector<std::vector<T>>& vecs, const void* ptr, size_t& offset) {
  size_t vec_size = DeserializePod<size_t>(ptr, offset);

  for (size_t i = 0; i < vec_size; ++i) {
    std::vector<T> vec;
    DeserializeVector(vec, ptr, offset);
    vecs.push_back(vec);
  }
}

inline void SerializeStatus(const Status& value, void* ptr, size_t& offset) {
  SerializeEnum<RetCode>(value.GetCode(), ptr, offset);
  SerializeString(value.GetMessage(), ptr, offset);
}

inline Status DeserializeStatus(const void* ptr, size_t& offset) {
  RetCode ret_code = DeserializeEnum<RetCode>(ptr, offset);
  std::string message = DeserializeString(ptr, offset);
  return Status(ret_code, message);
}

}  // namespace ksana_llm
