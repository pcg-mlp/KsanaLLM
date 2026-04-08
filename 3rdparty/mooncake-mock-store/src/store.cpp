/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "mooncake/store.h"

#include <cstdlib>
#include <cstring>

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mooncake {

// ---- helpers ----

std::string MockStore::path(const char* key) const { return dir_ + "/" + key; }

static bool write_all(int fd, const void* data, size_t n) {
  const char* p = (const char*)data;
  while (n > 0) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= w;
  }
  return true;
}

static bool read_all(int fd, void* buf, size_t n) {
  char* p = (char*)buf;
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r <= 0) return false;
    p += r;
    n -= r;
  }
  return true;
}

// ---- MockStore ----

MockStore::MockStore() = default;
MockStore::~MockStore() { close(); }

int MockStore::setup(const char* local_hostname, const char* metadata_server, uint64_t, uint64_t, const char*,
                     const char*, const char*) {
  if (initialized_) return -1;
  hostname_ = local_hostname ? local_hostname : "localhost";
  dir_ = (metadata_server && metadata_server[0]) ? metadata_server : "/tmp/mooncake_store";
  ::mkdir(dir_.c_str(), 0755);
  initialized_ = true;
  return 0;
}

int MockStore::close() {
  initialized_ = false;
  return 0;
}
const char* MockStore::get_hostname() const { return hostname_.c_str(); }

int MockStore::put(const char* key, const void* data, size_t size, const ReplicateConfig*) {
  if (!initialized_ || !key) return -1;
  int fd = ::open(path(key).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return -1;
  ::flock(fd, LOCK_EX);
  bool ok = write_all(fd, data, size);
  ::flock(fd, LOCK_UN);
  ::close(fd);
  return ok ? 0 : -1;
}

int64_t MockStore::get(const char* key, void** out_data) {
  if (!initialized_ || !key || !out_data) return -1;
  int fd = ::open(path(key).c_str(), O_RDONLY);
  if (fd < 0) return -1;
  ::flock(fd, LOCK_SH);
  struct stat st;
  if (::fstat(fd, &st) < 0) {
    ::flock(fd, LOCK_UN);
    ::close(fd);
    return -1;
  }
  size_t sz = st.st_size;
  void* buf = std::malloc(sz);
  if (!buf) {
    ::flock(fd, LOCK_UN);
    ::close(fd);
    return -1;
  }
  bool ok = read_all(fd, buf, sz);
  ::flock(fd, LOCK_UN);
  ::close(fd);
  if (!ok) {
    std::free(buf);
    return -1;
  }
  *out_data = buf;
  return (int64_t)sz;
}

int MockStore::remove(const char* key) {
  if (!initialized_ || !key) return -1;
  return ::unlink(path(key).c_str()) == 0 ? 0 : -1;
}

long MockStore::remove_all() {
  long n = 0;
  DIR* d = ::opendir(dir_.c_str());
  if (!d) return 0;
  struct dirent* e;
  while ((e = ::readdir(d))) {
    if (e->d_name[0] == '.') continue;
    ::unlink((dir_ + "/" + e->d_name).c_str());
    n++;
  }
  ::closedir(d);
  return n;
}

int MockStore::is_exist(const char* key) {
  if (!initialized_ || !key) return -1;
  return ::access(path(key).c_str(), F_OK) == 0 ? 1 : 0;
}

int64_t MockStore::get_size(const char* key) {
  if (!initialized_ || !key) return -1;
  struct stat st;
  return ::stat(path(key).c_str(), &st) == 0 ? (int64_t)st.st_size : -1;
}

int MockStore::register_buffer(uintptr_t, size_t) { return 0; }
int MockStore::unregister_buffer(uintptr_t) { return 0; }

int MockStore::put_from(const char* key, uintptr_t ptr, size_t size, const ReplicateConfig* c) {
  return put(key, (const void*)ptr, size, c);
}

int64_t MockStore::get_into(const char* key, uintptr_t ptr, size_t size) {
  if (!initialized_ || !key) return -1;
  int fd = ::open(path(key).c_str(), O_RDONLY);
  if (fd < 0) return -1;
  ::flock(fd, LOCK_SH);
  struct stat st;
  if (::fstat(fd, &st) < 0 || (size_t)st.st_size > size) {
    ::flock(fd, LOCK_UN);
    ::close(fd);
    return -1;
  }
  size_t sz = st.st_size;
  bool ok = read_all(fd, (void*)ptr, sz);
  ::flock(fd, LOCK_UN);
  ::close(fd);
  return ok ? (int64_t)sz : -1;
}

int MockStore::put_batch(const char** keys, const void** values, const size_t* sizes, size_t count,
                         const ReplicateConfig* c) {
  for (size_t i = 0; i < count; ++i)
    if (put(keys[i], values[i], sizes[i], c) != 0) return -1;
  return 0;
}

int MockStore::get_batch(const char** keys, size_t count, void** out_datas, int64_t* out_sizes) {
  for (size_t i = 0; i < count; ++i) out_sizes[i] = get(keys[i], &out_datas[i]);
  return 0;
}

}  // namespace mooncake
