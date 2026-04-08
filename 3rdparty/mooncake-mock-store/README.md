# Mooncake Mock Store

[Mooncake Store](https://github.com/kvcache-ai/Mooncake) 的极简 mock 实现，用于上游项目功能验证。

## 原理

每个 key 对应共享目录中的一个文件，多进程/多线程通过 `flock()` 保证读写一致性（写排他锁 / 读共享锁）。不同进程创建各自的 `MockStore` 实例并指向同一目录即可模拟多节点的 put/get。

## 项目结构

```
mooncake-mock-store/
├── CMakeLists.txt
├── README.md
├── include/mooncake/store.h      # 头文件（C++ 接口）
├── src/store.cpp                  # 实现（~160 行）
└── tests/
    ├── test_store.cpp             # 单元测试
    └── cross_process/
        ├── writer.cpp             # 跨进程写端
        ├── reader.cpp             # 跨进程读端
        └── run.sh                 # 跨进程测试脚本
```

## 编译

```bash
mkdir build && cd build
cmake ..
make -j
```

产物：`libmooncake_store.dylib`（Linux 为 `.so`），以及测试程序 `test_store`、`writer`、`reader`。

所有可执行文件通过 RPATH 自动找到动态库，无需手动设置 `LD_LIBRARY_PATH`。

## 测试

```bash
cd build && ctest --output-on-failure
```

包含两组测试：
- **test_store** — 基本 KV、零拷贝、批量、多线程、大 tensor
- **test_cross_process** — writer 进程写入，reader 进程读取并校验

## 使用

```cpp
#include "mooncake/store.h"

mooncake::MockStore store;
// 第二个参数为共享目录路径，多个进程/线程指向同一目录即可互通
store.setup("node0", "/tmp/mooncake_shared");

// 写入
store.put("key", data, size);

// 读取（返回字节数，out 需 free）
void* out = nullptr;
int64_t n = store.get("key", &out);
// use out ...
free(out);

// 零拷贝风格
store.put_from("key2", (uintptr_t)buf, len);
store.get_into("key2", (uintptr_t)recv_buf, len);

store.close();
```

### 多进程（模拟多节点）

```cpp
// 进程 A
mooncake::MockStore nodeA;
nodeA.setup("nodeA", "/tmp/mooncake_shared");
nodeA.put("kv_0", data, size);

// 进程 B（独立进程，同一目录）
mooncake::MockStore nodeB;
nodeB.setup("nodeB", "/tmp/mooncake_shared");
void* out = nullptr;
nodeB.get("kv_0", &out);  // 可以读到进程 A 写入的数据
free(out);
```

## API

| 方法 | 说明 |
|---|---|
| `setup(hostname, dir, ...)` | 初始化，`dir` 为共享目录路径，其余参数保持 API 兼容但不使用 |
| `close()` | 关闭 |
| `put(key, data, size)` | 存储 |
| `get(key, &out)` → `int64_t` | 取回（返回字节数，out 需 `free`） |
| `remove(key)` | 删除 |
| `remove_all()` | 清空目录内所有 key |
| `is_exist(key)` | 1=存在, 0=不存在 |
| `get_size(key)` | 返回字节数 |
| `put_from(key, ptr, size)` | 零拷贝风格写入 |
| `get_into(key, ptr, size)` | 零拷贝风格读取 |
| `put_batch` / `get_batch` | 批量操作 |
| `register_buffer` / `unregister_buffer` | 保持 API 兼容，实际为空操作 |
