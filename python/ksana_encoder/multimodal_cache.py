# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""多模态缓存模块

提供基于LRU策略的Tensor字典缓存实现，用于缓存多模态编码结果。
支持多线程环境下的安全访问。
"""

import threading
from collections import OrderedDict
from typing import Dict, Optional

import torch


class MultimodalCache:
    """多模态Tensor字典缓存（线程安全）

    基于LRU（最近最少使用）策略的缓存实现，用于存储Dict[str, torch.Tensor]数据。
    使用OrderedDict实现基于字节大小限制的LRU淘汰策略。
    使用threading.Lock确保多线程环境下的安全访问。

    特性:
        - 支持配置总容量（GB单位）
        - 自动统计存储的所有Tensor占用空间
        - 空间不足时按LRU策略自动淘汰旧数据
        - 碎片化存储，每个Tensor字典独立维护
        - 存取时使用clone深拷贝，确保数据独立性
        - 线程安全：所有公共方法都使用锁保护

    Attributes:
        _max_size_bytes: 缓存最大容量（字节）
        _current_size_bytes: 当前已使用容量（字节）
        _cache: 有序字典，存储key到(tensor_dict, size)的映射
        _lock: 线程锁，保护缓存的并发访问
    """

    def __init__(self, max_size_gb: float):
        """初始化多模态缓存

        Args:
            max_size_gb: 缓存最大容量，单位为GB（float类型）
        """
        self._max_size_bytes: int = int(max_size_gb * (1 << 30))  # 1 << 30 = 1GB
        self._current_size_bytes: int = 0
        self._cache: OrderedDict[int, tuple[Dict[str, torch.Tensor], int]] = (
            OrderedDict()
        )
        self._lock: threading.Lock = threading.Lock()

    @staticmethod
    def _calculate_dict_size(tensor_dict: Dict[str, torch.Tensor]) -> int:
        """计算Tensor字典的总字节大小

        Args:
            tensor_dict: Tensor字典

        Returns:
            int: 所有Tensor的总字节大小
        """
        return sum(tensor.nbytes for tensor in tensor_dict.values())

    @staticmethod
    def _clone_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """深拷贝Tensor字典

        Args:
            tensor_dict: 要拷贝的Tensor字典

        Returns:
            Dict[str, torch.Tensor]: 深拷贝后的Tensor字典
        """
        return {key: tensor.clone() for key, tensor in tensor_dict.items()}

    def get(self, key: int) -> Optional[Dict[str, torch.Tensor]]:
        """获取缓存中的Tensor字典（线程安全）

        如果key存在，返回Tensor字典的深拷贝并更新LRU顺序。
        如果key不存在，返回None。

        Args:
            key: 缓存键（uint64哈希值，使用int存储）

        Returns:
            Optional[Dict[str, torch.Tensor]]: 存在时返回Tensor字典的深拷贝，不存在时返回None
        """
        with self._lock:
            if key not in self._cache:
                return None

            self._cache.move_to_end(key)
            tensor_dict, _ = self._cache[key]
            return self._clone_dict(tensor_dict)

    def put(self, key: int, tensor_dict: Dict[str, torch.Tensor]) -> bool:
        """存入Tensor字典到缓存（线程安全）

        如果key不存在则插入新数据，如果存在则更新数据。
        空间不足时按LRU策略从头部淘汰旧数据。
        单个Tensor字典超出缓存总大小则不插入。

        Args:
            key: 缓存键（uint64哈希值，使用int存储）
            tensor_dict: 要存储的Tensor字典（CPU Tensor）

        Returns:
            bool: 插入/更新成功返回True，失败返回False
        """
        new_size = self._calculate_dict_size(tensor_dict)

        if new_size > self._max_size_bytes:
            return False

        with self._lock:
            # 如果key已存在，先删除旧数据
            if key in self._cache:
                _, old_size = self._cache.pop(key)
                self._current_size_bytes -= old_size

            # 淘汰旧数据直到有足够空间
            while (
                self._current_size_bytes + new_size > self._max_size_bytes
                and self._cache
            ):
                _, (_, size) = self._cache.popitem(last=False)
                self._current_size_bytes -= size

            self._cache[key] = (self._clone_dict(tensor_dict), new_size)
            self._current_size_bytes += new_size
            return True

    def size(self) -> int:
        """获取已存储的Tensor总字节大小（线程安全）"""
        with self._lock:
            return self._current_size_bytes

    def clear(self) -> None:
        """清空缓存（线程安全）"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0

    def contains(self, key: int) -> bool:
        """检查缓存中是否存在指定key（线程安全，不影响LRU顺序）"""
        with self._lock:
            return key in self._cache
