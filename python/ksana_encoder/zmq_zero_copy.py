# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""ZMQ零拷贝传输工具模块

提供tensor的零拷贝发送和接收功能，用于encoder_client和encoder_server之间的高效通信。
"""

import pickle
from typing import Any, Dict, Tuple

import numpy as np
import torch
import zmq


def send_zero_copy(socket: zmq.Socket, data: dict) -> Tuple[int, int]:
    """零拷贝发送数据，分离多个Tensor和元数据

    Args:
        socket: ZMQ socket
        data: 要发送的数据字典，可包含torch.Tensor

    Returns:
        Tuple[int, int]: (元数据大小, tensor总大小)
    """
    # 查找所有torch.Tensor加入零拷贝
    tensor_keys = []
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            tensor_keys.append(key)

    # 提取所有tensor并记录元数据
    tensors = []
    tensor_metas = {}
    for key in tensor_keys:
        tensor = data.pop(key, None)
        if tensor is not None:
            tensor_metas[key] = {
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "index": len(tensors),  # 记录在frames中的索引
            }
            tensors.append(tensor)

    # 元数据中记录所有tensor信息
    data["__tensor_metas__"] = tensor_metas

    # 构建multipart消息: [metadata, tensor1_bytes, tensor2_bytes, ...]
    frames = [pickle.dumps(data)]
    total_tensor_size = 0

    for tensor in tensors:
        tensor_bytes = tensor.view(torch.uint8).numpy()
        frames.append(zmq.Frame(memoryview(tensor_bytes), copy=False))
        total_tensor_size += tensor_bytes.nbytes

    # 发送multipart消息
    socket.send_multipart(frames, copy=False)

    return len(frames[0]), total_tensor_size


def recv_zero_copy(socket: zmq.Socket) -> Dict[str, Any]:
    """零拷贝接收数据，支持多个tensor

    Args:
        socket: ZMQ socket

    Returns:
        Dict[str, Any]: 接收的数据字典，包含重建的tensor数据
    """
    # 接收multipart消息
    frames = socket.recv_multipart(copy=False)

    # 解析元数据
    data = pickle.loads(frames[0].buffer)

    # 如果有tensor数据，重建它们
    tensor_metas = data.pop("__tensor_metas__", None)
    if tensor_metas and len(frames) > 1:
        for key, meta in tensor_metas.items():
            # 获取对应的frame（索引+1，因为第一个frame是元数据）
            frame_index = meta["index"] + 1
            if frame_index < len(frames):
                # 从原始字节重建tensor
                uint8_array = np.frombuffer(frames[frame_index].buffer, dtype=np.uint8)
                tensor = (
                    torch.from_numpy(uint8_array.copy())
                    .view(meta["dtype"])
                    .reshape(meta["shape"])
                )
                data[key] = tensor

    return data
