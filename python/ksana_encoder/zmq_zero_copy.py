# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""ZMQ零拷贝传输工具模块

提供tensor的零拷贝发送和接收功能，用于encoder_client和encoder_server之间的高效通信。
使用msgpack替代pickle，规避反序列化安全风险。

支持两种传输模式：
  - 标准模式 (REQ/ROUTER): 一次性发送所有数据
  - 流式模式 (DEALER/ROUTER): 逐步发送中间结果（如deepstack features）
"""

from typing import Any, Dict, List, Tuple

import msgpack
import numpy as np
import torch
import zmq

_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_to_str(dtype: torch.dtype) -> str:
    """将 torch.dtype 转换为字符串用于序列化。"""
    s = _DTYPE_TO_STR.get(dtype)
    if s is None:
        raise ValueError(f"Unsupported torch.dtype for serialization: {dtype}")
    return s


def _str_to_dtype(s: str) -> torch.dtype:
    """将字符串恢复为 torch.dtype。"""
    dtype = _STR_TO_DTYPE.get(s)
    if dtype is None:
        raise ValueError(f"Unknown dtype string: {s}")
    return dtype


def _frame_to_tensor(frame, dtype_str: str, shape: list) -> torch.Tensor:
    """从 ZMQ frame 零拷贝重建单个 tensor。"""
    uint8_array = np.frombuffer(frame.buffer, dtype=np.uint8)
    return torch.from_numpy(uint8_array).view(_str_to_dtype(dtype_str)).reshape(shape)


def _extract_tensors(data: dict) -> List[torch.Tensor]:
    """in-place 提取 data 中的 tensor 和 tensor list，替换为元数据描述。

    Returns:
        按索引顺序排列的 tensor 列表。
    """
    tensor_keys = []
    tensor_list_keys = []
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            tensor_keys.append(key)
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            tensor_list_keys.append(key)

    tensors: List[torch.Tensor] = []
    tensor_metas: dict = {}

    for key in tensor_keys:
        tensor = data.pop(key)
        tensor_metas[key] = {
            "shape": list(tensor.shape),
            "dtype": _dtype_to_str(tensor.dtype),
            "index": len(tensors),
        }
        tensors.append(tensor)

    for key in tensor_list_keys:
        tensor_list = data.pop(key)
        start_idx = len(tensors)
        tensors.extend(tensor_list)
        tensor_metas[key] = {
            "type": "tensor_list",
            "indices": list(range(start_idx, start_idx + len(tensor_list))),
            "shapes": [list(t.shape) for t in tensor_list],
            "dtypes": [_dtype_to_str(t.dtype) for t in tensor_list],
        }

    data["__tensor_metas__"] = tensor_metas
    return tensors


def _pack_frames(data: dict, tensors: List[torch.Tensor]) -> Tuple[List, int, int]:
    """将 data（已含 __tensor_metas__）和 tensors 打包为 ZMQ frames。

    Returns:
        (frames, meta_size, tensor_total_size)
    """
    meta_bytes = msgpack.packb(data, use_bin_type=True)
    frames = [meta_bytes]
    total_tensor_size = 0

    for tensor in tensors:
        tensor_bytes = tensor.contiguous().view(torch.uint8).numpy()
        frames.append(zmq.Frame(memoryview(tensor_bytes), copy=False))
        total_tensor_size += tensor_bytes.nbytes

    return frames, len(meta_bytes), total_tensor_size


def _unpack_data(meta_frame, data_frames) -> Dict[str, Any]:
    """从 msgpack 元数据帧 + tensor 数据帧重建完整字典（零拷贝）。"""
    buf = meta_frame.buffer if hasattr(meta_frame, "buffer") else meta_frame
    data = msgpack.unpackb(buf, raw=False)

    tensor_metas = data.pop("__tensor_metas__", None)
    if tensor_metas and len(data_frames) > 1:
        for key, meta in tensor_metas.items():
            if meta.get("type") == "tensor_list":
                data[key] = [
                    _frame_to_tensor(
                        data_frames[idx + 1], meta["dtypes"][i], meta["shapes"][i]
                    )
                    for i, idx in enumerate(meta["indices"])
                ]
            else:
                data[key] = _frame_to_tensor(
                    data_frames[meta["index"] + 1], meta["dtype"], meta["shape"]
                )

        # 保留 frame 引用，防止 tensor 零拷贝视图的底层内存被提前释放。
        data["__zmq_frames__"] = data_frames

    return data


def _send_zero_copy_impl(
    socket: zmq.Socket, data: dict, prefix_frames: list = ()
) -> Tuple[int, int]:
    """核心零拷贝发送：提取 tensor、打包、拼接前缀帧、发送。"""
    tensors = _extract_tensors(data)
    data_frames, meta_size, tensor_size = _pack_frames(data, tensors)
    socket.send_multipart(list(prefix_frames) + data_frames, copy=False)
    return meta_size, tensor_size


def _strip_dealer_delimiter(frames: list) -> list:
    """剥离 DEALER/ROUTER 回复中的前导空分隔符帧。"""
    if frames and len(frames[0]) == 0:
        return frames[1:]
    return frames


def send_zero_copy(socket: zmq.Socket, data: dict) -> Tuple[int, int]:
    """零拷贝发送数据，分离多个Tensor和元数据

    Args:
        socket: ZMQ socket
        data: 要发送的数据字典，可包含torch.Tensor或list[torch.Tensor]

    Returns:
        Tuple[int, int]: (元数据大小, tensor总大小)
    """
    return _send_zero_copy_impl(socket, data)


def recv_zero_copy(socket: zmq.Socket) -> Dict[str, Any]:
    """零拷贝接收数据，支持多个tensor和tensor列表"""
    frames = socket.recv_multipart(copy=False)
    return _unpack_data(frames[0], frames)


def recv_zero_copy_router(socket: zmq.Socket) -> Tuple[List[zmq.Frame], Dict[str, Any]]:
    """ROUTER socket零拷贝接收数据，将identity frames与数据分离。

    Returns:
        (identity_frames, data)：identity_frames 含空分隔符，回复时需原样发回
    """
    frames = socket.recv_multipart(copy=False)

    delimiter_idx = None
    for i, frame in enumerate(frames):
        if len(frame) == 0:
            delimiter_idx = i
            break

    if delimiter_idx is None:
        raise ValueError("ROUTER message missing empty delimiter frame")

    identity_frames = frames[: delimiter_idx + 1]
    data_frames = frames[delimiter_idx + 1 :]

    data = _unpack_data(data_frames[0], data_frames)
    return identity_frames, data


def send_zero_copy_router(
    socket: zmq.Socket, identity_frames: List[zmq.Frame], data: dict
) -> Tuple[int, int]:
    """ROUTER socket零拷贝发送数据"""
    return _send_zero_copy_impl(socket, data, prefix_frames=identity_frames)


def send_zero_copy_dealer(socket: zmq.Socket, data: dict) -> Tuple[int, int]:
    """DEALER socket零拷贝发送，手动prepend空分隔符以兼容ROUTER协议。"""
    return _send_zero_copy_impl(socket, data, prefix_frames=[zmq.Frame(b"")])


def recv_zero_copy_dealer(socket: zmq.Socket) -> Dict[str, Any]:
    """DEALER socket零拷贝接收，剥离ROUTER回复中的前导空分隔符帧。"""
    frames = _strip_dealer_delimiter(socket.recv_multipart(copy=False))
    return _unpack_data(frames[0], frames)


def recv_zero_copy_streaming(socket: zmq.Socket) -> List[Dict[str, Any]]:
    """DEALER socket流式接收多条消息，直到收到最终消息。

    流式协议：中间消息含 __stream_seq__，最终消息含 __stream_final__，
    无上述字段则视为非流式单条消息。
    """
    results = []
    while True:
        frames = _strip_dealer_delimiter(socket.recv_multipart(copy=False))
        data = _unpack_data(frames[0], frames)
        is_final = data.pop("__stream_final__", False)
        seq = data.pop("__stream_seq__", None)
        results.append(data)
        if is_final or seq is None:
            break
    return results


def mark_stream_intermediate(data: dict, seq: int):
    """标记数据为流式中间消息。"""
    data["__stream_seq__"] = seq


def mark_stream_final(data: dict):
    """标记数据为流式最终消息。"""
    data["__stream_final__"] = True


def peek_stream_metadata(meta_frame) -> bool:
    """从msgpack元数据帧判断是否为流式中间消息，用于Gateway的peek/pop决策。"""
    buf = meta_frame.buffer if hasattr(meta_frame, "buffer") else meta_frame
    meta = msgpack.unpackb(buf, raw=False)
    return "__stream_seq__" in meta and "__stream_final__" not in meta
