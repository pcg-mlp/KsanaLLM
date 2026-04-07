# pylint: disable=broad-except
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""EPD编码器客户端模块
提供线程安全的多模态编码器客户端，通过ZeroMQ与编码器服务端通信，支持零拷贝数据传输。
"""

import threading
import time
from typing import Any, Dict, List, Optional
import shortuuid
import zmq

from utilize.logger import get_logger
from ksana_encoder.zmq_zero_copy import (
    recv_zero_copy,
    recv_zero_copy_dealer,
    recv_zero_copy_streaming,
    send_zero_copy,
    send_zero_copy_dealer,
)

logger = get_logger("ksana_encoder")

# 响应中需要移除的内部字段
_INTERNAL_FIELDS = ("status", "encode_time", "queue_time", "server_recv_timestamp")


class _BaseEncoderClient:
    """编码器客户端基类

    提供线程安全的ZMQ连接池管理、请求构造、响应处理等通用逻辑。
    子类只需指定 _SOCKET_TYPE 并实现 _send_recv 方法。

    Attributes:
        encoder_url: 编码器服务端地址，格式为"tcp://host:port"
        timeout: 发送/接收超时时间(毫秒)
        context: ZeroMQ上下文对象
    """

    _SOCKET_TYPE: int  # 子类指定: zmq.REQ 或 zmq.DEALER

    def __init__(
        self,
        encoder_url: str = "tcp://localhost:5555",
        timeout: int = 30000,
    ):
        self.encoder_url = encoder_url
        self.timeout = timeout
        self.context = zmq.Context()
        self._thread_local = threading.local()
        self._all_sockets: List[zmq.Socket] = []
        self._sockets_lock = threading.Lock()

    def _get_socket(self) -> zmq.Socket:
        """获取当前线程的socket连接(懒加载)，线程安全"""
        if (
            not hasattr(self._thread_local, "socket")
            or self._thread_local.socket is None
        ):
            socket = self.context.socket(self._SOCKET_TYPE)
            socket.connect(self.encoder_url)
            socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            socket.setsockopt(zmq.LINGER, 0)

            self._thread_local.socket = socket
            with self._sockets_lock:
                self._all_sockets.append(socket)

            logger.debug(
                f"Created new {self._SOCKET_TYPE} socket for "
                f"thread {threading.current_thread().ident}"
            )

        return self._thread_local.socket

    def _build_request(self, messages: Dict) -> tuple:
        """构造带元信息的请求数据，返回 (request_id, client_send_timestamp, request_dict)"""
        request_id = shortuuid.uuid()
        client_send_timestamp = time.time()
        request = {
            "request_id": request_id,
            "client_send_timestamp": client_send_timestamp,
            **messages,
        }
        return request_id, client_send_timestamp, request

    def _process_response(
        self,
        response: Dict,
        request_id: str,
        client_send_timestamp: float,
        total_time: float,
    ) -> Optional[Dict[str, Any]]:
        """计算耗时、校验状态、清理内部字段。成功返回response，失败返回None。"""
        encode_time = response.get("encode_time", 0)
        server_recv_timestamp = response.get("server_recv_timestamp")

        if server_recv_timestamp is not None:
            send_time = server_recv_timestamp - client_send_timestamp
            queue_time = response.get("queue_time", 0)
            recv_time = total_time - send_time - queue_time - encode_time
        else:
            send_time = queue_time = recv_time = 0

        logger.info(
            f"({request_id}) Times: total={total_time:.3f}s, "
            f"send={send_time:.3f}s, queue={queue_time:.3f}s, "
            f"encode={encode_time:.3f}s, recv={recv_time:.3f}s"
        )

        if response.get("status") != "success":
            error_msg = response.get("message", "Unknown error")
            logger.error(f"({request_id}) Request failed: {error_msg}")
            return None

        # 写入分阶段耗时
        response["encoder_send_time"] = send_time
        response["encoder_queue_time"] = queue_time
        response["encoder_encode_time"] = encode_time
        response["encoder_recv_time"] = recv_time

        # 移除内部字段
        for key in _INTERNAL_FIELDS:
            response.pop(key, None)

        return response

    def _send_recv(self, socket: zmq.Socket, request: Dict, request_id: str) -> Dict:
        """子类实现：发送请求并接收响应，返回响应字典"""
        raise NotImplementedError

    def __call__(self, messages: Dict) -> Optional[Dict[str, Any]]:
        """执行多模态编码请求

        Args:
            messages: 待编码的多模态数据字典

        Returns:
            成功时返回编码结果字典，失败时返回None
        """
        request_id, client_send_timestamp, request = self._build_request(messages)
        logger.info(f"({request_id}) Request starting...")

        try:
            t0 = time.perf_counter()
            socket = self._get_socket()
            response = self._send_recv(socket, request, request_id)
            total_time = time.perf_counter() - t0

            return self._process_response(
                response, request_id, client_send_timestamp, total_time
            )

        except zmq.Again:
            logger.error(f"({request_id}) Request timeout")
            return None

        except Exception as e:
            logger.error(f"({request_id}) Request error: {e}")
            return None

    def close(self):
        """关闭客户端，释放所有资源（线程安全）"""
        with self._sockets_lock:
            for socket in self._all_sockets:
                socket.close()
            self._all_sockets.clear()

        if self.context is not None:
            self.context.term()
            self.context = None

        logger.info("Encoder client closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EncoderClient(_BaseEncoderClient):
    """EPD多模态编码器客户端（REQ/REP模式）

    线程安全的编码器客户端，每个线程维护独立的REQ socket连接。
    支持零拷贝传输大型tensor数据，适用于高并发场景。
    """

    _SOCKET_TYPE = zmq.REQ

    def __init__(
        self,
        encoder_url: str = "tcp://localhost:5555",
        timeout: int = 30000,
    ):
        super().__init__(encoder_url, timeout)
        logger.info(
            f"Encoder client initialized for {encoder_url} "
            "(thread-safe with connection pool)"
        )

    def _send_recv(self, socket, request, request_id):
        meta_size, tensor_size = send_zero_copy(socket, request)
        logger.info(
            f"({request_id}) Sent: meta={meta_size/1024:.1f}KB, "
            f"tensor={tensor_size/1024/1024:.1f}MB"
        )
        return recv_zero_copy(socket)


class StreamingEncoderClient(_BaseEncoderClient):
    """EPD多模态编码器流式客户端（DEALER模式）

    使用DEALER socket替代REQ，支持接收多条流式消息（deepstack中间结果边算边传）。
    手动prepend空分隔符帧，与ROUTER侧线路格式兼容。
    """

    _SOCKET_TYPE = zmq.DEALER

    def __init__(
        self,
        encoder_url: str = "tcp://localhost:5555",
        timeout: int = 30000,
        streaming: bool = True,
    ):
        super().__init__(encoder_url, timeout)
        self.streaming = streaming
        logger.info(
            f"Streaming encoder client initialized for {encoder_url} "
            f"(streaming={streaming}, DEALER socket, thread-safe)"
        )

    @staticmethod
    def _merge_streaming_results(
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """合并流式消息：收集中间消息的deepstack_feature，合入最终消息。"""
        # 最终消息始终是最后一条（由协议保证）
        final = messages[-1] if messages else {}
        deepstack_features = [
            msg["deepstack_feature"]
            for msg in messages[:-1]
            if "deepstack_feature" in msg
        ]
        if deepstack_features:
            final["deepstack_features"] = deepstack_features
        return final

    def _send_recv(self, socket, request, request_id):
        meta_size, tensor_size = send_zero_copy_dealer(socket, request)
        logger.info(
            f"({request_id}) Sent (DEALER): meta={meta_size/1024:.1f}KB, "
            f"tensor={tensor_size/1024/1024:.1f}MB"
        )
        if self.streaming:
            all_messages = recv_zero_copy_streaming(socket)
            return self._merge_streaming_results(all_messages)
        return recv_zero_copy_dealer(socket)
