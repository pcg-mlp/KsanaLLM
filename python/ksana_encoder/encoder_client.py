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
from codetiming import Timer

from utilize.logger import get_logger
from ksana_encoder.zmq_zero_copy import recv_zero_copy, send_zero_copy

logger = get_logger("ksana_encoder")


class EncoderClient:
    """EPD多模态编码器客户端

    线程安全的编码器客户端，使用ZeroMQ REQ/REP模式与服务端通信。
    每个线程维护独立的socket连接，支持高并发场景。

    特性:
        - 线程安全: 使用thread-local存储为每个线程创建独立socket
        - 零拷贝传输: 使用自定义协议高效传输大型tensor数据
        - 连接池管理: 统一管理所有线程的socket，支持优雅关闭
        - 超时控制: 支持发送/接收超时配置

    Attributes:
        encoder_url: 编码器服务端地址，格式为"tcp://host:port"
        timeout: 发送/接收超时时间(毫秒)
        context: ZeroMQ上下文对象
    """

    def __init__(
        self,
        encoder_url: str = "tcp://localhost:5555",
        timeout: int = 30000,
    ):
        """初始化编码器客户端

        Args:
            encoder_url: 编码器服务端ZeroMQ地址，默认"tcp://localhost:5555"
            timeout: 发送/接收超时时间(毫秒)，默认30000ms(30秒)
        """
        self.encoder_url = encoder_url
        self.timeout = timeout

        logger.info(f"Initializing encoder client for {encoder_url}")
        # 创建ZeroMQ上下文，所有socket共享此上下文
        self.context = zmq.Context()

        # 线程本地存储，每个线程有独立的socket
        self._thread_local = threading.local()
        # 记录所有创建的socket，用于统一关闭
        self._all_sockets: List[zmq.Socket] = []
        # 保护_all_sockets的线程锁
        self._sockets_lock = threading.Lock()

        logger.info(
            "Encoder client initialized successfully (thread-safe with connection pool)"
        )

    def _get_socket(self):
        """获取当前线程的socket连接(懒加载)

        为每个线程创建独立的ZeroMQ REQ socket，实现线程安全。
        首次调用时创建socket并连接到服务端，后续调用直接返回已有socket。

        Returns:
            zmq.Socket: 当前线程的REQ socket

        Note:
            - socket配置: RCVTIMEO/SNDTIMEO设置超时，LINGER=0确保快速关闭
            - 新创建的socket会被添加到_all_sockets列表，便于统一管理
        """
        if (
            not hasattr(self._thread_local, "socket")
            or self._thread_local.socket is None
        ):
            # 创建REQ类型socket(请求-响应模式)
            socket = self.context.socket(zmq.REQ)
            socket.connect(self.encoder_url)
            # 设置接收超时
            socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            # 设置发送超时
            socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            # LINGER=0: 关闭时立即丢弃未发送消息，避免阻塞
            socket.setsockopt(zmq.LINGER, 0)

            self._thread_local.socket = socket
            # 线程安全地添加到全局socket列表
            with self._sockets_lock:
                self._all_sockets.append(socket)

            logger.debug(
                f"Created new socket for thread {threading.current_thread().ident}"
            )

        return self._thread_local.socket

    def __call__(
        self,
        messages: Dict,
    ) -> Optional[Dict[str, Any]]:
        """执行多模态编码请求

        将输入数据发送到编码器服务端进行处理，返回编码结果。
        使用零拷贝协议高效传输大型tensor数据。

        Args:
            messages: 待编码的多模态数据字典，通常包含:
                - images: 图像数据列表(numpy数组或tensor)
                - text: 文本内容
                - 其他模态数据...

        Returns:
            Optional[Dict[str, Any]]: 编码结果字典，包含编码后的特征向量等。
                成功时返回结果字典，失败时返回None。

        Raises:
            不抛出异常，所有错误通过日志记录并返回None

        Note:
            - 自动生成唯一request_id用于追踪请求
            - 记录详细的时间统计: 总耗时、发送耗时、编码耗时、接收耗时
            - 响应中的status/encode_time/server_recv_timestamp字段会被移除后返回
        """
        # 生成唯一请求ID，用于日志追踪
        request_id = shortuuid.uuid()
        client_send_timestamp = time.time()
        # 构造请求数据，包含元信息和用户数据
        request = {
            "request_id": request_id,
            "client_send_timestamp": client_send_timestamp,
            **messages,
        }

        logger.info(f"({request_id}) Request starting...")

        try:
            with Timer(logger=None) as timer:
                socket = self._get_socket()
                # 使用零拷贝协议发送请求，返回元数据和tensor数据大小
                meta_size, tensor_size = send_zero_copy(socket, request)
                logger.info(
                    f"({request_id}) Sent: meta={meta_size/1024:.1f}KB, "
                    f"tensor={tensor_size/1024/1024:.1f}MB"
                )
                # 使用零拷贝协议接收响应
                response = recv_zero_copy(socket)

            # 计算各阶段耗时
            total_time = timer.last
            encode_time = response["encode_time"]
            server_recv_timestamp = response.get("server_recv_timestamp")
            send_time = server_recv_timestamp - client_send_timestamp
            recv_time = total_time - send_time - encode_time
            logger.info(
                f"({request_id}) Times: total={total_time:.3f}s, "
                f"send={send_time:.3f}s, encode={encode_time:.3f}s, "
                f"recv={recv_time:.3f}s"
            )

            # 检查服务端返回状态
            if response.get("status") != "success":
                error_msg = response.get("message", "Unknown error")
                logger.error(f"({request_id}) Request failed: {error_msg}")
                return None

            # 移除内部字段，只返回编码结果
            response.pop("status", None)
            response.pop("encode_time", None)
            response.pop("server_recv_timestamp", None)
            return response

        except zmq.Again:
            # ZeroMQ超时异常
            logger.error(f"({request_id}) Request timeout")
            return None

        except Exception as e:
            # 捕获所有其他异常
            logger.error(f"({request_id}) Request error: {e}")
            return None

    def close(self):
        """关闭客户端，释放所有资源

        关闭所有线程创建的socket连接，并终止ZeroMQ上下文。
        此方法是线程安全的，可以从任意线程调用。

        Note:
            - 关闭后客户端不可再使用
            - 建议使用上下文管理器(with语句)自动管理生命周期
        """
        # 线程安全地关闭所有socket
        with self._sockets_lock:
            for socket in self._all_sockets:
                socket.close()
            self._all_sockets.clear()

        # 终止ZeroMQ上下文
        if self.context is not None:
            self.context.term()
            self.context = None

        logger.info("Encoder client closed")

    def __enter__(self):
        """上下文管理器入口，支持with语句

        Returns:
            EncoderClient: 返回自身实例
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动关闭客户端

        Args:
            exc_type: 异常类型(如有)
            exc_val: 异常值(如有)
            exc_tb: 异常追踪信息(如有)
        """
        self.close()
