# pylint: disable=broad-except,unused-import
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""EPD编码器服务端
本模块实现了基于ZMQ的编码器服务端，用于接收客户端的编码请求，
调用插件进行编码处理，并返回编码结果。支持零拷贝传输以提高性能。
"""

import itertools
import os
import queue
import sys
import threading
import time
import zmq

try:
    import ksana_llm
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.insert(0, parent_dir)

from ksana_llm.arg_utils import EngineArgs
from ksana_llm.ksana_plugin import KsanaPlugin
from ksana_encoder.zmq_zero_copy import (
    recv_zero_copy_router,
    send_zero_copy_router,
    mark_stream_intermediate,
    mark_stream_final,
)
from utilize.logger import get_logger

logger = get_logger("ksana_encoder")


class EncoderServer:
    """EPD编码器服务端类
    该类实现了一个基于ZMQ ROUTER模式的编码服务端，主要功能包括：
    - 初始化ZMQ通信套接字，监听指定地址和端口
    - 加载并初始化编码插件（如视觉编码器）
    - 后台线程持续接收请求并缓存到队列，主线程从队列取请求处理
    - 支持零拷贝传输，优化大规模tensor数据的传输性能

    Attributes:
        config (EngineArgs): 引擎配置参数，包含host、port、model_dir等
        context (zmq.Context): ZMQ上下文对象
        socket (zmq.Socket): ZMQ ROUTER套接字
        plugin: 加载的编码插件实例，提供encode方法
        _request_queue: 接收线程与主线程之间的请求队列
    """

    def __init__(self, engine_args: EngineArgs):
        """初始化编码器服务端

        Args:
            engine_args (EngineArgs): 引擎配置参数对象，需包含以下字段：
                - host: 服务监听地址（如 "0.0.0.0"）
                - port: 服务监听端口（如 5555）
                - model_dir: 模型目录路径
                - model_type: 模型/插件类型
                - config_file: 配置文件路径（可选）
                - multimodal_vit_model_type: ViT模型类型（可选）

        Raises:
            RuntimeError: 插件加载失败时抛出
            Exception: 插件初始化失败时抛出
        """
        self.config = engine_args
        self._running = False

        # 初始化ZMQ ROUTER套接字
        logger.info(f"Initializing ZMQ on {self.config.host}:{self.config.port}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://{self.config.host}:{self.config.port}")

        # 请求队列：recv线程 -> 主线程
        self._request_queue = queue.Queue()

        # 加载编码插件
        logger.info(
            f"Loading plugin '{self.config.model_type}' from {self.config.model_dir}"
        )
        self._load_plugin()

        logger.info("Encoder server initialized successfully")

    def _load_plugin(self):
        """加载并初始化编码插件

        Raises:
            RuntimeError: 插件加载失败（找不到插件或加载返回None）
            Exception: 插件init_plugin方法执行失败
        """
        cfg = self.config

        plugin_path = KsanaPlugin.search_plugin_path(cfg.model_dir, cfg.model_type)
        self.plugin = KsanaPlugin.load_plugin(plugin_path)

        if self.plugin is None:
            error_msg = f"Failed to load plugin '{cfg.model_type}' from {cfg.model_dir}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Plugin loaded from: {plugin_path}")

        if hasattr(self.plugin, "init_plugin"):
            try:
                self.plugin.init_plugin(
                    preprocess=True,
                    postprocess=True,
                    model_path=cfg.model_dir,
                    config_file=cfg.config_file,
                    model_type=cfg.model_type,
                    vit_model_type=cfg.multimodal_vit_model_type or "",
                )
                logger.info(f"Plugin '{cfg.model_type}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize plugin: {e}")
                raise
        else:
            logger.warning("Plugin does not have 'init_plugin' method")

    def _recv_loop(self):
        """后台接收线程：持续从ROUTER socket接收请求并放入队列

        每个请求入队时附带server_recv_timestamp，用于计算queue_time。
        线程在self._running为False时退出。
        """
        while self._running:
            try:
                if self.socket.poll(timeout=1000):
                    identity_frames, request = recv_zero_copy_router(self.socket)
                    self._request_queue.put((identity_frames, request, time.time()))
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"Recv thread ZMQ error: {e}")
            except Exception as e:
                if self._running:
                    logger.error(f"Recv thread error: {e}")

    def run(self):
        """运行编码器服务端主循环

        启动后台接收线程持续接收请求放入队列，主线程从队列取请求处理：
        1. 从队列取请求，计算queue_time
        2. 调用插件的encode方法进行编码处理
        3. 构造响应（含queue_time）并使用ROUTER零拷贝方式发送
        4. 处理异常情况，返回错误响应
        """
        logger.info(f"Encoder server running on {self.config.host}:{self.config.port}")
        logger.info(f"Using plugin: {self.config.model_type}")
        logger.info("Zero-copy enabled, ROUTER mode with recv thread")

        streaming_enabled = os.environ.get("KSANA_ENABLE_ENCODER_STREAMING", "0") == "1"
        if streaming_enabled:
            logger.info("Streaming encode enabled (KSANA_ENABLE_ENCODER_STREAMING=1)")

        logger.info("Waiting for requests...")

        # 启动后台接收线程
        self._running = True
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()

        try:
            while self._running:
                try:
                    identity_frames, request, server_recv_timestamp = (
                        self._request_queue.get(timeout=1.0)
                    )
                except queue.Empty:
                    continue

                request_id = request.get("request_id", "unknown")
                client_send_timestamp = request.get("client_send_timestamp", None)

                queue_dequeue_time = time.time()
                send_time = server_recv_timestamp - client_send_timestamp
                queue_time = queue_dequeue_time - server_recv_timestamp
                logger.info(
                    f"({request_id}) Received request "
                    f"(send_time={send_time:.3f}s, queue_time={queue_time:.3f}s)"
                )

                try:
                    # 流式模式：构造中间结果回调
                    encode_kwargs = {}
                    if streaming_enabled:
                        seq = itertools.count()

                        def on_intermediate_ready(tensor_data: dict):
                            # NOTE: 此回调在主线程（plugin.encode内部）执行，而_recv_loop
                            # 后台线程同时在对self.socket执行poll/recv。ZMQ socket非线程安全，
                            # 理论上存在竞态风险。实际运行中由于poll(1000ms)等待窗口远大于
                            # send操作耗时，碰撞概率极低，目前未观察到异常。若未来并发压力增大
                            # 或出现偶发通信异常，应考虑为socket操作增加threading.Lock保护。
                            mark_stream_intermediate(tensor_data, next(seq))
                            send_zero_copy_router(
                                self.socket, identity_frames, tensor_data
                            )

                        encode_kwargs["on_intermediate_ready"] = on_intermediate_ready

                    # 编码
                    t0 = time.perf_counter()
                    result = self.plugin.encode(request, **encode_kwargs)
                    encode_time = time.perf_counter() - t0

                    # 构造响应（原地设置，避免拷贝整个result字典）
                    result["status"] = "success"
                    result["encode_time"] = encode_time
                    result["queue_time"] = queue_time
                    result["server_recv_timestamp"] = server_recv_timestamp

                    if streaming_enabled:
                        mark_stream_final(result)

                    # ROUTER零拷贝发送响应
                    meta_size, tensor_size = send_zero_copy_router(
                        self.socket, identity_frames, result
                    )
                    logger.info(
                        f"({request_id}) Response: meta={meta_size/1024:.1f}KB, "
                        f"tensor={tensor_size/1024/1024:.1f}MB"
                    )

                except Exception as e:
                    logger.error(f"({request_id}) Request failed: {e}")
                    send_zero_copy_router(
                        self.socket,
                        identity_frames,
                        {"status": "error", "message": str(e)},
                    )

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._running = False
            recv_thread.join(timeout=3.0)
            if recv_thread.is_alive():
                logger.warning(
                    "Recv thread did not exit within 3s, "
                    "forcing socket close (may cause ZMQError in recv thread)"
                )
            self.socket.close()
            self.context.term()


def main():
    """主入口函数"""
    engine_args = EngineArgs.from_command_line()
    server = EncoderServer(engine_args=engine_args)
    server.run()


if __name__ == "__main__":
    main()
