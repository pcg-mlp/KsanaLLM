# pylint: disable=broad-except
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""EPD编码器服务端
本模块实现了基于ZMQ的编码器服务端，用于接收客户端的编码请求，
调用插件进行编码处理，并返回编码结果。支持零拷贝传输以提高性能。
"""

import time
import zmq
from codetiming import Timer

from ksana_llm.arg_utils import EngineArgs
from ksana_llm.ksana_plugin import KsanaPlugin
from ksana_encoder.zmq_zero_copy import recv_zero_copy, send_zero_copy
from utilize.logger import get_logger

logger = get_logger("ksana_encoder")


class EncoderServer:
    """EPD编码器服务端类
    该类实现了一个基于ZMQ REP模式的编码服务端，主要功能包括：
    - 初始化ZMQ通信套接字，监听指定地址和端口
    - 加载并初始化编码插件（如视觉编码器）
    - 接收客户端编码请求，调用插件处理，返回编码结果
    - 支持零拷贝传输，优化大规模tensor数据的传输性能

    Attributes:
        config (EngineArgs): 引擎配置参数，包含host、port、model_dir等
        context (zmq.Context): ZMQ上下文对象
        socket (zmq.Socket): ZMQ REP套接字，用于请求-响应通信
        plugin: 加载的编码插件实例，提供encode方法
    """

    def __init__(self, engine_args: EngineArgs):
        """初始化编码器服务端

        执行以下初始化步骤：
        1. 保存配置参数
        2. 创建ZMQ上下文和REP套接字，绑定到指定地址
        3. 加载并初始化编码插件

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

        # 初始化ZMQ通信：创建上下文和REP套接字，绑定到tcp://host:port
        logger.info(f"Initializing ZMQ on {self.config.host}:{self.config.port}")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.config.host}:{self.config.port}")

        # 加载编码插件
        logger.info(
            f"Loading plugin '{self.config.model_type}' from {self.config.model_dir}"
        )
        self._load_plugin(self.config)

        logger.info("Encoder server initialized successfully")

    def _load_plugin(self, config: EngineArgs):
        """加载并初始化编码插件

        执行以下步骤：
        1. 根据model_dir和model_type搜索插件路径
        2. 加载插件模块
        3. 调用插件的init_plugin方法进行初始化

        Args:
            config (EngineArgs): 引擎配置参数，包含模型路径和类型信息

        Raises:
            RuntimeError: 插件加载失败（找不到插件或加载返回None）
            Exception: 插件init_plugin方法执行失败
        """
        # 使用KsanaPlugin的标准方法搜索插件路径
        plugin_path = KsanaPlugin.search_plugin_path(
            config.model_dir, config.model_type
        )
        self.plugin = KsanaPlugin.load_plugin(plugin_path)

        # 验证插件加载成功
        if self.plugin is None:
            error_msg = (
                f"Failed to load plugin '{config.model_type}' from {config.model_dir}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Plugin loaded from: {plugin_path}")

        # 初始化插件：构造参数并调用init_plugin方法
        if hasattr(self.plugin, "init_plugin"):
            kwargs = {
                "preprocess": True,  # 启用预处理
                "postprocess": True,  # 启用后处理
                "model_path": config.model_dir,
                "config_file": config.config_file,
                "model_type": config.model_type,
                "vit_model_type": config.multimodal_vit_model_type or "",
            }

            try:
                self.plugin.init_plugin(**kwargs)
                logger.info(f"Plugin '{config.model_type}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize plugin: {e}")
                raise
        else:
            logger.warning("Plugin does not have 'init_plugin' method")

    def run(self):
        """运行编码器服务端主循环

        启动服务端主循环，持续监听并处理客户端请求：
        1. 使用零拷贝方式接收请求数据
        2. 记录请求接收时间，计算网络传输耗时
        3. 调用插件的encode方法进行编码处理
        4. 构造响应并使用零拷贝方式发送
        5. 处理异常情况，返回错误响应

        请求数据格式（dict）：
            - request_id: 请求唯一标识
            - client_send_timestamp: 客户端发送时间戳
            - 其他编码所需的数据（如图像数据）

        响应数据格式（dict）：
            成功时：
                - status: "success"
                - encode_time: 编码耗时（秒）
                - server_recv_timestamp: 服务端接收时间戳
                - 编码结果数据（如embeddings tensor）
            失败时：
                - status: "error"
                - message: 错误信息

        Note:
            - 使用Ctrl+C可优雅关闭服务
            - 关闭时会自动清理ZMQ资源
        """
        logger.info(f"Encoder server running on {self.config.host}:{self.config.port}")
        logger.info(f"Using plugin: {self.config.model_type}")
        logger.info("Zero-copy enabled")
        logger.info("Waiting for requests...")

        try:
            while True:
                # 零拷贝接收请求，避免大tensor数据的内存拷贝
                request = recv_zero_copy(self.socket)
                server_recv_timestamp = time.time()

                # 提取请求元信息
                request_id = request.get("request_id", "unknown")
                client_send_timestamp = request.get("client_send_timestamp", None)

                # 计算网络传输耗时（客户端发送到服务端接收）
                send_time = server_recv_timestamp - client_send_timestamp
                logger.info(
                    f"({request_id}) Received request (send_time={send_time:.3f}s)"
                )

                try:
                    # 调用插件进行编码，使用Timer记录耗时
                    with Timer(logger=None) as timer:
                        result = self.plugin.encode(request)

                    # 构造成功响应，包含编码结果和性能指标
                    response = {
                        **result,
                        "status": "success",
                        "encode_time": timer.last,
                        "server_recv_timestamp": server_recv_timestamp,
                    }

                    # 零拷贝发送响应，返回元数据和tensor数据的大小
                    meta_size, tensor_size = send_zero_copy(self.socket, response)
                    logger.info(
                        f"({request_id}) Response: meta={meta_size/1024:.1f}KB, "
                        f"tensor={tensor_size/1024/1024:.1f}MB"
                    )

                except Exception as e:
                    # 编码失败，记录错误并返回错误响应
                    logger.error(f"({request_id}) Request failed: {e}")
                    response = {
                        "status": "error",
                        "message": str(e),
                    }
                    send_zero_copy(self.socket, response)

        except KeyboardInterrupt:
            # 捕获Ctrl+C，优雅关闭
            logger.info("Shutting down...")
        finally:
            # 清理ZMQ资源
            self.socket.close()
            self.context.term()


def main():
    """主入口函数

    解析命令行参数，创建并启动编码器服务端。
    使用EngineArgs.from_command_line()自动处理以下参数：
    - --config_file: 配置文件路径
    - --host: 服务监听地址
    - --port: 服务监听端口
    - --model_dir: 模型目录
    - --model_type: 模型类型
    - 其他EngineArgs支持的参数
    """
    engine_args = EngineArgs.from_command_line()
    server = EncoderServer(engine_args=engine_args)
    server.run()


if __name__ == "__main__":
    main()
