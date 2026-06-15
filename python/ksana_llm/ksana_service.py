# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Dict

import orjson
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import status as http_status
from transformers import logging as hf_logging

import ksana_llm
from ksana_llm.arg_utils import EngineArgs
from ksana_llm.exceptions import KsanaEngineError
from request_id import generate_request_id

logger = logging.getLogger(__name__)

try:
    from openaiapi.openai_adapter import add_openai_routes, OpenAIConfig
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"\033[33m------OpenAI modules not available: {e}--------\033[0m")
    print("OpenAI compatible routes will be disabled.")
    OPENAI_AVAILABLE = False

# 配置常量
CONFIG = {
    "TIMEOUT_KEEP_ALIVE": 200,
    "FIELDS_TO_EXTRACT": [
        "x-remote-ip",
        "kv-comm-group-key",
        "request-id",
        "traceparent",
        'Content-Type'],
    "DEFAULT_LOG_LEVEL": "INFO",
}

# Health check timeout in seconds
HEALTH_CHECK_TIMEOUT = 60


class LLMService:
    def __init__(self, engine_args: EngineArgs):
        self.app = FastAPI(title="Ksana LLM Server", version="API-1.0.0")
        self.tokenizer = None
        self.pre_post_processor = None
        self.engine_args = engine_args
        self._setup_routes()
        self.model = ksana_llm.KsanaLLMEngine.from_engine_args(self.engine_args)

    def init_ipc(self):
        """Create ipc resource.
        """
        self.model.init_ipc()

    def initialize(self) -> None:
        """初始化模型"""
        self.tokenizer = self.model.tokenizer
        self.pre_post_processor = self.model.pre_post_processor
        self.reasoning_parser = getattr(self.engine_args, 'reasoning_parser', None)

        # Prepare think_end_token_id if reasoning_parser is enabled
        think_end_token_id = None
        if self.reasoning_parser is not None:
            try:
                from openaiapi.reasoning import ReasoningParserManager
                reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
                if reasoning_parser_class is None:
                    raise ValueError(f"Failed to create reasoning parser instance for: {self.reasoning_parser}")
                think_end_token = getattr(reasoning_parser_class, 'think_end_token', None)
                if think_end_token is None:
                    raise ValueError(f"Reasoning parser class '{self.reasoning_parser}' does not have think_end_token")
                vocab_dict = self.tokenizer.get_vocab()
                think_end_token_id = vocab_dict.get(think_end_token)
            except (ValueError, AttributeError, TypeError) as e:
                raise RuntimeError(f"Failed to get think_end_token_id: {e}") from e
        self.model.initialize(think_end_token_id=think_end_token_id)

        if OPENAI_AVAILABLE:
            try:
                # 创建OpenAI配置，传入解析器参数, 添加路由函数
                openai_config = OpenAIConfig(
                    tool_call_parser=getattr(self.engine_args, 'tool_call_parser', None), 
                    reasoning_parser=self.reasoning_parser, 
                    enable_auto_tool_choice=getattr(self.engine_args, 'enable_auto_tool_choice', False), 
                    tool_parser_plugin=getattr(self.engine_args, 'tool_parser_plugin', None))

                chat_template = getattr(self.engine_args, 'chat_template', None)
                chat_template_content_format = getattr(
                    self.engine_args, 'chat_template_content_format', None)

                add_openai_routes(
                    self,
                    openai_config,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format
                )
                print(
                    "\033[32m======OpenAI compatible routes added successfully========\033[0m\n")
            except AttributeError as e:
                print(
                    f"\033[33m------OpenAI configuration error: {e}--------\033[0m")
                print("OpenAI compatible routes will be disabled.")
        else:
            print(
                "\033[33m------OpenAI modules not available, skipping OpenAI routes--------\033[0m")

    def _setup_routes(self) -> None:
        """设置路由"""
        self.app.post("/generate")(self.generate)
        self.app.post("/forward")(self.forward)
        self.app.get("/health")(self.health_check)

    @staticmethod
    def get_trace_context(request: Request) -> Dict[str, str]:
        """获取请求的追踪上下文.

        若上游（例如 simple_router）已下发 ``request-id`` header，则透传；否则
        在此处生成一个与 C++ ``IdGenerator`` 同布局的 ID，注入回 ``req_ctx`` 后
        再交给 C++。这样 Python 接入层与 C++ 引擎可以用同一个 ID 串联 access log。
        """
        req_ctx = {
            field: request.headers.get(field)
            for field in CONFIG["FIELDS_TO_EXTRACT"]
            if request.headers.get(field) is not None
        }
        if "request-id" not in req_ctx:
            req_ctx["request-id"] = str(generate_request_id())
        return req_ctx

    async def format_streaming_output(self, ksana_python_output_iterator):
        """Perform streaming generation."""

        async def stream_results():
            try:
                async for ksana_python_output in ksana_python_output_iterator:
                    if ksana_python_output is None:
                        return

                    # Check finish_status for errors from C++ engine
                    if hasattr(ksana_python_output, 'finish_status') and ksana_python_output.finish_status:
                        if hasattr(ksana_python_output.finish_status, 'OK') and \
                                not ksana_python_output.finish_status.OK():
                            error_msg = ""
                            error_code = 500
                            if hasattr(ksana_python_output.finish_status, 'GetMessage'):
                                error_msg = ksana_python_output.finish_status.GetMessage()
                            if hasattr(ksana_python_output.finish_status, 'GetCode'):
                                error_code = ksana_python_output.finish_status.GetCode().value
                            error_ret = {
                                "error": {
                                    "message": error_msg or "Generation failed",
                                    "type": "internal_server_error",
                                    "code": error_code,
                                    "param": None,
                                }
                            }
                            yield orjson.dumps(error_ret) + b"\0"
                            return

                    input_token_ids = ksana_python_output.input_tokens
                    output_texts = []
                    output_token_ids = []
                    for request_output in ksana_python_output.output_tokens:
                        # Decode the output tokens using the tokenizer
                        output_token_ids.append(request_output)
                        try:
                            output_text = self.pre_post_processor.decode(
                                request_output, True)
                        except Exception as e:
                            logger.exception(e)
                            raise ValueError("Invalid token ids!") from e
                        output_texts.append(output_text)
                    ret = {
                        "texts": output_texts,
                        "output_token_ids": output_token_ids,  # the output token IDs
                        "logprobs": ksana_python_output.logprobs,
                        "input_token_ids": input_token_ids,  # the input token IDs
                        "cache_stat": ksana_python_output.cache_stat,
                    }
                    yield orjson.dumps(ret) + b"\0"
            except Exception as e:  # pylint: disable=broad-except
                if isinstance(e, KsanaEngineError):
                    error_code = e.error_code
                    error_message = str(e)
                else:
                    logger.exception(e)
                    error_code = 500
                    error_message = "Internal server error"
                error_ret = {
                    "error": {
                        "message": error_message,
                        "type": "internal_server_error",
                        "code": error_code,
                        "param": None,
                    }
                }
                yield orjson.dumps(error_ret) + b"\0"
            finally:
                # Whether the generator exits via normal completion, exception,
                # GeneratorExit (client disconnect → Starlette aclose), or
                # CancelledError, force the C++ StreamingIterator to wake up so
                # KV blocks of an abandoned request are released promptly.
                ksana_python_output_iterator.abort()

        # Return the asynchronous generator
        return stream_results()

    def format_output(self, ksana_python_output):
        # Decode the output tokens into a human-readable text using the
        # tokenizer
        output_text = []
        for tokens in ksana_python_output.output_tokens:
            try:
                output_text.append(
                    self.pre_post_processor.decode(
                        tokens, False))
            except Exception as e:
                logger.exception(e)
                raise ValueError("Invalid token ids!") from e

        # Create a response with the generated text and token IDs
        return {
            "texts": output_text,  # the generated text
            "output_token_ids": ksana_python_output.output_tokens,  # the generated token IDs
            "logprobs": ksana_python_output.logprobs,
            "input_token_ids": ksana_python_output.input_tokens,  # the input token IDs
            "cache_stat": ksana_python_output.cache_stat,
        }

    async def generate(self, request: Request) -> Response:
        """Generate completion for the request."""
        try:
            req_ctx = self.get_trace_context(request)
            request_dict = orjson.loads(await request.body())
            enable_streaming = request_dict.get("stream", True)
            status, output, _plugin_timing_metrics = await self.model.generate(
                request_dict=request_dict,
                streamer=enable_streaming,
                req_ctx=req_ctx,
            )
            if not status.OK():
                return JSONResponse(
                    status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "error": {
                            "message": status.GetMessage(),
                            "type": "internal_server_error",
                            "code": status.GetCode().value,
                            "param": None,
                        }
                    },
                )

            if enable_streaming:
                response_stream = await self.format_streaming_output(output)
                return StreamingResponse(response_stream)
            else:
                response_data = self.format_output(output)
                return JSONResponse(response_data)

        except Exception as e:  # pylint: disable=broad-except
            if isinstance(e, KsanaEngineError):
                error_code = e.error_code
                error_message = str(e)
            else:
                logger.exception(e)
                error_code = 500
                error_message = "Internal server error"
            return JSONResponse(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "message": error_message,
                        "type": "internal_server_error",
                        "code": error_code,
                        "param": None,
                    }
                },
            )

    async def forward(self, request: Request) -> Response:
        """Generate next token for the request.

        The request should be a JSON object packed by msgpack.
        """
        try:
            req_ctx = self.get_trace_context(request)
            request_bytes = await request.body()
            status, response_bytes = await self.model.forward(request_bytes, req_ctx)
            if not status.OK() or response_bytes is None:
                return JSONResponse(
                    status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "error": {
                            "message": status.GetMessage(),
                            "type": "internal_server_error",
                            "code": status.GetCode().value,
                            "param": None,
                        }
                    },
                )

            return Response(
                content=response_bytes,
                media_type="application/x-msgpack")

        except Exception as e:  # pylint: disable=broad-except
            if isinstance(e, KsanaEngineError):
                error_code = e.error_code
                error_message = str(e)
            else:
                logger.exception(e)
                error_code = 500
                error_message = "Internal server error"
            return JSONResponse(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "message": error_message,
                        "type": "internal_server_error",
                        "code": error_code,
                        "param": None,
                    }
                },
            )

    async def health_check(self, request: Request) -> Response:
        """Health check endpoint.

        Verifies the server is healthy by sending a minimal generation request
        to the engine. If the engine responds within the timeout,
        the server is considered healthy.

        Similar to sglang's /health endpoint implementation.
        """
        # If the model is not yet initialized, return 503
        if self.tokenizer is None or self.model is None:
            return Response(status_code=503)

        try:
            # Use a small but sufficient token sequence for the health check.
            # A single token like [0] can cause DeepGEMM CUDA errors (M dimension
            # too small for MoE routing), so use a few tokens instead.
            health_request_dict = {
                "input_tokens": [1, 2, 3, 4],
                "sampling_config": {
                    "num_beams": 1,
                    "topk": 1,
                    "topp": 0.0,
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                },
                "stream": False,
            }
            req_ctx = {"request-id": f"health_check_{int(time.time() * 1000)}"}

            # Use asyncio.wait_for to enforce timeout
            status, _, _ = await asyncio.wait_for(
                self.model.generate(
                    request_dict=health_request_dict,
                    streamer=False,
                    req_ctx=req_ctx,
                ),
                timeout=HEALTH_CHECK_TIMEOUT,
            )

            if status.OK():
                return Response(status_code=200)
            else:
                logger.error(
                    f"Health check failed: engine returned error: {status.GetMessage()}"
                )
                return Response(status_code=503)

        except asyncio.TimeoutError:
            logger.error(
                f"Health check failed: no response from engine within "
                f"{HEALTH_CHECK_TIMEOUT} seconds."
            )
            return Response(status_code=503)
        except (RuntimeError, OSError) as e:
            logger.error(f"Health check failed with exception: {e}")
            return Response(status_code=503)


class KsanaService:
    """ The ksana service.
    """

    def __init__(self, engine_args: EngineArgs):
        """Initialize a KsanaService instance.

        Args:
            engine_args: The serving arguments.
        """
        self._engine_args = engine_args

        # The service instance.
        self._service = LLMService(self._engine_args)

    def init_ipc(self):
        """Create ipc resource.
        """
        self._service.init_ipc()

    def start(self):
        """Start the service process.
        """
        uvloop.install()
        hf_logging.set_verbosity_error()

        self._service.initialize()

        if self._engine_args.endpoint != "python":
            signal.pause()
            sys.exit(0)

        # Set the log level of uvicorn based on KLLM_LOG_LEVEL.
        log_level = os.getenv("KLLM_LOG_LEVEL", "INFO").upper()
        if log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            log_level = log_level.lower()
        else:
            log_level = "info"
            print(
                f"Uvicorn's logging not support env: KLLM_LOG_LEVEL={log_level}, keep it as defalt(info)."
            )

        # distributed config.
        world_size = int(os.environ.get('WORLD_SIZE', "1"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))

        # For standalone or distributed master node, listen on service port.
        # For distributed worker node, wait until cluster destroyed.
        if world_size == 1 or node_rank == 0:
            self._service.app.root_path = self._engine_args.root_path
            uvicorn.run(
                self._service.app,
                host=self._engine_args.host,
                port=self._engine_args.port,
                log_level=log_level,
                access_log=self._engine_args.access_log,
                timeout_keep_alive=CONFIG["TIMEOUT_KEEP_ALIVE"],
                ssl_keyfile=self._engine_args.ssl_keyfile,
                ssl_certfile=self._engine_args.ssl_certfile,
            )
        else:
            # In multi-node cases, non-zero rank nodes launch a dummy health
            # check server (similar to sglang) so that the load balancer /
            # orchestrator can verify the worker is alive.
            self._launch_dummy_health_check_server(
                self._engine_args.host, self._engine_args.port
            )
            print("Uvicorn running on NONE.")
            import threading
            threading.Event().wait()

    @staticmethod
    def _launch_dummy_health_check_server(host: str, port: int):
        """Launch a lightweight HTTP server on worker nodes for health checks.

        Similar to sglang's launch_dummy_health_check_server. Only provides
        /health endpoint that always returns 200.
        """
        import threading

        dummy_app = FastAPI()

        @dummy_app.get("/health")
        async def _health():
            return Response(status_code=200)

        config = uvicorn.Config(
            dummy_app,
            host=host,
            port=port,
            log_level="warning",
            timeout_keep_alive=CONFIG["TIMEOUT_KEEP_ALIVE"],
        )
        server = uvicorn.Server(config=config)

        def run_server():
            try:
                import asyncio
                asyncio.run(server.serve())
            except (RuntimeError, OSError) as e:
                logger.error(f"Dummy health check server failed: {e}")

        thread = threading.Thread(
            target=run_server, daemon=True, name="dummy-health-check"
        )
        thread.start()
        logger.info(
            f"Dummy health check server started at {host}:{port} for worker node"
        )
