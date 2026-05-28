# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import libtorch_serving
from transformers import GenerationConfig, PreTrainedTokenizerFast

from utilize.logger import init_logger
from .arg_utils import EngineArgs
from .exceptions import KsanaEngineError
from .ksana_plugin import KsanaPlugin, PluginConfig, PluginTimingMetrics
from .processor_op_base import TokenizerProcessorOpBase

logger = init_logger(__name__)

# Adjust the max_workers to a reasonable number based on the number of CPUs
plugin_executor = ThreadPoolExecutor(max_workers=32)
model_executor = ThreadPoolExecutor(max_workers=256)


@dataclass
class EndpointConfig:
    endpoint: str = "python"  # endpoint type
    host: str = "0.0.0.0"  # endpoint host address
    port: int = 8080  # endpoint port
    access_log: bool = True  # whether to enable the endpoint access log


class PyAsyncStreamingIterator:
    """The streaming iterator."""

    def __init__(
        self,
        serving_iterator: libtorch_serving.StreamingIterator,
        ksana_plugin: KsanaPlugin,
        ksana_python_input: libtorch_serving.KsanaPythonInput,
        start_time: float = None,
        input_tokens: int = 0,
        headers_info: str = "",
        plugin_timing_metrics: PluginTimingMetrics = None,
    ):
        self._serving_iterator = serving_iterator
        self._ksana_plugin = ksana_plugin
        self._ksana_python_input = ksana_python_input
        self._start_time = start_time or time.time()
        self._input_tokens = input_tokens
        self._headers_info = headers_info
        self._output_tokens = 0
        self.plugin_timing_metrics = plugin_timing_metrics or PluginTimingMetrics()

    def __aiter__(self):
        return self

    def abort(self):
        """Abort the underlying C++ streaming iterator.

        Idempotent and safe to call from any thread / from a finally block.
        Wakes any pending GetNext() so the C++ shared_ptr can be released and
        ~StreamingIterator() can run, which is what releases KV blocks held by
        a request whose client disconnected mid-stream.
        """
        try:
            self._serving_iterator.Abort()
        except Exception:  # pylint: disable=broad-except
            pass

    # Define an asynchronous iterator method
    async def __anext__(self):
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Run the GetNext method of the serving iterator in an executor
        status, ksana_python_output = await loop.run_in_executor(
            model_executor, self._serving_iterator.GetNext
        )

        # Check the status of the iteration
        if status.OK():
            # If the iteration is successful, return the output
            self._ksana_plugin.postprocess(
                self._ksana_python_input, ksana_python_output
            )
            # 累加输出 token 数（每次迭代返回的是新增的 token）
            if hasattr(ksana_python_output, 'output_tokens') and ksana_python_output.output_tokens:
                self._output_tokens += len(ksana_python_output.output_tokens)
            return ksana_python_output
        elif status.GetCode() == libtorch_serving.RetCode.RET_STOP_ITERATION:
            # If the iteration has finished, log completion and raise StopAsyncIteration
            elapsed_ms = (time.time() - self._start_time) * 1000
            logger.info(f"ACCESS| streaming=true input_tokens={self._input_tokens} "
                       f"output_tokens={self._output_tokens} status=OK elapsed={elapsed_ms:.2f}ms {self._headers_info}")
            raise StopAsyncIteration(
                f"Iterator finished, ret code {status.GetCode()}, message {status.GetMessage()}."
            )
        else:
            # If an error occurred during iteration, log error and raise RuntimeError
            elapsed_ms = (time.time() - self._start_time) * 1000
            logger.info(f"ACCESS| streaming=true input_tokens={self._input_tokens} "
                       f"output_tokens={self._output_tokens} status=FAILED "
                       f"elapsed={elapsed_ms:.2f}ms {self._headers_info}")
            raise KsanaEngineError(status.GetMessage(), status.GetCode().value)


class KsanaLLMEngine:
    """The LLM serving model instance."""

    def __init__(
        self,
        config_file: str,
        plugin_config: PluginConfig,
        endpoint_config: EndpointConfig,
        pre_post_processor: TokenizerProcessorOpBase,
        json_model_override_args: str = None,
    ):
        """Initialize a KsanaLLMEngine instance.

        Args:
            config_file: The serving config file.
            plugin_config: Configuration for the Ksana plugin.
            endpoint_config: Configuration for the endpoint.
            pre_post_processor: The pre/post processor that contains the tokenizer.
            json_model_override_args: Optional JSON string to override ModelConfig fields,
                e.g. '{"num_layer":32,"rope_theta":500000.0}'.
        """
        self._config_file = config_file

        import os

        log_level = os.environ.get("KLLM_LOG_LEVEL", "INFO")
        logger.debug(f"KsanaLLMEngine initializing with config_file={config_file}, KLLM_LOG_LEVEL={log_level}")

        # Pass json_model_override_args directly to the C++ Serving constructor so
        # that it reaches Environment::ParseConfig before FLAGS_json_model_override_args
        # is read.  The old sys.argv injection approach never worked because
        # gflags::ParseCommandLineFlags() is never called in the pybind11 path.
        _json_override = (
            json_model_override_args if json_model_override_args is not None else ""
        )
        if _json_override:
            logger.debug(
                f"Passing json_model_override_args to Serving: {_json_override}"
            )

        # The serving instance.
        self._serving = libtorch_serving.Serving(
            self._config_file, True, -1, _json_override
        )

        self._ksana_plugin = KsanaPlugin(plugin_config)
        logger.debug(f"KsanaPlugin initialized with plugin_name={plugin_config.plugin_name}")

        if endpoint_config.endpoint != "python":
            self._serving.endpoint_config.type = libtorch_serving.EndpointType.RPC
            self._serving.endpoint_config.rpc_plugin_name = endpoint_config.endpoint

        self._serving.endpoint_config.host = endpoint_config.host
        self._serving.endpoint_config.port = endpoint_config.port
        self._serving.endpoint_config.access_log = endpoint_config.access_log

        self.pre_post_processor = pre_post_processor
        self.tokenizer = self.pre_post_processor.inner_tokenizer

        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            print(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )

        self.eos_token_id = None
        if hasattr(self.tokenizer, "eos_token_id"):
            self.eos_token_id = self.tokenizer.eos_token_id

    def init_ipc(self):
        """Create ipc resource.
        """
        self._serving.init_ipc()

    def initialize(self, think_end_token_id: Optional[int] = None) -> None:
        """Initialize the model.
        
        Args:
            think_end_token_id: Optional token ID marking the end of reasoning phase.
                              For reasoning models like DeepSeek-R1, this token marks
                              when to resume normal constrained decoding.
        """
        # Convert None to -1 for C++ compatibility
        token_id = think_end_token_id if think_end_token_id is not None else -1
        self._serving.init_serving(token_id)

    @classmethod
    def from_engine_args(cls, args: EngineArgs) -> "KsanaLLMEngine":
        """Create an instance from EngineArgs."""
        if args.config_file is None:
            raise ValueError("config_file is required.")
        plugin_config = PluginConfig(
            model_dir=args.model_dir,
            config_file=args.config_file,
            plugin_name=args.model_type,
            enable_trt=args.plugin_model_enable_trt,
            num_threads_per_encoder=args.plugin_num_threads_per_encoder,
            vit_model_type=args.multimodal_vit_model_type,
            encoder_url=args.encoder_url,
            encoder_timeout=args.encoder_timeout,
            multimodal_cache_size=args.multimodal_cache_size,
        )

        endpoint_config = EndpointConfig(
            endpoint=args.endpoint,
            host=args.host,
            port=args.port,
            access_log=args.access_log,
        )

        pre_post_processor = TokenizerProcessorOpBase(
            args.model_dir,
            args.tokenizer_path,
            args.tokenization
        )

        return cls(
            config_file=args.config_file,
            plugin_config=plugin_config,
            endpoint_config=endpoint_config,
            pre_post_processor=pre_post_processor,
            json_model_override_args=args.json_model_override_args,
        )

    def _build_generator_config(
            self, sampling_config: dict) -> GenerationConfig:
        """
        Build the transformers GenerationConfig object Validate the
        values of the attributes.
        """
        def _get_sampling_value(
                sampling_config: dict,
                key: str,
                default_val=None):
            """Get value from sampling_config dict, return default if key not exists.
            """
            return sampling_config[key] if key in sampling_config else default_val

        stop_token_ids = _get_sampling_value(
            sampling_config, "stop_token_ids", None)
        if not stop_token_ids:
            stop_token_ids = self.pre_post_processor.get_stop_token_ids()
        ignore_eos = _get_sampling_value(sampling_config, "ignore_eos", False)

        if (
            self.pre_post_processor.get_tokenizer_stop_token_ids() is not None
            and not ignore_eos
        ):
            eos_token_id = self.pre_post_processor.get_tokenizer_stop_token_ids()
            if isinstance(eos_token_id, list):
                for token_id in eos_token_id:
                    if token_id not in stop_token_ids:
                        stop_token_ids.append(token_id)
            else:
                if eos_token_id not in stop_token_ids:
                    stop_token_ids.append(eos_token_id)

        generation_config = GenerationConfig(
            top_k=_get_sampling_value(sampling_config, "topk", 1),
            do_sample=_get_sampling_value(sampling_config, "do_sample", None),
            top_p=_get_sampling_value(sampling_config, "topp", 1.0),
            temperature=_get_sampling_value(sampling_config, "temperature", 1.0),
            logprobs_num=_get_sampling_value(sampling_config, "logprobs", 0),
            repetition_penalty=_get_sampling_value(sampling_config,
                                                   "repetition_penalty", 1.0),
            no_repeat_ngram_size=_get_sampling_value(sampling_config,
                                                     "no_repeat_ngram_size", 0),
            encoder_no_repeat_ngram_size=_get_sampling_value(sampling_config,
                                                             "encoder_no_repeat_ngram_size", 0),
            decoder_no_repeat_ngram_size=_get_sampling_value(sampling_config,
                                                             "decoder_no_repeat_ngram_size", 0),
            enable_structured_output=_get_sampling_value(sampling_config,
                                                         "enable_structured_output", False),
            json_schema=_get_sampling_value(sampling_config,
                                            "json_schema", ""),
            num_beams=_get_sampling_value(sampling_config,
                                          "num_beams", 1),
            num_return_sequences=_get_sampling_value(sampling_config,
                                                     "num_return_sequences", 1),
            length_penalty=_get_sampling_value(sampling_config,
                                               "length_penalty", 1.0),
            stop_strings=_get_sampling_value(sampling_config,
                                             "stop_strings", []),
            enable_thinking=_get_sampling_value(sampling_config,
                                                "enable_thinking", True),
            ptp_token_num=_get_sampling_value(sampling_config,
                                              "ptp_token_num", -1),
            stop_token_ids=stop_token_ids,
            ignore_eos=ignore_eos
        )
        if "max_new_tokens" in sampling_config:
            generation_config.max_new_tokens = sampling_config["max_new_tokens"]

        return generation_config

    def _build_python_input(
        self,
        request_dict: Dict[str, Any] = None,
        generation_config: GenerationConfig = None,
    ) -> libtorch_serving.KsanaPythonInput:
        """Build the KsanaPythonInput object."""

        prompt_text: Optional[str] = request_dict.pop("prompt", None)
        input_tokens = request_dict.pop("input_tokens", None)

        # `messages` is the OpenAI Chat Completion API that can contain visual input
        # `additonal_params` are model specific params packed in a dict, e.g.,
        # `max_pixels`, `fps` for imgs and videos
        messages: Optional[List[Dict]] = request_dict.pop("messages", None)
        use_chat_template = request_dict.pop("use_chat_template", False)

        if input_tokens is None and prompt_text:
            model_type = request_dict.get("model_type", "empty")
            prompt_text = self.pre_post_processor.build_prompt(
                prompt_text, model_type, use_chat_template)
            input_tokens = self.pre_post_processor.encode(prompt_text)

        # Reject empty inputs early so we never feed an empty token vector into the
        # C++ inference path (which historically segfaults on empty input).
        # Common cause: clients posting an unsupported schema (e.g. sglang's
        # {"text": ..., "sampling_params": {...}}) — Ksana /generate only
        # understands "prompt" / "input_tokens" / "messages".
        if (input_tokens is None or len(input_tokens) == 0) and messages is None:
            raise KsanaEngineError(
                "Empty input: request must contain non-empty 'prompt', 'input_tokens', "
                "or 'messages'. (Hint: Ksana /generate does NOT accept sglang's "
                "{\"text\", \"sampling_params\"} schema; use --backend ksana / "
                "ksana-server / openai-chat instead.)",
                400,
            )

        # Create a KsanaPythonInput object
        ksana_python_input = libtorch_serving.KsanaPythonInput()
        ksana_python_input.model_name = request_dict.pop("model_name", "")
        ksana_python_input.return_cache_stat = request_dict.pop("cache_stat", False)

        # Set input tokens to the ksana_python_input
        if input_tokens is not None:
            ksana_python_input.input_tokens = input_tokens

        # Set the generation parameters to the ksana_python_input
        # sampling_config
        sampling_config = ksana_python_input.sampling_config

        def _get_generation_value(
            generation_config: GenerationConfig, key: str, default_val
        ):
            """Get value from generation_config, return default if key not exists.
            """
            value = getattr(generation_config, key, default_val)
            return default_val if value is None else value

        sampling_config.num_beams = _get_generation_value(
            generation_config, "num_beams", 1
        )
        sampling_config.topk = _get_generation_value(
            generation_config, "top_k", 1)
        sampling_config.topp = _get_generation_value(
            generation_config, "top_p", 1.0)
        sampling_config.temperature = _get_generation_value(
            generation_config, "temperature", 1.0
        )
        sampling_config.max_new_tokens = _get_generation_value(
            generation_config, "max_new_tokens", -1
        )
        sampling_config.logprobs_num = _get_generation_value(
            generation_config, "logprobs_num", 0
        )
        sampling_config.num_return_sequences = _get_generation_value(
            generation_config, "num_return_sequences", 1
        )
        sampling_config.repetition_penalty = _get_generation_value(
            generation_config, "repetition_penalty", 1.0
        )
        sampling_config.no_repeat_ngram_size = _get_generation_value(
            generation_config, "no_repeat_ngram_size", 0
        )
        sampling_config.encoder_no_repeat_ngram_size = _get_generation_value(
            generation_config, "encoder_no_repeat_ngram_size", 0
        )
        sampling_config.decoder_no_repeat_ngram_size = _get_generation_value(
            generation_config, "decoder_no_repeat_ngram_size", 0
        )
        sampling_config.enable_structured_output = _get_generation_value(
            generation_config, "enable_structured_output", False
        )
        sampling_config.json_schema = _get_generation_value(
            generation_config, "json_schema", ""
        )
        sampling_config.length_penalty = _get_generation_value(
            generation_config, "length_penalty", 1.0
        )
        sampling_config.stop_token_ids = _get_generation_value(
            generation_config, "stop_token_ids", []
        )
        sampling_config.ignore_eos = _get_generation_value(
            generation_config, "ignore_eos", False
        )
        sampling_config.stop_strings = _get_generation_value(
            generation_config, "stop_strings", []
        )
        sampling_config.enable_thinking = _get_generation_value(
            generation_config, "enable_thinking", True
        )
        sampling_config.ptp_token_num = _get_generation_value(
            generation_config, "ptp_token_num", -1
        )

        def _check_do_sample_params():
            do_sample = True
            if (_get_generation_value(generation_config, "do_sample", None)
                    is False or sampling_config.topk == 1):
                do_sample = False

            if (sampling_config.topk == 1 and _get_generation_value(
                    generation_config, "do_sample", None) is True):
                print(
                    "Generation parameter topk cannot be 1 when do_sample is explicitly set to True!"
                )

            if not do_sample:
                sampling_config.topk = 1
                sampling_config.topp = 1.0
                sampling_config.temperature = 1.0

        _check_do_sample_params()

        # Set additional parameters with the request_dict
        if "input_refit_embedding" in request_dict:
            input_refit_embedding = request_dict["input_refit_embedding"]
            if "pos" in input_refit_embedding:
                ksana_python_input.input_refit_embedding.pos = input_refit_embedding[
                    "pos"
                ]
            if "embeddings" in input_refit_embedding:
                ksana_python_input.input_refit_embedding.embeddings = (
                    input_refit_embedding["embeddings"]
                )
            if "embedding_bytes" in input_refit_embedding:
                ksana_python_input.input_refit_embedding.embedding_bytes = (
                    input_refit_embedding["embedding_bytes"]
                )
            if "embedding_dtype" in input_refit_embedding:
                ksana_python_input.input_refit_embedding.embedding_dtype = (
                    input_refit_embedding["embedding_dtype"]
                )
        else:
            # Support independent pos and embeddings parameters for faster
            # transmission
            if "pos" in request_dict:
                ksana_python_input.input_refit_embedding.pos = request_dict["pos"]
            if "embeddings" in request_dict:
                ksana_python_input.input_refit_embedding.embeddings = request_dict["embeddings"]
            if "embedding_bytes" in request_dict:
                ksana_python_input.input_refit_embedding.embedding_bytes = request_dict["embedding_bytes"]
            if "embedding_dtype" in request_dict:
                ksana_python_input.input_refit_embedding.embedding_dtype = request_dict["embedding_dtype"]

        if "structured_output_regex" in request_dict:
            ksana_python_input.structured_output_regex = request_dict[
                "structured_output_regex"
            ]

        if "draft_tokens" in request_dict:
            ksana_python_input.draft_tokens = request_dict["draft_tokens"]

        plugin_timing_metrics = PluginTimingMetrics()
        plugin_kwargs = {
            "messages": messages,
            "additional_params": request_dict.get("additional_params", {}),
            "prompt": prompt_text,
            "plugin_timing_metrics": plugin_timing_metrics,
        }

        # First exec preprocessing of ksana plugin before inference
        self._ksana_plugin.preprocess(ksana_python_input, **plugin_kwargs)

        return ksana_python_input, plugin_timing_metrics

    def _call_serving_generate(self, ksana_python_input, req_ctx):
        return self._serving.generate(ksana_python_input, req_ctx)

    def _call_serving_generate_streaming(self, ksana_python_input, req_ctx):
        return self._serving.generate_streaming(ksana_python_input, req_ctx)

    def _call_serving_forward(self, request_bytes, req_ctx):
        return self._serving.forward(request_bytes, req_ctx)

    def _format_access_log_headers(self, req_ctx: Dict[str, str]) -> str:
        """Format request headers for access log."""
        # 要记录的 header 字段
        header_fields = [
            "x-remote-ip",
            "kv-comm-group-key", 
            "request-id",
            "traceparent",
        ]

        parts = []
        for field in header_fields:
            value = req_ctx.get(field, "-") if req_ctx else "-"
            # 将 header 名称转换为下划线格式
            key = field.replace("-", "_")
            parts.append(f"{key}={value}")

        return " ".join(parts)

    async def generate(
        self,
        request_dict: Dict[str, Any],
        model_name: Optional[str] = None,
        streamer: Optional[bool] = None,
        req_ctx: Optional[Dict[str, str]] = None,
    ) -> Tuple[Any, Union[Dict[str, Any], AsyncGenerator], PluginTimingMetrics]:
        """The model generate interface."""
        start_time = time.time()

        if req_ctx is None:
            req_ctx = {}

        # 格式化 header 信息用于 access log
        headers_info = self._format_access_log_headers(req_ctx)

        # Build a  transformers generation_config from the request_dict
        # used to validate the values
        generation_config = self._build_generator_config(
            request_dict.get("sampling_config", {}))

        logger.debug(f"generation_config: max_new_tokens={generation_config.max_new_tokens}, "
                    f"temperature={generation_config.temperature}, "
                    f"top_p={generation_config.top_p}, top_k={generation_config.top_k}")

        # Build the ksana_python_input object in executor to avoid blocking event loop
        # This is especially important for KsanaPluginEPD which makes synchronous ZMQ calls
        loop = asyncio.get_running_loop()
        ksana_python_input, plugin_timing_metrics = await loop.run_in_executor(
            plugin_executor,
            self._build_python_input,
            request_dict,
            generation_config
        )

        # 获取输入 token 数量（用于 access log）
        input_tokens = len(ksana_python_input.input_tokens) if ksana_python_input.input_tokens else 0
        if streamer:
            status, streaming_iterator = await loop.run_in_executor(
                model_executor,
                self._call_serving_generate_streaming,
                ksana_python_input,
                req_ctx
            )

            if not status.OK():
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"ACCESS| streaming=true input_tokens={input_tokens} "
                           f"status=FAILED elapsed={elapsed_ms:.2f}ms {headers_info}")
                return status, None, plugin_timing_metrics

            # streaming 模式返回迭代器，完成日志在迭代器结束时记录
            return status, PyAsyncStreamingIterator(
                streaming_iterator, self._ksana_plugin, ksana_python_input,
                start_time=start_time, input_tokens=input_tokens, headers_info=headers_info,
                plugin_timing_metrics=plugin_timing_metrics
            ), plugin_timing_metrics

        status, ksana_python_output = await loop.run_in_executor(
            model_executor,
            self._call_serving_generate,
            ksana_python_input,
            req_ctx
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if not status.OK():
            logger.info(f"ACCESS| streaming=false input_tokens={input_tokens} "
                       f"output_tokens=0 status=FAILED elapsed={elapsed_ms:.2f}ms {headers_info}")
            return status, None, plugin_timing_metrics

        self._ksana_plugin.postprocess(ksana_python_input, ksana_python_output)

        # 获取输出 token 数量
        output_tokens = len(ksana_python_output.output_tokens) \
            if hasattr(ksana_python_output, 'output_tokens') \
                and ksana_python_output.output_tokens else 0

        logger.info(f"ACCESS| streaming=false input_tokens={input_tokens} "
                   f"output_tokens={output_tokens} status=OK elapsed={elapsed_ms:.2f}ms {headers_info}")

        return status, ksana_python_output, plugin_timing_metrics

    async def forward(
        self, request_bytes: bytes, req_ctx: Optional[Dict[str, str]] = None
    ) -> Tuple[Any, Optional[bytes]]:
        """The model forward interface."""
        if req_ctx is None:
            req_ctx = {}
        loop = asyncio.get_running_loop()
        status, response_bytes = await loop.run_in_executor(
            model_executor,
            self._call_serving_forward,
            request_bytes,
            req_ctx
        )
        if status.OK():
            return status, response_bytes
        else:
            return status, None
