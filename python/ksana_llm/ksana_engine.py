# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import asyncio
import os
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

# Three separate pools, split by how long the work blocks. Do NOT merge them:
# mixing fast submits with calls that block for a whole request causes FIFO
# head-of-line blocking inside the pool. Symptom when that happens is nasty and
# silent -- engine throughput stays perfect, but token delivery arrives in
# multi-second bursts and some requests never even reach the engine.
#
#   plugin_executor       building the engine input (short, CPU-bound)
#   submit_executor       handing a request to the C++ engine (short)
#   engine_wait_executor  calls that block until the engine produces output:
#                         StreamingIterator.GetNext (once per streamed chunk),
#                         plus non-streaming Generate and forward (whole request)
#
# engine_wait_executor needs one thread per in-flight request, because GetNext
# parks its thread in step_waiter->Wait() until the next token exists. The
# bindings release the GIL first (see torch_op/serving_op.cpp), so parked threads
# are cheap and don't contend -- but the pool must still be sized above peak
# concurrency or delivery degrades as described above.
ENGINE_WAIT_THREADS = int(os.environ.get("KSANA_ENGINE_WAIT_THREADS", "2048"))

plugin_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="ksana-plugin")
submit_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="ksana-submit")
engine_wait_executor = ThreadPoolExecutor(
    max_workers=ENGINE_WAIT_THREADS, thread_name_prefix="ksana-wait"
)

_WAIT_POOL_WARNED = False


def _warn_if_wait_pool_backed_up() -> None:
    """Warn once if engine_wait_executor has run out of threads.

    Checked at submit time, so this costs one call per request rather than one
    per token. A backlog means queued waits are being served FIFO while requests
    parked in the engine's waiting queue hold threads they cannot use, so
    actively decoding requests wait seconds for their next token.

    A non-empty queue alone is not saturation: ThreadPoolExecutor puts the work item
    on the queue before it spawns a worker for it, so under a burst of submits the
    queue is transiently non-empty while the pool is still growing. Require the pool
    to have actually reached its thread ceiling as well.
    """
    global _WAIT_POOL_WARNED
    if _WAIT_POOL_WARNED:
        return
    work_queue = getattr(engine_wait_executor, "_work_queue", None)
    threads = getattr(engine_wait_executor, "_threads", None)
    if work_queue is None or threads is None:  # private attributes, tolerate removal
        return
    backlog = work_queue.qsize()
    if backlog > 0 and len(threads) >= ENGINE_WAIT_THREADS:
        _WAIT_POOL_WARNED = True
        logger.warning(
            f"engine_wait_executor exhausted: {backlog} engine waits queued behind "
            f"{ENGINE_WAIT_THREADS} threads. Token delivery will stall in bursts. "
            f"Set KSANA_ENGINE_WAIT_THREADS above the peak concurrent request count."
        )


# The C++ scheduler admits requests through a fixed-size staging buffer
# (setting.batch_scheduler.max_waiting_queue_len, default 256) that it drains in full
# exactly once per schedule step, in BatchScheduler::BalanceWaitingReqs. A burst larger
# than that buffer is rejected with RET_EXCEED_CAPACITY even though the engine is
# nowhere near its real capacity -- those requests just happened to arrive between two
# drains. So treat that one code as backpressure and retry rather than failing the
# request. The wait lands in TTFT, which is honest: the request really is queued.
#
# Only RET_EXCEED_CAPACITY is retried, and only until a deadline, so genuine sustained
# overload still surfaces as an error instead of hanging. The retry count per request is
# normally tiny -- the next drain frees the whole buffer -- so a fixed short interval is
# enough and needs no backoff.
ADMIT_RETRY_TIMEOUT = float(os.environ.get("KSANA_ADMIT_RETRY_TIMEOUT", "60.0"))
ADMIT_RETRY_INTERVAL = float(os.environ.get("KSANA_ADMIT_RETRY_INTERVAL", "0.02"))

_ADMIT_RETRY_WARNED = False


async def _submit_with_backpressure(loop, executor, fn, *fn_args):
    """Hand a request to the C++ engine, waiting out a full admission buffer.

    Returns whatever fn returns -- a (status, payload) tuple -- so the caller's
    error handling is unchanged. On timeout the last failing status is returned as-is.
    """
    global _ADMIT_RETRY_WARNED
    deadline = time.time() + ADMIT_RETRY_TIMEOUT
    attempts = 0
    while True:
        result = await loop.run_in_executor(executor, fn, *fn_args)
        status = result[0]
        if status.OK() or status.GetCode() != libtorch_serving.RetCode.RET_EXCEED_CAPACITY:
            if attempts:
                logger.debug(
                    f"request admitted after {attempts} retr{'y' if attempts == 1 else 'ies'} "
                    f"waiting for engine admission buffer space"
                )
            return result
        if time.time() >= deadline:
            logger.warning(
                f"engine admission buffer still full after {ADMIT_RETRY_TIMEOUT}s "
                f"({attempts} retries); failing request. Raise "
                f"setting.batch_scheduler.max_waiting_queue_len if this is steady state "
                f"rather than a burst."
            )
            return result
        if not _ADMIT_RETRY_WARNED:
            _ADMIT_RETRY_WARNED = True
            logger.info(
                "engine admission buffer full, applying backpressure (retrying every "
                f"{ADMIT_RETRY_INTERVAL * 1000:.0f}ms). Expected on a large arrival burst; "
                "if it persists, raise setting.batch_scheduler.max_waiting_queue_len."
            )
        attempts += 1
        await asyncio.sleep(ADMIT_RETRY_INTERVAL)


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
        access_log_enabled: bool = True,
    ):
        self._serving_iterator = serving_iterator
        self._ksana_plugin = ksana_plugin
        self._ksana_python_input = ksana_python_input
        self._start_time = start_time or time.time()
        self._input_tokens = input_tokens
        self._headers_info = headers_info
        self._output_tokens = 0
        self._access_log_enabled = access_log_enabled
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
            engine_wait_executor, self._serving_iterator.GetNext
        )

        # Check the status of the iteration
        if status.OK():
            # If the iteration is successful, return the output
            self._ksana_plugin.postprocess(
                self._ksana_python_input, ksana_python_output
            )
            # 累加输出 token 数（每次迭代返回的是新增的 token）
            if hasattr(ksana_python_output, 'output_tokens') and ksana_python_output.output_tokens:
                # output_tokens is List[List[int]] (per-beam); use beam-0 token count.
                self._output_tokens = len(ksana_python_output.output_tokens[0])
            return ksana_python_output
        elif status.GetCode() == libtorch_serving.RetCode.RET_STOP_ITERATION:
            # If the iteration has finished, log completion and raise StopAsyncIteration
            elapsed_ms = (time.time() - self._start_time) * 1000
            if self._access_log_enabled:
                # Re-snapshot timestamps so we capture finish_time_us, which is stamped
                # only after the worker calls SyncStopRequest (after the last GetNext).
                final_snapshot = self._serving_iterator.SnapshotTimestamps()
                timing = KsanaLLMEngine._format_access_log_timing(final_snapshot, self._output_tokens)
                logger.info(
                    f"ACCESS| streaming=true input_tokens={self._input_tokens} "
                    f"output_tokens={self._output_tokens} status=OK elapsed={elapsed_ms:.2f}ms "
                    f"{timing} {self._headers_info}"
                )
            raise StopAsyncIteration(
                f"Iterator finished, ret code {status.GetCode()}, message {status.GetMessage()}."
            )
        else:
            # If an error occurred during iteration, log error and raise RuntimeError
            elapsed_ms = (time.time() - self._start_time) * 1000
            if self._access_log_enabled:
                final_snapshot = self._serving_iterator.SnapshotTimestamps()
                timing = KsanaLLMEngine._format_access_log_timing(final_snapshot, self._output_tokens)
                status_str = KsanaLLMEngine._format_access_log_status(status)
                logger.info(
                    f"ACCESS| streaming=true input_tokens={self._input_tokens} "
                    f"output_tokens={self._output_tokens} status={status_str} "
                    f"elapsed={elapsed_ms:.2f}ms {timing} {self._headers_info}"
                )
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
        self._access_log_enabled = endpoint_config.access_log

        self._access_log = endpoint_config.access_log

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

        # Ensure stop_token_ids is a clean list[int]; drop any None entries that
        # may come from an incomplete generation_config.json (e.g. some VL models).
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

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
            regex=_get_sampling_value(sampling_config, "regex", ""),
            grammar=_get_sampling_value(sampling_config, "grammar", ""),
            structural_tag=_get_sampling_value(sampling_config,
                                               "structural_tag", ""),
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
        sampling_config.regex = _get_generation_value(
            generation_config, "regex", ""
        )
        sampling_config.grammar = _get_generation_value(
            generation_config, "grammar", ""
        )
        sampling_config.structural_tag = _get_generation_value(
            generation_config, "structural_tag", ""
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

    def has_healthy_prefill_peer(self) -> bool:
        """Return True iff this Decode engine sees at least one healthy pd_v2
        Prefill peer.

        Returns False when this process is not a pd_v2 Decode node, or when the
        underlying binding is unavailable (older engine build).
        """
        getter = getattr(self._serving, "has_healthy_prefill_peer", None)
        if getter is None:
            return False
        return getter()

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

    @staticmethod
    def _format_access_log_status(status) -> str:
        """Render access-log status as OK or FAILED:<RetCode>."""
        if status is None:
            return "FAILED:-"
        if status.OK():
            return "OK"
        return f"FAILED:{int(status.GetCode())}"

    @staticmethod
    def _format_pd_v2_access_fields(output) -> str:
        """pd_v2 Decode-side phase breakdown for the access log.

        Absolute decode-clock timestamps (us) are kept for correlation/ordering; the segments
        are also emitted as ready-to-read ms deltas so no hand math is needed:
          admit_ms      arrive -> blocks allocated (decode admission / KV-pool wait)
          remote_rtt_ms prefill dispatched -> PrefillComplete (full remote round trip, decode clock)
          pf_queue_ms / pf_compute_ms / pf_send_ms  peer-Prefill's own-clock segments
          kv_transfer_ms  remote_rtt - (pf_queue+compute+send): network + KV-transfer residual
          decode_queue_ms first_token -> scheduled: the request already streamed its first
                          token (produced by the remote Prefill and delivered via PdV2DecodeHook's
                          NotifyStep) but still waits in the Decode node's decoding queue for a
                          running-batch slot (max_batch_size / logits budget / KV-block) before
                          token #2 is generated. This is the "stall after first token" gap and,
                          unlike waiting_ms, it is the part of the wait that happens AFTER TTFT.
                          Note: because in pd_v2 the first token comes from Prefill before local
                          decode is scheduled, scheduled >= first_token, so this is >= 0 while a
                          negative value (-1.00) means the request never entered the running batch.
        Durations subtract fine across nodes (same tick rate); only absolute us are non-comparable.
        For local prefill (prefill_dispatch_us=0) the remote-only fields render -1.00.
        """
        arrive = getattr(output, "arrive_time_us", 0)
        blocks_alloc = getattr(output, "pd_v2_blocks_allocated_time_us", 0)
        dispatch = getattr(output, "pd_v2_prefill_dispatched_time_us", 0)
        complete = getattr(output, "pd_v2_prefill_complete_time_us", 0)
        scheduled = getattr(output, "scheduled_time_us", 0)
        first_token = getattr(output, "first_token_time_us", 0)
        peer = getattr(output, "pd_v2_prefill_peer", "") or "-"
        pf_prefix = getattr(output, "pd_v2_pf_prefix_cache_hit_tokens", 0)
        pf_queue_ms = getattr(output, "pd_v2_pf_queue_us", 0) / 1000.0
        pf_compute_ms = getattr(output, "pd_v2_pf_compute_us", 0) / 1000.0
        pf_send_ms = getattr(output, "pd_v2_pf_send_us", 0) / 1000.0

        def _ms(end: int, start: int) -> float:
            return (end - start) / 1000.0 if end and start and end >= start else -1.0

        admit_ms = _ms(blocks_alloc, arrive)
        remote_rtt_ms = _ms(complete, dispatch)
        kv_transfer_ms = (remote_rtt_ms - pf_queue_ms - pf_compute_ms - pf_send_ms
                          if remote_rtt_ms >= 0 else -1.0)
        decode_queue_ms = _ms(scheduled, first_token)

        return (f"blocks_alloc_us={blocks_alloc} prefill_dispatch_us={dispatch} "
                f"prefill_complete_us={complete} prefill_peer={peer} "
                f"pf_prefix_cache_hit_tokens={pf_prefix} "
                f"admit_ms={admit_ms:.2f} remote_rtt_ms={remote_rtt_ms:.2f} "
                f"pf_queue_ms={pf_queue_ms:.2f} pf_compute_ms={pf_compute_ms:.2f} "
                f"pf_send_ms={pf_send_ms:.2f} kv_transfer_ms={kv_transfer_ms:.2f} "
                f"decode_queue_ms={decode_queue_ms:.2f}")

    @staticmethod
    def _format_access_log_timing(output, output_tokens: int) -> str:
        """Format C++ scheduler timestamps into waiting/ttft/tpot/total fields,
        plus draft hit rate and prefix cache hit tokens.

        Always emits the same field layout; missing data renders as 0 (timestamps /
        counts) or -1.00 (deltas / hit rate) so log parsers see a fixed schema across
        OK / failed / early-failure paths.
        """
        if output is None or not hasattr(output, "arrive_time_us"):
            return ("arrive_us=0 waiting_ms=-1.00 "
                    "ttft_ms=-1.00 tpot_ms=-1.00 total_ms=-1.00 "
                    "draft_hit_rate=-1.00 draft_hit_tokens=0 draft_tokens=0 "
                    "prefix_cache_hit_tokens=0 "
                    "blocks_alloc_us=0 prefill_dispatch_us=0 prefill_complete_us=0 prefill_peer=- "
                    "pf_prefix_cache_hit_tokens=0 admit_ms=-1.00 remote_rtt_ms=-1.00 "
                    "pf_queue_ms=0.00 pf_compute_ms=0.00 pf_send_ms=0.00 kv_transfer_ms=-1.00 "
                    "decode_queue_ms=-1.00")
        arrive = output.arrive_time_us
        sched = output.scheduled_time_us
        ftt = output.first_token_time_us
        fin = output.finish_time_us

        # Human-readable request arrival wall-clock (arrive_us is epoch microseconds). The log
        # preamble shows completion time (the line is written when the request finishes); this
        # adds when it actually arrived, in the same MM-DD HH:MM:SS.mmm format.
        def _delta_ms(end: int, start: int) -> float:
            return (end - start) / 1000.0 if end and start and end >= start else -1.0

        waiting_ms = _delta_ms(sched, arrive)
        ttft_ms = _delta_ms(ftt, arrive)
        total_ms = _delta_ms(fin, arrive)
        # TPOT excludes the first token from the denominator.
        if fin and ftt and fin >= ftt and output_tokens > 1:
            tpot_ms = (fin - ftt) / 1000.0 / (output_tokens - 1)
        else:
            tpot_ms = -1.0

        # Draft acceptance hit rate; -1.00 when no draft tokens were proposed.
        draft_hit_tokens = getattr(output, "draft_hit_tokens", 0)
        draft_tokens = getattr(output, "draft_tokens", 0)
        draft_hit_rate = draft_hit_tokens / draft_tokens if draft_tokens > 0 else -1.0
        prefix_cache_hit_tokens = getattr(output, "prefix_cache_hit_tokens", 0)

        return (f"arrive_us={arrive} waiting_ms={waiting_ms:.2f} "
                f"ttft_ms={ttft_ms:.2f} tpot_ms={tpot_ms:.2f} total_ms={total_ms:.2f} "
                f"draft_hit_rate={draft_hit_rate:.2f} draft_hit_tokens={draft_hit_tokens} "
                f"draft_tokens={draft_tokens} "
                f"prefix_cache_hit_tokens={prefix_cache_hit_tokens} "
                f"{KsanaLLMEngine._format_pd_v2_access_fields(output)}")

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
            _warn_if_wait_pool_backed_up()
            status, streaming_iterator = await _submit_with_backpressure(
                loop,
                submit_executor,
                self._call_serving_generate_streaming,
                ksana_python_input,
                req_ctx
            )

            if not status.OK():
                elapsed_ms = (time.time() - start_time) * 1000
                if self._access_log_enabled:
                    status_str = self._format_access_log_status(status)
                    timing = self._format_access_log_timing(None, 0)
                    logger.info(
                        f"ACCESS| streaming=true input_tokens={input_tokens} "
                        f"output_tokens=0 status={status_str} "
                        f"elapsed={elapsed_ms:.2f}ms {timing} {headers_info}"
                    )
                return status, None, plugin_timing_metrics

            # streaming 模式返回迭代器，完成日志在迭代器结束时记录
            return status, PyAsyncStreamingIterator(
                streaming_iterator, self._ksana_plugin, ksana_python_input,
                start_time=start_time, input_tokens=input_tokens, headers_info=headers_info,
                plugin_timing_metrics=plugin_timing_metrics,
                access_log_enabled=self._access_log_enabled,
            ), plugin_timing_metrics

        _warn_if_wait_pool_backed_up()
        status, ksana_python_output = await _submit_with_backpressure(
            loop,
            engine_wait_executor,
            self._call_serving_generate,
            ksana_python_input,
            req_ctx
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if not status.OK():
            if self._access_log_enabled:
                status_str = self._format_access_log_status(status)
                timing = self._format_access_log_timing(ksana_python_output, 0)
                logger.info(
                    f"ACCESS| streaming=false input_tokens={input_tokens} "
                    f"output_tokens=0 status={status_str} elapsed={elapsed_ms:.2f}ms "
                    f"{timing} {headers_info}"
                )
            return status, None, plugin_timing_metrics

        self._ksana_plugin.postprocess(ksana_python_input, ksana_python_output)

        # 获取输出 token 数量（output_tokens 是 List[List[int]]，取 beam-0 长度）
        output_tokens = (
            len(ksana_python_output.output_tokens[0])
            if hasattr(ksana_python_output, 'output_tokens') and ksana_python_output.output_tokens
            else 0
        )

        if self._access_log_enabled:
            timing = self._format_access_log_timing(ksana_python_output, output_tokens)
            logger.info(
                f"ACCESS| streaming=false input_tokens={input_tokens} "
                f"output_tokens={output_tokens} status=OK elapsed={elapsed_ms:.2f}ms "
                f"{timing} {headers_info}"
            )

        return status, ksana_python_output, plugin_timing_metrics

    async def forward(
        self, request_bytes: bytes, req_ctx: Optional[Dict[str, str]] = None
    ) -> Tuple[Any, Optional[bytes]]:
        """The model forward interface."""
        if req_ctx is None:
            req_ctx = {}
        loop = asyncio.get_running_loop()
        status, response_bytes = await loop.run_in_executor(
            engine_wait_executor,
            self._call_serving_forward,
            request_bytes,
            req_ctx
        )
        if status.OK():
            return status, response_bytes
        else:
            return status, None
