# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
from typing import Callable, List, Optional, Dict
from concurrent import futures
from dataclasses import dataclass

import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

import libtorch_serving
from .ksana_plugin import KsanaPlugin, PluginConfig

model_executor = futures.ThreadPoolExecutor(max_workers=256)


@dataclass
class EndpointConfig:
    endpoint: str = "python"  # endpoint type
    host: str = "0.0.0.0"  # endpoint host address
    port: int = 8080  # endpoint port
    access_log: bool = True  # whether to enable the endpoint access log


class PyAsyncStreamingIterator(object):
    """The streaming iterator.
    """

    def __init__(self, 
                 serving_iterator: libtorch_serving.StreamingIterator,
                 ksana_plugin : KsanaPlugin,
                 ksana_python_input : libtorch_serving.KsanaPythonInput):
        self._serving_iterator = serving_iterator
        self._ksana_plugin = ksana_plugin
        self._ksana_python_input = ksana_python_input

    def __aiter__(self):
        return self

    # Define an asynchronous iterator method
    async def __anext__(self):
        # Get the current event loop
        loop = asyncio.get_event_loop()

        # Run the GetNext method of the serving iterator in an executor to avoid blocking the event loop
        status, ksana_python_output = await loop.run_in_executor(
            model_executor,  # specify the executor to use
            self._serving_iterator.GetNext  # method to call
        )

        # Check the status of the iteration
        if status.OK():
            # If the iteration is successful, return the token ID
            self._ksana_plugin.postprocess(self._ksana_python_input, ksana_python_output)
            return ksana_python_output
        elif status.GetCode() == libtorch_serving.RetCode.RET_STOP_ITERATION:
            # If the iteration has finished, raise a StopAsyncIteration exception
            raise StopAsyncIteration(
                "Iterator finished, ret code {}, message {}.".format(
                    status.GetCode(), status.GetMessage()))
        else:
            # If an error occurred during iteration, raise a RuntimeError exception
            raise RuntimeError(
                "Iterator error, ret code {}, message {}.".format(
                    status.GetCode(), status.GetMessage()))


class ServingModel(object):
    """The LLM serving model instance.
    """

    def __init__(self, config_file: str):
        """Initialize a ServingModel instance.

        Args:
            config_file: The serving config file.
        """
        self._config_file = config_file
        self._serving_cls = libtorch_serving.Serving

        # The serving instance.
        self._serving = self._serving_cls()

    def init_serving(
        self,
        plugin_config: PluginConfig,
        endpoint_config: EndpointConfig,
    ):
        """Initialize the serving endpoint.

        Args:
            plugin_config: The plugin config.
            endpoint_config: The endpoint config.
        """
        # Initialize the plugin
        self._ksana_plugin = KsanaPlugin(plugin_config)

        if endpoint_config.endpoint != "python":
            self._serving.endpoint_config.type = libtorch_serving.EndpointType.RPC
            self._serving.endpoint_config.rpc_plugin_name = endpoint_config.endpoint
        self._serving.endpoint_config.host = endpoint_config.host
        self._serving.endpoint_config.port = endpoint_config.port
        self._serving.endpoint_config.access_log = endpoint_config.access_log

        # Then initialize the serving server
        self._serving.init_serving(self._config_file)

    @torch.no_grad()
    def generate(
        self,
        model_name: Optional[str] = None,
        inputs: Optional[List[int]] = None,
        generation_config: Optional[GenerationConfig] = None,
        messages: Optional[List[Dict]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                    List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        req_ctx: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[List[int]]:
        """The model generate interface, invoked by venus.
        """
        if req_ctx is None:
            req_ctx = {}

        ksana_python_input = libtorch_serving.KsanaPythonInput()
        ksana_python_input.model_name = model_name
        if inputs is not None:
            ksana_python_input.input_tokens = inputs
        sampling_config = ksana_python_input.sampling_config

        def get_generation_value(generation_config: GenerationConfig, key: str, default_val):
            """Get value from generation_config, return default if key not exists.
            """
            value = getattr(generation_config, key, default_val)
            return default_val if value is None else value
        
        sampling_config.num_beams = \
            get_generation_value(generation_config, 'num_beams', 1)
        sampling_config.topk = get_generation_value(generation_config, 'top_k', 1)
        sampling_config.topp = \
            get_generation_value(generation_config, 'top_p', 0.0)
        sampling_config.temperature = \
            get_generation_value(generation_config, 'temperature', 0.0)
        sampling_config.max_new_tokens = \
            get_generation_value(generation_config, 'max_new_tokens', -1)
        sampling_config.logprobs_num = \
            get_generation_value(generation_config, 'logprobs_num', 0)
        sampling_config.num_return_sequences = \
            get_generation_value(generation_config, 'num_return_sequences', 1)
        sampling_config.repetition_penalty = \
            get_generation_value(generation_config, 'repetition_penalty', 1.0)
        sampling_config.no_repeat_ngram_size = \
            get_generation_value(generation_config, 'no_repeat_ngram_size', 0)
        sampling_config.encoder_no_repeat_ngram_size = \
            get_generation_value(generation_config, 'encoder_no_repeat_ngram_size', 0)
        sampling_config.length_penalty = \
            get_generation_value(generation_config, 'length_penalty', 1.0)
        sampling_config.stop_token_ids = \
            get_generation_value(generation_config, 'stop_token_ids', [])
        sampling_config.ignore_eos = \
            get_generation_value(generation_config, 'ignore_eos', False)
        sampling_config.stop_strings = \
            get_generation_value(generation_config, 'stop_strings', [])
        
        self._check_do_sample_params(generation_config, sampling_config, get_generation_value)
        if 'input_refit_embedding' in kwargs and 'pos' in kwargs['input_refit_embedding']:
            ksana_python_input.input_refit_embedding.pos = kwargs['input_refit_embedding']['pos']
        if 'input_refit_embedding' in kwargs and 'embeddings' in kwargs['input_refit_embedding']:
            ksana_python_input.input_refit_embedding.embeddings = kwargs['input_refit_embedding']['embeddings']
        if 'structured_output_regex' in kwargs:
            ksana_python_input.structured_output_regex = kwargs['structured_output_regex']

        # `messages` is the OpenAI Chat Completion API that can contain visual input
        # `additonal_params` are model specific params packed in a dict, e.g., `max_pixels`, `fps` for imgs and videos
        plugin_kwargs = {
            "messages": messages,
            "additional_params": kwargs.get('additional_params', {}),
        }

        # If user does not provide the input embeddings, compute it by the preprocessing of ksana plugin
        if not ksana_python_input.input_refit_embedding.pos:
            self._ksana_plugin.preprocess(ksana_python_input, **plugin_kwargs)

        if streamer is None:
            status, ksana_python_output = self._serving.generate(ksana_python_input, req_ctx)
            if not status.OK():
                return status, None
            self._ksana_plugin.postprocess(ksana_python_input, ksana_python_output)
            return status, ksana_python_output
        else:
            status, streaming_iterator = self._serving.generate_streaming(ksana_python_input, req_ctx)
            if not status.OK():
                return status, None
            return status, PyAsyncStreamingIterator(streaming_iterator, self._ksana_plugin, ksana_python_input)

    @torch.no_grad()
    def forward(self, request_bytes: bytes,
                req_ctx: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        """The model forward interface.
        This function just forwards the raw request bytes to the serving in the C++ side.
        """
        if req_ctx is None:
            req_ctx = {}
        status, response_bytes = self._serving.forward(request_bytes, req_ctx)
        if status.OK():
            return status, response_bytes
        else:  # Failed to get the response
            return status, None

    def _check_do_sample_params(self, generation_config, sampling_config, get_generation_value):
        do_sample = True
        if get_generation_value(generation_config, 'do_sample', None) is False or (sampling_config.topk == 1):
            do_sample = False
        
        if sampling_config.topk == 1 and get_generation_value(generation_config, 'do_sample', None) is True:
            print(f"Generation parameter topk cannot be 1 when do_sample is explicitly set to True!")
        
        # if do_sample=False, then set topk, topp and temperature to default value
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L560
        if not do_sample:
            sampling_config.topk = 1
            sampling_config.topp = 1.0
            sampling_config.temperature = 1.0
