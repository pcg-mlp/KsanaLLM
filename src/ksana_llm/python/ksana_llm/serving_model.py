# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import os
import sys
import torch

from typing import Callable, List, Optional, Union

from transformers.generation.streamers import BaseStreamer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

from transformers.generation.utils import GenerateOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

import libtorch_serving

import asyncio
from concurrent import futures

model_executor = futures.ThreadPoolExecutor(max_workers=256)


class PyAsyncStreamingIterator(object):
    """The streaming iterator.
    """

    def __init__(self, serving_iterator: libtorch_serving.StreamingIterator):
        self._serving_iterator = serving_iterator

    def __aiter__(self):
        return self

    # Define an asynchronous iterator method
    async def __anext__(self):
        # Get the current event loop
        loop = asyncio.get_event_loop()

        # Run the GetNext method of the serving iterator in an executor to avoid blocking the event loop
        status, token_id, logprobs = await loop.run_in_executor(
            model_executor,  # specify the executor to use
            self._serving_iterator.GetNext  # method to call
        )

        # Check the status of the iteration
        if status.OK():
            # If the iteration is successful, return the token ID
            return token_id, logprobs
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
        self._serving_cls = libtorch_serving.Serving

        # The serving instance.
        self._serving = self._serving_cls()
        self._serving.init_serving(config_file)

    @torch.no_grad()
    def generate(
        self,
        model_name: str = None,
        inputs: list[int] = None,
        generation_config: GenerationConfig = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                    List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> list[List[int]]:
        """The model generate interface, invoked by venus.
        """
        sampling_config = libtorch_serving.SamplingConfig()

        def get_generation_value(generation_config: GenerationConfig, key: str, default_val):
            """Get value from generation_config, return default if key not exists.
            """
            value = getattr(generation_config, key, default_val)
            return default_val if value is None else value
        sampling_config.num_beams = \
            get_generation_value(generation_config, 'num_beams', 1)
        sampling_config.topk = \
            get_generation_value(generation_config, 'top_k', 1)
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
        sampling_config.length_penalty = \
            get_generation_value(generation_config, 'length_penalty', 1.0)
        sampling_config.stop_token_ids = \
            get_generation_value(generation_config, 'stop_token_ids', [])

        if sampling_config.num_beams > 1:
            raise RuntimeError(
                "Not supported for sampling_config.num_beams > 1. Current value {}.".format(
                    sampling_config.num_beams))

        subinput_pos = []
        if 'subinput_pos' in kwargs:
            subinput_pos = kwargs['subinput_pos']

        subinput_embedding = []
        if 'subinput_embedding' in kwargs:
            subinput_embedding = kwargs['subinput_embedding']

        if streamer is None:
            _, outputs, logprobs = self._serving.generate(model_name, inputs,
                                                          sampling_config,
                                                          subinput_pos, subinput_embedding)
            return outputs, logprobs
        else:
            _, streaming_iterator = self._serving.generate_streaming(
                model_name, inputs, sampling_config,
                subinput_pos, subinput_embedding)
            return PyAsyncStreamingIterator(streaming_iterator)
