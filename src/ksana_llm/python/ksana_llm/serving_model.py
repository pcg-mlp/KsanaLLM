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


class PyStreamingIterator(object):
    """The streaming iterator.
    """

    def __init__(self, serving_iterator: libtorch_serving.StreamingIterator, loop):
        self._serving_iterator = serving_iterator

        asyncio.set_event_loop(loop)
        self._queue = asyncio.Queue()

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        status, token_id = self._serving_iterator.GetNext()
        if status.OK():
            return token_id
        elif status.GetCode() == libtorch_serving.RetCode.RET_STOP_ITERATION:
            raise StopIteration(
                "Iterator finished, ret code {}, message {}.".format(
                    status.GetCode(), status.GetMessage()))
        else:
            raise RuntimeError(
                "Iterator error, ret code {}, message {}.".format(
                    status.GetCode(), status.GetMessage()))

    async def __anext__(self):
        status, token_id = self._serving_iterator.GetNext()
        if status.OK():
            self._queue.put_nowait(token_id)
        elif status.GetCode() == libtorch_serving.RetCode.RET_STOP_ITERATION:
            self._queue.put_nowait(None)
        else:
            self._queue.put_nowait(
                RuntimeError("Iterator error, ret code {}, message {}.".format(
                    status.GetCode(), status.GetMessage())))

        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


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

        # For streaming mode.
        self._event_loop = asyncio.new_event_loop()

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
        sampling_config.beam_width = generation_config.num_beams
        sampling_config.topk = generation_config.top_k
        sampling_config.topp = generation_config.top_p
        sampling_config.temperature = generation_config.temperature

        if streamer is None:
            _, outputs = self._serving.generate(model_name, inputs,
                                                sampling_config)
            return outputs
        else:
            _, streaming_iterator = self._serving.generate_streaming(
                model_name, inputs, sampling_config)
            return PyStreamingIterator(streaming_iterator, self._event_loop)
