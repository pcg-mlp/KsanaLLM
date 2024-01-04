# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

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


class ServingModel(object):
    """The LLM serving model instance.
    """

    def __init__(self, model_dir: str):
        """Initialize a ServingModel instance.

        Args:
            model_dir: The model directory path.
        """
        self._serving_cls = libtorch_serving.Serving

        # The serving instance.
        self._serving = self._serving_cls()
        self._serving.init_serving(model_dir)

    @torch.no_grad()
    def generate(
        self,
        model_name: str = None,
        inputs: list[List[int]] = None,
        generation_configs: list[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> list[List[int]]:
        """The model generate interface, invoked by venus.
        """
        sampling_configs = []
        for generation_config in generation_configs:
            sampling_config = libtorch_serving.SamplingConfig()
            sampling_config.beam_width = generation_config.num_beams
            sampling_config.topk = generation_config.top_k
            sampling_config.topp = generation_config.top_p
            sampling_config.temperature = generation_config.temperature
            sampling_configs.append(sampling_config)

        status, outputs = self._serving.generate(
            model_name, inputs, sampling_configs)
        return outputs
