# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import os
import sys
import torch
import importlib.util

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



class KsanaPlugin(object):
    """
    This class is designed to dynamically load and manage plugins for the Ksana framework.
    It allows for the initialization and post-processing of plugins specified by a file path.
    """

    def __init__(self, plugin_path: str):
        """
        Initializes the KsanaPlugin instance by loading a plugin from the given path.

        :param plugin_path: The file system path to the plugin module to be loaded.
        """

        self._ksana_plugin = self.load_plugin(plugin_path)
        if hasattr(self._ksana_plugin, 'init_plugin'):
            kwargs = {
                "postprocess" : True,
            }
            self._ksana_plugin.init_plugin(**kwargs)

    def load_plugin(self, plugin_path : str):
        """
        Dynamically loads the plugin module located at the specified path.

        :param plugin_path: The file system path to the plugin module to be loaded.
        :return: An instance of the loaded plugin class if successful, None otherwise.
        """

        try:
            plugin_path = plugin_path + "/ksana_plugin.py"
            spec = importlib.util.spec_from_file_location("ksana_plugin", plugin_path)
            if spec is None:
                print("spec is None")
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            class_name = "KsanaPlugin"
            if hasattr(module, class_name):
                KsanaPlugin = getattr(module, class_name)
                return KsanaPlugin()
            else:
                return None
        except Exception as e:
            print(f"Error loading plugin: {e}")
            return None


    def postprocess(self,
                    ksana_python_input : libtorch_serving.KsanaPythonInput,
                    ksana_python_output : libtorch_serving.KsanaPythonOutput):
        """
        Invokes the postprocess method of the loaded plugin, if available.
        """

        kwargs = {
            "ksana_python_input" : ksana_python_input,
            "ksana_python_output" : ksana_python_output,
            }
        if self._ksana_plugin is None:
            return ksana_python_output
        return self._ksana_plugin.postprocess(**kwargs)

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
        self._serving_cls = libtorch_serving.Serving

        # The serving instance.
        self._serving = self._serving_cls()
        self._serving.init_serving(config_file)
        self._ksana_plugin = KsanaPlugin(self._serving.plugin_path)

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

        ksana_python_input = libtorch_serving.KsanaPythonInput()
        ksana_python_input.model_name = model_name
        ksana_python_input.input_tokens = inputs
        sampling_config = ksana_python_input.sampling_config

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
        sampling_config.ignore_eos = \
            get_generation_value(generation_config, 'ignore_eos', False)


        if 'subinput_pos' in kwargs:
            ksana_python_input.subinput_pos = kwargs['subinput_pos']

        if 'subinput_embedding' in kwargs:
            ksana_python_input.subinput_embedding = kwargs['subinput_embedding']

        if 'subinput_url' in kwargs:
            ksana_python_input.subinput_url = kwargs['subinput_url']

        if 'prompt_probs_offset' in kwargs:
            ksana_python_input.prompt_probs_offset = kwargs['prompt_probs_offset']
            sampling_config.max_new_tokens = 1
            sampling_config.return_prompt_probs = True

        if streamer is None:
            _, ksana_python_output = self._serving.generate(ksana_python_input)
            self._ksana_plugin.postprocess(ksana_python_input, ksana_python_output)
            return ksana_python_output
        else:
            _, streaming_iterator = self._serving.generate_streaming(ksana_python_input)
            return PyAsyncStreamingIterator(streaming_iterator, self._ksana_plugin, ksana_python_input)
