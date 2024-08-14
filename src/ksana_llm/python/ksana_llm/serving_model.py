# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import importlib.util
from typing import Callable, List, Optional, Dict, Any
import asyncio
from concurrent import futures

import torch
from transformers.generation.configuration_utils import GenerationConfig

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

import libtorch_serving

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
                "model_path" : plugin_path,
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
                ksana_plugin = getattr(module, class_name)
                return ksana_plugin()
            else:
                return None
        # pylint: disable-next=broad-except
        except Exception:
            # TODO: need a better log
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

    def validate_slice_pair(self,
                            slice_pair: tuple,
                            previous_end: int,
                            input_tokens_num: int) -> bool:
        """
        Validates a slice pair to ensure it represents a valid interval within a sequence of input tokens.

        Parameters:
        - slice_pair (tuple): A tuple containing two integers where the first element is the start position,
                            and the second element is the end position of the interval.
        - previous_end (int): The end position of the previous interval to check for overlaps.
        - input_tokens_num (int): The total number of input tokens to ensure the interval does not exceed this number.

        Returns:
        - bool: True if the slice_pair represents a valid interval, otherwise raises a ValueError.
        """

        # Ensure slice_pair contains exactly two elements
        if len(slice_pair) != 2:
            raise ValueError(f"Error: {slice_pair} does not represent a valid interval "
                             "(expected a tuple of two elements).")

        # Check if the end position is greater than or equal to the start position
        if slice_pair[1] < slice_pair[0]:
            raise ValueError(
                f"Error: The end position of interval {slice_pair} is less than its start position.")

        # Validate that the end position does not exceed the number of input tokens
        if slice_pair[1] >= input_tokens_num:
            raise ValueError(
                f"Error: The end position of interval {slice_pair} exceeds the total number of input tokens "
                f"({input_tokens_num}).")

        # Check for overlap with the previous interval
        if slice_pair[0] <= previous_end:
            raise ValueError(
                f"Error: Interval {slice_pair} overlaps with the previous interval ending at position {previous_end}.")

        return True

    def parse_request_target(self,
                            request_target: List[Dict[str, Any]],
                            ksana_python_input: libtorch_serving.KsanaPythonInput):
        """
        Parses the request target and updates the ksana_python_input object with the processed information.

        This function iterates through each target specified in the request, processes its description,
        and updates the ksana_python_input object with the processed target information.

        Parameters:
        - request_target (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a target
                                                 with its description.
        - ksana_python_input (libtorch_serving.KsanaPythonInput): An object to be updated with the processed
                                                                  target information.

        Raises:
        - RuntimeError: If both 'token_id' and 'slice_pos' are specified for the same target, or if 'GATHER_TOKEN_ID'
                        is specified for a transformer or layernorm target, which is not supported.
        """

        # Iterate through each target specified in the request
        for target_desc in request_target:
            # Ensure 'target_name' is specified
            if 'target_name' not in target_desc:
                raise RuntimeError("Missing 'target_name' in target description.")
            target_name = target_desc["target_name"]
            target_describe = libtorch_serving.TargetDescribe()

            # If 'token_id' is specified, set it in the target description
            if 'token_id' in target_desc:
                target_describe.token_id = target_desc['token_id']

            # If 'slice_pos' is specified, validate and set it in the target description
            if 'slice_pos' in target_desc:
                previous_end = -1
                input_tokens_num = len(ksana_python_input.input_tokens)
                for slice_pair in target_desc['slice_pos']:
                    if self.validate_slice_pair(slice_pair, previous_end, input_tokens_num):
                        target_describe.slice_pos.append((slice_pair[0], slice_pair[1]))
                        previous_end = slice_pair[1]

            # Ensure that 'token_id' and 'slice_pos' are not both set for the same target
            if len(target_describe.token_id) > 0 and len(target_describe.slice_pos) > 0:
                raise RuntimeError("Unable to set both token_id and slice_pos at the same time")

            # Set the token reduce mode based on the specified mode in the target description
            if 'token_reduce_mode' in target_desc:
                if target_desc['token_reduce_mode'] == 'GATHER_ALL':
                    target_describe.token_reduce_mode = libtorch_serving.TokenReduceMode.GATHER_ALL
                elif target_desc['token_reduce_mode'] == 'GATHER_TOKEN_ID':
                    # Ensure 'GATHER_TOKEN_ID' is not used with transformer, layernorm targets
                    if target_name == "transformer" or target_name == "layernorm":
                        raise RuntimeError(
                            "The output of the {target_name} does not support GATHER_TOKEN_ID")
                    target_describe.token_reduce_mode = libtorch_serving.TokenReduceMode.GATHER_TOKEN_ID
            # TODO(zakwang): Enhance support for additional request parameters
            if target_name == "logits":
                # Verify if the 'GATHER_ALL' token reduction mode is not used, as it's unsupported for logits output
                if target_desc['token_reduce_mode'] == 'GATHER_ALL':
                    raise RuntimeError(
                        f"The output for {target_name} does not support the 'GATHER_ALL' reduction mode.")
                # Verify that no token IDs are specified, as they are not supported for logits output.
                if len(target_describe.token_id) > 0:
                    raise RuntimeError(
                        f"Specifying token_id for {target_name} output is not supported.")
                # Ensure the 'GATHER_TOKEN_ID' reduction mode does not use the last logits, as it might be unsupported
                # or not intended.
                if target_desc['token_reduce_mode'] == 'GATHER_TOKEN_ID' and \
                   target_desc['slice_pos'][-1][-1] == len(ksana_python_input.input_tokens) - 1:
                    raise RuntimeError(
                        f"The last logits is not supported for {target_name} in the 'GATHER_TOKEN_ID' "
                        "token reduction mode.")

            # Update the ksana_python_input object with the processed target information
            ksana_python_input.request_target[target_name] = target_describe

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
        sampling_config.length_penalty = \
            get_generation_value(generation_config, 'length_penalty', 1.0)
        sampling_config.stop_token_ids = \
            get_generation_value(generation_config, 'stop_token_ids', [])
        sampling_config.ignore_eos = \
            get_generation_value(generation_config, 'ignore_eos', False)
        
        self._check_do_sample_params(generation_config, sampling_config, get_generation_value)

        if 'input_refit_embedding' in kwargs and 'pos' in kwargs['input_refit_embedding']:
            ksana_python_input.input_refit_embedding.pos = kwargs['input_refit_embedding']['pos']

        if 'input_refit_embedding' in kwargs and 'embeddings' in kwargs['input_refit_embedding']:
            ksana_python_input.input_refit_embedding.embeddings = kwargs['input_refit_embedding']['embeddings']

        if 'request_target' in kwargs:
            sampling_config.max_new_tokens = 1
            streamer = None
            self.parse_request_target(kwargs["request_target"], ksana_python_input)

        if streamer is None:
            _, ksana_python_output = self._serving.generate(ksana_python_input)
            self._ksana_plugin.postprocess(ksana_python_input, ksana_python_output)
            return ksana_python_output
        else:
            _, streaming_iterator = self._serving.generate_streaming(ksana_python_input)
            return PyAsyncStreamingIterator(streaming_iterator, self._ksana_plugin, ksana_python_input)

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
