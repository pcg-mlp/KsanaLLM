# tests/test_models.py

import os
import sys
import tempfile
import shutil
import logging
import msgpack
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from utils import modify_yaml_field

# Adjust the system path to import custom modules
# Note: Ensure that the path adjustment is correct relative to the new file
# location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serving_forward_client import python_tensor_to_numpy  # noqa: E402
import ksana_llm  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPT_AFFIX_DICT = {
    "llama": "[INST]%s[/INST]",
    "llama-3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "baichuan": "<reserved_106>%s<reserved_107>",
    "qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "vicuna": "A chat between a curious user and an assistant. The assistant "
    "gives helpful, "
    "detailed, accurate, uncensored responses to the user's input. USER: %s "
    "ASSISTANT:",
    "yi": "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "chatglm": "<|system|>\nYou are a large language model trained by "
    "Zhipu.AI. Follow the user's instructions carefully."
    " Respond using markdown.\n<|user|>\n%s\n<|assistant|>\n",
    "empty": "%s",
}


def run_test(
    model_dir,
    prompt_affix_type,
    max_ppl,
    test_num,
    tensor_para_size,
    block_host_memory_factor,
    reserved_device_memory_ratio,
    max_batch_size,
    enable_auto_prefix_cache,
    default_ksana_yaml_path,
    benchmark_inputs,
):
    """
    Execute the model test within a temporary directory.

    Args:
        model_dir (str): Directory of the model.
        prompt_affix_type (str): Affix to prepend to prompts.
        max_ppl (float): Maximum permitted perplexity.
        test_num (int): Number of tests to run.
        tensor_para_size (int): Tensor parallel size.
        block_host_memory_factor (float): Block host memory factor.
        reserved_device_memory_ratio (float): Reserved device memory ratio.
        max_batch_size (int): Maximum batch size.
        enable_auto_prefix_cache (bool): Enable auto prefix cache.
        default_ksana_yaml_path (str): Path to the default ksana YAML config.
        benchmark_inputs (List[str]): List of input prompts.
    """
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {temp_dir}")

    try:
        # Copy the default YAML to the temporary directory
        ksana_yaml_path = os.path.join(temp_dir, "ksana.yaml")
        shutil.copyfile(default_ksana_yaml_path, ksana_yaml_path)
        assert os.path.exists(ksana_yaml_path), "Failed to copy ksana.yaml"

        # Prepare inputs
        inputs = benchmark_inputs[:test_num]
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            use_fast=True,
        )

        # Modify YAML configuration
        yaml_modifications = {
            "setting.global.tensor_para_size": tensor_para_size,
            "setting.block_manager.block_host_memory_factor": block_host_memory_factor,
            "setting.block_manager.reserved_device_memory_ratio": reserved_device_memory_ratio,
            "setting.batch_scheduler.max_batch_size": max_batch_size,
            "setting.batch_scheduler.enable_auto_prefix_cache": enable_auto_prefix_cache,
            "model_spec.base_model.model_dir": model_dir,
            "setting.global.is_version_report": False,
            "setting.profiler.stat_interval_second": 0,
        }

        for field_path, value in yaml_modifications.items():
            modify_yaml_field(ksana_yaml_path, field_path, value)

        # Initialize the model
        model = ksana_llm.AutoModel.from_config(ksana_yaml_path)
        model.init_serving(ksana_llm.PluginConfig(), ksana_llm.EndpointConfig())
        logger.debug("Initialized ksana_llm model.")

        generate_results = []

        def generate_for_prompt(prompt):
            formatted_prompt = PROMPT_AFFIX_DICT[prompt_affix_type].replace(
                "%s", prompt
            )
            input_tokens = tokenizer.encode(formatted_prompt)
            generation_config = GenerationConfig()

            _, output = model.generate(
                model_name="",  # Specify the model name if needed
                inputs=input_tokens,
                generation_config=generation_config,
                streamer=None,
            )
            return input_tokens, output.output_tokens[0]

        # Generate outputs concurrently
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(generate_for_prompt, prompt): prompt
                for prompt in inputs
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    generate_results.append(result)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        f"Error generating for prompt {futures[future]}: {e}"
                    )

        # Update YAML with the max batch size from results
        max_generated_length = max(len(b) for _, b in generate_results)
        modify_yaml_field(
            ksana_yaml_path,
            "setting.batch_scheduler.max_batch_size",
            max_generated_length,
        )
        modify_yaml_field(
            ksana_yaml_path,
            "setting.batch_scheduler.enable_auto_prefix_cache",
            False,
        )

        # Re-initialize the model with updated YAML
        del model
        model = ksana_llm.AutoModel.from_config(ksana_yaml_path)
        model.init_serving(ksana_llm.PluginConfig(), ksana_llm.EndpointConfig())
        logger.debug(
            "Re-initialized ksana_llm model with updated configuration."
        )

        ppl_list = []
        for input_tokens, output_tokens in generate_results:
            if len(output_tokens) < 3:
                continue

            ppl_input_tokens = input_tokens + output_tokens
            data = {
                "requests": [
                    {
                        "input_tokens": ppl_input_tokens,
                        "request_target": [
                            {
                                "target_name": "logits",
                                "slice_pos": [
                                    [
                                        len(input_tokens),
                                        len(ppl_input_tokens) - 2,
                                    ]
                                ],
                                "token_reduce_mode": "GATHER_TOKEN_ID",
                            },
                        ],
                    },
                ]
            }

            packed_data = msgpack.packb(data)
            unpacked_result = msgpack.unpackb(model.forward(packed_data)[1])
            python_tensor = unpacked_result["responses"][0]["response"][0][
                "tensor"
            ]
            np_res = python_tensor_to_numpy(python_tensor)
            log_prob = -np.sum(np.log2(np_res)) / len(np_res)
            ppl = 2 ** log_prob
            ppl_list.append(ppl)

        if not ppl_list:
            logger.warning("No valid perplexity scores computed.")
            assert False, "Perplexity list is empty."

        ppl_average = sum(ppl_list) / len(ppl_list)
        logger.debug(f"Average Perplexity: {ppl_average}")

        assert (
            ppl_average < max_ppl
        ), f"Average PPL {ppl_average} exceeds maximum allowed {max_ppl}"

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        logger.debug(f"Deleted temporary directory: {temp_dir}")
