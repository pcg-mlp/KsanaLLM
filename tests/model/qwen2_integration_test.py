import sys
import argparse
import logging
import threading
import queue
import time

# Add paths for C++ lib python wrapper and python interface
sys.path.insert(0, '../../build/lib')
sys.path.insert(0, '../../src/ksana_llm/python')
sys.path.insert(0, '.')

import ksana_llm

# Set up logging configuration
logging.basicConfig()
# Set the logging level to DEBUG for the root logger
logging.getLogger().setLevel(logging.DEBUG)
# Create a StreamHandler for logging to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
# Define the format for log messages
formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
stream_handler.setFormatter(formatter)
# Add the StreamHandler to the root logger
logging.getLogger().addHandler(stream_handler)

from transformers import GenerationConfig, AutoTokenizer


# Define a function to parse command line arguments
def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="examples/ksana_llm.yaml",
                        help='serving config file')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="/model/qwen-hf/Qwen2-1.5B",
                        help='tokenizer dir')
    args = parser.parse_args()
    return args


# Define a function for inference
def infer(prompt, tokenizer, generation_config, model, queue=None, idx=0):
    # Encode the input prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt, add_special_tokens=True)

    # Set generation configuration parameters
    #if queue is not None and idx > 0:
    #time.sleep(1)
    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0,
                                         max_new_tokens=1024,
                                         logprobs_num=0,
                                         stop_token_ids=[])

    # Generate text using the model
    _, ksana_python_output = model.generate(model_name="",
                                         inputs=input_tokens,
                                         generation_config=generation_config,
                                         streamer=None)
    result = ksana_python_output.output_tokens[0]
    # Check if a queue is provided for storing results
    if queue is None:
        # If no queue is provided, return the decoded result
        return tokenizer.decode(result, skip_special_tokens=True)
    else:
        # If a queue is provided, put the result in the queue along with the
        # index
        queue.put((idx, tokenizer.decode(result, skip_special_tokens=True)))
        return


def is_run_on_npu_device() -> bool:
    try:
        # pylint: disable-next=unused-import
        import torch_npu
        return True
    except ImportError:
        return False


# Main function to run the script
if __name__ == "__main__":
    # Load the configuration arguments
    args = args_config()

    # Start loading the tokenizer
    logging.info("start load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                              trust_remote_code=True)
    # Finish loading the tokenizer
    logging.info("finish load tokenizer")

    # Start loading the model
    logging.info("start load model")
    start_time = time.perf_counter()
    model = ksana_llm.AutoModel.from_config(args.config_file)
    model.init_serving(ksana_llm.PluginConfig(), ksana_llm.EndpointConfig())
    end_time = time.perf_counter()
    model_load_time = end_time - start_time
    # Finish loading the model
    logging.info("finish load model")

    # Define warmup prompts and reference results
    warmup_prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n作为国际空间站上的宇航员，您意外地目睹了"
        "外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n作为国际空间站上的宇航员，您意外地目睹了"
        "外星实体接近空间站。您怎么吃饭？<|im_end|>\n<|im_start|>assistant\n"
    ]
    ref_result = [
        "1. 确保信息准确：首先，我需要确保我所观察到的外星实体是真实的，而不是幻觉。我需要确保我所观察到的实体是外星生物，而不"
        "是其他任何可能的物体。\n\n2. 保持冷静：在面对未知的外星实体时，保持冷静是非常重要的。我需要保持冷静，以便能够有效地传"
        "达我的观察结果和建议。\n\n3. 保持开放和尊重的态度：我需要保持开放和尊重的态度，以便能够与地面控制团队进行有效的沟通。"
        "我需要尊重他们的观点和决定，同时也需要尊重他们的工作。\n\n4. 提供信息：我需要提供我所观察到的外星实体的信息，以便他们"
        "能够理解我的观察结果。我需要提供我所观察到的外星实体的特征，以及我所观察到的外星实体的行为。\n\n5. 提供建议：我需要提"
        "供我所观察到的外星实体的建议，以便他们能够更好地理解我的观察结果。我需要提供我所观察到的外星实体的可能的解决方案，以及"
        "我所观察到的外星实体可能遇到的问题。\n\n6. 保持沟通：我需要保持与地面控制团队的沟通，以便他们能够及时了解我的观察结果"
        "和建议。我需要保持与地面控制团队的沟通，以便他们能够及时了解我的观察结果和建议。\n\n7. 保持专业：我需要保持专业，以便"
        "能够有效地传达我的观察结果和建议。我需要保持专业，以便能够有效地传达我的观察结果和建议。",
        "作为国际空间站上的宇航员，我无法直接在空间站上吃饭，因为这需要特殊的设备和环境。但是，我可以提供一些基本的建议。\n\n首"
        "先，你需要确保你的食物和水都是安全的。在太空中，食物和水可能会受到极端的温度和压力的影响，因此你需要确保它们是安全的。"
        "你可以在太空中携带一些食品和水，例如水和食物，但请注意，这些食品和水可能会受到极端的温度和压力的影响。\n\n其次，你需要"
        "确保你的食物和水都是安全的。在太空中，食物和水可能会受到极端的温度和压力的影响，因此你需要确保它们是安全的。你可以在太"
        "空中携带一些食品和水，例如水和食物，但请注意，这些食品和水可能会受到极端的温度和压力的影响。\n\n最后，你需要确保你的宇"
        "航服是安全的。在太空中，宇航服可能会受到极端的温度和压力的影响，因此你需要确保它们是安全的。你可以在太空中携带一些宇航"
        "服，例如宇航服，但请注意，这些宇航服可能会受到极端的温度和压力的影响。\n\n总的来说，你需要确保你的食物和水都是安全的，"
        "你的宇航服是安全的，你的宇航站是安全的。"
    ]

    none_prefix_prompt = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好。<|im_end|>\n<|im_start|>assistant\n"
    ]
    none_prefix_prompt_ref = ["你好！有什么我可以帮助你的吗？"]

    # Set the generation configuration parameters
    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0)

    # Set the length of the prefix to 32
    multi_thread_list = []  # List to store multiple threads
    multi_thread_queue = queue.Queue(
    )  # Queue for storing results from multiple threads

    # Log a message indicating the start of warmup
    logging.info("start warmup")
    start_time = time.perf_counter()  # Record the start time for warmup
    # Perform warmup by generating results for each warmup prompt
    for idx, prompt in enumerate(warmup_prompts):
        result = infer(prompt, tokenizer, generation_config, model)
        assert result.replace("\n", "") == ref_result[idx].replace("\n", "")

    end_time = time.perf_counter()
    warmup_time = end_time - start_time  # Calculate the warmup time
    logging.info("finish warmup")  # Log a message indicating the end of warmup

    # Log a message indicating the start of inference
    logging.info("start infer")
    start_time = time.perf_counter()  # Record the start time for inference

    for idx, prompt in enumerate(warmup_prompts):
        thread = threading.Thread(target=infer,
                                  args=(
                                      prompt,
                                      tokenizer,
                                      generation_config,
                                      model,
                                      multi_thread_queue,
                                      idx,
                                  ))
        thread.start()
        multi_thread_list.append(thread)

    # Wait for all threads to finish
    for thread in multi_thread_list:
        thread.join()
    end_time = time.perf_counter()
    total_infer_time = end_time - start_time  # Calculate the total inference time
    logging.info(
        "finish infer")  # Log a message indicating the end of inference
    # Ensure the total inference time is less than the warmup time
    assert total_infer_time < warmup_time

    while not multi_thread_queue.empty():
        idx, result = multi_thread_queue.get(block=False)
        # consistent when making concurrent requests.
        assert result.replace("\n", "") == ref_result[idx].replace("\n", "")

    print(f"Total model load time: {model_load_time:.2f} s")
    print(f"Total warmup time: {warmup_time:.2f} s")
    print(f"Total infer time: {total_infer_time:.2f} s")

    result = infer(none_prefix_prompt[0], tokenizer, generation_config, model)
    assert result.replace("\n", "") == none_prefix_prompt_ref[0]

    print("Integration test PASS")
