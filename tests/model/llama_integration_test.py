import sys
import argparse
import logging
import threading
import queue
import time
import typing
import pkgutil

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
from rouge_score import rouge_scorer


# Define a function to parse command line arguments
def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="examples/ksana_llm.yaml",
                        help='serving config file')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="/model/llama-hf/13B",
                        help='tokenizer dir')
    args = parser.parse_args()
    return args


# Define a function for inference
def infer(prompt, tokenizer, generation_config, model, queue=None, idx=0):
    # Encode the input prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt, add_special_tokens=True)

    # Set generation configuration parameters
    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0,
                                         max_new_tokens=1024,
                                         logprobs_num=0,
                                         stop_token_ids=[])

    # Generate text using the model
    ksana_python_output = model.generate(model_name="",
                                         inputs=input_tokens,
                                         generation_config=generation_config,
                                         streamer=None)
    result = ksana_python_output.output_tokens[0]
    # Check if a queue is provided for storing results
    if queue is None:
        # If no queue is provided, return the decoded result
        return tokenizer.decode(result)
    else:
        # If a queue is provided, put the result in the queue along with the
        # index
        queue.put((idx, tokenizer.decode(result)))
        return


def infer_test(prompts: typing.Optional[list] = None,
               ref_results: typing.Optional[list] = None,
               is_async: typing.Optional[bool] = False,
               results: typing.Optional[queue.Queue] = None) -> float:
    results = list()
    multi_thread_list = list()
    multi_thread_queue = queue.Queue(
    )  # Queue for storing results from multiple threads
    start_time = time.perf_counter()  # Record the start time for infer
    # Perform infer by generating results for each infer prompt
    for idx, prompt in enumerate(prompts):
        if is_async:
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
        else:
            results.append(infer(prompt, tokenizer, generation_config, model))
    if is_async:
        # Wait for all threads to finish
        for thread in multi_thread_list:
            thread.join()
    end_time = time.perf_counter()

    if is_async:
        results = ["" for i in range(multi_thread_queue.qsize())]
        while not multi_thread_queue.empty():
            idx, result = multi_thread_queue.get(block=False)
            results[idx] = result

    # Check results correctness
    for idx, result in enumerate(results):
        # Ensure the generated result matches the reference
        format_output = result.replace("\n", "")
        assert format_output == ref_results[
            idx], f"\ninfer output: {format_output} \nreal output: {ref_results[idx]}"

    warmup_time = end_time - start_time  # Calculate the runing time
    return warmup_time


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
    end_time = time.perf_counter()
    model_load_time = end_time - start_time
    # Finish loading the model
    logging.info("finish load model")

    # Define warmup prompts and reference results
    warmup_prompts = [
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？[/INST]",
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何在太空吃饭？[/INST]"
    ]

    ref_result = [
        '如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会'
        '立即向地面控制团队发出紧急通信。我会使用空间站上的通信设备，'
        '向地面控制团队报告我的观察结果和建议。首先，我会向地面控制团'
        '队报告我所看到的外星实体的位置、形状、大小和颜色等详细信息。'
        '我会尽可能提供准确的描述，以便地面控制团队能够更好地了解外星'
        '实体的情况。其次，我会向地面控制团队报告外星实体的行动轨迹和'
        '速度。我会尽可能提供详细的信息，以便地面控制团队能够更好地了'
        '解外星实体的行动方式。最后，我会向地面控制团队提出建议。我会'
        '建议地面控制团队采取适当的措施，以确保空间站和宇航员的安全。'
        '我会建议地面控制团队立即启动应急计划，并与国际空间站上的其他'
        '宇航员和地面控制团队成员保持联系。总之，如果我是国际空间站上'
        '的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发'
        '出紧急通信，并提供尽可能详细的观察结果和建议，以确保空间站和'
        '宇航员的安全。</s>', '作为一名国际空间站上的宇航员，如果我意外地目睹了外星实体接近'
        '空间站，我的第一反应可能是惊恐和惊讶。但是，我们的训练和准备'
        '使我们能够在任何情况下保持冷静和专业。在太空中吃饭是一项非常'
        '重要的任务，因为我们需要保持身体健康和精神状态良好，以便完成'
        '我们的任务。在太空中，我们使用特殊的食品和餐具，以确保我们的'
        '食物和饮料不会漂浮到空间站的其他部分或飘散到太空中。我们的食'
        '物通常是预包装的，并且在食品包装上有一个小孔，可以让我们用吸'
        '管或吸管吸食。这样可以避免食物漂浮到空间站的其他部分或飘散到'
        '太空中。我们还使用特殊的餐具，例如勺子和叉子，这些餐具都有特'
        '殊的设计，以确保它们不会漂浮到空间站的其他部分或飘散到太空中'
        '。我们的食物通常是干燥或罐装的，以便在太空中保持新鲜和营养。'
        '我们还有一些特殊的食品，例如冻干食品和软膏，这些食品可以在太'
        '空中保持新鲜和营养。在太空中，我们的饮料通常是封闭的，以避免'
        '液体漂浮到空间站的其他部分或飘散到太空中。我们使用特殊的吸管'
        '或吸管来喝水或其他液体。总之，在太空中吃饭是一项非常重要的任'
        '务，我们需要使用特殊的食品和餐具，以确保我们的食物和饮料不会'
        '漂浮到空间站的其他部分或飘散到太空中。我们的食物通常是干燥或'
        '罐装的，以便在太空中保持新鲜和营养。我们还需要保持冷静和专业'
        '，以便在任何情况下都能够完成我们的任务。</s>'
    ]

    none_prefix_prompt = ["[INST]你好。[/INST]"]
    none_prefix_prompt_ref = ["你好，有什么我可以帮助你的吗？</s>"]

    # Set the generation configuration parameters
    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0)
    # Log a message indicating the start of warmup
    logging.info("start warmup")
    warmup_time = infer_test(warmup_prompts, ref_result, False)
    logging.info("finish warmup")  # Log a message indicating the end of warmup

    # Log a message indicating the start of inference
    logging.info("start async infer")
    total_infer_time = infer_test(warmup_prompts, ref_result, True)
    total_infer_time = 0
    # Log a message indicating the end of inference
    logging.info("finish infer")

    print(f"Total model load time: {model_load_time:.2f} s")
    print(f"Total warmup time: {warmup_time:.2f} s")
    print(f"Total infer time: {total_infer_time:.2f} s")

    result = infer(none_prefix_prompt[0], tokenizer, generation_config, model)
    assert result == none_prefix_prompt_ref[0]

    print("Integration test PASS")
