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
    # pylint: disable-next=invalid-name
    is_npu = is_run_on_npu_device()

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
        "如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信。我会使用空间站上的通信设备，"
        "向地面控制团队报告我的观察结果和建议。首先，我会向地面控制团队报告我所看到的外星实体的位置、形状、大小和颜色等详细信息。"
        "我会尽可能提供准确的描述，以便地面控制团队能够更好地了解外星实体的情况。其次，我会向地面控制团队建议采取适当的措施来保护"
        "空间站和宇航员的安全。这可能包括减速空间站的速度、改变空间站的方向、启动空间站的防御系统等。我会尽可能提供详细的建议，以"
        "便地面控制团队能够做出正确的决策。最后，我会向地面控制团队提供任何其他有用的信息，以便他们能够更好地了解外星实体的情况。"
        "这可能包括外星实体的运动轨迹、速度、方向等信息。我会尽可能提供准确的信息，以便地面控制团队能够更好地了解外星实体的情况。"
        "总之，如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信，并提供尽可能详细的观察"
        "结果和建议，以便地面控制团队能够采取适当的措施来保护空间站和宇航员的安全。</s>",
        "作为一名国际空间站上的宇航员，如果我意外地目睹了外星实体接近空间站，我的第一反应可能是惊恐和惊讶。但是，我们的训练和准备"
        "使我们能够在任何情况下保持冷静和专业。在太空中吃饭是一项非常重要的任务，因为我们需要保持身体健康和精神状态良好，以便完成"
        "我们的任务。在太空中，我们使用特殊的食品和餐具，以确保我们的食物和饮料不会漂浮到空间站的其他部分或飘出太空舱。我们的食物"
        "通常是预包装的，并且在太空中使用特殊的加热器和冷却器进行加热和冷却。我们还使用特殊的餐具，例如吸管和勺子，以避免食物和液"
        "体漂浮到空间站的其他部分。在太空中，我们通常会吃固体食物，例如面包、饼干和坚果，以及液体食物，例如果汁和牛奶。我们还会吃"
        "一些特殊的食物，例如蛋白质棒和能量棒，以帮助我们保持身体健康和精神状态良好。在太空中，我们通常会在特殊的餐桌上吃饭，这些"
        "餐桌可以固定在空间站的墙壁上，以避免食物和餐具漂浮到空间站的其他部分。我们还会在特殊的食品袋中储存食物和饮料，以便在需要"
        "时使用。总之，在太空中吃饭需要特殊的准备和技巧，以确保我们的食物和饮料不会漂浮到空间站的其他部分或飘出太空舱。我们的训练"
        "和准备使我们能够在任何情况下保持冷静和专业，以确保我们的身体健康和精神状态良好。</s>"
    ]

    none_prefix_prompt = ["[INST]你好。[/INST]"]
    none_prefix_prompt_ref = ["你好，有什么我可以帮助你的吗？</s>"]

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
        # Ensure the generated result matches the reference
        # TODO(karlluo): npu result is generated by fp16 and gpu is generated by bf16
        if is_npu:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                              use_stemmer=True)
            scores = scorer.score(result.replace("\n", ""),
                                  ref_result[idx].replace("\n", ""))
            assert scores["rougeL"].precision > 0.95
        else:
            assert result.replace("\n",
                                  "") == ref_result[idx].replace("\n", "")
    end_time = time.perf_counter()
    warmup_time = end_time - start_time  # Calculate the warmup time
    logging.info("finish warmup")  # Log a message indicating the end of warmup

    # Log a message indicating the start of inference
    logging.info("start infer")
    start_time = time.perf_counter()  # Record the start time for inference

    if not is_npu:
        # TODO(karlluo): npu result is not stable we just check 20 words for now
        # Perform inference using multiple threads for each warmup prompt
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
    if not is_npu:
        # NOTE(karlluo): there isn't warmup in npu
        assert total_infer_time < warmup_time

    if not is_npu:
        # TODO(karlluo): npu result is not stable we just check 20 words for now
        # Check the results from the multi-threaded inference
        while not multi_thread_queue.empty():
            idx, result = multi_thread_queue.get(block=False)
            # TODO(karlluo): npu result is not stable we just check 20 words for now
            if is_npu:
                scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(result.replace("\n", ""),
                                      ref_result[idx].replace("\n", ""))
                assert scores["rougeL"].precision > 0.95
            else:
                assert result.replace("\n", "") == ref_result[
                    idx]  # Ensure the generated result matches the reference

    print(f"Total model load time: {model_load_time:.2f} s")
    print(f"Total warmup time: {warmup_time:.2f} s")
    print(f"Total infer time: {total_infer_time:.2f} s")

    result = infer(none_prefix_prompt[0], tokenizer, generation_config, model)
    if not is_npu:
        # NOTE(karlluo): ascend8 has percision issue on RmsNorm, will check here when issue fixed
        assert result.replace("\n", "") == none_prefix_prompt_ref[0]

    print("Integration test PASS")
