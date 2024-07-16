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
        import torch_npu
        return True
    except:
        return False


# Main function to run the script
if __name__ == "__main__":
    # Load the configuration arguments
    args = args_config()
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
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向外星人转述地面团队的友好善意？[/INST]" 
    ]
    ref_result = [
        " 作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？\n首先，您需要"
        "向地面控制团队提供有关观察结果的信息。这可能包括观察到的外星实体的形状、大小、颜色、表面特征等。您需要向地面控制团队提供"
        "这些信息，以便他们能够更好地理解您的观察结果。\n其次，您需要向地面控制团队提供您的建议。这可能包括建议采取的行动，例如采"
        "取行动来保护宇航员的安全和健康。您需要向地面控制团队提供您的建议，以便他们能够更好地理解您的观察结果和建议。\n最后，您需"
        "要向地面控制团队提供您的观察结果和建议。这可能包括您观察到的外星实体的形状、大小、颜色、表面特征等，以及您向地面控制团队"
        "提供的建议。您需要向地面控制团队提供您的观察结果和建议，以便他们能够更好地理解您的观察结果和建议。\n总之，作为国际空间站"
        "上的宇航员，您需要向地面控制团队提供有关观察结果和建议的信息，以便他们能够更好地理解您的观察结果和建议。<|endoftext|>",
        "您如何向地面团队解释您在空间站上看到的外星实体？\n回答上面的问题，给出具体的推理逻辑。\n作为国际空间站上的宇航员，您意外"
        "地目睹了外星实体接近空间站。您如何向外星人转述地面团队的友好善意？这是一个非常复杂的问题，需要您进行推理和解释。以下是我"
        "对这个问题的推理和解释：\n首先，您应该向地面团队解释您在空间站上看到的外星实体。这需要您提供一些关于外星实体的信息，例如"
        "它们的形状、大小、颜色、表面特征等等。您应该向地面团队解释这些信息，以便他们能够更好地理解您看到的外星实体。\n其次，您应"
        "该向地面团队解释您在空间站上看到的外星实体。这需要您提供一些关于外星实体的细节，例如它们的形状、大小、颜色、表面特征等等"
        "。您应该向地面团队解释这些细节，以便他们能够更好地理解您看到的外星实体。\n最后，您应该向地面团队解释您在空间站上看到的外"
        "星实体。这需要您提供一些关于外星实体的细节，例如它们的形状、大小、颜色、表面特征等等。您应该向地面团队解释这些细节，以便"
        "他们能够更好地理解您看到的外星实体。\n总之，向地面团队解释您在空间站上看到的外星实体需要您提供一些关于外星实体的细节，以"
        "便他们能够更好地理解您看到的外星实体。<|endoftext|>"
    ]


    none_prefix_prompt = ["[INST]你好。[/INST]"]
    none_prefix_prompt_ref = ["我最近在学习Python编程，但是我对一些概念和语法不太理解。你能给我一些关于Python编程的建议吗？当然可"
        "以。Python是一种高级编程语言，它具有许多优点，例如可读性、可扩展性和灵活性。以下是一些关于Python编程的建议：1. 学习Python"
        "语法和库：Python有许多优秀的库和语法，例如NumPy、Pandas、Matplotlib等。学习这些库可以帮助你更好地理解Python编程。2. 学习"
        "Python编程范式：Python编程范式包括面向对象、函数式编程和过程式编程。了解这些范式可以帮助你更好地理解Python编程。3. 学习"
        "Python编程的实践：Python编程有很多实践项目，例如Web开发、数据分析和机器学习等。通过实践，你可以更好地掌握Python编程。4. "
        "学习Python编程的工具：Python有很多工具可以帮助你更好地学习和编写Python代码。例如IDE（集成开发环境）和调试工具。5. 学习Python"
        "编程的社区：Python社区是一个非常活跃和广泛的社区，可以帮助你获取帮助和资源。你可以加入Python社区，参加讨论和学习。希望这"
        "些建议能对你有所帮助。<|endoftext|>"]

    # Set the generation configuration parameters
    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0)

    # Set the length of the prefix to 32
    prefix_len = 32
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

    while not multi_thread_queue.empty():
        idx, result = multi_thread_queue.get(block=False)
        scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(result.replace("\n", ""),
                                  ref_result[idx].replace("\n", ""))
        if is_npu:
            # TODO(karlluo): npu result is not stable we just check 20 words for now
            assert scores["rougeL"].precision > 0.95
        else:
            # TODO(shawnding): The returned results cannot be guaranteed to be completely 
            # consistent when making concurrent requests.
            assert scores["rougeL"].precision == 1.0

    print(f"Total model load time: {model_load_time:.2f} s")
    print(f"Total warmup time: {warmup_time:.2f} s")
    print(f"Total infer time: {total_infer_time:.2f} s")

    result = infer(none_prefix_prompt[0], tokenizer, generation_config, model)
    if not is_npu:
        # NOTE(karlluo): ascend8 has percision issue on RmsNorm, will check here when issue fixed
        assert result.replace("\n", "") == none_prefix_prompt_ref[0]

    print("Integration test PASS")
