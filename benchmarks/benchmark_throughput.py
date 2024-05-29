# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import argparse
import json
import time
import argparse
import asyncio
import json
import time
import aiohttp
import numpy as np
import csv
import requests

from typing import AsyncGenerator, List, Tuple
from tqdm.asyncio import tqdm

import prefix_cache_reader

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, int, int, float]] = []
PROMPT_AFFIX_DICT = {
    "llama":
    "[INST]%s[/INST]",
    "llama-3":
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "baichuan":
    "<reserved_106>%s<reserved_107>",
    "qwen":
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "empty":
    "%s",
}
DEFAULT_STOP_TOKEN_IDS = {
    "llama-3": [128009],
    "qwen": [151643, 151645],
}

def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default="0.0.0.0",
                        help='server host address')
    parser.add_argument('--port', type=int, default=8888, help='server port')
    parser.add_argument('--input_csv',
                        type=str,
                        default="benchmark_input.csv",
                        help='input data for benchmark')
    parser.add_argument('--col_idx',
                        type=int,
                        default=0,
                        help='col_idx to be read from the input csv')
    parser.add_argument('--output_csv',
                        type=str,
                        default="",
                        help='output csv file path')
    parser.add_argument("--request_rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--mode",
                        type=str,
                        default="async",
                        choices=['async', 'sync'],
                        help="requests send with async mode or sync mode")
    parser.add_argument('--stream',
                        action='store_true',
                        help='Whether to use stream mode for the request')
    parser.add_argument(
        '--backend',
        type=str,
        default="ksana",
        choices=[
            'ksana', 'vllm', 'ksana-server', 'vllm-server', 'trt-llm', 'evart'
        ],
        help='serving backend, ksana or vllm or evart or online server')
    parser.add_argument('--prompt_num',
                        type=int,
                        default=0,
                        help='number of input prompts')
    parser.add_argument(
        '--model_type',
        type=str,
        default="llama",
        choices=['llama', 'llama-3', 'baichuan', 'qwen', 'empty'],
        help=
        'serving model type, used to add prefixes and suffixes to the prompt.')
    parser.add_argument('--use_prefix_cache_prompts',
                        action='store_true',
                        help='test with prompts with very long prefix cache')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=1024,
                        help="The maximum numbers of tokens to generate, ignoring"
                             " the number of tokens in the prompt.")
    parser.add_argument('--temperature',
                        type=float,
                        default=0.0,
                        help="The value used to modulate the next token probabilities.")
    parser.add_argument('--topk',
                        type=int,
                        default=1,
                        help="The number of highest probability vocabulary tokens"
                             " to keep for top-k-filtering.")
    parser.add_argument('--topp',
                        type=float,
                        default=1.0,
                        help="If set to float < 1, only the smallest set of most"
                             " probable tokens with probabilities that add up to"
                             " top_p or higher are kept for generation.")
    parser.add_argument('--repetition_penalty',
                        type=float,
                        default=1.0,
                        help="The parameter for repetition penalty. 1.0 means no penalty.")
    parser.add_argument('--length_penalty',
                        type=float,
                        default=1.0,
                        help="Exponential penalty to the length that is used with" 
                             " beam-based generation.")
    parser.add_argument('--num_beams',
                        type=int,
                        default=1,
                        help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument('--num_return_sequences',
                        type=int,
                        default=1,
                        help="The number of independently computed returned sequences"
                             " for each element in the batch.")
    parser.add_argument('--logprobs',
                        type=int,
                        default=0,
                        help="Whether to return log probabilities of the output tokens"
                             " or not. ")
    parser.add_argument('--stop_token_ids',
                        nargs='+',
                        type=int,
                        default=[],
                        help="A list of token id that should terminate generation if the"
                             " model outputs them.")
    parser.add_argument('--client_timeout',
                        type=int,
                        default=3*3600,
                        help="The timeout limit for the aiohttp client,"
                             "(default is 3 hour).")
    args = parser.parse_args()
    return args


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    if remove_head:
        next(csv_reader)
    return [row[col_idx] for row in csv_reader]

def analyze_stream_jsons(data_str: str):
    brace_count = 0
    start_index = 0
    result_json = {}
    json_str = ""
    for i, char in enumerate(data_str):
        if (char == '\x00'):
            start_index = i+1
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if brace_count == 0 and char == '}':
            json_str = data_str[start_index:i+1]
            start_index = i+1

    return json.loads(str(json_str), strict=False)

async def generate_prompt_async(
    input_requests: List[str],
    request_rate: float,
) -> AsyncGenerator[str, None]:
    input_requests = enumerate(input_requests)
    for req_id, request in input_requests:
        yield req_id, request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request_async(args: argparse.Namespace, prompt: str, api_url: str,
                             req_id: int, result_list: List, pbar: tqdm):
    request_start_time = time.perf_counter()
    headers = {"User-Agent": "Benchmark Client"}
    if not args.stop_token_ids:
        args.stop_token_ids = DEFAULT_STOP_TOKEN_IDS.get(args.model_type, [])
    if args.backend == "ksana":
        data = {
            "prompt": prompt,
            "sampling_config": {
                "temperature": args.temperature,
                "topk": args.topk,
                "topp": args.topp,
                "num_beams": args.num_beams,
                "num_return_sequences": args.num_return_sequences,
                "length_penalty": args.length_penalty,
                "repetition_penalty": args.repetition_penalty,
                "logprobs": args.logprobs,
                "max_new_tokens": args.max_new_tokens,
                "stop_token_ids": args.stop_token_ids
            },
            "stream": args.stream,
        }
    elif args.backend == "trt-llm":
        data = {
            "text_input": prompt,
            "max_tokens": args.max_new_tokens,
            "bad_words": "",
            "stop_words": "",
            "top_k": args.topk,
        }
    elif args.backend in ["vllm", "evart"]:
        # max outputlen is 1024.
        data = {
            "prompt": prompt,
            "use_beam_search": False,
            "n": 1,
            "temperature": args.temperature,
            "max_tokens": args.max_new_tokens,
            "logprobs": args.logprobs,
            "repetition_penalty": args.repetition_penalty,
            "stop_token_ids": args.stop_token_ids,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "top_p": args.topp,
            "top_k": args.topk,
        }
    elif args.backend in ["ksana-server", "vllm-server"]:
        data = {
            "model": "default_model",
            "prompt": prompt,
            "top_p": args.topp,
            "temperature": args.temperature,
            "top_k": args.topk,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "logprobs": args.logprobs,
            "n": 1,
            "task_id": time.time(),
            "delete_prompt_from_output": 0,
            "stream": args.stream,
            "stop_token_ids": args.stop_token_ids
        }

    # Set a timeout of 3 hours for the aiohttp client
    timeout = aiohttp.ClientTimeout(total=args.client_timeout)

    # Create an asynchronous client session with the specified timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Loop indefinitely until the request is successful
        while True:
            # Send a POST request to the API URL with the specified headers and data
            async with session.post(api_url, headers=headers,
                                    json=data) as response:
                # Initialize an empty list to store the response chunks
                chunks = []
                # Iterate over the response chunks and append them to the list
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            # Join the chunks into a single byte string and decode it to UTF-8
            output = b"".join(chunks).decode("utf-8")
            # Parse the output as JSON
            if "server" in args.backend and args.stream:
                data_segments = output.strip().split("\n\n")
                texts = ""
                for segment in data_segments:
                    json_string = segment.split(': ', 1)[1]
                    data = json.loads(json_string)
                    texts += data["choices"][0]["delta"]["content"]
                output = json.loads(data_segments[-1].split(': ', 1)[1])
                output["choices"][0]["delta"]["content"] = texts
            elif args.stream:
                output = analyze_stream_jsons(output)
            else:
                output = json.loads(output)

            # If the response does not contain an "error" key, break out of the loop
            if "error" not in output:
                break

    # Record the end time of the request
    request_end_time = time.perf_counter()
    # Calculate the latency of the request
    request_latency = request_end_time - request_start_time

    output_token_num = len(output.get("output_token_ids", [""])[0])
    input_token_num = len(output.get("input_token_ids", ""))

    server_map_idx = "delta" if args.stream else "message"
    if args.backend == "ksana":
        output_text = output.get("texts", [""])[0].strip()
    elif args.backend == "trt-llm":
        output_text = output.get("text_output", "").strip()
    elif args.backend == "vllm":
        prompt_len = len(prompt)
        output_text = output["text"][0][prompt_len:].strip()
        output_token_num = len(output.get("output_token_ids", [[0]])[0])
    elif args.backend == "evart":
        prompt_len = len(prompt)
        output_text = output["text"][0].strip()
        output_token_num = len(output.get("output_token_ids")[0])
    elif args.backend == "ksana-server":
        output_text = output['choices'][0][server_map_idx]['content']
        input_token_num = output['usage']['prompt_tokens']
        output_token_num = output['usage']['completion_tokens']
    elif args.backend == "vllm-server":
        prompt_len = len(prompt)
        output_text = output['choices'][0][server_map_idx]['content'][
            prompt_len:].strip()

    output_len = len(output_text)
    result_list[req_id] = output_text
    print("", output_text)
    REQUEST_LATENCY.append(
        (len(prompt), output_len if output_len > 0 else 1, input_token_num,
         output_token_num, request_latency))
    pbar.update(1)


# Define an asynchronous function to benchmark the API
async def benchmark_async(args: argparse.Namespace, api_url: str,
                          inputs: List[str]):
    # Initialize a list to store the asynchronous tasks
    tasks: List[asyncio.Task] = []
    # Create a progress bar with a total count equal to the number of inputs
    pbar = tqdm(total=len(inputs))
    # Initialize a result list with empty strings, one for each input
    result_list = [""] * len(inputs)
    # Asynchronously generate prompts with the specified request rate
    async for req_id, prompt in generate_prompt_async(inputs,
                                                      args.request_rate):
        # Format the prompt using the affix dictionary for the specified model type
        prompt = PROMPT_AFFIX_DICT[args.model_type].replace("%s", prompt)
        # Create an asynchronous task to send the request
        task = asyncio.create_task(
            send_request_async(args, prompt, api_url, req_id, result_list, pbar))
        # Add the task to the list of tasks
        tasks.append(task)
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    # Close the progress bar
    pbar.close()
    # Return the result list
    return result_list


async def benchmark_sync(args: argparse.Namespace, api_url: str, inputs: List[str]):
    # Initialize a list to store the asynchronous tasks
    tasks: List[asyncio.Task] = []
    # Create a progress bar with a total count equal to the number of inputs
    pbar = tqdm(total=len(inputs))
    # Initialize a result list with empty strings, one for each input
    result_list = [""] * len(inputs)
    # Asynchronously generate prompts with the specified request rate
    async for req_id, prompt in generate_prompt_async(inputs, args.request_rate):
        # Format the prompt using the affix dictionary for the specified model type
        prompt = PROMPT_AFFIX_DICT[args.model_type].replace("%s", prompt)
        # Await until last request finished
        await send_request_async(args, prompt, api_url, req_id, result_list, pbar)
    # Close the progress bar
    pbar.close()
    # Return the result list
    return result_list


def adjust_list_length(inputs, args):
    if args.prompt_num == 0:
        # 如果args.prompt_num为0，不做任何改变
        return inputs
    elif args.prompt_num > len(inputs):
        # 如果args.prompt_num大于列表长度，尝试复制列表
        repeat_times = args.prompt_num // len(inputs)
        if len(inputs) * repeat_times != args.prompt_num:
            # 如果无法通过整数倍复制达到指定长度，抛出错误
            print(f"len = {len(inputs)}, prompt_num = {args.prompt_num}")
            raise ValueError("无法通过整数倍复制达到指定长度")
        return inputs * repeat_times
    else:
        # 如果args.prompt_num小于或等于列表长度，截断列表
        return inputs[:args.prompt_num]


def main(args: argparse.Namespace):
    api_url = "http://" + args.host + ":" + str(args.port) + "/generate"
    if args.backend == "trt-llm":
        api_url = "http://" + args.host + ":" + str(
            args.port) + "/v2/models/ensemble/generate"
    elif args.backend in ["ksana-server", "vllm-server"]:
        api_url = "http://" + args.host + ":" + str(args.port) + "/v1/chat"
        args.model_type = "empty"  # 在线服务不需要手动拼接前后缀

    # Initialize inputs to None
    inputs = None
    # If the use_prefix_cache_prompts flag is set, load prompts from the prefix cache
    if args.use_prefix_cache_prompts:
        # Load prompts from the prefix cache using the input CSV file
        _, inputs = prefix_cache_reader.load_prompts(input_csv=args.input_csv)
    else:
        # Otherwise, read inputs from the input CSV file
        inputs = read_from_csv(args.input_csv, args.col_idx)
    # Adjust the length of the input list based on the provided arguments
    inputs = adjust_list_length(inputs, args)
    # Record the start time of the benchmark
    benchmark_start_time = time.perf_counter()
    # Run the benchmark in either asynchronous or synchronous mode
    if args.mode == "async":
        # Run the asynchronous benchmark
        result_list = asyncio.run(benchmark_async(args, api_url, inputs))
    else:
        # Run the synchronous benchmark
        result_list = asyncio.run(benchmark_sync(args, api_url, inputs))
    # Record the end time of the benchmark
    benchmark_end_time = time.perf_counter()
    # Calculate the total benchmark time
    benchmark_time = benchmark_end_time - benchmark_start_time
    # Print the total time and throughput
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(inputs) / benchmark_time:.2f} requests/s")
    # Compute the latency statistics
    avg_latency = np.mean([latency for _, _, _, _, latency in REQUEST_LATENCY])
    avg_input_len = np.mean(
        [prompt_len for prompt_len, _, _, _, _ in REQUEST_LATENCY])
    avg_output_len = np.mean(
        [output_len for _, output_len, _, _, _ in REQUEST_LATENCY])
    avg_input_tokens_num = np.mean(
        [input_tokens_num for _, _, input_tokens_num, _, _ in REQUEST_LATENCY])
    avg_output_tokens_num = np.mean([
        output_tokens_num for _, _, _, output_tokens_num, _ in REQUEST_LATENCY
    ])
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"Average input len: {avg_input_len:.2f} chars")
    print(f"Average output len: {avg_output_len:.2f} chars")
    print(f"Average input len: {avg_input_tokens_num:.2f} tokens")
    print(f"Average output len: {avg_output_tokens_num:.2f} tokens")

    avg_per_char_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, _, _, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per char: {avg_per_char_latency:.2f} s")
    avg_per_output_char_latency = np.mean([
        latency / output_len
        for _, output_len, _, _, latency in REQUEST_LATENCY
    ])
    print("Average latency per output char: "
          f"{avg_per_output_char_latency:.2f} s")

    avg_per_token_sec = (avg_input_tokens_num + avg_output_tokens_num
                         ) * len(REQUEST_LATENCY) / benchmark_time
    print(f"Average token per sec: {avg_per_token_sec:.2f} tokens")
    avg_per_output_token_sec = avg_output_tokens_num * len(
        REQUEST_LATENCY) / benchmark_time
    print("Average output token per sec: "
          f"{avg_per_output_token_sec:.2f} tokens")

    if args.output_csv != "":
        with open(args.output_csv, "w", newline='') as fs:
            writer = csv.writer(fs)
            for idx in range(len(result_list)):
                result = result_list[idx]
                writer.writerow([result.replace("</s>", "")])


if __name__ == "__main__":
    args = args_config()
    main(args)
