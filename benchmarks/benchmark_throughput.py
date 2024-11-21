# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import asyncio
import csv
import http
import random
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Tuple, Union

import aiohttp
import numpy as np
import orjson
import uvloop
from tqdm.asyncio import tqdm
# NOTE(karlluo): mindie-service wont return tokens, we need encode tokens to get output tokens
from transformers import AutoTokenizer

# (prompt len, output len, input token num, output token num,
#  request latency, first token latency, inter token latencies)
REQUEST_LATENCY: List[Tuple[int, int, int, int, float, float, List[float]]] = []

# Chat template and stop token ids
# Refer to
# https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
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
    "vicuna":
    "A chat between a curious user and an assistant. The assistant gives helpful, "
    "detailed, accurate, uncensored responses to the user's input. USER: %s ASSISTANT:",
    "yi":
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "chatglm":
    "<|system|>\nYou are a large language model trained by Zhipu.AI. Follow the user's instructions carefully."
    " Respond using markdown.\n<|user|>\n%s\n<|assistant|>\n",
    "empty":
    "%s",
}
DEFAULT_STOP_TOKEN_IDS = {
    "llama-3": [128001, 128009],
    "qwen": [151643, 151644, 151645],
    "yi": [2, 6, 7, 8],
}


@dataclass
class BenchmarkMetrics:
    request_rate: float = 0.
    concurrency: int = 1
    total_latency: float = 0.
    request_throughput: float = 0.
    avg_latency: float = 0.
    avg_input_chars: float = 0.
    avg_output_chars: float = 0.
    avg_input_tokens: float = 0.
    avg_output_tokens: float = 0.
    avg_tokens_per_sec: float = 0.  # token throughput

    def __str__(self):
        return '\n'.join([
            f"Request rate: {self.request_rate:.3f} requests/s",
            f"Concurrency requests: {self.concurrency}",
            f"Total latency: {self.total_latency:.3f} s",
            f"Request throughput: {self.request_throughput:.3f} requests/s",
            f"Average latency: {self.avg_latency:.3f} s",
            f"Average input len: {self.avg_input_chars:.3f} chars",
            f"Average output len: {self.avg_output_chars:.3f} chars",
            f"Average input len: {self.avg_input_tokens:.3f} tokens",
            f"Average output len: {self.avg_output_tokens:.3f} tokens",
            f"Token throughput: {self.avg_tokens_per_sec:.3f} tokens/s",
        ])


@dataclass
class BenchmarkStreamMetrics:
    avg_first_token_latency: float = 0.0  # TTFT
    median_first_token_latency: float = 0.0
    percentiles_first_token_latency: List[Tuple[int, float]] = field(
        default_factory=list
    )
    avg_inter_token_latency: float = 0.0  # ITL
    median_inter_token_latency: float = 0.0
    percentiles_inter_token_latency: List[Tuple[int, float]] = field(
        default_factory=list
    )
    avg_latency_per_out_token: float = 0.0  # TPOT
    median_latency_per_out_token: float = 0.0
    percentiles_latency_per_out_token: List[Tuple[int, float]] = field(
        default_factory=list
    )

    def __str__(self):
        return '\n'.join(
            [f"Average TTFT: {self.avg_first_token_latency:.3f} s",
             f"Median TTFT: {self.median_first_token_latency:.3f} s"] +
            [f"P{percentile} TTFT: {percentile_ttft:.3f} s"
             for [percentile, percentile_ttft] in self.percentiles_first_token_latency] +
            [f"Average ITL: {self.avg_inter_token_latency:.3f} s",
             f"Median ITL: {self.median_inter_token_latency:.3f} s"] +
            [f"P{percentile} ITL: {percentile_itl:.3f} s"
             for [percentile, percentile_itl] in self.percentiles_inter_token_latency] +
            [f"Average TPOT: {self.avg_latency_per_out_token:.3f} s",
             f"Median TPOT: {self.median_latency_per_out_token:.3f} s"] +
            [f"P{percentile} TPOT: {percentile_tpot:.3f} s"
             for [percentile, percentile_tpot] in self.percentiles_latency_per_out_token]
        )


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
                        default=None,
                        help='output csv file path')
    parser.add_argument('--perf_csv',
                        type=str,
                        default=None,
                        help='performance result csv file path')
    parser.add_argument("--request_rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we synthesize the request arrival times.")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--random",
                        action="store_true",
                        help="Randomize request arrival time.")
    parser.add_argument("--chat_template",
                        action="store_true",
                        help="tokenizer apply_chat_template")
    parser.add_argument("--shuffle",
                        action="store_true",
                        help="shuffle input")
    parser.add_argument("--percentiles",
                        nargs='+',
                        type=int,
                        default=[99],
                        help="A list of percentiles for TTFT, ITL and TPOT. "
                        "To report 25-th, 50-th, and 75-th percentiles, use \"25 50 75\". "
                        "Default value is \"99\".")
    parser.add_argument("--request_rate_step",
                        type=float,
                        default=1.0,
                        help="Step for changing the request rate in each iteration.")
    parser.add_argument("--request_rate_num_iters",
                        type=int,
                        default=1,
                        help="Number of iterations for changing the request rate.")
    parser.add_argument("--max_avg_latency",
                        type=float,
                        default=float("inf"),
                        help="The max average latency(seconds).")
    parser.add_argument("--max_first_token_latency",
                        type=float,
                        default=float("inf"),
                        help="The max average first token latency(seconds).")
    parser.add_argument("--concurrency",
                        type=int,
                        default=1,
                        help="Number of requests launched concurrently at the same time.")
    parser.add_argument("--warmup_num_iters",
                        type=int,
                        default=0,
                        help="Number of warmup iterations.")
    parser.add_argument("--repeat_num_iters",
                        type=int,
                        default=1,
                        help="Number of iterations to repeat.")
    parser.add_argument("--mode",
                        type=str,
                        default="async",
                        choices=['async', 'sync'],
                        help="requests send with async mode or sync mode")
    parser.add_argument('--stream',
                        action='store_true',
                        help='Whether to use stream mode for the request')
    parser.add_argument('--backend',
                        type=str,
                        default="ksana",
                        choices=[
                            'ksana', 'vllm', 'ksana-server', 'vllm-server', 'trt-llm',
                            'evart', 'mindie-service', 'sglang'
                        ],
                        help='serving backend')
    parser.add_argument('--prompt_num',
                        type=int,
                        default=0,
                        help='number of input prompts')
    parser.add_argument('--model_type',
                        type=str,
                        default="llama",
                        choices=[
                            'llama', 'llama-3', 'baichuan', 'qwen', 'vicuna', 'yi',
                            'chatglm', 'empty'
                        ],
                        help="serving model type, used to add prefixes and suffixes"
                             " to the prompt.")
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
    parser.add_argument('--no_repeat_ngram_size',
                        type=int,
                        default=0,
                        help="If set to int > 0, all ngrams of that size can only occur once.")
    parser.add_argument('--encoder_no_repeat_ngram_size',
                        type=int,
                        default=0,
                        help="If set to int > 0, all ngrams of that size that occur in the \
                             `encoder_input_ids` cannot occur in the `decoder_input_ids`")
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
                        nargs='*',
                        type=int,
                        default=None,
                        help="A list of token id that should terminate generation if the"
                             " model outputs them.")
    parser.add_argument('--ignore_eos',
                        action="store_true",
                        help="Whether to ignore any EOS tokens.")
    parser.add_argument('--client_timeout',
                        type=int,
                        default=30*3600,
                        help="The timeout limit for the aiohttp client"
                             " (default is 3 hours).")
    parser.add_argument('--tokenizer_path',
                        type=str,
                        default=None,
                        help="mindie-service/TensorRT-LLM wont return tokens, we need"
                             " encode tokens to get output tokens")
    parser.add_argument('--stop_strings',
                        nargs='*',
                        type=str,
                        default=None,
                        help="A list of strings which will be used as stop string in token generation phase")
    args = parser.parse_args()
    return args


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    if remove_head:
        next(csv_reader)
    return [row[col_idx] for row in csv_reader]


async def generate_req_data_async(
    input_requests: List[Tuple[str, bytes]],
    request_rate: float,
    concurrency: int,
    random: bool,
) -> AsyncGenerator[int, Tuple[str, bytes]]:
    input_requests = enumerate(input_requests)
    # Number of requests already sent at the same time
    request_num = 0
    # Total number of requests processed
    total_requests = 0

    # Start time
    start_time = time.time()

    for req_id, request in input_requests:
        yield req_id, request

        request_num += 1
        total_requests += 1
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        if request_num < concurrency:
            # If the number of sent requests is less than the required number of concurrencies,
            # then we don't need to wait either.
            continue
        request_num = 0

        # Calculate the expected time to have sent total_requests at the current request_rate
        expected_time = total_requests / request_rate
        # Calculate the actual time elapsed
        actual_time = time.time() - start_time
        # Calculate the difference to adjust the sleep time
        interval_adjustment = expected_time - actual_time

        if random:
            # Sample the request interval from the exponential distribution and adjust
            interval = np.random.exponential(1.0 / (request_rate / concurrency)) + interval_adjustment
        else:
            # Request arrives uniformly and adjust
            interval = 1.0 / (request_rate / concurrency) + interval_adjustment

        # Ensure interval is not negative
        interval = max(interval, 0)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def construct_request_data(tokenizer: Union[None, AutoTokenizer], prompt: str,
                           args: argparse.Namespace) -> Tuple[str, bytes]:

    if not args.stop_token_ids:
        args.stop_token_ids = DEFAULT_STOP_TOKEN_IDS.get(args.model_type, [])
    if args.chat_template:
        # If chat template is enabled, apply the chat template to the prompt
        prompt = tokenizer.apply_chat_template(
            orjson.loads(prompt),
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # If chat template is not enabled,
        # replace the placeholder in the prompt affix dictionary
        prompt = PROMPT_AFFIX_DICT[args.model_type].replace("%s", prompt)
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
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "encoder_no_repeat_ngram_size": args.encoder_no_repeat_ngram_size,
                "logprobs": args.logprobs,
                "max_new_tokens": args.max_new_tokens,
                "stop_token_ids": args.stop_token_ids,
                "stop_strings": args.stop_strings,
                "ignore_eos": args.ignore_eos,
            },
            "stream": args.stream,
        }
    elif args.backend == "trt-llm":
        data = {
            "accumulate_tokens": True,
            "text_input": prompt,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "min_tokens": args.max_new_tokens if args.ignore_eos else 0,
            "bad_words": "",
            "stop_words": "",
            "top_p": args.topp,
            "top_k": args.topk,
            "stream": args.stream,
        }
    elif args.backend in ["vllm", "evart", "mindie-service"]:
        data = {
            "prompt": prompt,
            "n": 1,
            "temperature": args.temperature,
            "max_tokens": args.max_new_tokens,
            "logprobs": args.logprobs,
            "repetition_penalty": args.repetition_penalty,
            "stop_token_ids": args.stop_token_ids,
            "ignore_eos": args.ignore_eos,
            "min_tokens": args.max_new_tokens if args.ignore_eos else 0,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "top_p": args.topp,
            "top_k": args.topk,
            "stream": args.stream
        }
    elif args.backend == "sglang":
        data = {
            "text": prompt,
            "sampling_params": {
                "n": 1,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "stop_token_ids": args.stop_token_ids,
                "ignore_eos": args.ignore_eos,
                "top_p": args.topp,
                "top_k": args.topk,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "repetition_penalty": args.repetition_penalty,
            },
            "stream": args.stream
        }
    elif args.backend in ["ksana-server", "vllm-server"]:
        data = {
            "model": "default_model",
            "prompt": prompt,
            "top_p": args.topp,
            "temperature": args.temperature,
            "top_k": args.topk,
            "repetition_penalty": args.repetition_penalty,
            "logprobs": args.logprobs,
            "n": 1,
            "task_id": time.time(),
            "delete_prompt_from_output": 1,
            "stream": args.stream,
            "stop_token_ids": args.stop_token_ids,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": args.ignore_eos,
        }
    return prompt, orjson.dumps(data)


async def send_request_async(args: argparse.Namespace, prompt: int,
                             req_data: bytes, api_url: str,
                             req_id: int, result_list: List, pbar: tqdm,
                             tokenizer: Union[None, AutoTokenizer], max_retries=3):
    headers = {
        "User-Agent": "Benchmark Client",
        "Content-Type": "application/json",
    }

    # Set a timeout of 3 hours for the aiohttp client
    timeout = aiohttp.ClientTimeout(total=args.client_timeout)
    
    # Store the output of sever in stream mode
    server_stream_output = ""

    # Create an asynchronous client session with the specified timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Loop until the request is succeeds or the max_reties is reached
        retries = 0
        while True:
            # Record the start time of the request
            request_start_time = time.perf_counter()
            output = None

            # Send a POST request to the API URL with the specified headers and data
            async with session.post(api_url, headers=headers,
                                    data=req_data) as response:
                if response.status == 200:
                    first_token_latency = 0.
                    inter_token_latencies = []
                    most_recent_timestamp = request_start_time
                    # Store a temporarily incomplete response chunk
                    chunk_acc = ""
                    # Iterate over the response chunks and append them to the list
                    async for chunk_bytes, _ in response.content.iter_chunks():
                        # Record the current timestamp
                        timestamp = time.perf_counter()
                        chunk_bytes = chunk_bytes.strip(b'\x00')
                        if not chunk_bytes:
                            continue
                        chunk = chunk_bytes.decode("utf-8")
                        # Remove the optional prefix "data: " for the first chunk
                        if chunk_acc == "" and chunk.startswith("data: "):
                            chunk = chunk[len("data: "):]
                        # Response done
                        if chunk == "[DONE]":
                            break
                        # Accumulate this chunk
                        chunk_acc += chunk
                        # Response is error-prone
                        try:
                            output = orjson.loads(chunk_acc)
                            # Reset the chunk_acc
                            chunk_acc = ""
                            if "server" in args.backend and args.stream:
                                server_stream_output += output["choices"][0]["delta"]["content"]
                        except orjson.JSONDecodeError:
                            continue

                        # First token
                        if first_token_latency == 0.:
                            first_token_latency = timestamp - request_start_time
                        # Decoding phase
                        else:
                            inter_token_latencies.append(timestamp -
                                                most_recent_timestamp)
                        most_recent_timestamp = timestamp

            # If the response does not contain an "error" key, break out of the loop
            if (output is not None) and ("error" not in output):
                # Record the end time of the request as the most recent timestamp
                request_end_time = most_recent_timestamp
                break
            # Request failed, try again
            retries += 1
            if retries > max_retries:
                raise Exception(f"The request(req_id = {req_id}) failed.")

    # Calculate the latency of the request
    request_latency = request_end_time - request_start_time

    output_token_num = len(output.get("output_token_ids", [""])[0])
    input_token_num = len(output.get("input_token_ids", ""))

    server_map_idx = "delta" if args.stream else "message"
    if args.backend == "ksana":
        output_text = output.get("texts", [""])[0].strip()
    elif args.backend == "trt-llm":
        prompt_len = len(prompt)
        output_text = output.get("text_output", "").strip()
        if tokenizer is None:
            input_token_num = 0
            output_token_num = 0
        else:
            input_token_num = len(tokenizer.encode(prompt))
            output_token_num = len(tokenizer.encode(output_text))
    elif args.backend == "vllm":
        prompt_len = len(prompt)
        output_text = output["text"][0][prompt_len:].strip()
        if tokenizer is None:
            input_token_num = 0
            output_token_num = 0
        else:
            input_token_num = len(tokenizer.encode(prompt))
            output_token_num = len(tokenizer.encode(output_text))
    elif args.backend == "sglang":
        prompt_len = len(prompt)
        output_text = output["text"].strip()
        input_token_num = output["meta_info"]["prompt_tokens"]
        output_token_num = output["meta_info"]["completion_tokens"]
    elif args.backend == "evart":
        prompt_len = len(prompt)
        output_text = output["text"][0].strip()
        output_token_num = len(output.get("output_token_ids")[0])
    elif "server" in args.backend:
        if args.stream:
            output["choices"][0]["delta"]["content"] = server_stream_output
        output_text = output['choices'][0][server_map_idx]['content']
        input_token_num = output['usage']['prompt_tokens']
        output_token_num = output['usage']['completion_tokens']
    elif args.backend == "mindie-service":
        prompt_len = len(prompt)
        output_text = output["text"][0][prompt_len:].strip()
        if tokenizer is None:
            input_token_num = 0
            output_token_num = 0
        else:
            input_token_num = len(tokenizer.encode(prompt))
            output_token_num = len(tokenizer.encode(output_text))

    output_len = len(output_text)
    result_list[req_id] = output_text
    print(
        "req_id : {} input_token_num={}, output_token_num={}, output_text_len={}, output_text=\n{}"
        .format(req_id, input_token_num, output_token_num, output_len,
                output_text))
    REQUEST_LATENCY.append(
        (len(prompt), output_len if output_len > 0 else 1, input_token_num,
         output_token_num, request_latency,
         first_token_latency, inter_token_latencies))
    pbar.update(1)


# Define an asynchronous function to benchmark the API
async def benchmark_async(args: argparse.Namespace, api_url: str,
                          inputs: List[Tuple[str, bytes]], tokenizer: Union[None, AutoTokenizer]):
    # Initialize a list to store the asynchronous tasks
    tasks: List[asyncio.Task] = []
    # Create a progress bar with a total count equal to the number of inputs
    pbar = tqdm(total=len(inputs))
    # Initialize a result list with empty strings, one for each input
    result_list = [""] * len(inputs)
    # Asynchronously generate req_datas with the specified request rate
    async for req_id, (prompt, req_data) in generate_req_data_async(inputs,
                                                                    args.request_rate,
                                                                    args.concurrency,
                                                                    args.random):

        # Create an asynchronous task to send the request
        task = asyncio.create_task(
            send_request_async(args, prompt, req_data, api_url, req_id, result_list, pbar,
                               tokenizer))
        # Add the task to the list of tasks
        tasks.append(task)
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    # Close the progress bar
    pbar.close()
    # Return the result list
    return result_list


async def benchmark_sync(args: argparse.Namespace, api_url: str,
                         inputs: List[Tuple[str, bytes]], tokenizer: Union[None, AutoTokenizer]):
    # Create a progress bar with a total count equal to the number of inputs
    pbar = tqdm(total=len(inputs))
    # Initialize a result list with empty strings, one for each input
    result_list = [""] * len(inputs)
    # Asynchronously generate req_datas with the specified request rate
    async for req_id, (prompt, req_data) in generate_req_data_async(inputs, args.request_rate,
                                                                    args.concurrency,
                                                                    args.random):
        # Await until last request finished
        await send_request_async(args, prompt, req_data, api_url, req_id, result_list, pbar,
                                 tokenizer)
    # Close the progress bar
    pbar.close()
    # Return the result list
    return result_list


def adjust_list_length(inputs: List[str], args: argparse.Namespace):
    if args.prompt_num == 0:
        # 如果args.prompt_num为0，不做任何改变
        args.prompt_num = len(inputs)
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


def run_benchmark(args: argparse.Namespace, api_url: str, inputs: List[Tuple[str, bytes]],
                  tokenizer: Union[None, AutoTokenizer]):
    if args.mode == "async":
        # Run the asynchronous benchmark
        return asyncio.run(benchmark_async(args, api_url, inputs, tokenizer))
    else:
        # Run the synchronous benchmark
        return asyncio.run(benchmark_sync(args, api_url, inputs, tokenizer))


def search_request_rate(args: argparse.Namespace, request_rate_list: List[Tuple[float, float]]):
    def round_to_tenth(number):
        # When searching for the optimal request rate, the minimum precision is 0.1.
        return max(round(number * 10) / 10, 0.1)
    step = len(request_rate_list)
    request_rate = -1
    if step < args.request_rate_num_iters:
        request_rate = args.request_rate + (args.request_rate_step if step > 0 else 0)
    elif args.max_avg_latency != float("inf") or args.max_first_token_latency != float("inf"):
        request_rate_list.sort(key=lambda x: x[0])
        if request_rate_list[-1][1] <= args.max_avg_latency and \
           request_rate_list[-1][2] <= args.max_first_token_latency:
            request_rate = min(request_rate_list[-1][0] * 2, args.prompt_num)
        elif request_rate_list[0][1] > args.max_avg_latency or \
             request_rate_list[0][2] > args.max_first_token_latency:
            request_rate = round_to_tenth(request_rate_list[0][0] / 2)
        else:
            rate_left = max(filter(lambda x: x[1] <= args.max_avg_latency and \
                                             x[2] <= args.max_first_token_latency, request_rate_list),
                                   key=lambda x: x[0])[0]
            rate_right = min(filter(lambda x: x[1] > args.max_avg_latency or \
                                              x[2] > args.max_first_token_latency, request_rate_list),
                                    key=lambda x: x[0])[0]
            request_rate = round_to_tenth((rate_left + rate_right) / 2)
        if any(ite[0] == request_rate for ite in request_rate_list):
            print(f"Duplicate request rate detected: {request_rate}. Terminating the search prematurely.")
            request_rate = -1
    return request_rate


def main(args: argparse.Namespace):
    global REQUEST_LATENCY

    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer = None
    api_url = "http://" + args.host + ":" + str(args.port) + "/generate"
    if args.backend == "trt-llm":
        api_url = "http://" + args.host + ":" + str(
            args.port) + "/v2/models/ensemble/generate"
        if args.stream:
            api_url += "_stream"  # generate_stream
    elif args.backend in ["ksana-server", "vllm-server"]:
        api_url = "http://" + args.host + ":" + str(args.port) + "/v1/chat"
        args.model_type = "empty"  # 在线服务不需要手动拼接前后缀

    if args.chat_template and args.tokenizer_path is None:
        raise ValueError(
            "When chat_template is enabled, tokenizer_path cannot be None.")
    # NOTE: mindie-service/TensorRT-LLM wont return tokens, we need encode tokens to get output tokens
    if args.backend in ["mindie-service", "trt-llm", "vllm"] or args.chat_template:
        if args.tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                revision=None,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=True,
                use_fast=True
            )

    # Read inputs from the input CSV file
    inputs = read_from_csv(args.input_csv, args.col_idx)
    # Adjust the length of the input list based on the provided arguments
    if args.shuffle:
        random.shuffle(inputs)
    inputs = adjust_list_length(inputs, args)
    inputs = [construct_request_data(tokenizer, input, args) for input in inputs]
    perf_result_list: List[Tuple[BenchmarkMetrics, BenchmarkStreamMetrics]] = []
    # requst_rate_list: List[Tuple[request_rate, avg_latency, avg_TTFT]]
    request_rate_list: List[Tuple[float, float, float]] = []
    while True:
        # cmake -DWITH_CLEAR_CACHE=ON
        clear_cache_data = {
            "input_tokens": [0, 0],
            "sampling_config": {
                "max_new_tokens": 1
            }
        }
        conn = http.client.HTTPConnection(args.host + ":" + str(args.port))
        conn.request("POST", '/generate',
                     body=orjson.dumps(clear_cache_data),
                     headers={'Content-Type': 'application/json'})
        conn.getresponse()
        metrics = BenchmarkMetrics()
        metrics.request_rate = search_request_rate(args, request_rate_list)
        args.request_rate = metrics.request_rate
        if metrics.request_rate == -1:
            break
        metrics.concurrency = args.concurrency
        for iter in range(args.warmup_num_iters):
            print(f"Start warmup iteration {iter} with request rate {metrics.request_rate:.3f}")
            run_benchmark(args, api_url, inputs, tokenizer)
        REQUEST_LATENCY.clear()

        # Record the start time of the benchmark
        benchmark_start_time = time.perf_counter()
        for iter in range(args.repeat_num_iters):
            print(f"Start profile iteration {iter} with request rate {metrics.request_rate:.3f}")
            result_list = run_benchmark(args, api_url, inputs, tokenizer)
        # Record the end time of the benchmark
        benchmark_end_time = time.perf_counter()

        # Calculate the total benchmark time
        metrics.total_latency = (
            benchmark_end_time - benchmark_start_time
        ) / args.repeat_num_iters
        # Calculate the request throughput
        metrics.request_throughput = len(inputs) / metrics.total_latency

        # Compute the latency statistics
        metrics.avg_latency = np.mean([latency for _, _, _, _, latency, _, _ in REQUEST_LATENCY])

        metrics.avg_input_chars = np.mean(
            [prompt_len for prompt_len, _, _, _, _, _, _ in REQUEST_LATENCY]
        )
        metrics.avg_output_chars = np.mean(
            [output_len for _, output_len, _, _, _, _, _ in REQUEST_LATENCY]
        )
        metrics.avg_input_tokens = np.mean(
            [input_tokens_num for _, _, input_tokens_num, _, _, _, _ in REQUEST_LATENCY]
        )
        metrics.avg_output_tokens = np.mean(
            [output_tokens_num for _, _, _, output_tokens_num, _, _, _ in REQUEST_LATENCY]
        )

        # Calculate the token throughput
        metrics.avg_tokens_per_sec = (metrics.avg_input_tokens + metrics.avg_output_tokens
            ) * len(REQUEST_LATENCY) / metrics.total_latency / args.repeat_num_iters

        print(metrics)

        stream_metrics = BenchmarkStreamMetrics()
        if args.stream:  # TTFT, TPOT and ITL are only available in stream mode
            first_token_latencies = [
                first_token_latency
                for _, _, _, _, _, first_token_latency, _ in REQUEST_LATENCY
            ]
            if len(first_token_latencies) > 0:
                stream_metrics.avg_first_token_latency = np.mean(first_token_latencies)
                stream_metrics.median_first_token_latency = np.median(first_token_latencies)
                stream_metrics.percentiles_first_token_latency = [
                    (percentile, np.percentile(first_token_latencies, percentile))
                    for percentile in args.percentiles
                ]

            inter_token_latencies = [
                inter_token_latency for _, _, _, _, _, _, inter_token_latencies in REQUEST_LATENCY
                for inter_token_latency in inter_token_latencies
            ]
            if len(inter_token_latencies) > 0:
                stream_metrics.avg_inter_token_latency = np.mean(inter_token_latencies)
                stream_metrics.median_inter_token_latency = np.median(inter_token_latencies)
                stream_metrics.percentiles_inter_token_latency = [
                    (percentile, np.percentile(inter_token_latencies, percentile))
                    for percentile in args.percentiles
                ]

            latencies_per_out_token = [
                (latency - first_token_latency) / (output_tokens_num - 1)
                for _, _, _, output_tokens_num, latency, first_token_latency, _ in REQUEST_LATENCY
                if output_tokens_num > 1
            ]
            if len(latencies_per_out_token) > 0:
                stream_metrics.avg_latency_per_out_token = np.mean(latencies_per_out_token)
                stream_metrics.median_latency_per_out_token = np.median(latencies_per_out_token)
                stream_metrics.percentiles_latency_per_out_token = [
                    (percentile, np.percentile(latencies_per_out_token, percentile))
                    for percentile in args.percentiles
                ]

            print(stream_metrics)

        perf_result_list.append((metrics, stream_metrics))
        request_rate_list.append((metrics.request_rate, metrics.avg_latency, stream_metrics.avg_first_token_latency))
        REQUEST_LATENCY.clear()

    if args.output_csv is not None:
        with open(args.output_csv, "w", newline='') as fs:
            writer = csv.writer(fs)
            for idx in range(len(result_list)):
                result = result_list[idx]
                writer.writerow([result.replace("</s>", "")])

    if args.perf_csv is not None:
        with open(args.perf_csv, "w", newline='') as fs:
            writer = csv.writer(fs)
            header = ["Request rate", "Concurrency", "Total latency", "Request throughput", "Avg latency",
                      "Avg input chars", "Avg output chars", "Avg input tokens", "Avg output tokens",
                      "Token throughput"]
            if args.stream:
                header.extend(["Avg TTFT", "Median TTFT"] +
                              [f"P{percentile} TTFT" for percentile in args.percentiles] +
                              ["Avg ITL", "Median ITL"] +
                              [f"P{percentile} ITL" for percentile in args.percentiles] +
                              ["Avg TPOT", "Median TPOT"] +
                              [f"P{percentile} TPOT" for percentile in args.percentiles])
            writer.writerow(header)
            for (metrics, stream_metrics) in perf_result_list:
                row = [f"{value:.3f}" for value in metrics.__dict__.values()]
                if args.stream:
                    for value in stream_metrics.__dict__.values():
                        if isinstance(value, list):
                            row.extend([f"{percentile_value[1]:.3f}" for percentile_value in value])
                        else:
                            row.append(f"{value:.3f}")
                writer.writerow(row)


if __name__ == "__main__":
    uvloop.install()
    args = args_config()
    main(args)
