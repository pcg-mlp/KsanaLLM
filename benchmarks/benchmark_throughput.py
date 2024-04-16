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
    "llama": "[INST]%s[/INST]",
    "baichuan": "<reserved_106>%s<reserved_107>",
    "qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
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
    parser.add_argument('--backend',
                        type=str,
                        default="ksana",
                        choices=['ksana', 'vllm', 'ksana-server', 'vllm-server', 'trt-llm', 'evart'],
                        help='serving backend, ksana or vllm or evart or online server')
    parser.add_argument('--prompt_num',
                        type=int,
                        default=0,
                        help='number of input prompts')
    parser.add_argument('--model_type',
                        type=str,
                        default="llama",
                        choices=['llama', 'baichuan', 'qwen'],
                        help='serving model type, used to add prefixes and suffixes to the prompt.')
    parser.add_argument('--use_prefix_cache_prompts',
                        action='store_true',
                        help='test with prompts with very long prefix cache')
    args = parser.parse_args()
    return args


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    if remove_head:
        next(csv_reader)
    return [row[0] for row in csv_reader]


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


def generate_prompt_sync(
    input_requests: List[str],
    request_rate: float,
):
    input_requests = enumerate(input_requests)
    for req_id, request in input_requests:
        yield req_id, request
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        time.sleep(interval)


async def send_request_async(prompt: str, api_url: str, req_id: int,
                             result_list: List, pbar: tqdm, backend: str):
    request_start_time = time.perf_counter()
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "ksana":
        data = {
            "prompt": prompt,
            "sampling_config": {
                "temperature": 0.0,
                "topk": 1,
                "topp": 0.0,
                "repetition_penalty" : 1.0,
                "max_new_tokens": 1024,
            },
            "stream": False,
        }
    elif backend == "trt-llm":
        data = {
            "text_input": prompt, 
            "max_tokens": 1024, 
            "bad_words": "", 
            "stop_words": "",
            "top_k": 1, 
        }
    elif backend in ["vllm", "evart"]:
        # max outputlen is 1024.
        data = {
            "prompt": prompt,
            "use_beam_search": False,
            "n": 1,
            "temperature": 0.00000001,
            "max_tokens": 1024,
            "repetition_penalty" : 1.0
        }
    elif backend in ["ksana-server", "vllm-server"]:
        data = {
            "model": "default_model",
            "prompt": prompt,
            "top_p": 1.0,
            "temperature": 1,
            "top_k": 1,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "n": 1,
            "task_id": time.time(),
            "delete_prompt_from_output": 0,
        }

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers,
                                    json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time

    output_token_num = len(output.get("output_token_ids", ""))
    input_token_num = len(output.get("input_token_ids", ""))
    if backend == "ksana":
        output_text = output.get("texts", "").strip()
    elif backend == "trt-llm":
        output_text = output.get("text_output", "").strip()
    elif backend == "vllm":
        prompt_len = len(prompt)
        output_text = output["text"][0][prompt_len:].strip()
        output_token_num = len(output.get("output_token_ids")[0])
    elif backend == "evart":
        prompt_len = len(prompt)
        output_text = output["text"][0].strip()
        output_token_num = len(output.get("output_token_ids")[0])
    elif backend == "ksana-server":
        output_text = output['choices'][0]['message']['content']
        input_token_num = output['usage']['prompt_tokens']
        output_token_num = output['usage']['completion_tokens']
    elif backend == "vllm-server":
        prompt_len = len(prompt)
        output_text = output['choices'][0]['message']['content'][prompt_len:].strip()

    output_len = len(output_text)
    result_list[req_id] = output_text
    print("", output_text)
    REQUEST_LATENCY.append(
        (len(prompt), output_len if output_len > 0 else 1, input_token_num, output_token_num, request_latency))
    pbar.update(1)


def send_request_sync(prompt: str, api_url: str, req_id: int,
                      result_list: List, pbar: tqdm, backend: str):
    request_start_time = time.perf_counter()
    headers = {"User-Agent": "Benchmark Client"}
    data = {
        "prompt": prompt,
        "sampling_config": {
            "temperature": 0.0,
            "topk": 1,
            "topp": 0.0,
            "max_new_tokens" : 1024,
            "repetition_penalty" : 1.0
        },
        "stream": False,
    }
    timeout = 3 * 3600
    resp = requests.post(api_url, headers=headers, data=data, timeout=timeout)
    output = json.loads(resp.content)
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    output_len = len(output.get("texts", ""))
    result_list[req_id] = output.get("texts", "")
    REQUEST_LATENCY.append((len(prompt), output_len, request_latency))
    pbar.update(1)


async def benchmark_async(args: argparse.Namespace, api_url: str,
                          inputs: List[str]):
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(inputs))
    result_list = [""] * len(inputs)
    async for req_id, prompt in generate_prompt_async(inputs,
                                                      args.request_rate):
        prompt = PROMPT_AFFIX_DICT[args.model_type].replace("%s", prompt)
        task = asyncio.create_task(
            send_request_async(prompt, api_url, req_id, result_list, pbar,
                               args.backend))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()
    return result_list


def benchmark_sync(args: argparse.Namespace, api_url: str, inputs: List[str]):
    pbar = tqdm(total=len(inputs))
    result_list = [""] * len(inputs)
    for req_id, prompt in generate_prompt_sync(inputs, args.request_rate):
        prompt = "[INST]%s[/INST]" % prompt
        send_request_sync(prompt, api_url, req_id, result_list, pbar,
                          args.backend)
    pbar.close()
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
        api_url = "http://" + args.host + ":" + str(args.port) + "/v2/models/ensemble/generate"
    elif args.backend == "server":
        api_url = "http://" + args.host + ":" + str(args.port) + "/v1/chat"

    inputs = None
    if args.use_prefix_cache_prompts:
        _, inputs = prefix_cache_reader.load_prompts(input_csv=args.input_csv)
    else:
        inputs = read_from_csv(args.input_csv)

    inputs = adjust_list_length(inputs, args)

    benchmark_start_time = time.perf_counter()
    if args.mode == "async":
        result_list = asyncio.run(benchmark_async(args, api_url, inputs))
    else:
        result_list = benchmark_sync(args, api_url, inputs)
    benchmark_end_time = time.perf_counter()

    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(inputs) / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, _, _, latency in REQUEST_LATENCY])
    avg_input_len = np.mean(
        [prompt_len for prompt_len, _, _, _, _ in REQUEST_LATENCY])
    avg_output_len = np.mean(
        [output_len for _, output_len, _, _, _ in REQUEST_LATENCY])
    avg_input_tokens_num = np.mean(
        [input_tokens_num for _, _, input_tokens_num, _, _ in REQUEST_LATENCY])
    avg_output_tokens_num = np.mean(
        [output_tokens_num for _, _, _, output_tokens_num, _ in REQUEST_LATENCY])
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
    avg_per_output_char_latency = np.mean(
        [latency / output_len for _, output_len, _, _, latency in REQUEST_LATENCY])
    print("Average latency per output char: "
          f"{avg_per_output_char_latency:.2f} s")

    avg_per_token_sec = (avg_input_tokens_num + avg_output_tokens_num) * len(REQUEST_LATENCY) / benchmark_time
    print(f"Average token per sec: {avg_per_token_sec:.2f} tokens")
    avg_per_output_token_sec = avg_output_tokens_num * len(REQUEST_LATENCY) / benchmark_time
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
