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
from typing import AsyncGenerator, List, Tuple
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
import csv

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


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
    args = parser.parse_args()
    return args


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    if remove_head:
        next(csv_reader)
    for line in csv_reader:
        yield line[col_idx]


async def generate_prompt(
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


async def send_request(prompt: str, api_url: str, req_id: int, result_list: List, pbar: tqdm):
    request_start_time = time.perf_counter()
    headers = {"User-Agent": "Benchmark Client"}
    data = {
        "model_name": "llama",
        "prompt": prompt,
        "sampling_config": {
            "temperature": 0.0,
            "topk": 1,
            "topp": 0.0
        },
        "stream": False,
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
    output_len = len(output.get("texts", ""))
    print(f"\nQuestion {req_id}: {prompt}\nAnswer: ", output.get("texts", ""))
    result_list[req_id] = output.get("texts", "")
    REQUEST_LATENCY.append((len(prompt), output_len, request_latency))
    pbar.update(1)


async def benchmark(args: argparse.Namespace, api_url: str, inputs: List[str]):
    # inputs = inputs[0:8]
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(inputs))
    result_list = [""] * len(inputs)
    async for req_id, prompt in generate_prompt(inputs, args.request_rate):
        prompt = "[INST]%s[/INST]" % prompt
        task = asyncio.create_task(send_request(prompt, api_url, req_id, result_list, pbar))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()
    return result_list

def main(args: argparse.Namespace):
    api_url = "http://" + args.host + ":" + str(args.port) + "/generate"
    input_generator = read_from_csv(args.input_csv)
    inputs = list(input_generator)

    benchmark_start_time = time.perf_counter()
    result_list = asyncio.run(benchmark(args, api_url, inputs))

    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {len(inputs) / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")

    if args.output_csv != "":
        with open(args.output_csv, "w", newline='') as fs:
            writer = csv.writer(fs)
            for idx in range(len(result_list)):
                result = result_list[idx]
                writer.writerow([result.replace("</s>", "")])

if __name__ == "__main__":
    args = args_config()
    main(args)
