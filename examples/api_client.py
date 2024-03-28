"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model_name": "llama",
        "prompt": prompt,
        "n": n,
        "stream": stream,
        "sampling_config": {
            "temperature": 0.7,
            "topk": 1,
            "topp": 0.0
        },
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    try:
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["texts"]
                yield output
    except requests.exceptions.ChunkedEncodingError as ex:
        print(f"Invalid chunk encoding {str(ex)}")


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["texts"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Shanghai is")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)

    result = ""

    if stream:
        num_printed_lines = 0
        for output in get_streaming_response(response):
            num_printed_lines = len(result.splitlines())
            clear_line(num_printed_lines)
            result += (output + " ")
            print(result, flush=True)
    else:
        output = get_response(response)
        print(output, flush=True)
