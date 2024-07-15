# -*- coding: utf-8 -*-

import argparse
import json
from typing import Iterable, List

import requests


# Define a function to get streaming response from the server
def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    try:
        # Iterate over the chunks of response data
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b"\0"):
            if chunk:
                # Load the chunk as JSON and extract the output text
                data = json.loads(chunk.decode("utf-8"))
                output = data["texts"]
                yield output
    except requests.exceptions.ChunkedEncodingError as ex:
        print(f"Invalid chunk encoding {str(ex)}")


# Define a function to test the HTTP chat API
def test_http_chat(prompt, url):

    # Prepare the request data
    req = {
        "prompt": prompt,
        "sampling_config": {
            "temperature": 0.0,
            "topk": 1,
            "topp": 0.0
        },
        "stream": True,
    }

    # Send the request to the server
    response = requests.post(url,
                             headers={"User-Agent": "Test Client"},
                             json=req,
                             stream=True)
    # Print the output from the streaming response
    for output in get_streaming_response(response):
        print(output, flush=True)


# Main function to run the test
if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--prompt",
        type=str,
        default="作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    # Construct the API URL
    url = f"http://{args.host}:{args.port}/generate"
    # Wrap the prompt with [INST] tags (for llama)
    prompt = f"[INST]{args.prompt}[/INST]"

    # Run the test with the given prompt and API URL
    test_http_chat(prompt, url)
