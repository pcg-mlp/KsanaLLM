"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    line_up = '\033[1A'
    line_clear = '\x1b[2K'
    for _ in range(n):
        print(line_up, end=line_clear, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    """
    Send a POST request to the API with the given prompt and parameters.

    Args:
        prompt (str): The input prompt to send to the API.
        api_url (str): The URL of the API endpoint.
        n (int, optional): The number of responses to generate. Defaults to 1.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        requests.Response: The response from the API.
    """
    # Set the User-Agent header to "Test Client"
    headers = {"User-Agent": "Test Client"}
    # Create a JSON payload with the prompt, n, stream, and sampling configuration
    pload = {
        "prompt": prompt,
        "n": n,
        "stream": stream,
        "sampling_config": {
            "temperature": 0.7,
            "topk": 1,
            "topp": 0.0
        },
    }
    # Send a POST request to the API with the payload and headers
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    # Return the response
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    """
    Iterate over the streaming response from the API.

    Args:
        response (requests.Response): The response from the API.

    Yields:
        Iterable[List[str]]: An iterable of lists of strings, where each list contains the output texts.
    """
    try:
        # Iterate over the response chunks with a chunk size of 8192 bytes
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b"\0"):
            if chunk:
                # Parse the chunk as JSON and extract the output texts
                data = json.loads(chunk.decode("utf-8"))
                output = data["texts"]
                yield output
    except requests.exceptions.ChunkedEncodingError as ex:
        # Handle chunked encoding errors
        print(f"Invalid chunk encoding {str(ex)}")


def get_response(response: requests.Response) -> List[str]:
    """
    Get the response from the API.

    Args:
        response (requests.Response): The response from the API.

    Returns:
        List[str]: A list of output texts.
    """
    # Parse the response content as JSON and extract the output texts
    data = json.loads(response.content)
    output = data["texts"]
    return output


if __name__ == "__main__":
    """
    Main entry point of the script.
    """
    # Create an argument parser to parse command-line arguments
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Shanghai is")
    parser.add_argument("--stream", action="store_true")
    # Parse the command-line arguments
    args = parser.parse_args()
    # Extract the prompt, API URL, n, and stream flag from the parsed arguments
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream
    # Print the prompt to the console
    print(f"Prompt: {prompt!r}\n", flush=True)
    # Send a POST request to the API with the prompt and parameters
    response = post_http_request(prompt, api_url, n, stream)
    # Initialize an empty result string
    # pylint: disable-next=invalid-name
    RESULT = str()
    # Handle the response based on the stream flag
    if stream:
        # Iterate over the streaming response
        for output in get_streaming_response(response):
            # Clear the previous lines
            num_printed_lines = len(RESULT.splitlines())
            clear_line(num_printed_lines)

            # Append the output to the result string
            RESULT += (output + " ")
            # Print the result string to the console
            print(RESULT, flush=True)
    else:
        # Get the response output
        output = get_response(response)
        # Print the output to the console
        print(output, flush=True)
