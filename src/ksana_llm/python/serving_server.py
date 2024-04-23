# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import ksana_llm
import argparse
import json
import yaml
import uvicorn
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import GenerationConfig, AutoTokenizer
from fastapi import FastAPI

import asyncio
from functools import partial
from concurrent import futures

model_executor = futures.ThreadPoolExecutor(max_workers=256)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
model = None
tokenizer = None


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="examples/ksana_llm.yaml",
                        help='serving config file')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="",
                        help='the model tokenizer dir. By default, the YAML configuration will be used.')
    parser.add_argument('--host',
                        type=str,
                        default="0.0.0.0",
                        help='server host address')
    parser.add_argument('--port', type=int, default=8888, help='server port')
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    args = parser.parse_args()
    return args


def streaming_generate(model_name, input_tokens, generation_config):
    """Perform streaming generation.
    """
    # Create a results iterator for the model's generation
    results_iterator = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        streamer=True  # enable streaming generation
    )

    # Define an asynchronous function to stream the results
    async def stream_results():
        unfinished_token = []  # store unfinished tokens
        async for request_output in results_iterator:  # iterate over the results
            if request_output is not None:  # if the output is not empty
                # Decode the output tokens using the tokenizer
                output_text = tokenizer.decode(
                    unfinished_token +
                    [request_output
                     ],  # combine unfinished tokens with the new output
                    skip_special_tokens=True  # skip special tokens
                )
                # Check if the output ends with a special token (\uFFFD)
                if output_text[-1:] == "\uFFFD":
                    # If it does, add the output to the unfinished tokens and reset the output text
                    unfinished_token = unfinished_token + [request_output]
                    output_text = ""
                else:
                    # If it doesn't, reset the unfinished tokens
                    unfinished_token = []
                # Create a response dictionary with the output text
                ret = {"texts": output_text}
                # Yield the response as a JSON-encoded string with a null character (\0) at the end
                yield (json.dumps(ret) + "\0").encode("utf-8")
            else:
                # If the output is empty, return from the function
                return

    # Return a StreamingResponse object with the streamed results
    return StreamingResponse(stream_results())


def batch_generate(model_name, input_tokens, generation_config):
    """Perform batch generation.
    """
    # Generate output tokens using the model
    results_tokens = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        streamer=None  # disable streaming generation
    )

    # Decode the output tokens into a human-readable text using the tokenizer
    output_text = tokenizer.decode(results_tokens, skip_special_tokens=True)

    # Create a JSON response with the generated text and token IDs
    return JSONResponse({
        "texts": output_text,  # the generated text
        "output_token_ids": results_tokens,  # the generated token IDs
        "input_token_ids": input_tokens  # the input token IDs
    })


def get_sampling_value(sampling_config: dict, key: str, default_val=None):
    """Get value from sampling_config dict, return default if key not exists.
    """
    return sampling_config[key] if key in sampling_config else default_val


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model_name: the model your wanner to infer.
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    """

    request_dict = await request.json()
    model_name = request_dict.pop("model_name", "")
    prompt_text = request_dict.pop("prompt")
    enable_streaming = request_dict.pop("stream", True)
    sampling_config = request_dict.pop("sampling_config", None)

    input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    generation_config = GenerationConfig(
        num_beams=1,
        top_k=get_sampling_value(sampling_config, "topk", 1),
        top_p=get_sampling_value(sampling_config, "topp", 0.0),
        temperature=get_sampling_value(sampling_config, "temperature", 0.0),
        max_new_tokens=get_sampling_value(sampling_config, "max_new_tokens",
                                          -1),
        repetition_penalty=get_sampling_value(sampling_config,
                                              "repetition_penalty", 1.0))

    # Get the current event loop
    loop = asyncio.get_event_loop()

    # Determine whether to use streaming generation or batch generation
    if enable_streaming:
        # Use streaming generation
        # Run the streaming_generate function in an executor to avoid blocking the event loop
        results = await loop.run_in_executor(
            model_executor,  # specify the executor to use
            partial(
                streaming_generate,  # partial function to call
                model_name=model_name,  # pass model name as an argument
                input_tokens=input_tokens,  # pass input tokens as an argument
                generation_config=generation_config
            )  # pass generation config as an argument
        )
    else:
        # Use batch generation
        # Run the batch_generate function in an executor to avoid blocking the event loop
        results = await loop.run_in_executor(
            model_executor,  # specify the executor to use
            partial(
                batch_generate,  # partial function to call
                model_name=model_name,  # pass model name as an argument
                input_tokens=input_tokens,  # pass input tokens as an argument
                generation_config=generation_config
            )  # pass generation config as an argument
        )

    # Return the results of the generation
    return results


if __name__ == "__main__":
    args = args_config()
    if not args.tokenizer_dir:
        with open(args.config_file, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            args.tokenizer_dir = yaml_data["model_spec"]["base_model"]["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                              trust_remote_code=True)
    model = ksana_llm.AutoModel.from_config(args.config_file)

    # Use multithread to support parallelism.
    log_level = os.getenv("NLLM_LOG_LEVEL", "INFO").upper()
    if log_level in ["DEBUG", "INFO", "ERROR"]:
        log_level = log_level.lower()
    else:
        log_level = "info"
        print(
            f"Uvicorn's logging not support env: NLLM_LOG_LEVEL={log_level}, keep it as defalt(info)."
        )
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
