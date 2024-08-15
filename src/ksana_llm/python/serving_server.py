# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import json
import os
import asyncio
from typing import Dict, Any, Optional
from functools import partial
from concurrent import futures

import ksana_llm
import yaml
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import GenerationConfig, AutoTokenizer, PreTrainedTokenizerFast

model_executor = futures.ThreadPoolExecutor(max_workers=256)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
# pylint: disable=invalid-name
model = None
tokenizer = None
# pylint: enable=invalid-name


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


def streaming_generate(model_name, input_tokens, generation_config, **kwargs):
    """Perform streaming generation.
    """
    # Create a results iterator for the model's generation
    results_iterator = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        streamer=True,  # enable streaming generation
        **kwargs,
    )

    # Define an asynchronous function to stream the results
    async def stream_results():
        async for ksana_python_output in results_iterator:  # iterate over the results
            if ksana_python_output is None:
                return
            output_texts = []
            output_token_ids = []
            for request_output in ksana_python_output.output_tokens:
                # Decode the output tokens using the tokenizer
                request_output = request_output[len(input_tokens):]
                output_token_ids.append(request_output)
                try:
                    output_text = tokenizer.decode(
                        request_output,
                        skip_special_tokens=True  # skip special tokens
                    ).rstrip('\ufffd')
                except:
                    print("except occurred, invalid token ids:", input_tokens)
                    raise ValueError("Invalid token ids!")
                output_texts.append(output_text)
            ret = {
                "texts": output_texts,
                "output_token_ids": output_token_ids,
                "logprobs": ksana_python_output.logprobs,
                "input_token_ids": input_tokens
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")

    # Return a StreamingResponse object with the streamed results
    return stream_results()


def batch_generate(model_name, input_tokens, generation_config, **kwargs):
    """Perform batch generation.
    """
    # Generate output tokens using the model
    ksana_python_output = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        streamer=None,  # disable streaming generation
        **kwargs,
    )

    # Decode the output tokens into a human-readable text using the tokenizer
    output_text = []
    for tokens in ksana_python_output.output_tokens:
        try:
            output_text.append(tokenizer.decode(tokens, skip_special_tokens=True))
        except:
            print("except occurred, invalid token ids:", tokens)
            raise ValueError("Invalid token ids!")

    # Create a JSON response with the generated text and token IDs
    return {
        "texts": output_text,  # the generated text
        "output_token_ids": ksana_python_output.output_tokens,  # the generated token IDs
        "logprobs": ksana_python_output.logprobs,
        "input_token_ids": input_tokens  # the input token IDs
    }


def get_sampling_value(sampling_config: dict, key: str, default_val=None):
    """Get value from sampling_config dict, return default if key not exists.
    """
    return sampling_config[key] if key in sampling_config else default_val


async def process_request(request_dict: Dict[str, Any]) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model_name: the model your wanner to infer.
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    """

    model_name = request_dict.pop("model_name", "")
    prompt_text = request_dict.pop("prompt", None)
    enable_streaming = request_dict.pop("stream", True)
    sampling_config = request_dict.pop("sampling_config", None)

    input_refit_embedding = request_dict.pop("input_refit_embedding", None)

    input_tokens = request_dict.pop("input_tokens", None)
    if input_tokens is None:
        input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    kwargs = {
        "input_refit_embedding": {}
    }

    if input_refit_embedding is not None and "pos" in input_refit_embedding:
        kwargs['input_refit_embedding']["pos"] = input_refit_embedding["pos"]

    if input_refit_embedding is not None and "embeddings" in input_refit_embedding:
        kwargs['input_refit_embedding']["embeddings"] = input_refit_embedding["embeddings"]

    stop_token_ids = get_sampling_value(sampling_config, "stop_token_ids", [])
    ignore_eos = get_sampling_value(sampling_config, "ignore_eos", False)
    if (
        tokenizer.eos_token_id is not None
        and not ignore_eos
        and not tokenizer.eos_token_id in stop_token_ids
    ):
        stop_token_ids.append(tokenizer.eos_token_id)

    generation_config = GenerationConfig(
        top_k=get_sampling_value(sampling_config, "topk", 1),
        do_sample=get_sampling_value(sampling_config, "do_sample", None),
        top_p=get_sampling_value(sampling_config, "topp", 0.0),
        temperature=get_sampling_value(sampling_config, "temperature", 0.0),
        max_new_tokens=get_sampling_value(sampling_config, "max_new_tokens",
                                          -1),
        logprobs_num=get_sampling_value(sampling_config, "logprobs", 0),
        repetition_penalty=get_sampling_value(sampling_config,
                                              "repetition_penalty", 1.0),
        num_beams=get_sampling_value(sampling_config,
                                              "num_beams", 1),
        num_return_sequences=get_sampling_value(sampling_config,
                                              "num_return_sequences", 1),
        length_penalty=get_sampling_value(sampling_config,
                                              "length_penalty", 1.0),
        stop_token_ids=stop_token_ids,
        ignore_eos=ignore_eos
    )

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
                generation_config=generation_config,  # pass generation config as an argument
                **kwargs,
            )
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
                generation_config=generation_config,  # pass generation config as an argument
                **kwargs,
            )
        )

    # Return the results of the generation
    return results


async def forward_request(request_bytes: bytes) -> Optional[bytes]:
    """Forward the raw request bytes to the serving model.
    """

    # Get the current event loop
    loop = asyncio.get_event_loop()

    response_bytes = await loop.run_in_executor(
        model_executor,  # specify the executor to use
        partial(
            model.forward,  # partial function to call
            request_bytes  # pass request_bytes as an argument
        )
    )
    return response_bytes


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model_name: the model your wanner to infer.
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    """
    request_dict = await request.json()
    enable_streaming = request_dict.get("stream", True)
    response_data = await process_request(request_dict)
    if enable_streaming:
        return StreamingResponse(response_data)
    else:
        return JSONResponse(response_data)


@app.post("/forward")
async def forward(request: Request):
    """Generate next token for the request.

    The request should be a JSON object packed by msgpack.
    """
    request_bytes = await request.body()
    response_bytes = await forward_request(request_bytes)
    if response_bytes is not None:
        return Response(content=response_bytes, media_type="application/x-msgpack")
    else:  # Bad request
        return Response(status_code=status.HTTP_400_BAD_REQUEST, media_type="application/x-msgpack")


if __name__ == "__main__":
    args = args_config()
    if not args.tokenizer_dir:
        with open(args.config_file, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            args.tokenizer_dir = os.path.abspath(yaml_data["model_spec"]["base_model"]["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                              trust_remote_code=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        print(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    model = ksana_llm.AutoModel.from_config(args.config_file)

    # Use multithread to support parallelism.
    LOG_LEVEL = os.getenv("KLLM_LOG_LEVEL", "INFO").upper()
    if LOG_LEVEL in ["DEBUG", "INFO", "ERROR"]:
        LOG_LEVEL = LOG_LEVEL.lower()
    else:
        LOG_LEVEL = "info"
        print(
            f"Uvicorn's logging not support env: KLLM_LOG_LEVEL={LOG_LEVEL}, keep it as defalt(info)."
        )
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=LOG_LEVEL,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
