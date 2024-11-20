# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import asyncio
import os
import signal
import sys
from concurrent import futures
from functools import partial
from typing import Any, Dict, Optional

import orjson
import uvicorn
import uvloop
import yaml
from fastapi import FastAPI, Request
from fastapi import status as http_status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import (
   AutoProcessor, AutoTokenizer, LlamaTokenizer, VideoLlavaProcessor,
   GenerationConfig, PreTrainedTokenizerFast, logging
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config

import ksana_llm

model_executor = futures.ThreadPoolExecutor(max_workers=256)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
# pylint: disable=invalid-name
model = None
tokenizer = None
fields_to_extract = ['x-remote-ip', 'traceparent']


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
    parser.add_argument('--port', type=int, default=8080, help='server port')
    parser.add_argument("--endpoint",
                        type=str,
                        default=None,
                        help="server endpoint type (e.g., python/trpc). If not specified, "
                        "it will be inferred based on the config file.")
    parser.add_argument("--access-log",
                        action="store_true",
                        help="enable the endpoint access log")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--root-path",
                        type=str,
                        default=None,
                        help="FastAPI root_path when app is behind a path based routing proxy")
    args = parser.parse_args()

    if not args.tokenizer_dir:
        with open(args.config_file, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            args.tokenizer_dir = os.path.abspath(yaml_data["model_spec"]["base_model"]["model_dir"])
    if args.endpoint is None:
        with open(args.config_file, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            if "endpoint_type" in yaml_data["setting"]:
                args.endpoint = yaml_data["setting"]["endpoint_type"]
            else:  # Use Python endpoint by default
                args.endpoint = "python"
    # normalize the endpoint type
    args.endpoint = args.endpoint.lower()

    return args


def streaming_generate(model_name, input_tokens, generation_config, req_ctx, **kwargs):
    """Perform streaming generation.
    """
    # Create a results iterator for the model's generation
    status, results_iterator = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        streamer=True,  # enable streaming generation
        req_ctx = req_ctx,  # request trace context
        **kwargs,
    )

    if not status.OK():
        return status, None
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
            yield orjson.dumps(ret) + b"\0"

    # Return a StreamingResponse object with the streamed results
    return status, stream_results()


def batch_generate(model_name, input_tokens, generation_config, req_ctx, **kwargs):
    """Perform batch generation.
    """
    # Generate output tokens using the model
    status, ksana_python_output = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        streamer=None,  # disable streaming generation
        req_ctx = req_ctx,  # request trace context
        **kwargs,
    )
    if not status.OK():
        return status, None

    # Decode the output tokens into a human-readable text using the tokenizer
    output_text = []
    for tokens in ksana_python_output.output_tokens:
        try:
            output_text.append(tokenizer.decode(tokens, skip_special_tokens=True))
        except:
            print("except occurred, invalid token ids:", tokens)
            raise ValueError("Invalid token ids!")

    # Create a JSON response with the generated text and token IDs
    return status, {
        "texts": output_text,  # the generated text
        "output_token_ids": ksana_python_output.output_tokens,  # the generated token IDs
        "logprobs": ksana_python_output.logprobs,
        "input_token_ids": input_tokens  # the input token IDs
    }


def get_sampling_value(sampling_config: dict, key: str, default_val=None):
    """Get value from sampling_config dict, return default if key not exists.
    """
    return sampling_config[key] if key in sampling_config else default_val


def get_trace_context(request: Request) -> Dict[str, str]:
    return {
        field: request.headers.get(field) 
        for field in fields_to_extract 
        if request.headers.get(field) is not None
    }


async def process_request(request_dict: Dict[str, Any], req_ctx: Dict[str, str]) -> Response:
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
        "input_refit_embedding": {},
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
        no_repeat_ngram_size=get_sampling_value(sampling_config,
                                              "no_repeat_ngram_size", 0),
        encoder_no_repeat_ngram_size=get_sampling_value(sampling_config,
                                              "encoder_no_repeat_ngram_size", 0),
        num_beams=get_sampling_value(sampling_config,
                                              "num_beams", 1),
        num_return_sequences=get_sampling_value(sampling_config,
                                              "num_return_sequences", 1),
        length_penalty=get_sampling_value(sampling_config,
                                              "length_penalty", 1.0),
        stop_strings=get_sampling_value(sampling_config,
                                              "stop_strings", []),
        stop_token_ids=stop_token_ids,
        ignore_eos=ignore_eos
    )

    # Get the current event loop
    loop = asyncio.get_event_loop()

    # Determine whether to use streaming generation or batch generation
    if enable_streaming:
        # Use streaming generation
        # Run the streaming_generate function in an executor to avoid blocking the event loop
        status, results = await loop.run_in_executor(
            model_executor,  # specify the executor to use
            partial(
                streaming_generate,  # partial function to call
                model_name=model_name,  # pass model name as an argument
                input_tokens=input_tokens,  # pass input tokens as an argument
                generation_config=generation_config,  # pass generation config as an argument
                req_ctx = req_ctx,  # request trace context
                **kwargs,
            )
        )
    else:
        # Use batch generation
        # Run the batch_generate function in an executor to avoid blocking the event loop
        status, results = await loop.run_in_executor(
            model_executor,  # specify the executor to use
            partial(
                batch_generate,  # partial function to call
                model_name=model_name,  # pass model name as an argument
                input_tokens=input_tokens,  # pass input tokens as an argument
                generation_config=generation_config,  # pass generation config as an argument
                req_ctx = req_ctx,  # request trace context
                **kwargs,
            )
        )

    # Return the results of the generation
    return status, results


async def forward_request(request_bytes: bytes, req_ctx: Dict[str, str]) -> Optional[bytes]:
    """Forward the raw request bytes to the serving model.
    """

    # Get the current event loop
    loop = asyncio.get_event_loop()

    status, response_bytes = await loop.run_in_executor(
        model_executor,  # specify the executor to use
        partial(
            model.forward,  # partial function to call
            request_bytes,  # pass request_bytes as an argument
            req_ctx         # request trace context
        )
    )
    return status, response_bytes


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model_name: the model your wanner to infer.
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    """
    req_ctx = get_trace_context(request)
    request_dict = orjson.loads(await request.body())
    enable_streaming = request_dict.get("stream", True)
    status, response_data = await process_request(request_dict, req_ctx)
    if not status.OK():
        error_response = {"Message": status.GetMessage(), "code": status.GetCode().value}
        return Response(content = orjson.dumps(error_response),
             status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
    if enable_streaming:
        return StreamingResponse(response_data)
    else:
        return JSONResponse(response_data)


@app.post("/forward")
async def forward(request: Request):
    """Generate next token for the request.

    The request should be a JSON object packed by msgpack.
    """
    req_ctx = get_trace_context(request)
    request_bytes = await request.body()

    status, response_bytes = await forward_request(request_bytes, req_ctx)
    if status.OK() and response_bytes is not None:
        return Response(content=response_bytes, media_type="application/x-msgpack")
    else:  # Bad request
        error_response = {"Message": status.GetMessage(), "code": status.GetCode().value}
        return Response(content = orjson.dumps(error_response),
             status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


def load_tokenizer(model_path):
    tokenizer_config = get_tokenizer_config(model_path)
    if tokenizer_config.get("processor_class", "") == "VideoLlavaProcessor":
        return VideoLlavaProcessor.from_pretrained(model_path)
    if tokenizer_config.get("tokenizer_class", "") == "LlamaTokenizer":
        return LlamaTokenizer.from_pretrained(model_path)

    if os.path.exists(model_path + "/preprocessor_config.json"):
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if __name__ == "__main__":
    uvloop.install()
    args = args_config()

    # Initialize model serving based on configs.
    model = ksana_llm.AutoModel.from_config(args.config_file)
    endpoint_config = ksana_llm.EndpointConfig(args.endpoint, args.host,
                                               args.port, args.access_log)
    model.init_serving(endpoint_config)
    if args.endpoint != "python":
        signal.pause()
        sys.exit(0)

    # Set the verbosity of transformers to ERROR.
    logging.set_verbosity_error()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        print(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    # Set the log level of uvicorn based on KLLM_LOG_LEVEL.
    LOG_LEVEL = os.getenv("KLLM_LOG_LEVEL", "INFO").upper()
    if LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]:
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
                access_log=args.access_log,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
