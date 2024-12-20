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
from typing import Any, Dict, List, Optional

import msgpack
import orjson
import uvicorn
import uvloop
import yaml
from fastapi import FastAPI, Request
from fastapi import status as http_status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import (
   AutoConfig, AutoProcessor, AutoTokenizer, LlamaTokenizer,
   VideoLlavaProcessor, GenerationConfig, PreTrainedTokenizerFast, logging
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
                        type=bool,
                        default=True,
                        help="enable the endpoint access log")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--root-path",
                        type=str,
                        default=None,
                        help="FastAPI root_path when app is behind a path based routing proxy")
    args = parser.parse_args()
    return args


def prepare_config():
    """ Automatically adjust the configuration.
    """
    # Parse the yaml config file
    with open(args.config_file, "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    args.model_dir = os.path.abspath(yaml_config["model_spec"]["base_model"]["model_dir"])
    if not args.tokenizer_dir:
        args.tokenizer_dir = args.model_dir

    args.plugin_model_enable_trt = True
    if ("plugin_model" in yaml_config["model_spec"] and
            "enable_tensorrt" in yaml_config["model_spec"]["plugin_model"]):
        args.plugin_model_enable_trt = \
            yaml_config["model_spec"]["plugin_model"]["enable_tensorrt"]

    if args.endpoint is None:
        # Use Python endpoint by default
        args.endpoint = yaml_config["setting"].get("endpoint_type", "python")
    # Normalize the endpoint type
    args.endpoint = args.endpoint.lower()

    # Parse the model json config
    model_config = AutoConfig.from_pretrained(args.model_dir,
                                              trust_remote_code=True).to_dict()
    args.model_type = model_config["model_type"]
    # Adjust the model type for qwen_vl
    if args.model_type == "qwen" and "visual" in model_config:
        args.model_type = "qwen_vl"


def streaming_generate(model_name, input_tokens, messages, generation_config, req_ctx, **kwargs):
    """Perform streaming generation.
    """
    # Create a results iterator for the model's generation
    status, results_iterator = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        messages=messages,  # pass the OpenAI chat messages
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
            input_token_ids = ksana_python_output.input_tokens
            output_texts = []
            output_token_ids = []
            for request_output in ksana_python_output.output_tokens:
                # Decode the output tokens using the tokenizer
                output_token_ids.append(request_output)
                try:
                    output_text = tokenizer.decode(
                        request_output,
                        skip_special_tokens=True  # skip special tokens
                    ).rstrip('\ufffd')
                except:
                    print("except occurred, invalid token ids:", input_token_ids)
                    raise ValueError("Invalid token ids!")
                output_texts.append(output_text)
            ret = {
                "texts": output_texts,
                "output_token_ids": output_token_ids,  # the output token IDs
                "logprobs": ksana_python_output.logprobs,
                "input_token_ids": input_token_ids  # the input token IDs
            }
            yield orjson.dumps(ret) + b"\0"

    # Return a StreamingResponse object with the streamed results
    return status, stream_results()


def batch_generate(model_name, input_tokens, messages, generation_config, req_ctx, **kwargs):
    """Perform batch generation.
    """
    # Generate output tokens using the model
    status, ksana_python_output = model.generate(
        model_name=model_name,  # specify the model name
        inputs=input_tokens,  # provide the input tokens
        generation_config=generation_config,  # configure the generation
        messages=messages,  # pass the OpenAI chat messages
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
        "input_token_ids": ksana_python_output.input_tokens  # the input token IDs
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
    prompt_text: Optional[str] = request_dict.pop("prompt", None)
    # `messages` is compatible with the OpenAI Chat Completion API.
    # Unlike `prompt`, it is a List[Dict] that can contain visual information,
    # e.g., image url and base64 encoded images.
    # See https://platform.openai.com/docs/guides/vision for details.
    messages: Optional[List[Dict]] = request_dict.pop("messages", None)
    enable_streaming = request_dict.pop("stream", True)
    sampling_config = request_dict.pop("sampling_config", None)
    structured_output_regex = request_dict.pop("structured_output_regex", None)
    input_tokens = request_dict.pop("input_tokens", None)
    if input_tokens is None and prompt_text is not None:
        input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    kwargs = {
        "input_refit_embedding": {},
        "additional_params": {},  # any model-specific parameters packed in a dict
        "prompt": prompt_text,
    }

    input_refit_embedding = request_dict.pop("input_refit_embedding", None)
    if input_refit_embedding is not None and "pos" in input_refit_embedding:
        kwargs['input_refit_embedding']["pos"] = input_refit_embedding["pos"]
    if input_refit_embedding is not None and "embeddings" in input_refit_embedding:
        kwargs['input_refit_embedding']["embeddings"] = input_refit_embedding["embeddings"]
    additional_params: Optional[Dict] = request_dict.pop("additional_params", None)
    if additional_params is not None:
        kwargs['additional_params'] = additional_params
    if structured_output_regex is not None:
        kwargs['structured_output_regex'] = structured_output_regex

    stop_token_ids = get_sampling_value(sampling_config, "stop_token_ids", [])
    ignore_eos = get_sampling_value(sampling_config, "ignore_eos", False)
    if (
        hasattr(tokenizer, "eos_token_id")
        and tokenizer.eos_token_id is not None
        and not ignore_eos
        and not tokenizer.eos_token_id in stop_token_ids
    ):
        stop_token_ids.append(tokenizer.eos_token_id)

    generation_config = GenerationConfig(
        top_k=get_sampling_value(sampling_config, "topk", 1),
        do_sample=get_sampling_value(sampling_config, "do_sample", None),
        top_p=get_sampling_value(sampling_config, "topp", 1.0),
        temperature=get_sampling_value(sampling_config, "temperature", 1.0),
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
                messages=messages,  # pass OpenAI chat messages as an argument
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
                messages=messages,  # pass OpenAI chat messages as an argument
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
    if not status.OK():  # Request failed
        error_response = {"message": status.GetMessage(), "code": status.GetCode().value}
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
    else:  # Request failed
        error_response = {"message": status.GetMessage(), "code": status.GetCode().value}
        return Response(content = msgpack.packb(error_response),
                        media_type="application/x-msgpack",
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
    # Set the verbosity of transformers to ERROR.
    logging.set_verbosity_error()

    args = args_config()
    prepare_config()

    # Initialize model serving based on configs.
    model = ksana_llm.AutoModel.from_config(args.config_file)
    plugin_config = ksana_llm.PluginConfig(args.model_dir, args.config_file,
                                           args.model_type, args.plugin_model_enable_trt)
    endpoint_config = ksana_llm.EndpointConfig(args.endpoint, args.host,
                                               args.port, args.access_log)
    model.init_serving(plugin_config, endpoint_config)
    if args.endpoint != "python":
        signal.pause()
        sys.exit(0)

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

    # distributed config.
    world_size = int(os.environ.get('WORLD_SIZE', "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))

    # For standalone or distributed master node, listen on server port.
    # For distributed worker node, wait until cluster destroyed.
    if world_size == 1 or node_rank == 0:
        app.root_path = args.root_path
        uvicorn.run(app,
                    host=args.host,
                    port=args.port,
                    log_level=LOG_LEVEL,
                    access_log=args.access_log,
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    ssl_keyfile=args.ssl_keyfile,
                    ssl_certfile=args.ssl_certfile)
    else:
        print("Uvicorn running on NONE.")
        import threading
        threading.Event().wait()
