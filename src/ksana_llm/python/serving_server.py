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
from transformers import GenerationConfig, AutoTokenizer, AutoConfig
from fastapi import FastAPI

import asyncio
from functools import partial
from concurrent import futures

model_executor = futures.ThreadPoolExecutor(max_workers=256)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
model = None
tokenizer = None
model_config = None


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
                output_text = tokenizer.decode(
                    request_output,
                    skip_special_tokens=True  # skip special tokens
                ).rstrip('\ufffd')
                output_texts.append(output_text)
            ret = {
                "texts": output_texts,
                "output_token_ids": output_token_ids,
                "prompt_probs": ksana_python_output.prompt_probs,
                "logprobs": ksana_python_output.logprobs,
                "input_token_ids": input_tokens
            }
            yield (json.dumps(ret) + "\0").encode("utf-8")

    # Return a StreamingResponse object with the streamed results
    return StreamingResponse(stream_results())


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
        output_text.append(tokenizer.decode(tokens, skip_special_tokens=True))

    # Create a JSON response with the generated text and token IDs
    return JSONResponse({
        "texts": output_text,  # the generated text
        "output_token_ids": ksana_python_output.output_tokens,  # the generated token IDs
        "logprobs": ksana_python_output.logprobs,
        "prompt_probs": ksana_python_output.prompt_probs,
        "input_token_ids": input_tokens  # the input token IDs
    })


def get_sampling_value(sampling_config: dict, key: str, default_val=None):
    """Get value from sampling_config dict, return default if key not exists.
    """
    return sampling_config[key] if key in sampling_config else default_val

def update_resources(input_tokens, kwargs):
    """Update parameters for special models
    """
    # Follow the preprocess from: https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/modeling_qwen.py#L554
    if model_config.model_type == "qwen" and "visual" in model_config.to_dict().keys():
        subinput_pos = [int(pos+1) for pos, ids in enumerate(input_tokens) if ids == model_config.visual["image_start_id"]]
        subinput_end = [int(pos-1) for pos, ids in enumerate(input_tokens) if ids == model_config.visual["image_start_id"]+1]

        # check token
        if len(subinput_pos) != len(subinput_end):
            raise RuntimeError(f"len(subinput_pos) != len(subinput_end), please check your prompt.")
        
        subinput_url = []
        for i in range(len(subinput_pos)):
            url = input_tokens[subinput_pos[i]:subinput_end[i]]
            url = url[:url.index(model_config.visual['image_start_id'] + 2)]
            subinput_url.append(bytes(url).decode('utf-8'))

        kwargs["subinput_pos"] = subinput_pos
        kwargs["subinput_url"] = subinput_url

    return kwargs

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

    subinput_pos = request_dict.pop("subinput_pos", None)
    subinput_embedding = request_dict.pop("subinput_embedding", None)
    subinput_url = request_dict.pop("subinput_url", None)
    prompt_probs_offset = request_dict.pop("prompt_probs_offset", None)

    input_tokens = request_dict.pop("input_tokens", None)
    if input_tokens is None:
       input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

    kwargs = {}
    if subinput_pos is not None:
        kwargs['subinput_pos'] = subinput_pos
    if subinput_embedding is not None:
        kwargs['subinput_embedding'] = subinput_embedding
    if subinput_url is not None:
        kwargs['subinput_url'] = subinput_url
    if prompt_probs_offset is not None:
        kwargs['prompt_probs_offset'] = prompt_probs_offset

    kwargs = update_resources(input_tokens, kwargs)

    generation_config = GenerationConfig(
        top_k=get_sampling_value(sampling_config, "topk", 1),
        do_sample=get_sampling_value(sampling_config, "topk", 1) != 1,
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
        stop_token_ids=get_sampling_value(sampling_config,
                                          "stop_token_ids", []),
        ignore_eos=get_sampling_value(sampling_config, "ignore_eos", False)
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
                generation_config=generation_config,
                **kwargs,
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
                generation_config=generation_config,
                **kwargs,
            )  # pass generation config as an argument
        )

    # Return the results of the generation
    return results


if __name__ == "__main__":
    args = args_config()
    if not args.tokenizer_dir:
        with open(args.config_file, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            args.tokenizer_dir = os.path.abspath(yaml_data["model_spec"]["base_model"]["model_dir"])
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
