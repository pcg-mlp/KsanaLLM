# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import ksana_llm
import argparse
import json
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
                        default="/model/llama-hf/13B",
                        help='tokenizer dir')
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
    """Do the streaming generate.
    """
    results_iterator = model.generate(model_name=model_name,
                                      inputs=input_tokens,
                                      generation_config=generation_config,
                                      streamer=True)

    def stream_results():
        unfinished_token = []
        for request_output in results_iterator:
            if request_output is not None:
                output_text = tokenizer.decode(
                    unfinished_token + [request_output], skip_special_tokens=True)
                if output_text[-1:] == "\uFFFD":
                    unfinished_token = unfinished_token + [request_output]
                    output_text = ""
                else:
                    unfinished_token = []
                ret = {"texts": output_text}
                yield (json.dumps(ret) + "\0").encode("utf-8")
            else:
                return

    return StreamingResponse(stream_results())


def batch_generate(model_name, input_tokens, generation_config):
    """Do the batch generate.
    """
    results_tokens = model.generate(model_name=model_name,
                                    inputs=input_tokens,
                                    generation_config=generation_config,
                                    streamer=None)

    output_text = tokenizer.decode(results_tokens, skip_special_tokens=True)
    return JSONResponse(
        {"texts": output_text, "output_token_ids": results_tokens})


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model_name: the model your wanner to infer.
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    """

    request_dict = await request.json()
    model_name = request_dict.pop("model_name")
    prompt_text = request_dict.pop("prompt")
    enable_streaming = request_dict.pop("stream", True)
    sampling_config = request_dict.pop("sampling_config", None)

    input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    generation_config = GenerationConfig(
        num_beams=1,
        top_k=sampling_config["topk"],
        top_p=sampling_config["topp"],
        temperature=sampling_config["temperature"])

    loop = asyncio.get_event_loop()
    if enable_streaming:
        results = await loop.run_in_executor(
            model_executor,
            partial(streaming_generate,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    generation_config=generation_config))
    else:
        results = await loop.run_in_executor(
            model_executor,
            partial(batch_generate,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    generation_config=generation_config))
    return results


if __name__ == "__main__":
    args = args_config()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
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
