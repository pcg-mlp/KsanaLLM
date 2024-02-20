# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import json
import uvicorn
import os

from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import GenerationConfig, LlamaTokenizer
from fastapi import FastAPI

import ksana_llm

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
model = None
tokenizer = None


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default="/model/llama-ft/13B/2-gpu",
                        help='model dir')
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


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model_name: the model your wanner to infer.
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """

    request_dict = await request.json()
    model_name = request_dict.pop("model_name")
    prompt_text = request_dict.pop("prompt")
    prefix_pos = request_dict.pop("prefix_pos", None)
    enable_streaming = request_dict.pop("stream", True)
    sampling_config = request_dict.pop("sampling_config", None)

    input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    generation_config = GenerationConfig(
        num_beams=1,
        top_k=sampling_config["topk"],
        top_p=sampling_config["topp"],
        temperature=sampling_config["temperature"])

    results_generator = model.generate(model_name=model_name,
                                       inputs=input_tokens,
                                       generation_config=generation_config,
                                       streamer=True)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            if request_output:
                ret = {"texts": tokenizer.decode([request_output])}
                yield (json.dumps(ret) + "\0").encode("utf-8")
            else:
                return

    if enable_streaming:
        return StreamingResponse(stream_results())

    output_text = tokenizer.decode(results_generator)
    return JSONResponse({"texts": output_text})


if __name__ == "__main__":
    args = args_config()
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    model = ksana_llm.AutoModel.from_pretrained(args.model_dir)

    # Use multithread to support parallelism.
    log_level = os.getenv("NLLM_LOG_LEVEL", "info")
    if log_level in ["DEBUG", "INFO"]:
        log_level = log_level.lower()
    else:
        log_level = "info"
        print(
            f"Not support env: NLLM_LOG_LEVEL={log_level}, keep it as defalt(info)."
        )
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
