# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import json

from transformers import GenerationConfig
from transformers import LlamaTokenizer
from flask import Flask, request, jsonify, make_response

import numerous_llm

app = Flask('numerous-llm')


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
    args = parser.parse_args()
    return args


@app.route('/generate', methods=["POST"])
def generate():
    http_body = request.get_data().decode()
    json_request = json.loads(http_body)

    model_name = json_request["model_name"]
    prompt_text = json_request["prompt"]
    sampling_config = json_request["sampling_config"]

    tokenizer = app.config["tokenizer"]
    model = app.config["model"]

    input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    generation_config = GenerationConfig(
        num_beams=1,
        top_k=sampling_config["topk"],
        top_p=sampling_config["topp"],
        temperature=sampling_config["temperature"])

    try:
        enable_streaming = True
        if enable_streaming:
            streaming_iter = model.generate(
                model_name=model_name,
                inputs=input_tokens,
                generation_config=generation_config,
                streamer=True)
            output_tokens = []
            for output_token in streaming_iter:
                # print("Get step token:", output_token, flush=True)
                output_tokens.append(output_token)

        else:
            output_tokens = model.generate(model_name=model_name,
                                           inputs=input_tokens,
                                           generation_config=generation_config)
    except Exception as exc:  # pylint: disable=W0718
        return make_response(jsonify({"texts": str(exc)}), 500)

    output_text = tokenizer.decode(output_tokens)
    return make_response(jsonify({"texts": output_text}), 200)


if __name__ == "__main__":
    args = args_config()
    app.config["tokenizer_dir"] = args.tokenizer_dir
    app.config["tokenizer"] = LlamaTokenizer.from_pretrained(
        app.config["tokenizer_dir"])
    app.config["model_dir"] = args.model_dir
    app.config["model"] = numerous_llm.AutoModel.from_pretrained(
        app.config["model_dir"])

    # Use multithread to support parallelism.
    app.run(host=args.host, port=args.port, threaded=True)
