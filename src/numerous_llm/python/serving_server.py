# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
import json
import torch

import numerous_llm

from transformers import GenerationConfig
from transformers import AutoTokenizer, LlamaTokenizer
from flask import Flask, request, jsonify, make_response

app = Flask('numerous-llm')

# Autokenizer is very slow.
tokenizer_dir = "/model/llama-hf/13B"
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)

model_dir = "/model/llama-ft/13B/2-gpu"
model = numerous_llm.AutoModel.from_pretrained(model_dir)


@app.route('/generate', methods=["POST"])
def generate():
    http_body = request.get_data().decode()
    json_request = json.loads(http_body)

    model_name = json_request["model_name"]
    prompt_text = json_request["prompt"]
    sampling_config = json_request["sampling_config"]

    input_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    generation_config = GenerationConfig(
        num_beams=1,
        top_k=sampling_config["topk"],
        top_p=sampling_config["topp"],
        temperature=sampling_config["temperature"])

    try:
        output_tokens = model.generate(model_name=model_name,
                                       inputs=input_tokens,
                                       generation_config=generation_config)
    except Exception as e:
        return make_response(jsonify({"texts": str(e)}), 500)

    output_text = tokenizer.decode(output_tokens)
    return make_response(jsonify({"texts": output_text}), 200)


# Use multithread to support parallelism.
app.run(host="0.0.0.0", port=8888, threaded=True)
