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
    prompt_texts = json_request["prompts"]
    sampling_configs = json_request["sampling_configs"]

    generate_input_tokens = []
    for prompt_text in prompt_texts:
        input_token = tokenizer.encode(prompt_text, add_special_tokens=True)
        generate_input_tokens.append(input_token)

    generate_sampling_configs = []
    for sampling_config in sampling_configs:
        generation_config = GenerationConfig(
            num_beams=1,
            top_k=sampling_config["topk"],
            top_p=sampling_config["topp"],
            temperature=sampling_config["temperature"])
        generate_sampling_configs.append(generation_config)

    try:
        outputs = model.generate(
            model_name=model_name,
            inputs=generate_input_tokens,
            generation_configs=generate_sampling_configs)
    except Exception as e:
        return make_response(jsonify({"texts": str(e)}), 500)

    output_texts = []
    for output in outputs:
        output_text = tokenizer.decode(output)
        output_texts.append(output_text)

    return make_response(jsonify({"texts": output_texts}), 200)


# Use multithread to support parallelism.
app.run(host="0.0.0.0", port=8888, threaded=True)
