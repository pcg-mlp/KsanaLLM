# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import json
from glob import glob
import importlib.util

import torch
from transformers import AutoConfig
from safetensors.torch import load


def load_safetensors(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    loaded = load(data)
    return loaded


class KsanaPlugin:
    """
    Define a class named KsanaPlugin
    """

    # Plugin initialization is automatically invoked upon service startup.
    def init_plugin(self, **kwargs):
        if "preprocess" in kwargs:
            modelpath = kwargs["model_path"]

            # read config
            self.config = AutoConfig.from_pretrained(modelpath, trust_remote_code=True)
            precision = self.config.torch_dtype

            # get visual model
            spec = importlib.util.spec_from_file_location(modelpath, f'{modelpath}/visual.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            vision_transformer = getattr(module, "VisionTransformer")
            visual = vision_transformer(**self.config.visual)

            # read weight map
            weight_map_json = glob(os.path.join(modelpath, "*index.json"))
            assert len(weight_map_json) == 1
            with open(weight_map_json[0]) as file:
                weight_map_files = json.load(file)
            weight_map_files = weight_map_files["weight_map"]
            # get visual weight files
            filtered_values = {value for key, value in weight_map_files.items() if "transformer.visual" in key}
            weight_map_files = list(filtered_values)
            # read weight
            visual_weights = {}
            for weight_map_file in weight_map_files:
                weight_file = os.path.join(modelpath, weight_map_file)
                if os.path.splitext(weight_file)[1] == ".safetensors":
                    weights = load_safetensors(weight_file)
                else:
                    weights = torch.load(weight_file, map_location=torch.device('cpu'))
                for name, tensor in weights.items():
                    if "transformer.visual." in name:
                        visual_weights[name.replace("transformer.visual.", "")] = tensor

            # assign gpu and precision
            visual = visual.to(dtype=precision)
            visual.load_state_dict(visual_weights)
            visual = visual.to(device="cuda")

            self.visual = visual

            # free cache
            torch.cuda.empty_cache()

        if "postprocess" in kwargs:
            return

    def check_intput(self, **kwargs):
        input_list = [
            'ksana_python_input',
        ]
        for input_name in input_list:
            if input_name not in kwargs:
                print(f"input {input_name} not found.")
                return False

    # Method for pre-processing
    def preprocess(self, **kwargs):
        if self.check_intput(**kwargs):
            raise RuntimeError(f"Check input failed.")
        ksana_python_input = kwargs['ksana_python_input']

        input_tokens = ksana_python_input.input_tokens
        url_srt = [int(pos+1) for pos, ids in enumerate(input_tokens) if ids == self.config.visual["image_start_id"]]
        url_end = [int(pos-1) for pos, ids in enumerate(input_tokens) if ids == self.config.visual["image_start_id"]+1]
        image_url = []
        for i in range(len(url_srt)):
            url = input_tokens[url_srt[i]:url_end[i]]
            url = url[:url.index(self.config.visual['image_start_id']+2)]
            image_url.append(bytes(url).decode('utf-8'))

        if (len(image_url) == 0):
            return
        with torch.no_grad():
            image_embedding = self.visual.encode(image_url)

        ksana_python_input.input_refit_embedding.pos = url_srt
        ksana_python_input.input_refit_embedding.embedding_tensors = torch.unbind(image_embedding.cpu().float())

    # Method for post-processing
    def postprocess(self, **kwargs):
        return
