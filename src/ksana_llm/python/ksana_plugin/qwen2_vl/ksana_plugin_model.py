# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import json
import os
import sys

from glob import glob
from transformers import AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

import torch

# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from plugin_utils import free_cache, load_safetensors


class VITModel:

    def __init__(self, model_path):
        # read config
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.precision = self.config.torch_dtype

        # Initialize the model device, assume on GPU
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def get_model(self, model_path, precision=None):
        if precision is None:
            precision = self.precision

        # init vit model
        visual = Qwen2VisionTransformerPretrainedModel._from_config(self.config.vision_config,
                                                                    attn_implementation="flash_attention_2",
                                                                    torch_dtype=precision)

        # read weight map
        weight_map_json = glob(os.path.join(model_path, "*index.json"))
        assert len(weight_map_json) == 1
        with open(weight_map_json[0]) as file:
            weight_map_files = json.load(file)
        weight_map_files = weight_map_files["weight_map"]
        # get visual weight files
        filtered_values = {value for key, value in weight_map_files.items() if "visual." in key}
        weight_map_files = list(filtered_values)
        # read weight
        visual_weights = {}
        for weight_map_file in weight_map_files:
            weight_file = os.path.join(model_path, weight_map_file)
            if os.path.splitext(weight_file)[1] == ".safetensors":
                weights = load_safetensors(weight_file)
            else:
                weights = torch.load(weight_file, map_location=torch.device('cpu'))
            for name, tensor in weights.items():
                if "visual." in name:
                    visual_weights[name.replace("visual.", "")] = tensor

        # assign gpu and precision
        visual = visual.to(dtype=precision)
        visual.load_state_dict(visual_weights)
        visual = visual.to(device=self.device)

        free_cache()
        return visual
