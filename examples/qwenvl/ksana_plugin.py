# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import json
from glob import glob
import importlib.util

import torch
from transformers import AutoConfig

class KsanaPlugin:
    """
    Define a class named KsanaPlugin
    """

    # Plugin initialization is automatically invoked upon service startup.
    def init_plugin(self, **kwargs):
        if "preprocess" in kwargs:
            modelpath = kwargs["model_path"]

            # read config
            config = AutoConfig.from_pretrained(modelpath, trust_remote_code=True)
            precision = config.torch_dtype

            # get visual model
            spec = importlib.util.spec_from_file_location(modelpath, f'{modelpath}/visual.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            VisionTransformer = getattr(module, "VisionTransformer")
            visual = VisionTransformer(**config.visual)

            # read weight map
            weight_map_json = glob(os.path.join(modelpath, "*index.json"))
            assert len(weight_map_json) == 1
            with open(weight_map_json[0]) as file:
                weight_map_files = json.load(file)
            weight_map_files = weight_map_files["weight_map"]
            # get visual weight files
            weight_map_files = list(set([value for key, value in weight_map_files.items() if "transformer.visual" in key]))
            # read weight
            visual_weights = {}
            for weight_map_file in weight_map_files:
                weights = torch.load(os.path.join(modelpath,weight_map_file), map_location=torch.device('cpu'))
                for name, tensor in weights.items():
                    if "transformer.visual." in name:
                        visual_weights[name.replace("transformer.visual.","")] = tensor

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

        image_url = ksana_python_input.subinput_url
        if (len(image_url) == 0):
            return
        with torch.no_grad():
            image_embedding = self.visual.encode(image_url)
        ksana_python_input.subinput_embedding_tensors = torch.split(image_embedding.cpu(), image_embedding.shape[0])

    # Method for post-processing
    def postprocess(self, **kwargs):
        return
