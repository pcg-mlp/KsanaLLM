# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import torch


def free_cache():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def load_safetensors(file_path: str):
    from safetensors.torch import load
    with open(file_path, "rb") as f:
        data = f.read()
    loaded = load(data)
    return loaded


def check_file_dir(file_path):
    import os
    file_dir = os.path.dirname(file_path)
    if not file_dir == '' and not os.path.exists(file_dir):
        os.makedirs(file_dir)


def get_module(module_name, py_path, class_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def adjust_device_memory_ratio(config_file: str, reserved_device_memory_ratio: float):
    """Adjust the memory ratio for multi-modal models
    """
    import re
    with open(config_file, "r") as yaml_file:
        yaml_data = yaml_file.read()
    # Overwrite the yaml config file
    # Use regular expressions to preserve the original order and comments
    match = re.search(r'(\s*reserved_device_memory_ratio:)(\s*\d+\.\d+|\d+)(.*)',
                      yaml_data)
    if match:
        yaml_data = re.sub(r'(\s*reserved_device_memory_ratio:)(\s*\d+\.\d+|\d+)(.*)',
                           f'{match.group(1)} {reserved_device_memory_ratio}{match.group(3)}',
                           yaml_data)
    with open(config_file, "w") as yaml_file:
        yaml_file.write(yaml_data)
