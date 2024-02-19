# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import argparse
import configparser
import numpy as np
from pathlib import Path

import os
import torch
from transformers import LlamaConfig

# using numpy extension: https://github.com/GreenWaves-Technologies/bfloat16
# install the library with `pip install bfloat16`
from bfloat16 import bfloat16


LORA_WEIGHTS_NAME = "adapter_model.bin"
LORA_CONFIG_NAME = "adapter_config.json"


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    elif data_type == "bf16":
        return bfloat16
    else:
        assert False, f"Invalid weight data type {data_type}"


def split_and_convert_process(saved_dir, factor, key, val):
    if key.find("lora_A.weight") != -1:
        saved_path = saved_dir + "/" + key + ".bin"
        val.tofile(saved_path)
    elif key.find("lora_B.weight") != -1:
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/" + key + ".%d.bin" % j
            split_vals[j].tofile(saved_path)
    else:
        print("[ERROR] cannot find key '{}'".format(key))


def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert(i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load config from base.
    hf_config = LlamaConfig.from_pretrained(args.base_file).to_dict()
    print(f"hf_config: {hf_config}")

    lora_dir = args.in_file
    if os.path.exists(os.path.join(lora_dir, LORA_WEIGHTS_NAME)):
        filename = os.path.join(lora_dir, LORA_WEIGHTS_NAME)
    lora_weights = torch.load(filename)

    print("named parameters:")
    for name, _ in lora_weights.items():
        print(f"- {name}")

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]
    kv_head_num = head_num
    if "num_key_value_heads" in hf_config:
        kv_head_num = hf_config["num_key_value_heads"]
    assert(head_num % kv_head_num == 0)
    assert(kv_head_num % factor == 0)
    kv_head_rep_num = head_num // kv_head_num

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    def param_to_weights(param):
        return param.detach().cpu().numpy().astype(np_weight_data_type)

    # layer-wise weights, example:
    # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    # base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
    # base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight
    # base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight
    # base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight
    # base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight
    # base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight
    # base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight
    for l in range(num_layers):
        print(f"converting layer {l}")

        # first merge QKV into a single weight
        if kv_head_rep_num == 1:
            # concat direct to FT shape: [hidden_size, 3, head_num, head_size]
            # copied from huggingface_gptj_ckpt_convert.py
            lora_A_qkv_weights = np.stack(
                [
                    param_to_weights(
                        lora_weights[f'base_model.model.model.layers.{l}.self_attn.q_proj.lora_A.weight']),
                    param_to_weights(
                        lora_weights[f'base_model.model.model.layers.{l}.self_attn.k_proj.lora_A.weight']),
                    param_to_weights(
                        lora_weights[f'base_model.model.model.layers.{l}.self_attn.v_proj.lora_A.weight']),
                ])
            lora_A_qkv_weights = np.transpose(lora_A_qkv_weights, (2, 0, 1))

            lora_B_qkv_weights = np.stack(
                [
                    param_to_weights(
                        lora_weights[f'base_model.model.model.layers.{l}.self_attn.q_proj.lora_B.weight']),
                    param_to_weights(
                        lora_weights[f'base_model.model.model.layers.{l}.self_attn.k_proj.lora_B.weight']),
                    param_to_weights(
                        lora_weights[f'base_model.model.model.layers.{l}.self_attn.v_proj.lora_B.weight']),
                ])
            lora_B_qkv_weights = np.transpose(lora_B_qkv_weights, (2, 0, 1))
        else:
            # for GQA
            # concat to FT shape: [hidden_size, kv_head_num * (kv_head_rep_num + 2) * head_size]
            # according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
            # [head_num * head_size, hidden_size]->[factor, head_num // factor * head_size, hidden_size]
            lora_A_q_weight = param_to_weights(
                lora_weights[f'base_model.model.model.layers.{l}.self_attn.q_proj.lora_A.weight']).reshape(
                factor, head_num // factor * head_size, hidden_size)
            # [kv_head_num * head_size, hidden_size]->[factor, kv_head_num // factor * head_size, hidden_size]
            lora_A_k_weight = param_to_weights(
                lora_weights[f'base_model.model.model.layers.{l}.self_attn.k_proj.lora_A.weight']).reshape(
                factor, kv_head_num // factor * head_size, hidden_size)
            # [kv_head_num * head_size, hidden_size]->[factor, kv_head_num // factor * head_size, hidden_size]
            lora_A_v_weight = param_to_weights(
                lora_weights[f'base_model.model.model.layers.{l}.self_attn.v_proj.lora_A.weight']).reshape(
                factor, kv_head_num // factor * head_size, hidden_size)
            # [factor, head_num // factor * head_size + 2 * kv_head_num // factor * head_size, hidden_size]
            lora_A_qkv_weights = np.concatenate(
                (lora_A_q_weight, lora_A_k_weight, lora_A_v_weight), axis=1)
            # [hidden_size, factor * (head_num // factor * head_size + 2 * kv_head_num // factor * head_size)]
            lora_A_qkv_weights = np.transpose(
                lora_A_qkv_weights, (2, 0, 1)).reshape(
                hidden_size, -1)

            lora_B_q_weight = param_to_weights(
                lora_weights[f'base_model.model.model.layers.{l}.self_attn.q_proj.lora_B.weight']).reshape(
                factor, head_num // factor * head_size, hidden_size)
            # [kv_head_num * head_size, hidden_size]->[factor, kv_head_num // factor * head_size, hidden_size]
            lora_B_k_weight = param_to_weights(
                lora_weights[f'base_model.model.model.layers.{l}.self_attn.k_proj.lora_B.weight']).reshape(
                factor, kv_head_num // factor * head_size, hidden_size)
            # [kv_head_num * head_size, hidden_size]->[factor, kv_head_num // factor * head_size, hidden_size]
            lora_B_v_weight = param_to_weights(
                lora_weights[f'base_model.model.model.layers.{l}.self_attn.v_proj.lora_B.weight']).reshape(
                factor, kv_head_num // factor * head_size, hidden_size)
            # [factor, head_num // factor * head_size + 2 * kv_head_num // factor * head_size, hidden_size]
            lora_B_qkv_weights = np.concatenate(
                (lora_B_q_weight, lora_B_k_weight, lora_B_v_weight), axis=1)
            # [hidden_size, factor * (head_num // factor * head_size + 2 * kv_head_num // factor * head_size)]
            lora_B_qkv_weights = np.transpose(
                lora_B_qkv_weights, (2, 0, 1)).reshape(
                hidden_size, -1)

        lora_A_qkv_weights_base_name = f'base_model.model.model.layers.{l}.attention.query_key_value.lora_A.weight'
        split_and_convert_process(
            saved_dir,
            factor,
            lora_A_qkv_weights_base_name,
            lora_A_qkv_weights)

        lora_B_qkv_weights_base_name = f'base_model.model.model.layers.{l}.attention.query_key_value.lora_B.weight'
        split_and_convert_process(
            saved_dir,
            factor,
            lora_B_qkv_weights_base_name,
            lora_B_qkv_weights)

        # attention dense
        lora_A_o_weight = param_to_weights(
            lora_weights[f'base_model.model.model.layers.{l}.self_attn.o_proj.lora_A.weight']).T
        lora_A_o_weight_base_name = f'base_model.model.model.layers.{l}.attention.dense.lora_A.weight'
        split_and_convert_process(
            saved_dir, factor, lora_A_o_weight_base_name, lora_A_o_weight)

        lora_B_o_weight = param_to_weights(
            lora_weights[f'base_model.model.model.layers.{l}.self_attn.o_proj.lora_B.weight']).T
        lora_B_o_weight_base_name = f'base_model.model.model.layers.{l}.attention.dense.lora_B.weight'
        split_and_convert_process(
            saved_dir, factor, lora_B_o_weight_base_name, lora_B_o_weight)

        print(f"done layer {l}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir',
                        '-o',
                        type=str,
                        help='file name of output file',
                        required=True)
    parser.add_argument('-base_file',
                        '-b',
                        type=str,
                        help='file name of base model checkpoint',
                        required=True)
    parser.add_argument('-in_file',
                        '-i',
                        type=str,
                        help='file name of input checkpoint file',
                        required=True)
    parser.add_argument('-trained_gpu_num',
                        '-t_g',
                        type=int,
                        help='How many gpus for train',
                        default=1)
    parser.add_argument('-infer_gpu_num',
                        '-i_g',
                        type=int,
                        help='How many gpus for inference',
                        required=True)
    parser.add_argument("-weight_data_type",
                        type=str,
                        default="fp32",
                        choices=[
                            "fp32",
                            "fp16",
                            "bf16"])
    parser.add_argument('-model_name',
                        '-m_n',
                        type=str,
                        help='model name',
                        required=True)

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)
