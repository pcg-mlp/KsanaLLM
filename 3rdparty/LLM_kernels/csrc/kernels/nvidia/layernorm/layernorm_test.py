# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--variance_epsilon", help="variance epsilon", type=float)

if __name__ == "__main__":
    args = parser.parse_args()

    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    input = torch.Tensor(
        np.load("layernorm_test_input.npy")).to(inference_data_type).cuda()
    weight = torch.Tensor(
        np.load("layernorm_test_weight.npy")).to(inference_data_type).cuda()
    bias = torch.Tensor(
        np.load("layernorm_test_bias.npy")).to(inference_data_type).cuda()

    layernorm = torch.nn.LayerNorm(normalized_shape=input.shape[1], eps=args.variance_epsilon)
    layernorm.weight.data = weight
    layernorm.bias.data = bias
    layernorm_output = layernorm(input)

    np.save("layernorm_test_output.npy", layernorm_output.cpu().detach().numpy())

    input_dtype = input.dtype
    hidden_states = input.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + args.variance_epsilon)
    rmsnorm_output = weight * hidden_states.to(input_dtype)

    np.save("rmsnorm_test_output.npy", rmsnorm_output.cpu().numpy())
