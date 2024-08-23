# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)
parser.add_argument("--activation", help="activation type", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    activation = "silu"
    if args.activation == "gelu":
        activation = "gelu"
    elif args.activation == "relu":
        activation = "relu"

    # input
    input = torch.Tensor(
        np.load("activation_test_input.npy")).to(inference_data_type).cuda()
    gated_weight = torch.Tensor(
        np.load("activation_test_gated_weight.npy")).to(inference_data_type).cuda()

    if activation == "silu":
        act = torch.nn.SiLU()
    elif activation == "gelu":
        act = torch.nn.GELU()
    else:  # activation == "relu"
        act = torch.nn.ReLU()
    output = act(input) * gated_weight

    np.save("activation_test_output.npy", output.cpu().numpy())
