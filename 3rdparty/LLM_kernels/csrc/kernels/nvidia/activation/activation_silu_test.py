# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="inference type", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    inference_data_type = torch.float32
    if args.type == "half":
        inference_data_type = torch.float16
    elif args.type == "bfloat16":
        inference_data_type = torch.bfloat16

    # input
    input = torch.Tensor(
        np.load("silu_test_input.npy")).to(inference_data_type).cuda()
    gated_weight = torch.Tensor(
        np.load("silu_test_gated_weight.npy")).to(inference_data_type).cuda()

    silu_act = torch.nn.SiLU()

    output = silu_act(input) * gated_weight

    np.save("silu_test_output.npy", output.cpu().numpy())
