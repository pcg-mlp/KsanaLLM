# Copyright 2024 Tencent Inc.  All rights reserved.

import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--layer_idx", help="layer idx of lora weight", type=int)
parser.add_argument("--scale", help="scale factor", type=float)

# reference compute from https://github.com/vllm-project/vllm/blob/891070257c145b506a20666a3cb70afcf674d4ca/tests/lora/test_punica.py
def lora_ref_impl(
    y_final: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indices: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    y_stage_1 = torch.empty(
        (x.size(0), wa_T_all.size(-2)),
        dtype=torch.float32,
        device=x.device,
    )
    bs = x.shape[0]
    s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    for i, lora_idx in zip(range(bs), indices.cpu().tolist()):
        xi = x[i].unsqueeze(0).to(torch.float32)

        wa = wa_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)
        wb = wb_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)

        tmp = xi @ wa
        y_stage_1[i] = tmp.squeeze(0)

        y_final[i] += (tmp @ wb).squeeze(0) * s
    return y_final, y_stage_1

if __name__ == "__main__":
    args = parser.parse_args()
    # output
    y_ref = torch.Tensor(np.load("y_ref.npy")).to(torch.float16).cuda()
    
    # input
    wa_T_all = torch.Tensor(np.load("wa_T_all.npy")).to(torch.float16).cuda()
    wb_T_all = torch.Tensor(np.load("wb_T_all.npy")).to(torch.float16).cuda()
    x = torch.Tensor(np.load("x.npy")).to(torch.float16).cuda()
    indices = torch.Tensor(np.load("indices.npy")).to(torch.long).cuda()

    y_final, y_stage_1 = lora_ref_impl(y_ref, x, wa_T_all, wb_T_all, indices, args.layer_idx, args.scale)

    np.save("y_ref.npy", y_ref.cpu().numpy())
    np.save("y_stage_1.npy", y_stage_1.cpu().numpy())