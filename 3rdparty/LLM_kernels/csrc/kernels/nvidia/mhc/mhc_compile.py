# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import math
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# The autotuner muse be first imported, otherwise the env will not take effect in matmul.
from autotuner import autotune

from mhc_kernel import (_mhc_pre_big_fuse, _mhc_pre_big_fuse_with_norm, _mhc_head_big_fuse)

import tilelang
import torch
from typing import Dict

import itertools


def build_mhc_pre_big_fuse(output_dir):

    def get_kernel_func(input_config, autotune_config):
        hidden_size = input_config['hidden_size']
        rms_eps = input_config['rms_eps']
        mhc_pre_eps = input_config['mhc_pre_eps']
        mhc_sinkhorn_eps = input_config['mhc_sinkhorn_eps']
        mhc_post_mult_value = input_config['mhc_post_mult_value']
        sinkhorn_repeat = input_config['sinkhorn_repeat']
        n_splits = input_config['n_splits']
        mhc_mult = input_config['mhc_mult']

        return _mhc_pre_big_fuse(hidden_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps, mhc_post_mult_value,
                                 sinkhorn_repeat, n_splits, mhc_mult)

    def get_kernel_input(input_config):
        hidden_size = input_config['hidden_size']
        rms_eps = input_config['rms_eps']
        mhc_pre_eps = input_config['mhc_pre_eps']
        mhc_sinkhorn_eps = input_config['mhc_sinkhorn_eps']
        mhc_post_mult_value = input_config['mhc_post_mult_value']
        sinkhorn_repeat = input_config['sinkhorn_repeat']
        n_splits = input_config['n_splits']
        mhc_mult = input_config['mhc_mult']

        n = 128
        mhc_mult3 = (2 + mhc_mult) * mhc_mult

        gemm_out_mul = torch.randn(n_splits, n, mhc_mult3, device="cuda", dtype=torch.float32)
        gemm_out_sqrsum = torch.randn(n_splits, n, device="cuda", dtype=torch.float32)
        mhc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
        mhc_base = torch.randn(mhc_mult3, device="cuda", dtype=torch.float32)
        residual = torch.randn(n, mhc_mult, hidden_size, device="cuda", dtype=torch.bfloat16)
        post_mix = torch.randn(n, mhc_mult, device="cuda", dtype=torch.float32)
        comb_mix = torch.randn(n, mhc_mult * mhc_mult, device="cuda", dtype=torch.float32)
        layer_input = torch.randn(n, hidden_size, device="cuda", dtype=torch.bfloat16)

        return (gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual, post_mix, comb_mix,
                layer_input)

    def get_kernel_name(kernel_name: str, input_config: Dict):
        return (
            f"{kernel_name}"
            f"_hidden_size_{input_config['hidden_size']}"
            f"_rms_eps_{input_config['rms_eps']:g}"
            f"_mhc_pre_eps_{input_config['mhc_pre_eps']:g}"
            f"_mhc_sinkhorn_eps_{input_config['mhc_sinkhorn_eps']:g}"
            f"_mhc_post_mult_value_{input_config['mhc_post_mult_value']:g}"
            f"_sinkhorn_repeat_{input_config['sinkhorn_repeat']}"
            f"_n_splits_{input_config['n_splits']}"
            f"_mhc_mult_{input_config['mhc_mult']}"
        )

    def get_input_configs():
        iter_params = dict(
            hidden_size=[4096, 7168],
            rms_eps=[1e-06],
            mhc_pre_eps=[1e-06],
            mhc_sinkhorn_eps=[1e-06],
            mhc_post_mult_value=[2.0],
            sinkhorn_repeat=[20],
            n_splits=[1,2,16,32,64],
            mhc_mult=[4],
        )
        return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

    def get_autotune_configs():
        return []

    pass_configs= {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    }

    autotune("mhc_pre_big_fuse_kernel",
             get_input_configs,
             get_autotune_configs,
             get_kernel_func,
             get_kernel_input,
             get_kernel_name,
             output_dir,
             target="cuda --arch=sm_90",
             verbose=True,
             pass_configs=pass_configs)


def build_mhc_pre_big_fuse_with_norm(output_dir):

    def get_kernel_func(input_config, autotune_config):
        hidden_size = input_config['hidden_size']
        rms_eps = input_config['rms_eps']
        mhc_pre_eps = input_config['mhc_pre_eps']
        mhc_sinkhorn_eps = input_config['mhc_sinkhorn_eps']
        mhc_post_mult_value = input_config['mhc_post_mult_value']
        sinkhorn_repeat = input_config['sinkhorn_repeat']
        n_splits = input_config['n_splits']
        mhc_mult = input_config['mhc_mult']

        return _mhc_pre_big_fuse_with_norm(hidden_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
                                           mhc_post_mult_value, sinkhorn_repeat,
                                           n_splits, mhc_mult)

    def get_kernel_input(input_config):
        hidden_size = input_config['hidden_size']
        rms_eps = input_config['rms_eps']
        mhc_pre_eps = input_config['mhc_pre_eps']
        mhc_sinkhorn_eps = input_config['mhc_sinkhorn_eps']
        mhc_post_mult_value = input_config['mhc_post_mult_value']
        sinkhorn_repeat = input_config['sinkhorn_repeat']
        n_splits = input_config['n_splits']
        mhc_mult = input_config['mhc_mult']

        n = 128
        mhc_mult3 = (2 + mhc_mult) * mhc_mult

        gemm_out_mul = torch.randn(n_splits, n, mhc_mult3, device="cuda", dtype=torch.float32)
        gemm_out_sqrsum = torch.randn(n_splits, n, device="cuda", dtype=torch.float32)
        mhc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
        mhc_base = torch.randn(mhc_mult3, device="cuda", dtype=torch.float32)
        residual = torch.randn(n, mhc_mult, hidden_size, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        post_mix = torch.randn(n, mhc_mult, device="cuda", dtype=torch.float32)
        comb_mix = torch.randn(n, mhc_mult * mhc_mult, device="cuda", dtype=torch.float32)
        layer_input = torch.randn(n, hidden_size, device="cuda", dtype=torch.bfloat16)

        return (gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual, norm_weight,
                post_mix, comb_mix, layer_input)

    def get_kernel_name(kernel_name: str, input_config: Dict):
        return (
            f"{kernel_name}"
            f"_hidden_size_{input_config['hidden_size']}"
            f"_rms_eps_{input_config['rms_eps']:g}"
            f"_mhc_pre_eps_{input_config['mhc_pre_eps']:g}"
            f"_mhc_sinkhorn_eps_{input_config['mhc_sinkhorn_eps']:g}"
            f"_mhc_post_mult_value_{input_config['mhc_post_mult_value']:g}"
            f"_sinkhorn_repeat_{input_config['sinkhorn_repeat']}"
            f"_n_splits_{input_config['n_splits']}"
            f"_mhc_mult_{input_config['mhc_mult']}"
        )

    def get_input_configs():
        iter_params = dict(
            hidden_size=[4096, 7168],
            rms_eps=[1e-06],
            mhc_pre_eps=[1e-06],
            mhc_sinkhorn_eps=[1e-06],
            mhc_post_mult_value=[2.0],
            sinkhorn_repeat=[20],
            n_splits=[1,2,16,32,64],
            mhc_mult=[4],
        )
        return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

    def get_autotune_configs():
        return []

    pass_configs= {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    }

    autotune("mhc_pre_big_fuse_with_norm_kernel",
             get_input_configs,
             get_autotune_configs,
             get_kernel_func,
             get_kernel_input,
             get_kernel_name,
             output_dir,
             target="cuda --arch=sm_90",
             verbose=True,
             pass_configs=pass_configs)


def build_mhc_head_big_fuse(output_dir):

    def get_kernel_func(input_config, autotune_config):
        hidden_size = input_config['hidden_size']
        rms_eps = input_config['rms_eps']
        mhc_pre_eps = input_config['mhc_pre_eps']
        n_splits = input_config['n_splits']
        mhc_mult = input_config['mhc_mult']

        return _mhc_head_big_fuse(hidden_size, rms_eps, mhc_pre_eps,
                                           n_splits, mhc_mult)

    def get_kernel_input(input_config):
        hidden_size = input_config['hidden_size']
        rms_eps = input_config['rms_eps']
        mhc_pre_eps = input_config['mhc_pre_eps']
        n_splits = input_config['n_splits']
        mhc_mult = input_config['mhc_mult']

        n = 128

        gemm_out_mul = torch.randn(n_splits, n, 2 * mhc_mult, device="cuda", dtype=torch.float32)
        gemm_out_sqrsum = torch.randn(n_splits, n, device="cuda", dtype=torch.float32)
        mhc_scale = torch.randn(1, device="cuda", dtype=torch.float32)
        mhc_base = torch.randn(mhc_mult, device="cuda", dtype=torch.float32)
        residual = torch.randn(n, mhc_mult, hidden_size, device="cuda", dtype=torch.bfloat16)
        layer_input = torch.randn(n, hidden_size, device="cuda", dtype=torch.bfloat16)

        return (gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual,
                layer_input)

    def get_kernel_name(kernel_name: str, input_config: Dict):
        return (
            f"{kernel_name}"
            f"_hidden_size_{input_config['hidden_size']}"
            f"_rms_eps_{input_config['rms_eps']:g}"
            f"_mhc_pre_eps_{input_config['mhc_pre_eps']:g}"
            f"_n_splits_{input_config['n_splits']}"
            f"_mhc_mult_{input_config['mhc_mult']}"
        )

    def get_input_configs():
        iter_params = dict(
            hidden_size=[4096, 7168],
            rms_eps=[1e-06],
            mhc_pre_eps=[1e-06],
            n_splits=[1,2,16,32,64],
            mhc_mult=[4],
        )
        return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

    def get_autotune_configs():
        return []

    pass_configs= {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    }

    autotune("mhc_head_big_fuse_kernel",
             get_input_configs,
             get_autotune_configs,
             get_kernel_func,
             get_kernel_input,
             get_kernel_name,
             output_dir,
             target="cuda --arch=sm_90",
             verbose=True,
             pass_configs=pass_configs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)
    output_dir = sys.argv[1]

    build_mhc_pre_big_fuse(output_dir)
    build_mhc_pre_big_fuse_with_norm(output_dir)
    build_mhc_head_big_fuse(output_dir)
