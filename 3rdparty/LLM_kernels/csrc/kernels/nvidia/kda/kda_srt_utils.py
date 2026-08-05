# Adapted from
# sglang/python/sglang/srt/utils (subset used by fused_norm_gate)
#
# Copyright 2026 Tencent Inc.  All rights reserved.

import triton


def cdiv(a, b):
    return triton.cdiv(a, b)


def next_power_of_2(n):
    return triton.next_power_of_2(n)


def is_cpu():
    return False


def is_npu():
    return False


def cpu_has_amx_support():
    return False
