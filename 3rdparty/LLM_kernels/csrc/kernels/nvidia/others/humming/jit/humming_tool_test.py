#!/usr/bin/env python3
# Copyright 2026 Tencent Inc. All rights reserved.
#
# Tests for humming_tool.py generate / generate-batch subcommands.
# Run: python3 humming_tool_test.py

import json
import os
import subprocess
import sys
import tempfile

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humming_tool.py")

VALID = {
    "size_n": 1024, "size_k": 4096, "num_experts": 256,
    "compute_dtype": 1, "gemm_type": 1,
    "block_shape": [64, 128, 64], "warp_shape": [64, 64, 64],
    "use_stream_k": True, "num_stages": 3, "num_ctas_per_sm": 1,
    "use_tma_a": True, "use_tma_b": True, "use_tma_c": True, "use_tma_bs": True,
    "use_mbarrier": True, "use_warp_spec": False, "use_f16_accum": False,
}


def test_generate_single_still_works():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "k.cu")
        proc = subprocess.run(
            ["python3", TOOL, "generate",
             "--size_n", "1024", "--size_k", "4096", "--num_experts", "256",
             "--compute_dtype", "1", "--gemm_type", "1",
             "--block_m", "64", "--block_n", "128", "--block_k", "64",
             "--warp_shape_m", "64", "--warp_shape_n", "64", "--warp_shape_k", "64",
             "--use_stream_k", "1", "--num_stages", "3", "--num_ctas_per_sm", "1",
             "--use_tma_a", "1", "--use_tma_b", "1", "--use_tma_c", "1", "--use_tma_bs", "1",
             "--use_mbarrier", "1", "--use_warp_spec", "0", "--use_f16_accum", "0",
             "--output", out],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        src = open(out).read()
        assert "humming" in src and "__humming_kernel_ptr" in src
    print("generate (single) PASSED")


def test_generate_batch():
    with tempfile.TemporaryDirectory() as tmp:
        ok_out = os.path.join(tmp, "ok.cu")
        bad_out = os.path.join(tmp, "bad.cu")
        # 预置陈旧文件,验证原子替换语义(不出现半写状态)
        with open(ok_out, "w") as f:
            f.write("stale")
        manifest = os.path.join(tmp, "manifest.json")
        # 坏 config: 缺 num_stages 键 → 渲染抛 KeyError → 记入 failed,不影响其它 config
        bad_entry = {k: v for k, v in VALID.items() if k != "num_stages"}
        with open(manifest, "w") as f:
            json.dump({"kernels": [dict(VALID, output=ok_out),
                                   dict(bad_entry, output=bad_out)]}, f)

        proc = subprocess.run(["python3", TOOL, "generate-batch", "--manifest", manifest],
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        summary = json.loads(proc.stdout)  # stdout 必须是纯 JSON
        assert ok_out in summary["generated"], summary
        assert any(e["output"] == bad_out for e in summary["failed"]), summary
        src = open(ok_out).read()
        assert "stale" not in src and "__humming_kernel_ptr" in src
        # tmp 文件带 pid 后缀(子进程 pid 与本进程不同),用前缀匹配断言无残留
        assert not [f for f in os.listdir(tmp) if f.startswith("ok.cu.tmp")]
    print("generate-batch PASSED")


if __name__ == "__main__":
    test_generate_batch()
    test_generate_single_still_works()
