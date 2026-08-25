#!/usr/bin/env python3
# Copyright 2026 Tencent Inc. All rights reserved.
#
"""GDN cubin 一键生成。用法: python gen_gdn_cubins.py [all|list|<model>|ensure]"""
import glob
import hashlib
import os
import subprocess as sp
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

GDN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gdn/ dir


# Output dir must match triton_wrapper.cpp InitKernelsDir().
def _resolve_out_dir():
    env_path = os.environ.get("KSANA_TRITON_KERNEL_PATH")
    if env_path:
        return os.path.join(env_path, "triton_kernel_files")
    home_dir = os.environ.get("HOME")
    if home_dir:
        return os.path.join(home_dir, ".cache/KsanaLLM/triton_kernel_files")
    return "./triton_kernel_files"


OUT = _resolve_out_dir()
PY = sys.executable
# Qwen3.5/3.6 GDN 只跑 BF16；FP16 无调用方，不生成。
DTYPES = ("BF16",)

CONFIGS = {
    "0.8B_tp1": {"H_key": 16, "HV": 16, "D": 6144},
    "0.8B_tp2": {"H_key":  8, "HV":  8, "D": 3072},
    "27B_tp2":  {"H_key":  8, "HV": 24, "D": 5120},
    "27B_tp1":  {"H_key": 16, "HV": 48, "D": 10240},
    "35B_tp2":  {"H_key":  8, "HV": 16, "D": 4096},
}

def _np2_statelen(sl):
    """next_power_of_2(sl); must match C++ NextPowerOfTwo."""
    return 1 << (sl - 1).bit_length()


def run(cmd, gpu_id, quiet=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if not quiet:
        print(f"[GPU {gpu_id}] {cmd}")
    r = sp.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"FAIL [GPU {gpu_id}]: {r.stderr[-200:]}")  # errors always shown
    elif not quiet:
        out = r.stdout.strip().split("\n")
        print(out[-1] if out else "OK")
    return r.returncode == 0


# ---------------------------------------------------------------------------
# 命令 + 产物路径成对构建。cubin 文件名模板必须与 cubin_gen/*.py 的
# kernel_name 一致（每族一个生成器；改生成器命名时必须同步这里）。
# ---------------------------------------------------------------------------

def _path(*parts):
    return os.path.join(OUT, "".join(parts))


def _universal_cmds(dts=None):
    """与模型 config 无关的通用 cubin（rmsnorm_gated）。dts 过滤 dtype（默认全量）。"""
    cmds = []
    for dt in DTYPES:
        if dts and dt not in dts:
            continue
        # 仅 W4（launch test 与生产 prefill/decode 共用此配置；W1 无调用方）。
        name = (f"rmsnorm_gated_kernel_N_128_BN_128_R_4_W_4_S_3_"
                f"HB_0_HZ_1_NBG_1_RMS_1_ACT_swish_compute_type_{dt}.cubin")
        cmds.append((
            f"{PY} {GDN}/cubin_gen/rmsnorm_gated.py --output_dir {OUT} "
            f"--N 128 --BLOCK_N 128 --ROWS_PER_BLOCK 4 --num_warps 4 "
            f"--num_stages 3 --HAS_BIAS False --HAS_Z True --NORM_BEFORE_GATE True "
            f"--IS_RMS_NORM True --ACTIVATION swish --compute_type {dt}",
            _path(name)))
    return cmds


def _model_cmds(H_key, HV, D, mtp, dts=None):
    """模型相关 cubin（chunk 族 + GQA 族 + conv1d）。mtp>0 时含 spec 变体。dts 过滤 dtype。"""
    cmds = []
    for dt in DTYPES:
        if dts and dt not in dts:
            continue
        # chunk 族：H=HV, Hg=H_key 原生 GQA（kernel 用 i_h//(H//Hg) 索引 q/k，
        # 无需显式 repeat_interleave）。chunk_h 文件名带 gdn_ 前缀：与 KDA 集
        # 合同名 kernel 的 cubin 命名空间隔离。
        chunk_h = (
            f"gdn_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_H_{HV}_Hg_"
            f"{H_key}_K_128_V_128_BT_64_BV_64_G_1_GK_0_INIT_1_STORE_1_SAVE_"
            f"1_VL_1_W_4_S_3_H0OFF_1_compute_type_{dt}.cubin"
        )
        cmds.append((f"{PY} {GDN}/cubin_gen/chunk_h.py --output_dir {OUT} "
                     f"--H {HV} --Hg {H_key} --K 128 --V 128 --BT 64 --BV 64 --USE_G True "
                     f"--USE_GK False --USE_INITIAL_STATE True --STORE_FINAL_STATE True "
                     f"--SAVE_NEW_VALUE True --IS_VARLEN True --num_warps 4 --num_stages 3 "
                     f"--compute_type {dt}", _path(chunk_h)))
        chunk_o = (
            f"chunk_fwd_kernel_o_H_{HV}_Hg_{H_key}_K_128_V_128_BT_64_BK_64_"
            f"BV_64_G_1_VL_1_W_4_S_3_compute_type_{dt}.cubin"
        )
        cmds.append((f"{PY} {GDN}/cubin_gen/chunk_o.py --output_dir {OUT} "
                     f"--H {HV} --Hg {H_key} --K 128 --V 128 --BT 64 --BK 64 --BV 64 "
                     f"--USE_G True --IS_VARLEN True --num_warps 4 --num_stages 3 "
                     f"--compute_type {dt}", _path(chunk_o)))
        for use_g, gval in ((True, 1),):
            kkt = (
                f"chunk_scaled_dot_kkt_fwd_kernel_H_{HV}_Hg_{H_key}_K_128_BT_"
                f"64_BK_64_VL_1_G_{gval}_W_4_S_3_compute_type_{dt}.cubin"
            )
            cmds.append((f"{PY} {GDN}/cubin_gen/chunk_scaled_dot_kkt.py --output_dir {OUT} "
                         f"--H {HV} --Hg {H_key} --K 128 --BT 64 --BK 64 --USE_G {use_g} "
                         f"--IS_VARLEN True --num_warps 4 --num_stages 3 --compute_type {dt}",
                         _path(kkt)))
        recompute = (
            f"gdn_recompute_w_u_fwd_kernel_H_{HV}_Hg_{H_key}_K_128_V_128_"
            f"BT_64_BK_64_BV_64_VL_1_W_4_S_3_compute_type_{dt}.cubin"
        )
        cmds.append((f"{PY} {GDN}/cubin_gen/wy_fast_recompute.py --output_dir {OUT} "
                     f"--H {HV} --Hg {H_key} --K 128 --V 128 --BT 64 --BK 64 --BV 64 "
                     f"--IS_VARLEN True --num_warps 4 --num_stages 3 --compute_type {dt}",
                     _path(recompute)))
        cumsum = f"chunk_local_cumsum_scalar_kernel_B_1_H_{HV}_BT_64_REV_0_VL_1_HF_0_W_4_S_1_compute_type_{dt}.cubin"
        cmds.append((f"{PY} {GDN}/cubin_gen/chunk_local_cumsum_scalar.py --output_dir {OUT} "
                     f"--B 1 --H {HV} --BT 64 --HEAD_FIRST False --REVERSE False "
                     f"--IS_VARLEN True --num_warps 4 --num_stages 1 --compute_type {dt}",
                     _path(cumsum)))
        tril = f"merge_16x16_to_64x64_inverse_kernel_H_{HV}_BT_64_TMA_0_VL_1_DP_ieee_W_4_S_3_compute_type_{dt}.cubin"
        cmds.append((f"{PY} {GDN}/cubin_gen/solve_tril_64.py --output_dir {OUT} "
                     f"--H {HV} --BT 64 --USE_TMA False --IS_VARLEN True "
                     f"--DOT_PRECISION ieee --num_warps 4 --num_stages 3 --compute_type {dt}",
                     _path(tril)))

        # GQA 族（H=H_key, HV=HV）
        post_conv = (
            f"fused_post_conv_kernel_H_{H_key}_HV_{HV}_K_128_V_128_BK_128_"
            f"BV_128_BT_16_L2_1_GEXP_0_W_4_S_2_compute_type_{dt}.cubin"
        )
        cmds.append((f"{PY} {GDN}/cubin_gen/fused_post_conv.py --output_dir {OUT} "
                     f"--H {H_key} --HV {HV} --K 128 --V 128 --BK 128 --BV 128 "
                     f"--BLOCK_T 16 --APPLY_L2NORM True --L2NORM_EPS 1e-6 "
                     f"--OUTPUT_G_EXP False --SOFTPLUS_THRESHOLD 20.0 "
                     f"--num_warps 4 --num_stages 2 --compute_type {dt}", _path(post_conv)))
        # fused conv+post：主 kernel NSEQ=0（prior/pool 由 boundary 串行修正）
        # + boundary kernel。文件名模板与 triton_wrapper.cpp kernel_key_map_
        # 两新条目一致（键序勿动）。
        c1_main = (f"fused_conv_post_c1_kernel_H_{H_key}_HV_{HV}_K_128_V_128_BT_16_L2_1_"
                   f"GEXP_0_NSEQ_0_W_4_S_2_compute_type_{dt}.cubin")
        cmds.append((f"{PY} {GDN}/cubin_gen/fused_conv_post_c1.py --output_dir {OUT} "
                     f"--H {H_key} --HV {HV} --K 128 --V 128 --BLOCK_T 16 --NSEQ 0 "
                     f"--num_warps 4 --num_stages 2 --compute_type {dt}", _path(c1_main)))
        c1_bnd = (f"fused_conv_post_c1_boundary_kernel_H_{H_key}_HV_{HV}_K_128_V_128_BT_8_"
                  f"L2_1_W_4_S_1_compute_type_{dt}.cubin")
        cmds.append((f"{PY} {GDN}/cubin_gen/fused_conv_post_c1.py --output_dir {OUT} "
                     f"--H {H_key} --HV {HV} --K 128 --V 128 --BLOCK_T 8 --boundary "
                     f"--num_warps 4 --num_stages 1 --compute_type {dt}", _path(c1_bnd)))
        # packed_decode: SMQ = conv_dim = 2*H*K + HV*V（与生成器 dummy 的
        # mixed_qkv.stride(0) 一致）；SA/SB = 2*HV（in_proj ba 合并后 a/b 为
        # interleaved 布局，dummy 也按 --interleaved_ba 构造）。
        smq = 2 * H_key * 128 + HV * 128
        pd = (
            f"fused_recurrent_gated_delta_rule_packed_decode_kernel_SMQ_"
            f"{smq}_SA_{2 * HV}_SB_{2 * HV}_SISQ_1_H_{H_key}_HV_{HV}_K_128_"
            f"V_128_BK_128_BV_32_SPT_20.0_L2N_1_W_1_S_3_compute_type_"
            f"{dt}.cubin"
        )
        cmds.append((f"{PY} {GDN}/cubin_gen/packed_decode.py --output_dir {OUT} "
                     f"--B 1 --H {H_key} --HV {HV} --K 128 --V 128 --BK 128 --BV 32 "
                     f"--SOFTPLUS_THRESHOLD 20.0 --USE_QK_L2NORM_IN_KERNEL True "
                     f"--interleaved_ba True "
                     f"--num_warps 1 --num_stages 3 --compute_type {dt}", _path(pd)))
        # Generic recurrent fwd（spec-decode 路径，仅 mtp>0 时生成）。
        # SISQ = 1+mtp（烘焙 stride_indices_seq：dummy (2, SISQ) → strides (SISQ,1)）。
        if mtp > 0:
            rec = (f"fused_recurrent_gated_delta_rule_fwd_kernel_H_{H_key}_HV_{HV}_K_128_V_128_"
                   f"BK_128_BV_32_SISQ_{1 + mtp}_SIT_1_UIS_1_INP_1_IBH_0_L2N_0_VAR_1_CB_1_"
                   f"SPEC_1_KDA_0_W_1_S_3_compute_type_{dt}.cubin")
            cmds.append((f"{PY} {GDN}/cubin_gen/fused_recurrent_fwd.py --output_dir {OUT} "
                         f"--H {H_key} --HV {HV} --K 128 --V 128 --BK 128 --BV 32 "
                         f"--SISQ {1 + mtp} --num_warps 1 --num_stages 3 --compute_type {dt}",
                         _path(rec)))

        # conv1d（仅 D，decode update）。prefill conv 已由 fused_conv_post_c1 取代。
        conv_upd = (
            f"gdn_causal_conv1d_update_kernel_D_{D}_KW_4_BN_256_NP2_4_SL_3_"
            f"HB_0_SILU_1_VL_0_APC_0_SPEC_0_NULL_1_W_4_S_2_compute_type_"
            f"{dt}.cubin"
        )
        cmds.append((f"{PY} {GDN}/cubin_gen/causal_conv1d_update.py --output_dir {OUT} "
                     f"--dim {D} --KERNEL_WIDTH 4 --HAS_BIAS False --SILU_ACTIVATION True "
                     f"--IS_VARLEN False --IS_APC_ENABLED False --IS_SPEC_DECODING False "
                     f"--HAS_NULL_BLOCK True --NP2_STATELEN 4 --STATE_LEN 3 --BLOCK_N 256 "
                     f"--num_warps 4 --num_stages 2 --compute_type {dt}", _path(conv_upd)))
        if mtp > 0:
            spec_sl = 3 + mtp  # KW-1 + mtp（KW=4）
            np2 = _np2_statelen(spec_sl)
            conv_spec = (f"gdn_causal_conv1d_update_kernel_D_{D}_KW_4_BN_256_NP2_{np2}_SL_{spec_sl}_"
                         f"HB_0_SILU_1_VL_1_APC_0_SPEC_1_NULL_1_W_4_S_2_compute_type_{dt}.cubin")
            cmds.append((f"{PY} {GDN}/cubin_gen/causal_conv1d_update.py --output_dir {OUT} "
                         f"--dim {D} --KERNEL_WIDTH 4 --HAS_BIAS False --SILU_ACTIVATION True "
                         f"--IS_VARLEN True --IS_APC_ENABLED False --IS_SPEC_DECODING True "
                         f"--HAS_NULL_BLOCK True --NP2_STATELEN {np2} --STATE_LEN {spec_sl} "
                         f"--BLOCK_N 256 --num_warps 4 --num_stages 2 --compute_type {dt}",
                         _path(conv_spec)))
    return cmds


def run_cmds(cmds, name, quiet=False):
    """并行编译（round-robin GPU 分配），返回 (ok, fail) 计数。"""
    total = len(cmds)
    ngpu = len(gpu_list)
    if ngpu == 0:
        raise RuntimeError("no GPU visible (CUDA_VISIBLE_DEVICES empty)")
    workers = ngpu  # 并发限制到 GPU 数，避免 16 路同时编译
    if not quiet:
        print(f"[{name}] {total} kernels, {workers} workers, GPUs {gpu_list}")
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, (cmd, _) in enumerate(cmds):
            gpu = gpu_list[i % ngpu]
            futures[pool.submit(run, cmd, gpu, quiet)] = cmd
        for f in as_completed(futures):
            if f.result():
                ok += 1
    fail = total - ok
    if not quiet:
        print(f"[{name}] {ok}/{total} OK, {fail} failed")
    return ok, fail


def gen_universal(quiet=False):
    return run_cmds(_universal_cmds(), "universal", quiet)


def gen_one(name, cfg, quiet=False, mtp=0):
    return run_cmds(_model_cmds(cfg["H_key"], cfg["HV"], cfg["D"], mtp), name, quiet)


def _kernel_complete(path):
    """cubin 与运行期必需的 json 必须都存在；json 缺失时服务在
    LoadTritonKernelJsonFile 直接 terminate。"""
    json_path = path[: -len(".cubin")] + ".json"
    return os.path.exists(path) and os.path.exists(json_path)


def gen_ensure(H_key, HV, D, mtp, dts=None):
    """幂等：全部 cubin+json 存在时静默；否则只编译缺失的，任一失败非零退出。
    文件锁防并发 TP-rank 重复生成。"""
    expected = _universal_cmds(dts) + _model_cmds(H_key, HV, D, mtp, dts)
    missing = [(cmd, path) for cmd, path in expected if not _kernel_complete(path)]
    if not missing:
        return
    # File lock so only one TP rank generates; others wait then re-check.
    lock_path = os.path.join(OUT, f".gdn_ensure_{D}.lock")
    os.makedirs(OUT, exist_ok=True)
    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        missing = [(cmd, path) for cmd, path in expected if not _kernel_complete(path)]
        if not missing:
            return  # another rank generated while we waited
        print(f"[GDN] ensure: {len(missing)}/{len(expected)} cubins missing "
              f"(D={D} mtp={mtp}), generating")
        _, fail = run_cmds(missing, "ensure", quiet=True)
        if fail:
            print(f"[GDN] ensure FAILED: {fail}/{len(missing)} kernels did not compile")
            sys.exit(1)
        # 后校验：模板若与生成器命名漂移，在这里暴露而非运行时。
        still_missing = [path for _, path in expected if not _kernel_complete(path)]
        if still_missing:
            print(f"[GDN] ensure INCOMPLETE: {len(still_missing)} expected cubins "
                  f"still missing after generation:")
            for path in still_missing:
                print(f"  {path}")
            sys.exit(1)
        print(f"[GDN] ensure done: all {len(expected)} cubins present")


if __name__ == "__main__":
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_list = [int(x.strip()) for x in cvd.split(",")]

    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for k, v in CONFIGS.items():
            print(f"  {k}: H_key={v['H_key']} HV={v['HV']} D={v['D']}")
        sys.exit(0)

    # ensure：幂等预检——按 cubin+json 双文件存在性判断（MD5 缓存不参与）。
    if sys.argv[1] == "ensure":
        kv = {}
        i = 2
        while i < len(sys.argv) - 1:
            kv[sys.argv[i].lstrip("-")] = sys.argv[i + 1]
            i += 2
        dts = kv.get("dtype")
        if dts is not None:
            dts = dts.split(",")  # --dtype 过滤 DTYPES 循环
        gen_ensure(int(kv["H_key"]), int(kv["HV"]), int(kv["D"]), int(kv.get("mtp", 0)),
                   dts)
        sys.exit(0)

    # MD5 cache: skip if all GDN .py sources unchanged since last generation.
    # 产物被删除但脚本未变时也必须重生成——skip 前按目标集合检查产物完整性。
    cache_file = os.path.join(OUT, ".gdn_cubin_md5")
    py_files = sorted(glob.glob(f"{GDN}/**/*.py", recursive=True))
    py_files.append(__file__)  # include this script
    h = hashlib.md5()
    for f in py_files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    current_md5 = h.hexdigest()
    target = sys.argv[1]
    if os.path.exists(cache_file):
        with open(cache_file) as fh:
            cached_md5 = fh.read().strip()
        if cached_md5 == current_md5:
            missing = [(cmd, path) for cmd, path in _universal_cmds() if not _kernel_complete(path)]
            if target != "all":
                for name, cfg in CONFIGS.items():
                    if target == name:
                        missing += [(c, p) for c, p in _model_cmds(cfg["H_key"], cfg["HV"], cfg["D"], 0)
                                    if not _kernel_complete(p)]
            if not missing:
                print(f"[MD5] {current_md5[:12]} unchanged, skipping cubin generation")
                sys.exit(0)
            print(f"[MD5] sources unchanged but {len(missing)} artifacts missing; regenerating")
    all_fail = 0
    _, fail = gen_universal()
    all_fail += fail
    if target == "all":
        for name, cfg in CONFIGS.items():
            _, fail = gen_one(name, cfg)
            all_fail += fail
    elif target in CONFIGS:
        _, fail = gen_one(target, CONFIGS[target])
        all_fail += fail
    else:
        print(f"Unknown: {target}. Try 'list' or 'all'")
        sys.exit(1)
    if all_fail:
        print(f"[MD5] generation had {all_fail} failures; NOT saving cache marker")
        sys.exit(1)

    with open(cache_file, "w") as fh:
        fh.write(current_md5)
    print(f"[MD5] saved {current_md5[:12]}")
