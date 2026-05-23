# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import os
import shutil

TILELANG_CACHE_DIR = "/tmp/tilelang_autotuner/cache"
if os.path.exists(TILELANG_CACHE_DIR):
    shutil.rmtree(TILELANG_CACHE_DIR)
os.environ['TILELANG_CACHE_DIR'] = TILELANG_CACHE_DIR


def compile_cuda(prim_func, 
                 kernel_name, 
                 output_dir, 
                 *args, 
                 **kwargs):
    s_dynamic_smem_buf = None

    # Tilelang detect wrong cuda version when cuda-python version is 13.x but cuda version is 12.x.
    def fix_cuda_version():
        try:
            import cuda.bindings.driver as cuda
            OLD_CUDA_VERSION = cuda.CUDA_VERSION

            err, = cuda.cuInit(0)
            err, version = cuda.cuDriverGetVersion()
            cuda.CUDA_VERSION = version
            print(f"Reset CUDA_VERSION from {OLD_CUDA_VERSION} to {version} succ.")
        except AttributeError:
            print("Reset CUDA_VERSION failed.")

    # Hack tilelang, so we can extract the shared memory size.
    def hack_update_lib_code():
        from tilelang.jit.adapter.nvrtc.wrapper import TLNVRTCSourceWrapper
        from tilelang.jit.adapter.cutedsl.wrapper import TLCuTeDSLSourceWrapper

        nvrtc_update_lib_code = TLNVRTCSourceWrapper.update_lib_code
        cutedsl_update_lib_code = TLCuTeDSLSourceWrapper.update_lib_code

        def hacked_nvrtc_update_lib_code(self, code: str):
            res = nvrtc_update_lib_code(self, code)
            nonlocal s_dynamic_smem_buf
            s_dynamic_smem_buf = self.dynamic_smem_buf
            return res

        def hacked_cutedsl_update_lib_code(self, code: str):
            res = cutedsl_update_lib_code(self, code)
            nonlocal s_dynamic_smem_buf
            s_dynamic_smem_buf = self.dynamic_smem_buf
            return res

        TLNVRTCSourceWrapper.update_lib_code = hacked_nvrtc_update_lib_code 
        TLCuTeDSLSourceWrapper.update_lib_code = hacked_cutedsl_update_lib_code 

    fix_cuda_version()
    hack_update_lib_code()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import tilelang
    jit_kernel = tilelang.compile(prim_func, execution_backend="nvrtc", *args, **kwargs)
    print("Extract shm:", s_dynamic_smem_buf)

    # Save cu and cubin.
    cubin_src = jit_kernel.adapter.lib_generator.libpath
    cubin_dst = output_dir + "/" + kernel_name + ".cubin"
    shutil.copy2(cubin_src, cubin_dst)

    cu_src = jit_kernel.adapter.lib_generator.srcpath
    cu_dst = output_dir + "/" + kernel_name + ".cu"
    shutil.copy2(cu_src, cu_dst)

    py_src = jit_kernel.adapter.lib_generator.pypath
    py_dst = output_dir + "/" + kernel_name + ".py"
    shutil.copy2(py_src, py_dst)

    # write config.
    import json
    with open(f"{output_dir}/{kernel_name}.json", "w") as f:
        primfunc_name = prim_func.attrs["global_symbol"]
        shm_key = f"{primfunc_name}_kernel"
        config_content = {
            "shm_size": 0 if s_dynamic_smem_buf[shm_key] is None else s_dynamic_smem_buf[shm_key],
        }
        json.dump(config_content, f)


def profile_kernel(fn, 
                   num_iters: int = 10, 
                   warmup_iters: int = 1):
    # compile
    fn()

    # warmup
    for _ in range(warmup_iters):
        fn()

    import torch
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # test
    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)


def run_autotune(kernel_name, 
                 input_config, 
                 get_autotune_configs_fn, 
                 get_kernel_func_fn, 
                 get_kernel_input_fn, 
                 get_kernel_name_fn, 
                 output_dir, 
                 *args, 
                 **kwargs):
    best_latency = float('inf')
    best_config = None

    # Create a empty dict if autotune configs is empty.
    autotune_configs = get_autotune_configs_fn()
    if len(autotune_configs) == 0:
        autotune_configs.append({})

    import tilelang
    for autotune_config in autotune_configs:
        print(f"Start to test config {autotune_config}")
        func = get_kernel_func_fn(input_config, autotune_config)
        jit_kernel = tilelang.compile(func, execution_backend="auto", *args, **kwargs)

        def fn():
            args = get_kernel_input_fn(input_config)
            jit_kernel(*args)

        avg_ms = profile_kernel(fn, 10, 1)
        print(f"Config {autotune_config}: {avg_ms:.4f} ms")

        if avg_ms < best_latency:
            best_latency = avg_ms
            best_config = autotune_config 

    kernel_name = get_kernel_name_fn(kernel_name, input_config)

    print("-" * 30)
    print(f"Autotune {kernel_name} finishied!")
    print(f"Best latency: {best_latency:.4f} ms")
    print(f"Best config: {best_config}")

    # Use nvrtc backend to get shm size.
    opt_func = get_kernel_func_fn(input_config, best_config)
    compile_cuda(opt_func, kernel_name, output_dir, *args, **kwargs)


def autotune(kernel_name, 
             get_input_configs_fn, 
             get_autotune_configs_fn, 
             get_kernel_func_fn, 
             get_kernel_input_fn, 
             get_kernel_name_fn, 
             output_dir, 
             *args, 
             **kwargs):
    for input_config in get_input_configs_fn():
        run_autotune(kernel_name, 
                     input_config, 
                     get_autotune_configs_fn, 
                     get_kernel_func_fn, 
                     get_kernel_input_fn, 
                     get_kernel_name_fn, 
                     output_dir, 
                     *args, 
                     **kwargs)
