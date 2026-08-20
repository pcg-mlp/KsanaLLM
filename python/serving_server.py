# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import atexit
import os
import signal

from utilize.utils import sync_visible_devices
sync_visible_devices()

from ksana_encoder.utils import pre_configure_cuda
pre_configure_cuda()

import torch
import torch.multiprocessing as mp

from ksana_llm.arg_utils import EngineArgs
from ksana_llm import KsanaService
from ksana_llm import KsanaExecutor
from ksana_encoder.utils import (
    start_encoder_processes,
    wait_encoders_ready,
    build_encoder_urls,
    cleanup_encoder_processes,
)

_encoder_processes = []


def start_executor(i, engine_args: EngineArgs, rank_id: int):
    """ Start executor processes.
    """
    torch.cuda.set_device(rank_id)
    executor = KsanaExecutor(engine_args, rank_id)
    executor.init_ipc()
    executor.start()


def _signal_handler(signum, frame):
    print(f"[Embedded Encoder] Received signal {signum}, shutting down...",
          flush=True)
    cleanup_encoder_processes(_encoder_processes)
    os._exit(0)


if __name__ == '__main__':
    engine_args = EngineArgs.from_command_line()

    embedded_encoder = (
        engine_args.encoder_gpu is not None
        and engine_args.encoder_url is None
    )

    if embedded_encoder:
        if engine_args.llm_gpu is None or engine_args.encoder_base_port is None:
            raise ValueError(
                "llm_gpu and encoder_base_port must both be configured "
                "when encoder_gpu is set"
            )

        print(f"[Embedded Encoder] LLM GPUs: {engine_args.llm_gpu}, "
              f"Encoder GPUs: {engine_args.encoder_gpu}", flush=True)

        _encoder_processes.extend(start_encoder_processes(engine_args))

        atexit.register(cleanup_encoder_processes, _encoder_processes)
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        wait_encoders_ready(engine_args, _encoder_processes)

        engine_args.encoder_url = build_encoder_urls(engine_args)
        print(f"[Embedded Encoder] Encoder URLs: {engine_args.encoder_url}",
              flush=True)

    # Note: create ipc resource before executors started.
    service = KsanaService(engine_args)
    service.init_ipc()

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # context already set (e.g. by XPU runtime or test harness)
    num_gpus = torch.cuda.device_count()
    # TODO(zhongzhicao): 下面的 xpu device_count fallback 可能已不再需要（sync_visible_devices 已解决），验证后删除。
    # On XPU/Kunlun, cuda.device_count() returns 0; fall back to xpu device count.
    if num_gpus == 0 and hasattr(torch, 'xpu') and callable(getattr(torch.xpu, 'device_count', None)):
        num_gpus = torch.xpu.device_count()
    # Respect tensor_para_size from config: only spawn that many executors
    if hasattr(engine_args, 'tensor_para_size') and engine_args.tensor_para_size is not None:
        num_gpus = min(num_gpus, engine_args.tensor_para_size)
    elif num_gpus > 1:
        # Fallback: read from config file
        import yaml
        with open(engine_args.config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        tp = cfg.get('setting', {}).get('global', {}).get('tensor_para_size', num_gpus)
        num_gpus = min(num_gpus, tp)
    # Final safety: if still 0, default to tensor_para_size=1 from config.
    if num_gpus == 0:
        import yaml
        with open(engine_args.config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        num_gpus = cfg.get('setting', {}).get('global', {}).get('tensor_para_size', 1)

    print(f"[engine] num_gpus={num_gpus}, spawning executors...", flush=True)
    processes_list = []
    for rank_id in range(num_gpus):
        os.environ['KSANA_EXECUTOR_ID'] = str(rank_id)
        processes_list.append(
            mp.spawn(
                start_executor,
                nprocs=1,
                args=(
                    engine_args,
                    rank_id
                ),
                join=False))

    # Start service process.
    print("[engine] all executors spawned, starting service...", flush=True)
    service.start()

    # Wait child processes.
    for ctx in processes_list:
        ctx.join()

    if embedded_encoder:
        cleanup_encoder_processes(_encoder_processes)

    print("Ksana serving finished.", flush=True)
