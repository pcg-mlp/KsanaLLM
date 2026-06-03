# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import atexit
import os
import signal

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

    mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()

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
    service.start()

    # Wait child processes.
    for ctx in processes_list:
        ctx.join()

    if embedded_encoder:
        cleanup_encoder_processes(_encoder_processes)

    print("Ksana serving finished.", flush=True)
