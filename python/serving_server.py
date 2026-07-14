# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import torch
import torch.multiprocessing as mp

from ksana_llm.arg_utils import EngineArgs
from ksana_llm import KsanaService
from ksana_llm import KsanaExecutor


def start_executor(i, engine_args: EngineArgs, rank_id: int):
    """ Start executor processes.
    """
    torch.cuda.set_device(rank_id)
    executor = KsanaExecutor(engine_args, rank_id)
    executor.init_ipc()
    executor.start()


if __name__ == '__main__':
    # Note: create ipc resource before executors started.
    engine_args = EngineArgs.from_command_line()
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
    for processes in processes_list:
        for p in processes:
            p.join
    print("Ksana serving finished.", flush=True)
