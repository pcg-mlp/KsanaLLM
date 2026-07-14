# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import libtorch_serving

from ksana_llm.arg_utils import EngineArgs


class KsanaExecutor:
    """ The ksana executor.
    """

    def __init__(self, engine_args: EngineArgs, executor_id: int):
        """Initialize a KsanaExecutor instance.

        Args:
            engine_args: The serving arguments.
            executor_id: The rank id.
        """
        # The serving instance.
        self._serving = libtorch_serving.Serving(
            engine_args.config_file, False, executor_id)

    def init_ipc(self):
        """Attach ipc resource.
        """
        self._serving.init_ipc()

    def start(self):
        """Start the executor process.
        """
        # Should never return.
        self._serving.start_executor()
