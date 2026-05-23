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
        # Pass json_model_override_args directly to the C++ Serving constructor so
        # that it reaches Environment::ParseConfig before FLAGS_json_model_override_args
        # is read.  The old sys.argv injection approach never worked because
        # gflags::ParseCommandLineFlags() is never called in the pybind11 path.
        _json_override = getattr(engine_args, "json_model_override_args", None) or ""

        # The serving instance.
        self._serving = libtorch_serving.Serving(
            engine_args.config_file, False, executor_id, _json_override
        )

    def init_ipc(self):
        """Attach ipc resource.
        """
        self._serving.init_ipc()

    def start(self):
        """Start the executor process.
        """
        # Should never return.
        self._serving.start_executor()
