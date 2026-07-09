# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

# Load library path.
sys.path.append(os.path.abspath("./lib"))

# pd_v2 forks ksana_executor child processes that dlopen
# libpd_v2_connector_mooncake.so via inference_executor.cpp. The .so has
# undefined references to IpcFuncWrapper::PdV2GetKvPoolBaseVa etc. that
# live in libtorch_serving.so. Python's default RTLD_LOCAL import keeps
# those symbols private to libtorch_serving, so the executor child's
# dlopen fails with "undefined symbol". Switch to RTLD_GLOBAL for the
# libtorch_serving import so the dynamic linker can resolve transitive
# symbols against libtorch_serving's full symbol table.
_old_flags = sys.getdlopenflags()
sys.setdlopenflags(_old_flags | os.RTLD_GLOBAL)
try:
    from ksana_llm import openaiapi, utilize, libtorch_serving, ksana_encoder
    sys.modules['openaiapi'] = openaiapi
    sys.modules['utilize'] = utilize
    sys.modules['libtorch_serving'] = libtorch_serving
    sys.modules['ksana_encoder'] = ksana_encoder
except ImportError:
    try:
        import openaiapi
        import utilize
        import ksana_encoder
    except ImportError:
        print(f"Warning: Failed to import optional modules, please check")
finally:
    sys.setdlopenflags(_old_flags)

# Expose request_id as a top-level module so bare `from request_id import ...`
# works both in dev mode (python/request_id.py on sys.path) and after wheel
# install (request_id.py is copied into the ksana_llm package by setup.py).
try:
    from ksana_llm import request_id
    sys.modules['request_id'] = request_id
except ImportError:
    pass


from .exceptions import KsanaEngineError
from .ksana_executor import KsanaExecutor
from .ksana_service import KsanaService
from .ksana_engine import KsanaLLMEngine
from .arg_utils import EngineArgs
from .ksana_plugin import PluginConfig
from .processor_op_base import TokenizerProcessorOpBase
