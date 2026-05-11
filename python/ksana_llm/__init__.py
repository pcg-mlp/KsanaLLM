# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

# Load library path.
sys.path.append(os.path.abspath("./lib"))

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
