# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================


class KsanaEngineError(RuntimeError):
    """Engine-level error carrying a machine-readable error code.

    Inherits from RuntimeError so existing ``except RuntimeError`` handlers
    still work, while callers that need the code can ``except KsanaEngineError``
    for precise matching.
    """

    def __init__(self, message: str, error_code: int = 500):
        super().__init__(message)
        self.error_code = error_code
