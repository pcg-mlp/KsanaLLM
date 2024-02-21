# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig

from .serving_model import ServingModel


class AutoModel(object):
    """Auto llm model, return a callable model instance.
    """

    @classmethod
    def from_config(cls,
                    config_file: Optional[Union[str, os.PathLike]],
                    **kwargs):
        """The model loader interface, invoked by venus.
        """
        if not os.path.exists(config_file):
            raise RuntimeError(
                f"The config file {config_file} is not exists.")

        if not config_file.lower().endswith('.yaml'):
            raise RuntimeError(
                f"The config file {config_file} must be YAML format.")

        serving_model = ServingModel(config_file)
        return serving_model
