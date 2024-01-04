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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        """The model loader interface, invoked by venus.
        """
        if not os.path.exists(pretrained_model_name_or_path):
            raise RuntimeError(
                f"The model dir {pretrained_model_name_or_path} is not exists.")

        serving_model = ServingModel(pretrained_model_name_or_path)
        return serving_model

