# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import importlib.util
import os
import sys
from concurrent import futures
import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch
import libtorch_serving

from utilize.logger import get_logger
from ksana_encoder.multimodal_cache import MultimodalCache

logger = get_logger(__name__)


@dataclass
class PluginConfig:
    model_dir: str = ""  # the directory of the serving model
    config_file: str = ""  # the path of the yaml config file
    plugin_name: str = ""  # the name of this plugin
    # Whether to enable TensorRT inference for plugin model (e.g., ViT).
    # The default value is true. But if trt inference is abnormal (NaN),
    # you need to manually configure it in yaml to adjust to False to use
    # the Torch inference.
    enable_trt: bool = True
    num_threads_per_encoder: int = 1  # The thread size for each multimodal encoder
    # The type of the vit model
    # Some models shares the same base LLM model architecture but have different vit model 
    # such as InternVL2_5 and InternVL2, then we need to specify the vit model type so that 
    # we can share the same ksana_plugin.py
    vit_model_type: str = ""
    # Multimodal Encoder separation parameters
    encoder_url: Optional[List[str]] = None  # URLs of remote Encoder services (e.g., ['tcp://localhost:5555'])
    encoder_timeout: int = 30  # Timeout for encoder requests in seconds
    multimodal_cache_size: float = -1.0  # Cache size in GB (-1.0 means disabled)


class SimplePluginRouter:
    def __init__(self, max_server_num: int):
        self._cycle = itertools.cycle(range(max_server_num))

    def get_next(self):
        return next(self._cycle)


class KsanaPlugin(object):
    """
    This class is designed to dynamically load and manage plugins for the Ksana framework.
    It allows for the initialization and post-processing of plugins specified by a file path.
    """

    def __init__(self, config: PluginConfig):
        """
        Initializes the KsanaPlugin instance by loading a plugin from the given path.

        :param plugin_config: The specified plugin configuration.
        """
        self.remote_mode = config.encoder_url is not None
        if not self.remote_mode:
            config.num_threads_per_encoder = 1
        else:
            config.num_threads_per_encoder = len(config.encoder_url) * config.num_threads_per_encoder

        self.preprocess_router = SimplePluginRouter(config.num_threads_per_encoder)
        self.postprocess_router = SimplePluginRouter(config.num_threads_per_encoder)

        if config.multimodal_cache_size > 0:
            self.multimodal_cache = MultimodalCache(max_size_gb=config.multimodal_cache_size)
            logger.info(f"Multimodal cache created with size {config.multimodal_cache_size} GB")
        else:
            self.multimodal_cache = None
            logger.info("Multimodal cache is disabled")

        # 批量创建
        plugin_path = KsanaPlugin.search_plugin_path(config.model_dir,
                                                     config.plugin_name)
        self._ksana_plugin = []
        for _ in range(config.num_threads_per_encoder):
            cur_ksana_plugin = KsanaPlugin.load_plugin(plugin_path)
            if hasattr(cur_ksana_plugin, 'init_plugin'):
                kwargs = {
                    "encoder_url": (
                        config.encoder_url[
                            self.preprocess_router.get_next() % len(config.encoder_url)
                        ]
                        if self.remote_mode
                        else None
                    ),
                    "encoder_timeout": config.encoder_timeout,
                    "preprocess": True,
                    "postprocess": True,
                    "model_path": config.model_dir,
                    "config_file": config.config_file,
                    "enable_trt": config.enable_trt,
                    "model_type": config.plugin_name,
                    "vit_model_type": config.vit_model_type,
                    "multimodal_cache": self.multimodal_cache
                }
                try:
                    cur_ksana_plugin.init_plugin(**kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(f"Init plugin failed: {e}")
                    cur_ksana_plugin = None
            if cur_ksana_plugin is not None:
                self._ksana_plugin.append(cur_ksana_plugin)
        if len(self._ksana_plugin) != config.num_threads_per_encoder:
            self._ksana_plugin = None

        # We utilize a thread pool to manage the execution of plugin operations, to control
        # the level of concurrency. This is primarily aimed at preventing concurrent model
        # inference task from competing for GPU memory and computational resources.
        # When `num_threads_per_encoder > 1`, the preprocess and postprocess methods in the
        # plugin must be **thread-safe**.
        logger.info(f"Plugin thread pool size is set to {config.num_threads_per_encoder}")
        self._thread_pool = futures.ThreadPoolExecutor(config.num_threads_per_encoder)

        # Initialize a non-default stream pool.
        # We place all computations within the plugin in these streams, to avoid affecting
        # the computations in KsanaLLM that use the default stream.
        self.streams = [torch.cuda.Stream() for _ in range(config.num_threads_per_encoder)]

    def __del__(self):
        """
        Clean up the thread pool when the instance is destroyed.
        """
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=True)

    @staticmethod
    def load_plugin(plugin_path: Optional[str]):
        """
        Dynamically loads the plugin module located at the specified path.

        :param plugin_path: The file system path to the plugin module to be loaded.
        :return: An instance of the loaded plugin class if successful, None otherwise.
        """
        # Plugin is not provided
        if plugin_path is None:
            return None
        try:
            spec = importlib.util.spec_from_file_location("ksana_plugin", plugin_path)
            if spec is None:
                logger.warning(f"Load plugin failed: {plugin_path} can not be imported.")
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            class_name = "KsanaPlugin"
            if hasattr(module, class_name):
                ksana_plugin = getattr(module, class_name)
                logger.info("Plugin is loaded.")
                return ksana_plugin()
            else:
                logger.warning("Load plugin failed: the plugin does not have KsanaPlugin class.")
                return None
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Load plugin failed: {e}")
            return None

    def _preprocess(self, router_id, **kwargs):
        """
        This is a wrapper around the preprocess method of the loaded plugin to run in a non-default stream.
        """
        with torch.cuda.stream(self.streams[router_id]):
            self._ksana_plugin[router_id].preprocess(**kwargs)

    def preprocess(self,
                   ksana_python_input: libtorch_serving.KsanaPythonInput,
                   **kwargs):
        """
        Invokes the preprocess method of the loaded plugin, if available.
        """
        if self._ksana_plugin is None:
            return

        kwargs.update({
            "ksana_python_input": ksana_python_input,
        })
        future = self._thread_pool.submit(self._preprocess, self.preprocess_router.get_next(), **kwargs)
        return future.result()

    def _postprocess(self, router_id, **kwargs):
        """
        This is a wrapper around the postprocess method of the loaded plugin to run in a non-default stream.
        """
        with torch.cuda.stream(self.streams[router_id]):
            self._ksana_plugin[router_id].postprocess(**kwargs)

    def postprocess(self,
                    ksana_python_input: libtorch_serving.KsanaPythonInput,
                    ksana_python_output: libtorch_serving.KsanaPythonOutput,
                    **kwargs):
        """
        Invokes the postprocess method of the loaded plugin, if available.
        """
        if self._ksana_plugin is None:
            return

        kwargs.update({
            "ksana_python_input": ksana_python_input,
            "ksana_python_output": ksana_python_output,
        })
        future = self._thread_pool.submit(self._postprocess, self.postprocess_router.get_next(), **kwargs)
        return future.result()

    @staticmethod
    def search_plugin_path(model_dir: str, plugin_name: str) -> Optional[str]:
        """
        Search the path where the ksana plugin is located.
        """
        sys_path: List[str] = sys.path
        # 1. Search within the model dir
        target_file = os.path.join(model_dir, "ksana_plugin.py")
        if os.path.isfile(target_file):
            return target_file

        # 2. Search within the current working directory
        python_dir = sys_path[0]
        target_file = os.path.join(python_dir, "ksana_plugin", plugin_name, "ksana_plugin.py")
        if os.path.isfile(target_file):
            return target_file

        # 3. Search within the ksana_llm package
        package_suffix = "site-packages"
        for python_dir in sys_path[1:]:
            if python_dir.endswith(package_suffix):
                target_file = os.path.join(python_dir, "ksana_llm", "ksana_plugin",
                                           plugin_name, "ksana_plugin.py")
                if os.path.isfile(target_file):
                    return target_file

        return None
