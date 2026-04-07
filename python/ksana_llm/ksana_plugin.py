# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import importlib.util
import os
import sys
import time
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
class PluginTimingMetrics:
    """Timing metrics collected during plugin processing."""
    preprocess_time_ms: float = 0.0  # Wall time of the preprocess (or preprocess_before_encode) call
    waiting_time_ms: float = 0.0  # Time waiting for a plugin thread pool slot before preprocess starts
    encoder_time_ms: float = 0.0  # Total round-trip time of the remote encoder call (send + queue + encode + recv)
    encoder_send_time_ms: float = 0.0  # Time spent serializing and sending the request to the encoder
    encoder_queue_time_ms: float = 0.0  # Time the request waited in the encoder's internal queue
    encoder_encode_time_ms: float = 0.0  # Time the encoder spent on the actual forward pass (e.g., ViT inference)
    encoder_recv_time_ms: float = 0.0  # Time spent receiving and deserializing the encoder response


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
        self._ksana_plugin = None
        plugin_path = KsanaPlugin.search_plugin_path(config.model_dir, config.plugin_name)
        if plugin_path is None:
            return

        self.remote_mode = config.encoder_url is not None
        if not self.remote_mode:
            config.num_threads_per_encoder = 1
        else:
            config.num_threads_per_encoder = len(config.encoder_url) * config.num_threads_per_encoder

        # Thread pool for plugin operations; controls GPU resource concurrency.
        # When num_threads_per_encoder > 1, plugin preprocess/postprocess must be thread-safe.
        logger.info(f"Plugin thread pool size is set to {config.num_threads_per_encoder}")
        self._thread_pool = futures.ThreadPoolExecutor(config.num_threads_per_encoder)

        # Non-default CUDA stream pool; isolates plugin computation from KsanaLLM default stream.
        self.streams = [torch.cuda.Stream() for _ in range(config.num_threads_per_encoder)]

        self.preprocess_router = SimplePluginRouter(config.num_threads_per_encoder)
        self.postprocess_router = SimplePluginRouter(config.num_threads_per_encoder)

        if config.multimodal_cache_size > 0:
            self.multimodal_cache = MultimodalCache(max_size_gb=config.multimodal_cache_size)
            logger.info(f"Multimodal cache created with size {config.multimodal_cache_size} GB")
        else:
            self.multimodal_cache = None
            logger.info("Multimodal cache is disabled")

        # Dedicated encoder I/O pool; only created when decoupled preprocessing is supported.
        self._encoder_pool = None

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

        # Enable decoupled preprocessing when the plugin implements preprocess_before/after_encode,
        # releasing the thread pool before blocking encoder I/O. Set KSANA_DISABLE_DECOUPLED_PREPROCESS=1 to disable.
        force_disable = os.environ.get("KSANA_DISABLE_DECOUPLED_PREPROCESS", "0") == "1"
        self._supports_decoupled = (
            self._ksana_plugin is not None
            and self.remote_mode
            and hasattr(self._ksana_plugin[0], 'preprocess_before_encode')
            and hasattr(self._ksana_plugin[0], 'preprocess_after_encode')
            and not force_disable
        )
        if self._supports_decoupled:
            # Encoder I/O pool size is 2x to avoid blocking (threads only wait on network).
            self._encoder_pool = futures.ThreadPoolExecutor(config.num_threads_per_encoder * 2)
            logger.info("Decoupled preprocessing enabled: preprocess threads will be released "
                        "before encoder I/O")
        elif force_disable:
            logger.info("Decoupled preprocessing forcibly disabled via "
                        "KSANA_DISABLE_DECOUPLED_PREPROCESS=1")

    def __del__(self):
        """
        Clean up the thread pool when the instance is destroyed.
        """
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=True)
        if hasattr(self, "_encoder_pool") and self._encoder_pool is not None:
            self._encoder_pool.shutdown(wait=True)

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

    def _preprocess(self, submit_time, router_id, **kwargs):
        """
        This is a wrapper around the preprocess method of the loaded plugin to run in a non-default stream.
        """
        plugin_timing_metrics = kwargs.get("plugin_timing_metrics", None)
        with torch.cuda.stream(self.streams[router_id]):
            try:
                start = time.time()
                if plugin_timing_metrics is not None:
                    plugin_timing_metrics.waiting_time_ms = (start - submit_time) * 1000
                self._ksana_plugin[router_id].preprocess(**kwargs)
                if plugin_timing_metrics is not None:
                    plugin_timing_metrics.preprocess_time_ms = (time.time() - start) * 1000
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"Plugin preprocess failed: {e}")

    def _preprocess_before_encode(self, submit_time, router_id, **kwargs):
        """Phase 1 of decoupled preprocessing (in preprocess thread pool): CPU work only.

        Returns (router_id, encode_input, context); encode_input is forwarded to encoder,
        context is opaque state passed to preprocess_after_encode.
        """
        plugin_timing_metrics = kwargs.get("plugin_timing_metrics", None)
        with torch.cuda.stream(self.streams[router_id]):
            try:
                start = time.time()
                if plugin_timing_metrics is not None:
                    plugin_timing_metrics.waiting_time_ms = (start - submit_time) * 1000
                encode_input, context = self._ksana_plugin[router_id].preprocess_before_encode(**kwargs)
                if plugin_timing_metrics is not None:
                    plugin_timing_metrics.preprocess_time_ms = (time.time() - start) * 1000
                return router_id, encode_input, context
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f"Plugin preprocess_before_encode failed: {e}")
                return router_id, None, None

    def _encode_remote(self, router_id, encode_input, plugin_timing_metrics):
        """Encoder I/O phase: runs in the encoder pool (NOT in preprocess pool)."""
        try:
            encode_start = time.time()
            encode_output = self._ksana_plugin[router_id].encode(encode_input)
            if encode_output is None:
                logger.error("Plugin encode returned None (timeout or encoder error)")
                return None
            if plugin_timing_metrics is not None:
                plugin_timing_metrics.encoder_time_ms = (time.time() - encode_start) * 1000
                plugin_timing_metrics.encoder_send_time_ms = encode_output.pop("encoder_send_time", 0) * 1000
                plugin_timing_metrics.encoder_queue_time_ms = encode_output.pop("encoder_queue_time", 0) * 1000
                plugin_timing_metrics.encoder_encode_time_ms = encode_output.pop("encoder_encode_time", 0) * 1000
                plugin_timing_metrics.encoder_recv_time_ms = encode_output.pop("encoder_recv_time", 0) * 1000
            return encode_output
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Plugin encode failed: {e}")
            return None

    def _preprocess_decoupled(self, ksana_python_input, **kwargs):
        """Decoupled preprocessing: phase1 (thread pool) → encode (encoder pool) → phase2 (inline)."""
        kwargs.update({
            "ksana_python_input": ksana_python_input,
        })

        # Phase 1: CPU preprocessing in thread pool.
        submit_time = time.time()
        router_id_for_phase1 = self.preprocess_router.get_next()
        before_encode_future = self._thread_pool.submit(
            self._preprocess_before_encode, submit_time, router_id_for_phase1, **kwargs)
        router_id, encode_input, context = before_encode_future.result()

        if encode_input is None:
            return  # preprocessing failed

        # Phase 2: encoder I/O in encoder pool (does not occupy preprocess thread).
        plugin_timing_metrics = kwargs.get("plugin_timing_metrics", None)
        encode_future = self._encoder_pool.submit(
            self._encode_remote, router_id, encode_input, plugin_timing_metrics)
        encode_output = encode_future.result()

        if encode_output is None:
            return  # encode failed

        # Phase 3: post-encode processing runs inline.
        try:
            self._ksana_plugin[router_id].preprocess_after_encode(
                encode_output=encode_output, context=context, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Plugin preprocess_after_encode failed: {e}")

    def preprocess(self,
                   ksana_python_input: libtorch_serving.KsanaPythonInput,
                   **kwargs):
        """
        Invokes the preprocess method of the loaded plugin, if available.
        Automatically uses decoupled path when the plugin supports it.
        """
        if self._ksana_plugin is None:
            return

        if self._supports_decoupled:
            return self._preprocess_decoupled(ksana_python_input, **kwargs)

        kwargs.update({
            "ksana_python_input": ksana_python_input,
        })
        submit_time = time.time()
        future = self._thread_pool.submit(self._preprocess, submit_time, self.preprocess_router.get_next(), **kwargs)
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
