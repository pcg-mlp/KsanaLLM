# pylint: disable=broad-except
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""Embedded encoder management utilities.

This module provides functions for the embedded encoder mode, where
serving_server automatically manages encoder subprocess lifecycle instead
of requiring separate manual encoder_server startup.
"""

import ctypes
import ctypes.util
import os
import signal
import subprocess
import sys
import time

import yaml

try:
    from utilize.logger import get_logger
    logger = get_logger("ksana_encoder")
except ImportError:
    import logging
    logger = logging.getLogger("ksana_encoder")

# Cache libc handle at module level to avoid repeated lookups in preexec_fn.
try:
    _LIBC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
except Exception:
    _LIBC = None


def pre_configure_cuda():
    """Pre-read YAML config to set CUDA_VISIBLE_DEVICES before torch import.

    When embedded encoder mode is configured (plugin_model.encoder section
    in YAML), restricts CUDA_VISIBLE_DEVICES to llm_gpu only.  This MUST
    happen before importing torch, as CUDA runtime caches device count at
    first initialization.
    """
    config_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config_file" and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break
        if arg.startswith("--config_file="):
            config_file = arg.split("=", 1)[1]
            break

    if not config_file or not os.path.exists(config_file):
        return

    with open(config_file) as f:
        yaml_config = yaml.safe_load(f)

    llm_gpu = (
        yaml_config.get("model_spec", {})
        .get("plugin_model", {})
        .get("encoder", {})
        .get("llm_gpu")
    )
    if llm_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = llm_gpu
        os.environ["XPU_VISIBLE_DEVICES"] = llm_gpu


def _parse_encoder_endpoints(engine_args):
    """Parse encoder GPU list and compute (host, port) pairs.

    Returns:
        tuple: (encoder_gpus, host, base_port)
            encoder_gpus: list of GPU id strings, e.g. ["6", "7"]
            host: bind host string
            base_port: int starting port
    """
    encoder_gpus = [g.strip() for g in engine_args.encoder_gpu.split(",")]
    return encoder_gpus, engine_args.host, engine_args.encoder_base_port


def _set_pdeathsig():
    """preexec_fn: set PR_SET_PDEATHSIG so the child receives SIGTERM
    when the parent dies (including SIGKILL)."""
    if _LIBC is not None:
        try:
            _LIBC.prctl(1, signal.SIGTERM)   # 1 == PR_SET_PDEATHSIG
        except Exception:
            pass


def start_encoder_processes(engine_args):
    """Start encoder subprocesses, one per GPU in *encoder_gpu*.

    Each encoder gets its own ``CUDA_VISIBLE_DEVICES`` and ZMQ port.

    Returns:
        list[subprocess.Popen]
    """
    encoder_gpus, host, base_port = _parse_encoder_endpoints(engine_args)

    encoder_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "encoder_server.py",
    )
    if not os.path.exists(encoder_script):
        raise FileNotFoundError(
            f"Encoder server script not found: {encoder_script}"
        )

    procs = []
    for i, gpu_id in enumerate(encoder_gpus):
        port = base_port + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["XPU_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            sys.executable, "-u", encoder_script,
            "--config_file", engine_args.config_file,
            "--host", host,
            "--port", str(port),
        ]

        logger.info(f"Starting encoder {i} on GPU {gpu_id}, port {port}")
        proc = subprocess.Popen(cmd, env=env, preexec_fn=_set_pdeathsig)
        procs.append(proc)

    return procs


def wait_encoders_ready(engine_args, encoder_processes, timeout=120):
    """Block until every encoder's ZMQ ROUTER socket is reachable.

    Uses a per-encoder DEALER socket monitor watching for EVENT_CONNECTED.

    Raises:
        RuntimeError: If any encoder is unreachable or exits prematurely.
    """
    import zmq
    from zmq.utils.monitor import recv_monitor_message

    urls = build_encoder_urls(engine_args)

    ctx = zmq.Context()
    try:
        for i, url in enumerate(urls):
            sock = ctx.socket(zmq.DEALER)
            monitor = sock.get_monitor_socket()
            sock.connect(url)

            deadline = time.time() + timeout
            ready = False
            while time.time() < deadline:
                if monitor.poll(timeout=1000):
                    evt = recv_monitor_message(monitor)
                    if evt["event"] == zmq.EVENT_CONNECTED:
                        ready = True
                        break

                if (i < len(encoder_processes)
                        and encoder_processes[i].poll() is not None):
                    raise RuntimeError(
                        f"Encoder {i} ({url}) exited prematurely "
                        f"with return code {encoder_processes[i].returncode}"
                    )

            sock.disable_monitor()
            monitor.close()
            sock.close()

            if not ready:
                raise RuntimeError(
                    f"Encoder at {url} not ready within {timeout}s"
                )
            logger.info(f"Encoder {i} at {url} is ready")
    finally:
        ctx.term()


def build_encoder_urls(engine_args):
    """Build the encoder URL list from configuration.

    Returns:
        list[str], e.g. ``["tcp://127.0.0.1:30000", "tcp://127.0.0.1:30001"]``
    """
    encoder_gpus, host, base_port = _parse_encoder_endpoints(engine_args)
    return [f"tcp://{host}:{base_port + i}" for i in range(len(encoder_gpus))]


def cleanup_encoder_processes(encoder_processes):
    """Gracefully terminate, then force-kill encoder subprocesses."""
    for proc in encoder_processes:
        if proc.poll() is None:
            logger.info(f"Terminating encoder process (pid={proc.pid})")
            proc.terminate()
    for proc in encoder_processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Force killing encoder process (pid={proc.pid})")
            proc.kill()
