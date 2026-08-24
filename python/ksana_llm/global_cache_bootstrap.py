# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""ENABLE_GLOBAL_CACHE_HOOK=1 时启动本地 Mooncake master。"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import time

logger = logging.getLogger(__name__)


def ensure_mooncake_master(
    http_port: int | None = None,
    rpc_port: int | None = None,
    metrics_port: int | None = None,
) -> None:
    """Hook 开启时拉起 mooncake_master；RPC 端口已监听则跳过。"""
    if os.environ.get("ENABLE_GLOBAL_CACHE_HOOK", "0") != "1":
        return

    if http_port is None or rpc_port is None or metrics_port is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mooncake-http-port", type=int, default=8080)
        parser.add_argument("--mooncake-rpc-port", type=int, default=50051)
        parser.add_argument("--mooncake-metrics-port", type=int, default=9003)
        args, _ = parser.parse_known_args()
        http_port = http_port if http_port is not None else args.mooncake_http_port
        rpc_port = rpc_port if rpc_port is not None else args.mooncake_rpc_port
        metrics_port = metrics_port if metrics_port is not None else args.mooncake_metrics_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", rpc_port)) == 0:
            logger.info("[mooncake] master already up on 127.0.0.1:%s", rpc_port)
            return

    master_bin = os.environ.get("MOONCAKE_MASTER_BIN", "").strip() or shutil.which("mooncake_master")
    if not master_bin or not os.path.isfile(master_bin):
        raise RuntimeError(
            "mooncake_master not found; set MOONCAKE_MASTER_BIN or install mooncake_master into PATH"
        )

    log_dir = "log"
    if os.environ.get("KLLM_LOG_FILE"):
        parent = os.path.dirname(os.path.abspath(os.environ["KLLM_LOG_FILE"]))
        if parent:
            log_dir = parent
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "mooncake_master.log")

    cmd = [
        master_bin,
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        f"--http_metadata_server_port={http_port}",
        f"--rpc_port={rpc_port}",
        "--rpc_address=0.0.0.0",
        "--rpc_thread_num=8",
        f"--metrics_port={metrics_port}",
        "--enable_metric_reporting=false",
    ]
    with open(log_path, "a", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True
        )

    ready = f"Master service started on port {rpc_port}"
    for i in range(1, 31):
        if proc.poll() is not None:
            raise RuntimeError(f"mooncake_master exited early, see {log_path}")
        with open(log_path, "r", encoding="utf-8") as log_fp:
            if ready in log_fp.read():
                logger.info("[mooncake] master pid=%s ready (%ss)", proc.pid, i)
                return
        time.sleep(1)
    raise RuntimeError(f"mooncake_master not ready within 30s, see {log_path}")
