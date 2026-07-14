# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import logging
from typing import Any, Tuple

import config
from .name_service import NameServiceProvider

try:  # pragma: no cover - optional dependency
    from polaris.api.consumer import (
        GetOneInstanceRequest,
        ServiceCallResult,
        create_consumer_by_default_config_file,
    )
    from polaris.wrapper import POLARIS_CALL_RET_ERROR, POLARIS_CALL_RET_OK
except ImportError:  # pragma: no cover - optional dependency not installed
    GetOneInstanceRequest = None  # type: ignore
    ServiceCallResult = None  # type: ignore
    create_consumer_by_default_config_file = None  # type: ignore
    POLARIS_CALL_RET_ERROR = -1  # type: ignore
    POLARIS_CALL_RET_OK = 0  # type: ignore


logger = logging.getLogger(__name__)

# consumer api 内部有缓存，全局只使用一个consumer api对象
G_CONSUMER_API = None


def get_consumer_api() -> Any:
    global G_CONSUMER_API
    if not G_CONSUMER_API:
        G_CONSUMER_API = create_consumer_by_default_config_file()
    return G_CONSUMER_API


class PolarisNameServiceProvider(NameServiceProvider):
    """Polaris-backed provider."""

    def __init__(self) -> None:
        super().__init__("polaris")
        if create_consumer_by_default_config_file:
            self._consumer_api = get_consumer_api()
        else:
            self._consumer_api = None

    def get_available_nodes(
        self, **kwargs: Any
    ) -> Tuple[Tuple[str, str], Tuple[str, str], Any, Any]:
        if self._consumer_api is None or GetOneInstanceRequest is None:
            raise RuntimeError("Polaris SDK is not available")

        settings = config.get_settings()
        namespace = settings.namespace
        prefill_service = settings.prefill_service
        decode_service = settings.decode_service

        try:
            request = GetOneInstanceRequest(
                namespace=namespace, service=prefill_service
            )
            prefill_instance = self._consumer_api.get_one_instance(request)
            prefill_node = (
                f"{prefill_instance.get_host()}:{prefill_instance.get_port()}"
            )

            request = GetOneInstanceRequest(namespace=namespace, service=decode_service)
            decode_instance = self._consumer_api.get_one_instance(request)
            decode_node = f"{decode_instance.get_host()}:{decode_instance.get_port()}"
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error getting available nodes from Polaris: %s", exc)
            raise RuntimeError("Failed to retrieve nodes from Polaris") from exc

        return (
            (prefill_node, prefill_node),
            (decode_node, decode_node),
            prefill_instance,
            decode_instance,
        )

    def update_nodes_call_result(
        self,
        prefill_instance: Any,
        decode_instance: Any,
        prefill_success: bool,
        decode_success: bool,
        **kwargs: Any,
    ) -> None:
        if self._consumer_api is None or ServiceCallResult is None:
            return

        settings = config.get_settings()
        namespace = settings.namespace

        try:
            if prefill_instance:
                result = ServiceCallResult(
                    namespace, settings.prefill_service, prefill_instance.get_id()
                )
                result.set_ret_status(
                    POLARIS_CALL_RET_OK if prefill_success else POLARIS_CALL_RET_ERROR
                )
                self._consumer_api.update_service_call_result(result)

            if decode_instance:
                result = ServiceCallResult(
                    namespace, settings.decode_service, decode_instance.get_id()
                )
                result.set_ret_status(
                    POLARIS_CALL_RET_OK if decode_success else POLARIS_CALL_RET_ERROR
                )
                self._consumer_api.update_service_call_result(result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error updating Polaris service call result: %s", exc)


def create_provider() -> NameServiceProvider:
    return PolarisNameServiceProvider()
