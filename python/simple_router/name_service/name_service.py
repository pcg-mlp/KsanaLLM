# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class NameServiceProvider(ABC):
    """Abstract base class for discovering serving nodes."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_available_nodes(
        self, **kwargs: Any
    ) -> Tuple[Tuple[str, str], Tuple[str, str], Optional[Any], Optional[Any]]:
        """Return a selected prefill/decode node pair."""

    def update_nodes_call_result(
        self,
        prefill_instance: Optional[Any],
        decode_instance: Optional[Any],
        prefill_success: bool,
        decode_success: bool,
        **kwargs: Any,
    ) -> None:
        """Optional hook invoked after requests are sent."""
        # Default implementation does nothing.


class NameServiceRegistry:
    """Helper to load providers via dotted path string."""

    @classmethod
    def get_provider_from_config(cls, module_path: str) -> NameServiceProvider:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import name service provider module '{module_path}'"
            ) from exc

        if not hasattr(module, "create_provider"):
            raise AttributeError(
                f"Module '{module_path}' does not expose a create_provider() function"
            )

        provider = module.create_provider()
        if not isinstance(provider, NameServiceProvider):
            raise TypeError(
                f"create_provider() in '{module_path}' returned {type(provider).__name__}, "
                f"expected a NameServiceProvider instance"
            )

        logger.info("Loaded name service provider: %s", provider.name)
        return provider
