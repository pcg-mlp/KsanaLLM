# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/abstract_tool_parser.py

import os
from json import JSONDecodeError
from collections.abc import Sequence
from typing import Callable, Optional, Union, Dict, List

import orjson

from openaiapi.openai_protocol import (ChatCompletionRequest,
                                       ChatCompletionToolsParam,
                                       DeltaFunctionCall,
                                       DeltaMessage,
                                       DeltaToolCall,
                                       ExtractedToolCallInformation)

from utilize.utils import import_from_path, is_list_of
from utilize.logger import get_logger

logger = get_logger(__name__)


class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    # XGrammar v0.2.5 model-format key. Parsers opt in only when their output
    # syntax exactly matches one of XGrammar's native structural-tag builders.
    # See csrc/runtime/structured_generation/README.md before adding a mapping.
    structural_tag_model: Optional[str] = None

    def __init__(self):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.temp_text: str = ""
        self.temp_tokens: Sequence[int] = []

        # Model structural tags are byte strings and may span token or stream
        # boundaries. Parsers with an outer tool section opt into the text
        # state machine through ``extract_structural_tag_streaming``.
        self.model_structural_tag_enabled: bool = False
        self._structural_tag_content_offset: Optional[int] = None
        self._structural_tag_start: Optional[int] = None
        self._structural_tag_end_search_offset: Optional[int] = None
        self._structural_tag_emitted: bool = False

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Static method that used to adjust the request parameters.
        """
        return request

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls_streaming has not been "
            "implemented!")

    @staticmethod
    def _partial_marker_length(text: str, marker: str) -> int:
        """Return the longest suffix of text that can begin marker."""
        for length in range(min(len(text), len(marker) - 1), 0, -1):
            if marker.startswith(text[-length:]):
                return length
        return 0

    def extract_structural_tag_streaming(
        self,
        previous_text: str,
        current_text: str,
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """Extract a complete outer structural-tag tool section.

        Text that may be the beginning of the outer marker is withheld until
        it can be classified. Once the outer end marker arrives, the existing
        full-response parser converts the complete section to tool-call deltas.
        """
        if self._structural_tag_emitted:
            return None

        start_marker = getattr(self, "tool_calls_start_token", None)
        end_marker = getattr(self, "tool_calls_end_token", None)
        if not isinstance(start_marker, str) or not isinstance(end_marker, str):
            return None

        if self._structural_tag_content_offset is None:
            self._structural_tag_content_offset = len(previous_text)

        content_offset = self._structural_tag_content_offset
        if self._structural_tag_start is None:
            start = current_text.find(start_marker, content_offset)
            if start < 0:
                pending = current_text[content_offset:]
                hold = self._partial_marker_length(pending, start_marker)
                safe_end = len(current_text) - hold
                if safe_end == content_offset:
                    return None
                self._structural_tag_content_offset = safe_end
                return DeltaMessage(content=current_text[content_offset:safe_end])
            self._structural_tag_start = start
            self._structural_tag_end_search_offset = start + len(start_marker)

        start = self._structural_tag_start
        content = current_text[content_offset:start] or None
        self._structural_tag_content_offset = start

        end = current_text.find(end_marker, self._structural_tag_end_search_offset)
        if end < 0:
            # Only an end-marker prefix at the current suffix can participate
            # in a match after more text arrives. Resume there instead of
            # rescanning the complete tool section on every stream update.
            self._structural_tag_end_search_offset = max(
                self._structural_tag_end_search_offset,
                len(current_text) - len(end_marker) + 1,
            )
            return DeltaMessage(content=content) if content else None

        end += len(end_marker)
        parsed = self.extract_tool_calls(current_text[start:end], request)
        self._structural_tag_emitted = True
        self._structural_tag_content_offset = end

        if not parsed.tools_called or not parsed.tool_calls:
            logger.warning("Unable to parse complete structural-tag tool section")
            return DeltaMessage(content=content) if content else None

        tool_calls = []
        for tool_call in parsed.tool_calls:
            index = len(self.prev_tool_call_arr)
            arguments = tool_call.function.arguments
            self.prev_tool_call_arr.append({
                "name": tool_call.function.name,
                "arguments": arguments,
            })
            self.streamed_args_for_tool.append(arguments)
            self.current_tool_id = index
            self.current_tool_name_sent = True
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=arguments,
                    ),
                ))

        return DeltaMessage(content=content, tool_calls=tool_calls)

    def flush_streaming(self, current_text: str) -> Union[DeltaMessage, None]:
        """Flush normal text withheld while matching a structural marker."""
        content_offset = self._structural_tag_content_offset
        if (not self.model_structural_tag_enabled
                or content_offset is None
                or self._structural_tag_start is not None
                or self._structural_tag_emitted):
            return None

        content = current_text[content_offset:]
        self._structural_tag_content_offset = len(current_text)
        return DeltaMessage(content=content) if content else None

    def is_json_tool_call_args(self, arguments: str) -> bool:
        """
        Method that should be implemented for checking if a tool call arguments
        is a JSON tool call.
        """
        try:
            orjson.loads(arguments)
            return True
        except JSONDecodeError:
            return False

    def _get_tool_indices(self, tools: List[ChatCompletionToolsParam]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }


class ToolParserManager:
    tool_parsers: dict[str, type] = {}

    @classmethod
    def get_tool_parser(cls, name) -> type:
        """
        Get tool parser by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        raise KeyError(f"tool helper: '{name}' not found in tool_parsers")

    @classmethod
    def _register_module(cls,
                         module: type,
                         module_name: Optional[Union[str, list[str]]] = None,
                         force: bool = True) -> None:
        if not issubclass(module, ToolParser):
            raise TypeError(
                f'module must be subclass of ToolParser, but got {type(module)}'
            )
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.tool_parsers:
                existed_module = cls.tool_parsers[name]
                raise KeyError(f'{name} is already registered '
                               f'at {existed_module.__module__}')
            cls.tool_parsers[name] = module

    @classmethod
    def register_module(
            cls,
            name: Optional[Union[str, list[str]]] = None,
            force: bool = True,
            module: Union[type, None] = None) -> Union[type, Callable]:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not 
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)
                or is_list_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined tool parser by the path of the tool parser define
        file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except (ImportError, ModuleNotFoundError, AttributeError, SyntaxError):
            logger.exception("Failed to load module '%s' from %s.",
                             module_name, plugin_path)
            return
