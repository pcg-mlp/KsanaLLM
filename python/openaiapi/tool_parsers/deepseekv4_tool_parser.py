# Adapted from DeepSeekV32ToolParser for DeepSeek-V4 DSML format.
# V4 uses <｜DSML｜tool_calls> instead of <｜DSML｜function_calls>.
from openaiapi.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from openaiapi.tool_parsers.abstract_tool_parser import ToolParserManager

from utilize.logger import get_logger

logger = get_logger(__name__)


@ToolParserManager.register_module("deepseek_v4")
class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    Detector for DeepSeek V4 DSML tool-call format.

    Identical to V3.2 except the outer block wrapper is
    ``<｜DSML｜tool_calls>...</｜DSML｜tool_calls>`` instead of
    ``<｜DSML｜function_calls>...</｜DSML｜function_calls>``. The inner
    ``<｜DSML｜invoke>`` / ``<｜DSML｜parameter>`` shape is unchanged.

    Example (XML parameters):
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
        </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
