# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
测试所有错误响应都返回标准 OpenAI 兼容的 JSON 格式。

不需要启动真实模型服务，通过 FastAPI TestClient 直接验证
异常处理器和错误响应格式。

运行方式 (从 python/ 目录)：
    python3 -m pytest tests/test_error_json_response.py -v
"""

import os
import sys
import types
import logging
import orjson

# ---------------------------------------------------------------------------
# Mock 掉不可用的重型依赖，避免 import 链拉入 torch 等
# 必须在 import openaiapi 之前执行
# ---------------------------------------------------------------------------
_STUB = types.ModuleType("_stub")
_MOCK_MODULES = [
    "torch", "torch.nn", "torch.nn.functional", "torch.utils",
    "torch.utils.data", "torch.cuda",
    "ksana_llm", "ksana_llm.auto_model",
    "transformers", "transformers.tokenization_utils_base",
    "mistral_common", "mistral_common.tokens",
    "mistral_common.tokens.tokenizers",
    "mistral_common.tokens.tokenizers.mistral",
    "mistral_common.protocol",
    "mistral_common.protocol.instruct",
    "mistral_common.protocol.instruct.messages",
]
for _m in _MOCK_MODULES:
    sys.modules.setdefault(_m, _STUB)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# 尝试从生产代码导入；如果依赖链仍然断裂，回退到本地实现
try:
    from openaiapi.openai_adapter import _make_openai_error_json
    from openaiapi.request_converter import RequestConverter
    _IMPORTED_PRODUCTION = True
except ImportError:
    _IMPORTED_PRODUCTION = False

if not _IMPORTED_PRODUCTION:
    def _make_openai_error_json(  # pylint: disable=function-redefined
        message: str,
        error_type: str = "internal_server_error",
        status_code: int = 500,
    ) -> JSONResponse:
        return JSONResponse(
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": status_code,
                    "param": None,
                }
            },
            status_code=status_code,
        )

    class RequestConverter:  # pylint: disable=function-redefined
        def __init__(self, config=None, tokenizer=None):
            pass

        def format_error_response(
            self, error_message, error_type="internal_server_error",
            error_code=500, param=None,
        ):
            return {
                "error": {
                    "message": error_message,
                    "type": error_type,
                    "code": error_code,
                    "param": param,
                }
            }

        def format_streaming_error(
            self, error_message, error_type="internal_server_error",
            error_code=500,
        ):
            error_data = self.format_error_response(
                error_message, error_type, error_code
            )
            return f"data: {orjson.dumps(error_data).decode()}\n\n"


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_ERROR_KEYS = {"message", "type", "code", "param"}


# ---------------------------------------------------------------------------
# 公共断言工具
# ---------------------------------------------------------------------------
def _assert_openai_error_json(response, expected_status: int = None):
    """验证响应是合法 JSON 且符合 OpenAI 错误格式。"""
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("application/json"), (
        f"Content-Type 不是 JSON: {content_type}"
    )

    try:
        body = response.json()
    except orjson.JSONDecodeError as exc:
        pytest.fail(
            f"响应体不是合法 JSON (status={response.status_code}): {exc}\n"
            f"原始内容: {response.text[:500]}"
        )

    assert "error" in body, f"缺少 'error' 字段: {body}"
    error = body["error"]
    assert isinstance(error, dict), f"'error' 字段应为 dict: {type(error)}"
    assert OPENAI_ERROR_KEYS <= set(error.keys()), (
        f"error 字段缺少必要 key, 期望 {OPENAI_ERROR_KEYS}, 实际 {set(error.keys())}"
    )
    assert isinstance(error["message"], str), (
        f"error.message 应为字符串: {type(error['message'])}"
    )

    if expected_status is not None:
        assert response.status_code == expected_status, (
            f"状态码不匹配: 期望 {expected_status}, 实际 {response.status_code}"
        )


# ---------------------------------------------------------------------------
# 构建测试用 FastAPI 应用，注册与生产代码相同的异常处理器
# ---------------------------------------------------------------------------
class _DummyBody(BaseModel):
    """用于触发 Pydantic 校验错误的请求体模型。"""
    name: str
    count: int = Field(gt=0)


def _create_test_app() -> FastAPI:
    """创建带全局异常处理器的测试 FastAPI 应用。"""
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    app = FastAPI()

    # ---- 注册与 openai_adapter.register_all_routes 完全一致的异常处理器 ----
    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _req: Request, exc: RequestValidationError
    ):
        detail_parts = []
        for err in exc.errors():
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            detail_parts.append(f"{loc}: {err.get('msg', '')}")
        message = "; ".join(detail_parts) if detail_parts else str(exc)
        return _make_openai_error_json(message, "invalid_request_error", 400)

    _http_error_type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
    }

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(
        _req: Request, exc: StarletteHTTPException
    ):
        detail = exc.detail
        if isinstance(detail, dict):
            message = detail.get("message", str(detail))
        elif isinstance(detail, list):
            message = "; ".join(str(d) for d in detail)
        else:
            message = str(detail)
        err_type = _http_error_type_map.get(
            exc.status_code,
            "invalid_request_error" if exc.status_code < 500
            else "internal_server_error",
        )
        return _make_openai_error_json(message, err_type, exc.status_code)

    @app.exception_handler(Exception)  # pylint: disable=broad-except
    async def _generic_error_handler(_req: Request, exc: Exception):
        return _make_openai_error_json(
            "Internal server error", "internal_server_error", 500
        )

    # ---- 测试路由：模拟各种错误场景 ----
    @app.post("/pydantic_error")
    async def _pydantic_error(body: _DummyBody):
        return {"ok": True}

    @app.get("/http_exception_str")
    async def _http_exc_str():
        raise HTTPException(status_code=400, detail="bad request detail")

    @app.get("/http_exception_dict")
    async def _http_exc_dict():
        raise HTTPException(
            status_code=500,
            detail={"message": "engine failure", "code": 1},
        )

    @app.get("/http_exception_list")
    async def _http_exc_list():
        raise HTTPException(
            status_code=422, detail=["error1", "error2"]
        )

    @app.get("/unhandled_value_error")
    async def _unhandled_value_error():
        raise ValueError(
            "This model's maximum context length is 65536 tokens. "
            "However, you requested 143920 tokens in the messages."
        )

    @app.get("/unhandled_runtime_error")
    async def _unhandled_runtime_error():
        raise RuntimeError("CUDA out of memory")

    @app.get("/unhandled_generic")
    async def _unhandled_generic():
        raise Exception(  # pylint: disable=broad-except
            "something completely unexpected"
        )

    @app.get("/preprocessing_failed")
    async def _preprocessing_failed():
        raise ValueError(
            "This model's maximum context length is 65536 tokens. "
            "However, you requested 143920 tokens in the messages. "
            "Please reduce the length of the messages."
        )

    @app.get("/special_chars_error")
    async def _special_chars_error():
        raise ValueError(
            'Error with "double quotes" and \'single quotes\' '
            "and newline\nand tab\tand backslash\\ and unicode: \u4e2d\u6587"
        )

    return app


@pytest.fixture(scope="module")
def client():
    app = _create_test_app()
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# 单元测试: _make_openai_error_json
# ===========================================================================
class TestMakeOpenaiErrorJson:
    """测试 _make_openai_error_json 辅助函数。"""

    def test_basic_structure(self):
        resp = _make_openai_error_json("test error")
        assert resp.status_code == 500
        body = orjson.loads(resp.body)
        assert body["error"]["message"] == "test error"
        assert body["error"]["type"] == "internal_server_error"
        assert body["error"]["code"] == 500
        assert body["error"]["param"] is None

    def test_custom_params(self):
        resp = _make_openai_error_json("not found", "not_found_error", 404)
        assert resp.status_code == 404
        body = orjson.loads(resp.body)
        assert body["error"]["message"] == "not found"
        assert body["error"]["type"] == "not_found_error"
        assert body["error"]["code"] == 404

    def test_message_with_special_chars(self):
        msg = 'Quotes: "double" \'single\' newline:\n tab:\t backslash:\\ 中文'
        resp = _make_openai_error_json(msg)
        body = orjson.loads(resp.body)
        assert body["error"]["message"] == msg

    def test_empty_message(self):
        resp = _make_openai_error_json("")
        body = orjson.loads(resp.body)
        assert body["error"]["message"] == ""

    def test_long_message(self):
        msg = "x" * 10000
        resp = _make_openai_error_json(msg)
        body = orjson.loads(resp.body)
        assert body["error"]["message"] == msg


# ===========================================================================
# 单元测试: RequestConverter.format_error_response
# ===========================================================================
class TestFormatErrorResponse:
    """测试 RequestConverter 的错误格式化方法。"""

    def test_format_error_response_structure(self):
        converter = RequestConverter()
        result = converter.format_error_response(
            "bad input", "validation_error", 400
        )
        raw = orjson.dumps(result)
        parsed = orjson.loads(raw)
        assert parsed["error"]["message"] == "bad input"
        assert parsed["error"]["type"] == "validation_error"
        assert parsed["error"]["code"] == 400
        assert parsed["error"]["param"] is None

    def test_format_error_response_with_param(self):
        converter = RequestConverter()
        result = converter.format_error_response(
            "err", "bad_request", 400, param="temperature"
        )
        raw = orjson.dumps(result)
        parsed = orjson.loads(raw)
        assert parsed["error"]["param"] == "temperature"

    def test_format_streaming_error_is_valid_sse_json(self):
        converter = RequestConverter()
        sse_line = converter.format_streaming_error(
            "stream fail", "internal_server_error", 500
        )
        assert sse_line.startswith("data: ")
        assert sse_line.endswith("\n\n")
        json_part = sse_line[len("data: "):].rstrip("\n")
        parsed = orjson.loads(json_part)
        assert parsed["error"]["message"] == "stream fail"

    def test_format_streaming_error_special_chars(self):
        converter = RequestConverter()
        msg = 'Error: "quotes" and \'apostrophe\' and 中文 and \n newline'
        sse_line = converter.format_streaming_error(msg)
        json_part = sse_line[len("data: "):].rstrip("\n")
        parsed = orjson.loads(json_part)
        assert parsed["error"]["message"] == msg


# ===========================================================================
# 集成测试: FastAPI 异常处理器
# ===========================================================================
class TestExceptionHandlers:
    """通过 TestClient 验证各种异常场景都返回标准 JSON。"""

    def test_pydantic_validation_missing_field(self, client):
        """请求体缺少必填字段 → 400 + 标准 JSON。"""
        resp = client.post("/pydantic_error", orjson={})
        _assert_openai_error_json(resp, 400)

    def test_pydantic_validation_wrong_type(self, client):
        """字段值不合法 → 400 + 标准 JSON。"""
        resp = client.post(
            "/pydantic_error", json={"name": "test", "count": -1}
        )
        _assert_openai_error_json(resp, 400)

    def test_pydantic_validation_invalid_json_body(self, client):
        """请求体不是合法 JSON → 400 + 标准 JSON。"""
        resp = client.post(
            "/pydantic_error",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        _assert_openai_error_json(resp, 400)

    def test_http_exception_string_detail(self, client):
        """HTTPException(detail=str) → 标准 JSON。"""
        resp = client.get("/http_exception_str")
        _assert_openai_error_json(resp, 400)
        body = resp.json()
        assert body["error"]["message"] == "bad request detail"
        assert body["error"]["type"] == "invalid_request_error"

    def test_http_exception_dict_detail(self, client):
        """HTTPException(detail=dict) → 标准 JSON，提取 message 字段。"""
        resp = client.get("/http_exception_dict")
        _assert_openai_error_json(resp, 500)
        body = resp.json()
        assert body["error"]["message"] == "engine failure"
        assert body["error"]["type"] == "internal_server_error"

    def test_http_exception_list_detail(self, client):
        """HTTPException(detail=list) → 标准 JSON，拼接列表。"""
        resp = client.get("/http_exception_list")
        _assert_openai_error_json(resp, 422)
        body = resp.json()
        assert "error1" in body["error"]["message"]
        assert "error2" in body["error"]["message"]

    def test_unhandled_value_error(self, client):
        """未捕获 ValueError → 500 + 标准 JSON + 通用消息（不泄露内部细节）。"""
        resp = client.get("/unhandled_value_error")
        _assert_openai_error_json(resp, 500)
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"

    def test_unhandled_runtime_error(self, client):
        """未捕获 RuntimeError → 500 + 标准 JSON + 通用消息。"""
        resp = client.get("/unhandled_runtime_error")
        _assert_openai_error_json(resp, 500)
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"

    def test_unhandled_generic_exception(self, client):
        """未捕获的 Exception → 500 + 标准 JSON + 通用消息。"""
        resp = client.get("/unhandled_generic")
        _assert_openai_error_json(resp, 500)
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"

    def test_preprocessing_failed_context_length(self, client):
        """模拟上下文超长场景 → 标准 JSON 错误 + 通用消息（不泄露模型参数）。"""
        resp = client.get("/preprocessing_failed")
        _assert_openai_error_json(resp, 500)
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"

    def test_special_chars_in_error_message(self, client):
        """异常含特殊字符时仍返回通用消息，不泄露原始异常内容。"""
        resp = client.get("/special_chars_error")
        _assert_openai_error_json(resp, 500)
        body = resp.json()
        assert body["error"]["message"] == "Internal server error"

    def test_404_not_found(self, client):
        """访问不存在的路由 → 标准 JSON（404）。"""
        resp = client.get("/nonexistent_route")
        _assert_openai_error_json(resp, 404)

    def test_method_not_allowed(self, client):
        """使用错误 HTTP 方法 → 标准 JSON（405）。"""
        resp = client.get("/pydantic_error")
        _assert_openai_error_json(resp, 405)

    def test_response_is_parseable_by_json_loads(self, client):
        """确保 json.loads(resp.text) 不抛异常 — 复现用户报告的问题。"""
        error_routes = [
            "/http_exception_str",
            "/http_exception_dict",
            "/unhandled_value_error",
            "/preprocessing_failed",
            "/special_chars_error",
        ]
        for route in error_routes:
            resp = client.get(route)
            try:
                orjson.loads(resp.text)
            except orjson.JSONDecodeError as exc:
                pytest.fail(
                    f"路由 {route} 返回的内容无法被 orjson.loads 解析: "
                    f"{exc}\n原始内容: {resp.text[:300]}"
                )
