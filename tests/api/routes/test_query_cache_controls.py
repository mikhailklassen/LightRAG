"""Per-request ``bypass_cache`` and ``temperature`` on /query and /query/stream.

Covers the API surface (validation, OpenAPI, threading into QueryParam) and the
server's LLM wrappers, which must let a per-call temperature override the
configured provider default instead of the other way round.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

pytestmark = pytest.mark.offline

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "OPENAI_LLM_TEMPERATURE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    # Importing the API package parses CLI args; keep pytest's argv out of it.
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])


def _query_request(**fields):
    from lightrag.api.routers.query_routes import QueryRequest

    return QueryRequest(**fields)


def _create_app(mock_rag):
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        with patch(
            "lightrag.api.lightrag_server.LightRAG", return_value=mock_rag
        ) as rag_cls:
            app = create_app(args)
        return app, rag_cls
    finally:
        sys.argv = original_argv


def _client_capturing_params():
    mock_rag = MagicMock()
    captured = {}

    async def _fake_aquery_llm(query, param, system_prompt=None):
        captured["param"] = param
        return {
            "llm_response": {"is_streaming": False, "content": "ok"},
            "data": {"references": []},
        }

    mock_rag.aquery_llm = AsyncMock(side_effect=_fake_aquery_llm)
    app, _ = _create_app(mock_rag)
    return TestClient(app), captured


def test_query_request_threads_fields_into_query_param():
    param = _query_request(
        query="what is it", bypass_cache=True, temperature=0.4
    ).to_query_params(is_stream=False)

    assert param.bypass_cache is True
    assert param.temperature == 0.4


def test_query_request_defaults_leave_cache_and_temperature_alone():
    param = _query_request(query="what is it").to_query_params(is_stream=False)

    assert param.bypass_cache is False
    assert param.temperature is None


@pytest.mark.parametrize("bad", [-0.1, 2.5])
def test_query_request_rejects_out_of_range_temperature(bad):
    with pytest.raises(ValidationError):
        _query_request(query="what is it", temperature=bad)


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
def test_routes_pass_bypass_cache_and_temperature(path):
    client, captured = _client_capturing_params()

    response = client.post(
        path, json={"query": "what is it", "bypass_cache": True, "temperature": 0.7}
    )

    assert response.status_code == 200
    assert captured["param"].bypass_cache is True
    assert captured["param"].temperature == 0.7


def test_openapi_documents_new_fields():
    client, _ = _client_capturing_params()
    schema = client.get("/openapi.json").json()["components"]["schemas"][
        "QueryRequest"
    ]["properties"]

    assert "bypass_cache" in schema
    assert "temperature" in schema
    assert "response LLM call" in schema["temperature"]["description"]


@pytest.mark.asyncio
async def test_openai_wrapper_lets_per_call_temperature_override_configured(
    monkeypatch,
):
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_LLM_TEMPERATURE", "1.0")
    _, rag_cls = _create_app(MagicMock())
    llm_model_func = rag_cls.call_args.kwargs["llm_model_func"]

    fake_complete = AsyncMock(return_value="ok")
    with patch("lightrag.llm.openai.openai_complete_if_cache", fake_complete):
        await llm_model_func("hello", temperature=0.1)
        await llm_model_func("hello")

    with_override, without_override = fake_complete.call_args_list
    assert with_override.kwargs["temperature"] == 0.1
    assert without_override.kwargs["temperature"] == 1.0
