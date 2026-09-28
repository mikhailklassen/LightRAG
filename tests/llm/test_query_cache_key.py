"""Query-response cache key and per-request cache/temperature controls.

The response cache must be partitioned by everything that changes the LLM
input: the system prompt template and the conversation history, not only the
query text and retrieval parameters. ``bypass_cache`` skips the response cache
entirely, and ``temperature`` reaches the response LLM call only.
"""

import json

import pytest

import lightrag.operate as operate
from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query

SYSTEM_PROMPT_A = "Persona A.\n{context_data}\n{response_type}{user_prompt}"
SYSTEM_PROMPT_B = "Persona B.\n{context_data}\n{response_type}{user_prompt}"
NAIVE_PROMPT_A = "Persona A.\n{content_data}\n{response_type}{user_prompt}"
NAIVE_PROMPT_B = "Persona B.\n{content_data}\n{response_type}{user_prompt}"


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}
        self.reads = 0

    async def get_by_id(self, key):
        self.reads += 1
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)

    def query_entries(self) -> dict:
        return {k: v for k, v in self._store.items() if ":query:" in k}


class _FakeChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "Query cache key test chunk.",
                "file_path": "test.md",
            }
        ]


class _RecordingLLM:
    """Fake response LLM that records kwargs and returns a distinct answer per call."""

    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return f"answer-{len(self.calls)}"


class _RecordingKeywordLLM:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return json.dumps(
            {"high_level_keywords": ["painting"], "low_level_keywords": ["mona lisa"]}
        )


def _global_config(query_llm, keyword_llm=None) -> dict:
    identity = {
        "binding": "openai",
        "model": "model-a",
        "host": "https://api.example.com/v1",
    }
    return {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {
            "query": query_llm,
            "keyword": keyword_llm or _RecordingKeywordLLM(),
        },
        "llm_cache_identities": {
            "query": {"role": "query", **identity},
            "keyword": {"role": "keyword", **identity},
        },
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


@pytest.fixture
def fake_context(monkeypatch):
    async def _fake_build_query_context(*_args, **_kwargs):
        return QueryContextResult(context="retrieved context", raw_data={})

    monkeypatch.setattr(operate, "_build_query_context", _fake_build_query_context)


async def _run_kg(query_param, llm, cache, system_prompt=None, keyword_llm=None):
    return await kg_query(
        "same query",
        None,
        None,
        None,
        None,
        query_param,
        _global_config(llm, keyword_llm),
        hashing_kv=cache,
        system_prompt=system_prompt,
    )


async def _run_naive(query_param, llm, cache, system_prompt=None):
    return await naive_query(
        "same query",
        _FakeChunksVDB(),
        query_param,
        _global_config(llm),
        hashing_kv=cache,
        system_prompt=system_prompt,
    )


def _kg_param(**overrides) -> QueryParam:
    # Pre-set keywords so kg_query skips keyword extraction unless a test wants it.
    defaults = dict(
        mode="mix", enable_rerank=False, hl_keywords=["art"], ll_keywords=["x"]
    )
    return QueryParam(**{**defaults, **overrides})


def _naive_param(**overrides) -> QueryParam:
    return QueryParam(**{"mode": "naive", "enable_rerank": False, **overrides})


# --- cache key: system prompt -------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_cache_is_partitioned_by_system_prompt(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    first = await _run_kg(_kg_param(), llm, cache, system_prompt=SYSTEM_PROMPT_A)
    second = await _run_kg(_kg_param(), llm, cache, system_prompt=SYSTEM_PROMPT_B)

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert len(llm.calls) == 2
    assert len(cache.query_entries()) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_cache_distinguishes_custom_prompt_from_default(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(_kg_param(), llm, cache)
    second = await _run_kg(_kg_param(), llm, cache, system_prompt=SYSTEM_PROMPT_A)

    assert second.content == "answer-2"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_cache_is_partitioned_by_system_prompt():
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    first = await _run_naive(_naive_param(), llm, cache, system_prompt=NAIVE_PROMPT_A)
    second = await _run_naive(_naive_param(), llm, cache, system_prompt=NAIVE_PROMPT_B)

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert len(llm.calls) == 2


# --- cache key: conversation history -----------------------------------------


HISTORY_1 = [
    {"role": "user", "content": "Who painted it?"},
    {"role": "assistant", "content": "I did."},
]
HISTORY_2 = [
    {"role": "user", "content": "Tell me about flight."},
    {"role": "assistant", "content": "Birds first."},
]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_cache_is_partitioned_by_conversation_history(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(_kg_param(conversation_history=HISTORY_1), llm, cache)
    second = await _run_kg(_kg_param(conversation_history=HISTORY_2), llm, cache)
    third = await _run_kg(_kg_param(), llm, cache)

    assert second.content == "answer-2"
    assert third.content == "answer-3"
    assert len(cache.query_entries()) == 3


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_cache_is_partitioned_by_conversation_history():
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_naive(_naive_param(conversation_history=HISTORY_1), llm, cache)
    second = await _run_naive(_naive_param(conversation_history=HISTORY_2), llm, cache)

    assert second.content == "answer-2"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_identical_requests_still_hit_the_cache(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()
    param = _kg_param(conversation_history=HISTORY_1)

    first = await _run_kg(param, llm, cache, system_prompt=SYSTEM_PROMPT_A)
    second = await _run_kg(param, llm, cache, system_prompt=SYSTEM_PROMPT_A)

    assert first.content == second.content == "answer-1"
    assert len(llm.calls) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cache_entry_records_system_prompt_and_history(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(
        _kg_param(conversation_history=HISTORY_1, temperature=0.3),
        llm,
        cache,
        system_prompt=SYSTEM_PROMPT_A,
    )

    (entry,) = cache.query_entries().values()
    queryparam = entry["queryparam"]
    assert queryparam["system_prompt"] == SYSTEM_PROMPT_A
    assert queryparam["conversation_history"] == HISTORY_1
    assert queryparam["temperature"] == 0.3


# --- bypass_cache ---------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_bypass_cache_never_reads_or_writes_query_cache(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(_kg_param(), llm, cache)  # seeds the cache
    assert len(cache.query_entries()) == 1
    reads_before = cache.reads

    bypassed = await _run_kg(_kg_param(bypass_cache=True), llm, cache)

    assert bypassed.content == "answer-2"
    assert cache.reads == reads_before  # no query-cache read (keywords are preset)
    (entry,) = cache.query_entries().values()
    assert entry["return"] == "answer-1"  # not overwritten


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_bypass_cache_does_not_save(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(_kg_param(bypass_cache=True), llm, cache)

    assert cache.query_entries() == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_bypass_cache_never_reads_or_writes_query_cache():
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_naive(_naive_param(), llm, cache)
    reads_before = cache.reads

    bypassed = await _run_naive(_naive_param(bypass_cache=True), llm, cache)

    assert bypassed.content == "answer-2"
    assert cache.reads == reads_before
    assert len(cache.query_entries()) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bypass_cache_keeps_keyword_cache(fake_context):
    cache, llm, kw_llm = _FakeKVStorage(), _RecordingLLM(), _RecordingKeywordLLM()
    param = QueryParam(mode="mix", enable_rerank=False, bypass_cache=True)

    await _run_kg(param, llm, cache, keyword_llm=kw_llm)
    await _run_kg(param, llm, cache, keyword_llm=kw_llm)

    assert len(kw_llm.calls) == 1  # second call served from keyword cache
    assert any(":keywords:" in k for k in cache._store)
    assert len(llm.calls) == 2


# --- temperature ----------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_temperature_reaches_response_call_only(fake_context):
    cache, llm, kw_llm = _FakeKVStorage(), _RecordingLLM(), _RecordingKeywordLLM()
    param = QueryParam(mode="mix", enable_rerank=False, temperature=0.2)

    await _run_kg(param, llm, cache, keyword_llm=kw_llm)

    assert llm.calls[0]["temperature"] == 0.2
    assert len(kw_llm.calls) == 1
    assert "temperature" not in kw_llm.calls[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_temperature_reaches_response_call():
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_naive(_naive_param(temperature=0.9), llm, cache)

    assert llm.calls[0]["temperature"] == 0.9


@pytest.mark.offline
@pytest.mark.asyncio
async def test_unset_temperature_is_not_sent(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(_kg_param(), llm, cache)

    assert "temperature" not in llm.calls[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_cache_is_partitioned_by_temperature(fake_context):
    cache, llm = _FakeKVStorage(), _RecordingLLM()

    await _run_kg(_kg_param(temperature=0.2), llm, cache)
    second = await _run_kg(_kg_param(temperature=0.8), llm, cache)
    third = await _run_kg(_kg_param(), llm, cache)

    assert second.content == "answer-2"
    assert third.content == "answer-3"
