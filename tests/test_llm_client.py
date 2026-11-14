from types import SimpleNamespace

import pytest
from openai import APIStatusError, RateLimitError

import llm_client
from llm_client import FileCache, LLMClient, TokenTracker, encode_image_base64

SAMPLE_IMAGE = llm_client.CACHE_DIR.parent.parent / "dataset" / "images" / "sample" / "case_001" / "img_1.jpg"


def _response(text: str, input_tokens: int = 10, output_tokens: int = 5):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    return SimpleNamespace(output_text=text, usage=usage)


def _status_error(cls, status: int, message: str):
    # Duck-typed HTTP response: avoids depending on the SDK's HTTP library.
    response = SimpleNamespace(status_code=status, request=SimpleNamespace(), headers={})
    return cls(message, response=response, body=None)


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)

    def factory(outcomes):
        client = LLMClient(tracker=TokenTracker())
        fake = FakeResponses(outcomes)
        client.client = SimpleNamespace(responses=fake)
        return client, fake

    return factory


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        LLMClient()


def test_parse_json_strips_code_fences():
    assert LLMClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert "parse_error" in LLMClient._parse_json("not json")


def test_file_cache_round_trip(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.get("key") is None
    cache.set("key", {"v": "é"})
    assert cache.get("key") == {"v": "é"}


def test_successful_call_is_cached_and_tracked(make_client):
    client, fake = make_client([_response('{"claim_summary": "x"}')])
    first = client.call_text(model="m", system_message="sys", user_message="hi")
    second = client.call_text(model="m", system_message="sys", user_message="hi")
    assert first == second == {"claim_summary": "x"}
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["name"] == "claim_extraction"
    summary = client.tracker.summary()
    assert summary["total_calls"] == 2
    assert summary["model_breakdown"]["m"]["cached"] == 1
    assert summary["total_input_tokens"] == 10


def test_malformed_response_is_not_cached(make_client):
    client, fake = make_client([_response("garbage"), _response('{"ok": true}')])
    assert "parse_error" in client.call_text(model="m", system_message="s", user_message="u")
    assert client.call_text(model="m", system_message="s", user_message="u") == {"ok": True}
    assert len(fake.requests) == 2


def test_rate_limit_is_retried(make_client):
    client, fake = make_client([_status_error(RateLimitError, 429, "slow down"), _response('{"ok": 1}')])
    assert client.call_text(model="m", system_message="s", user_message="u") == {"ok": 1}
    assert len(fake.requests) == 2


def test_unavailable_model_falls_back(make_client):
    client, fake = make_client(
        [_status_error(APIStatusError, 404, "model not found"), _response('{"ok": 1}')]
    )
    assert client.call_text(model="missing-model", system_message="s", user_message="u") == {"ok": 1}
    assert fake.requests[0]["model"] == "missing-model"
    assert fake.requests[1]["model"] == llm_client.FALLBACK_MODEL


def test_client_errors_are_raised(make_client):
    client, _ = make_client([_status_error(APIStatusError, 401, "bad key")])
    with pytest.raises(APIStatusError):
        client.call_text(model="m", system_message="s", user_message="u")


def test_vision_call_sends_images_and_reasoning_sets_effort(make_client):
    client, fake = make_client([_response('{"images": []}'), _response('{"claim_status": "supported"}')])
    client.call_vision(model="m", system_message="s", user_text="look", image_paths=[SAMPLE_IMAGE])
    client.call_reasoning(model="m", developer_message="d", user_message="u", reasoning_effort="low")
    content = fake.requests[0]["input"][0]["content"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert fake.requests[1]["reasoning"] == {"effort": "low"}


def test_encode_image_produces_jpeg():
    import base64

    data = base64.b64decode(encode_image_base64(SAMPLE_IMAGE))
    assert data[:2] == b"\xff\xd8"
