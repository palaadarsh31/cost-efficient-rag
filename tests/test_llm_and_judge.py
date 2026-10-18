import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from docqa.evaluation.judge import Judge, extract_json
from docqa.llm import ChatClient, Completion, LLMError
from docqa.passages import Passage


class _Handler(BaseHTTPRequestHandler):
    script: list = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        status, payload = self.script.pop(0)
        self.server.requests.append(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if status == 429:
            self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()


def _ok(text, prompt=50, completion=7):
    return 200, {"model": "m", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def test_chat_client_parses_usage_and_retries(server, monkeypatch):
    monkeypatch.setattr(ChatClient, "_sleep", staticmethod(lambda *a: None))
    _Handler.script = [(429, {"error": "slow down"}), _ok("hello [1]")]
    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m", api_key="k", max_tokens=99)
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out.text == "hello [1]" and out.prompt_tokens == 50 and out.completion_tokens == 7
    assert len(server.requests) == 2
    assert server.requests[-1]["max_tokens"] == 99 and server.requests[-1]["temperature"] == 0.0


def test_chat_client_drops_temperature_when_model_rejects_it(server):
    # newer Claude models on the OpenAI-compatible endpoint answer like this
    _Handler.script = [(400, {"error": {"message": "`temperature` is deprecated for this model."}}),
                       _ok("first"), _ok("second")]
    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m")
    assert client.complete([{"role": "user", "content": "a"}]).text == "first"
    assert client.complete([{"role": "user", "content": "b"}]).text == "second"
    assert "temperature" in server.requests[0]
    assert all("temperature" not in r for r in server.requests[1:])


def test_chat_client_adapts_parameters_even_when_extra_body_or_call_overrides_them(server):
    _Handler.script = [(400, {"error": {"message": "temperature is unsupported"}}),
                       (400, {"error": {"message": "Use max_completion_tokens instead of max_tokens"}}),
                       _ok("answer")]
    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m", temperature=None,
                        extra_body={"temperature": 0.1, "max_tokens": 20})
    assert client.complete([{"role": "user", "content": "q"}], temperature=0.2, max_tokens=30).text == "answer"
    assert "temperature" not in server.requests[-1]
    assert "max_tokens" not in server.requests[-1]
    assert server.requests[-1]["max_completion_tokens"] == 30


def test_chat_client_surfaces_client_errors(server):
    _Handler.script = [(400, {"error": {"message": "bad model"}})]
    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m")
    with pytest.raises(LLMError, match="bad model"):
        client.complete([{"role": "user", "content": "hi"}])


def test_extract_json_handles_messy_replies():
    assert extract_json('```json\n{"score": 4, "reason": "ok"}\n```') == {"score": 4, "reason": "ok"}
    assert extract_json('Sure! Here it is: {"label": "correct", "reason": "a {nested} brace"} thanks') == {
        "label": "correct", "reason": "a {nested} brace"}
    assert extract_json('{"claims": [{"claim": "x", "supported": true},]}') == {
        "claims": [{"claim": "x", "supported": True}]}
    assert extract_json("no json here") is None


class _ScriptedClient:
    model = "judge"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, messages, **_):
        self.seen.append(messages)
        return Completion(self.replies.pop(0), "judge", 10, 5, 1.0)


def test_judge_retries_once_on_bad_json():
    client = _ScriptedClient(["score: four", '{"score": 4, "reason": "fine"}'])
    j = Judge(client).relevance("q", "a")
    assert j.score == pytest.approx(0.75) and not j.parse_error
    assert len(j.raw) == 2 and "not valid JSON" in client.seen[1][-1]["content"]


def test_judge_gives_up_after_two_bad_replies():
    j = Judge(_ScriptedClient(["nope", '{"score": 9}'])).relevance("q", "a")
    assert j.score is None and j.parse_error


def test_faithfulness_score_is_supported_fraction():
    reply = json.dumps({"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False},
                                   {"claim": "c", "supported": True}, {"claim": "d", "supported": True}]})
    j = Judge(_ScriptedClient([reply])).faithfulness("q", "a [1]", ["passage"])
    assert j.score == pytest.approx(0.75)
    assert "passage" in j.prompt


@pytest.mark.parametrize("bad_claims", [
    [{"claim": "supported", "supported": True}, 0],
    [{"supported": True}],
    [{"claim": " ", "supported": True}],
    [{"claim": "a", "supported": 1}],
    [{"claim": "a", "supported": "true"}],
    [],
])
def test_faithfulness_rejects_entire_malformed_verdict(bad_claims):
    reply = json.dumps({"claims": bad_claims})
    judgement = Judge(_ScriptedClient([reply, reply])).faithfulness("q", "a", ["source"])
    assert judgement.score is None and judgement.parse_error
    assert judgement.failure_reason == "invalid_schema"
    assert len(judgement.attempts) == 2
    assert all(not attempt["valid"] for attempt in judgement.attempts)


@pytest.mark.parametrize("rating", [True, False, 4.9, "4", None, 0, 6])
def test_relevance_does_not_coerce_invalid_ratings(rating):
    reply = json.dumps({"score": rating, "reason": "reason"})
    judgement = Judge(_ScriptedClient([reply, reply])).relevance("q", "a")
    assert judgement.score is None and judgement.parse_error


@pytest.mark.parametrize("label", [[], {}, True, "CORRECT", None])
def test_correctness_rejects_invalid_labels_without_crashing(label):
    reply = json.dumps({"label": label, "reason": "reason"})
    judgement = Judge(_ScriptedClient([reply, reply])).correctness("q", "gold", "a")
    assert judgement.score is None and judgement.parse_error


def test_judge_preserves_canonical_sources_and_retry_messages():
    sources = [Passage(7, "chunk-a", "source-a.pdf", "Source A", "Fact A", page_start=3, page_end=4),
               Passage(2, "chunk-b", "source-b.md", "Source B", "Fact B")]
    client = _ScriptedClient(["bad", '{"claims":[{"claim":"Fact A","supported":true}]}'])
    judgement = Judge(client).faithfulness("q", "Fact A [7]", sources)
    assert judgement.score == 1.0
    assert "[7] Source A (source-a.pdf; pp. 3-4)" in judgement.prompt
    assert judgement.prompt.index("[7] Source A") < judgement.prompt.index("[2] Source B")
    assert judgement.attempts[0]["messages"] == client.seen[0]
    assert judgement.attempts[1]["messages"] == client.seen[1]
    assert judgement.attempts[0]["messages"][0]["role"] == "system"
    assert "untrusted data" in judgement.attempts[0]["messages"][0]["content"]
    assert judgement.attempts[1]["model"] == "judge"
    assert judgement.attempts[1]["raw"].startswith('{"claims"')
    assert judgement.prompt_tokens == 20 and judgement.completion_tokens == 10
    assert judgement.prompt_version


@pytest.mark.parametrize("payload, message", [
    ({"choices": []}, "no completion choices"),
    ({"choices": [{"message": {"content": " "}, "finish_reason": "stop"}]}, "empty completion"),
    ({"choices": [{"message": {"content": "Partial answer"}, "finish_reason": "length"}]}, "incomplete"),
    ({"choices": [{"message": {"content": ["not text"]}, "finish_reason": "stop"}]}, "not text"),
])
def test_chat_client_rejects_missing_empty_and_truncated_completions(server, payload, message):
    _Handler.script = [(200, payload)]
    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m")
    with pytest.raises(LLMError, match=message):
        client.complete([{"role": "user", "content": "q"}])


def test_chat_client_marks_usage_estimated_when_only_completion_usage_is_missing(server):
    status, payload = _ok("answer")
    del payload["usage"]["completion_tokens"]
    _Handler.script = [(status, payload)]
    completion = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m").complete(
        [{"role": "user", "content": "q"}])
    assert completion.usage_estimated and completion.prompt_tokens == 50
    assert completion.completion_tokens > 0


def test_snapshot_excludes_credentials_and_records_effective_parameters():
    client = ChatClient("https://name:private@example.test/v1?api_key=private", "model", api_key="private",
                        temperature=0.2, extra_body={"seed": 13, "api_key": "private", "top_p": 0.8})
    snapshot = client.snapshot()
    assert "private" not in json.dumps(snapshot)
    assert snapshot["parameters"]["temperature"] == 0.2
    assert snapshot["parameters"]["seed"] == 13
    assert snapshot["parameters"]["top_p"] == 0.8


def test_judge_retains_raw_usage_for_rejected_truncated_completion(server):
    status, payload = _ok('{"score":5,"reason":"partial"}')
    payload["choices"][0]["finish_reason"] = "length"
    _Handler.script = [(status, payload)]
    client = ChatClient(f"http://127.0.0.1:{server.server_port}/v1", "m")
    judgement = Judge(client).relevance("q", "a")
    assert judgement.score is None and judgement.parse_error
    assert judgement.prompt_tokens == 50 and judgement.completion_tokens == 7
    assert judgement.attempts[0]["finish_reason"] == "length"
    assert judgement.attempts[0]["raw"] == '{"score":5,"reason":"partial"}'
