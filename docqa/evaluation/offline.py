"""Stand-ins for the LLM so the harness can run end to end with no network.

`python -m docqa.evaluation.run --offline` uses these together with the hashing
embedder. The numbers it produces only prove the plumbing works; they say
nothing about answer quality and are written to a separate folder.
"""
from __future__ import annotations

import json
import re

from ..chunking import count_tokens
from ..llm import Completion

_PASSAGE_ONE = re.compile(r"\[1\][^\n]*\n(.+?)(?:\n\n\[2\]|\n\nQuestion:)", re.S)
_SENTENCE = re.compile(r"(.+?[.!?])(\s|$)", re.S)


class StubLLM:
    """Answers with the first sentence of passage [1]; judges with fixed verdicts."""

    model = "offline-stub"

    def complete(self, messages: list[dict], **_) -> Completion:
        prompt = messages[-1]["content"]
        if "Context passages:" in prompt and prompt.lstrip().startswith("Context passages:"):
            match = _PASSAGE_ONE.search(prompt)
            body = match.group(1).strip() if match else ""
            sentence = _SENTENCE.match(body)
            text = (sentence.group(1) if sentence else body[:200]).strip() + " [1]"
        elif "factual claims" in prompt:
            text = json.dumps({"claims": [{"claim": "stub claim", "supported": True}]})
        elif "Rate how well the answer" in prompt:
            text = json.dumps({"score": 4, "reason": "stub"})
        elif "Reference answer" in prompt:
            text = json.dumps({"label": "partial", "reason": "stub"})
        else:
            text = "I don't know based on the provided documents."
        prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
        return Completion(text=text, model=self.model, prompt_tokens=prompt_tokens,
                          completion_tokens=count_tokens(text), latency_ms=0.0,
                          usage_estimated=True)
