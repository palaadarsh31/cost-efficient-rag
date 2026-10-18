"""LLM backends with secrets read from environment only.

Supports both OpenAI-compatible chat/completions endpoints and Anthropic's
native Messages endpoint. No SDK dependency or secret persisted in artifacts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ''
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)


@dataclass
class ClientConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float | None = None
    timeout: float = 90.0
    provider: str = 'openai-compatible'

    @classmethod
    def from_env(cls, prefix: str, default_model: str = '') -> 'ClientConfig':
        provider = os.getenv(prefix + '_PROVIDER', 'openai-compatible').strip().lower()
        if provider not in ('openai-compatible', 'anthropic'):
            raise ValueError(f'{prefix}_PROVIDER must be anthropic or openai-compatible')
        base_default = 'https://api.anthropic.com/v1' if provider == 'anthropic' else 'https://api.openai.com/v1'
        # A single ANTHROPIC_API_KEY can be used by either selected Anthropic role.
        # Do not write it to .env, logs, CLI flags, or submitted reports.
        key = os.getenv(prefix + '_API_KEY', '')
        if not key and provider == 'anthropic':
            key = os.getenv('ANTHROPIC_API_KEY', '')
        temperature_raw = os.getenv(prefix + '_TEMPERATURE')
        if temperature_raw is None:
            # Current Claude models reject the legacy temperature field. Keep the
            # deterministic default for OpenAI-compatible backends, and let
            # Anthropic choose its model default unless the user explicitly opts in.
            temperature = None if provider == 'anthropic' else 0.0
        elif temperature_raw.strip().lower() in ('', 'none', 'omit'):
            temperature = None
        else:
            temperature = float(temperature_raw)
        return cls(base_url=os.getenv(prefix + '_BASE_URL', base_default),
                   model=os.getenv(prefix + '_MODEL', default_model),
                   api_key=key, provider=provider,
                   temperature=temperature,
                   timeout=float(os.getenv(prefix + '_TIMEOUT', '90')))


def family(model: str) -> str:
    m = model.lower()
    for key, fam in (('claude', 'anthropic'), ('gpt', 'openai'), ('o1', 'openai'),
                     ('gemini', 'google'), ('llama', 'meta'), ('qwen', 'alibaba'),
                     ('mistral', 'mistral'), ('deepseek', 'deepseek')):
        if key in m:
            return fam
    return 'unknown'


class OpenAICompatibleClient:
    def __init__(self, config: ClientConfig):
        if not config.model:
            raise ValueError('Model missing: set JUDGE_MODEL / GENERATOR_MODEL')
        if not config.api_key and not config.base_url.startswith(('http://localhost', 'http://127.0.0.1')):
            raise ValueError('API key missing; set the proper *_API_KEY environment variable')
        self.config = config

    def complete(self, messages: list[dict], *, max_tokens: int = 1600) -> Completion:
        base = self.config.base_url.rstrip('/')
        target = base if base.endswith('/chat/completions') else base + '/chat/completions'
        payload = {'model':self.config.model, 'messages':messages,
                   'max_tokens':max_tokens}
        if self.config.temperature is not None:
            payload['temperature'] = self.config.temperature
        req = Request(target, data=json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json',
                               'Authorization': 'Bearer ' + (self.config.api_key or 'local')},
                      method='POST')
        try:
            with urlopen(req, timeout=self.config.timeout) as res:
                body = json.load(res)
        except HTTPError as err:
            raise RuntimeError(f'LLM endpoint returned HTTP {err.code}') from None
        except (URLError, TimeoutError, OSError) as err:
            raise RuntimeError(f'LLM endpoint connection failed ({type(err).__name__})') from None
        try:
            raw = body['choices'][0]['message']['content']
            if isinstance(raw, list):
                raw = '\n'.join(b.get('text','') for b in raw if b.get('type')=='text')
            usage = body.get('usage') or {}
            return Completion(text=raw, prompt_tokens=int(usage.get('prompt_tokens') or 0),
                              completion_tokens=int(usage.get('completion_tokens') or 0),
                              model=body.get('model') or self.config.model,
                              finish_reason=body['choices'][0].get('finish_reason'), usage=usage)
        except (KeyError, IndexError, TypeError, ValueError) as err:
            raise RuntimeError('LLM returned unexpected response body; no verdict accepted') from err


class AnthropicClient:
    """Minimal native Anthropic /v1/messages adapter, including usage."""
    def __init__(self, config: ClientConfig):
        if not config.model:
            raise ValueError('Model missing: set JUDGE_MODEL / GENERATOR_MODEL')
        if not config.api_key:
            raise ValueError('API key missing; set ANTHROPIC_API_KEY or the role-specific *_API_KEY')
        self.config = config

    def complete(self, messages: list[dict], *, max_tokens: int = 1600) -> Completion:
        system_segments = [m['content'] for m in messages if m.get('role') == 'system']
        turns = [{'role':m['role'], 'content':m['content']} for m in messages
                 if m.get('role') in ('user', 'assistant')]
        if not turns:
            raise ValueError('Anthropic messages require at least one user/assistant turn')
        payload = {'model':self.config.model, 'messages':turns,
                   'max_tokens':max_tokens}
        if self.config.temperature is not None:
            payload['temperature'] = self.config.temperature
        if system_segments:
            payload['system'] = '\n\n'.join(system_segments)
        base = self.config.base_url.rstrip('/')
        url = base if base.endswith('/messages') else base + '/messages'
        req = Request(url, data=json.dumps(payload).encode('utf8'),
                      headers={'Content-Type':'application/json',
                               'x-api-key':self.config.api_key,
                               'anthropic-version':'2023-06-01'}, method='POST')
        try:
            with urlopen(req, timeout=self.config.timeout) as res:
                body = json.load(res)
        except HTTPError as err:
            # Never echo response headers, request details, or the supplied key.
            raise RuntimeError(f'Anthropic returned HTTP {err.code}') from None
        except (URLError, TimeoutError, OSError) as err:
            raise RuntimeError(f'Anthropic connection failed ({type(err).__name__})') from None
        try:
            blocks = body['content']
            text = '\n'.join(b['text'] for b in blocks if b.get('type') == 'text')
            if not text.strip():
                raise ValueError('No text blocks in completion')
            usage = body.get('usage') or {}
            return Completion(text=text,
                              prompt_tokens=int(usage.get('input_tokens') or 0),
                              completion_tokens=int(usage.get('output_tokens') or 0),
                              model=body.get('model') or self.config.model,
                              finish_reason=body.get('stop_reason'), usage=usage)
        except (KeyError, IndexError, TypeError, ValueError) as err:
            raise RuntimeError('Anthropic returned unexpected response body; no verdict accepted') from err


def make_client(config: ClientConfig):
    if config.provider == 'anthropic':
        return AnthropicClient(config)
    return OpenAICompatibleClient(config)
