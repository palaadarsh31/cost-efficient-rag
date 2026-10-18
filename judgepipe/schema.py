"""Input schema validation; deliberately dependency-free."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    system_prompt: str
    model_output: str
    candidate_b: str | None = None
    expected_output: str | None = None
    context: str | None = None
    criteria: tuple[str, ...] = ()
    gold_winner: str | None = None  # A, B, tie, independently labeled
    probe: str | None = None        # padded, confidently_wrong, ordinary


def load_cases(path: str | Path) -> list[Case]:
    path = Path(path)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        try:
            import yaml
        except ImportError as e:
            raise ValueError('YAML suites require PyYAML: pip install pyyaml (JSON works without it)') from e
        obj = yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        obj = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(obj, dict):
        obj = obj.get('cases')
    if not isinstance(obj, list) or not obj:
        raise ValueError('suite must be a nonempty list or {"cases": [...]}')
    cases: list[Case] = []
    seen = set()
    for i, raw in enumerate(obj):
        if not isinstance(raw, dict):
            raise ValueError(f'case {i} must be an object')
        for req in ('id', 'input', 'system_prompt'):
            if not isinstance(raw.get(req), str) or not raw[req].strip():
                raise ValueError(f'case {i}: {req} must be nonempty text')
        if not isinstance(raw.get('model_output', ''), str):
            raise ValueError(f'case {i}: model_output must be text')
        if raw['id'] in seen:
            raise ValueError(f'duplicate case id: {raw["id"]}')
        seen.add(raw['id'])
        for opt in ('candidate_b', 'expected_output', 'context'):
            if opt in raw and raw[opt] is not None and not isinstance(raw[opt], str):
                raise ValueError(f'case {raw["id"]}: {opt} must be a string or null')
        if raw.get('gold_winner') not in (None, 'A', 'B', 'tie'):
            raise ValueError(f'case {raw["id"]}: gold_winner must be A, B, or tie')
        if raw.get('gold_winner') is not None and raw.get('candidate_b') is None:
            raise ValueError(f'case {raw["id"]}: gold_winner requires candidate_b')
        criteria = raw.get('criteria', [])
        if not isinstance(criteria, list) or any(not isinstance(c, str) for c in criteria):
            raise ValueError(f'case {raw["id"]}: criteria must be a list of strings')
        cases.append(Case(id=raw['id'], input=raw['input'], system_prompt=raw['system_prompt'],
                          model_output=raw.get('model_output', ''), candidate_b=raw.get('candidate_b'),
                          expected_output=raw.get('expected_output'), context=raw.get('context'),
                          criteria=tuple(criteria), gold_winner=raw.get('gold_winner'),
                          probe=raw.get('probe')))
    return cases


def serialize_case(c: Case) -> dict[str, Any]:
    return {k:v for k,v in vars(c).items() if v is not None}
