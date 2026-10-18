"""Tolerant extraction; strict semantic validation. Invalid outputs never silently pass."""
from __future__ import annotations
import json
import re
from .rubric import RUBRIC, weighted_score


def json_object(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped, flags=re.I)
        stripped = re.sub(r'\s*```$', '', stripped)
    for start in (i for i,c in enumerate(stripped) if c == '{'):
        depth, quoted, escaped = 0, False, False
        for pos in range(start, len(stripped)):
            c = stripped[pos]
            if quoted:
                if escaped:
                    escaped = False
                elif c == '\\':
                    escaped = True
                elif c == '"':
                    quoted = False
            else:
                if c == '"':
                    quoted = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = stripped[start:pos+1]
                        for form in (candidate, re.sub(r',\s*([}\]])', r'\1', candidate)):
                            try:
                                val = json.loads(form)
                                if isinstance(val, dict):
                                    return val
                            except (ValueError, TypeError):
                                pass
                        break
    return None


def scores_valid(raw: object) -> dict | None:
    if not isinstance(raw, dict) or set(raw) != set(RUBRIC):
        return None
    clean = {}
    for name in RUBRIC:
        record = raw[name]
        if not isinstance(record, dict):
            return None
        n, reason = record.get('score'), record.get('rationale')
        if type(n) is not int or not 0 <= n <= 4:
            return None
        if not isinstance(reason, str) or len(reason.strip()) < 5:
            return None
        clean[name] = {'score': n, 'rationale': reason.strip()}
    return clean


def parse_pointwise(text: str) -> dict | None:
    obj = json_object(text)
    if not obj:
        return None
    # Early baseline prompts requested the five criteria at the root instead of
    # under ``scores``. Accept that exact legacy shape after the same strict
    # criterion, integer-range and rationale validation; no score is coerced.
    raw_scores = obj.get('scores')
    if raw_scores is None and set(obj) == set(RUBRIC):
        raw_scores = obj
    scores = scores_valid(raw_scores)
    if scores is None:
        return None
    return {'scores': scores, 'overall': weighted_score(scores)}


def parse_pairwise(text: str) -> dict | None:
    obj = json_object(text)
    if not obj or obj.get('winner') not in ('A','B','tie'):
        return None
    if not isinstance(obj.get('rationale'), str) or len(obj['rationale'].strip()) < 5:
        return None
    if type(obj.get('unsupported_length_penalty_applied')) is not bool:
        return None
    a, b = scores_valid(obj.get('scores_a')), scores_valid(obj.get('scores_b'))
    if a is None or b is None:
        return None
    return {'winner':obj['winner'], 'rationale':obj['rationale'].strip(),
            'scores_a':a, 'scores_b':b, 'overall_a':weighted_score(a),
            'overall_b':weighted_score(b), 'unsupported_length_penalty_applied':obj['unsupported_length_penalty_applied']}


def canonical_winner(shown_winner: str, a_is_first: bool) -> str:
    if shown_winner == 'tie': return 'tie'
    return shown_winner if a_is_first else ('B' if shown_winner == 'A' else 'A')


def consensus(first: str | None, second: str | None) -> tuple[str | None, bool | None]:
    if first is None or second is None:
        return None, None
    return (first if first == second else 'tie'), first != second
