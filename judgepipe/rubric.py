"""Explicit 0-4 scoring anchors and weights, shared between modes."""
from __future__ import annotations

RUBRIC = {
    'correctness': {
        'weight': 0.30,
        'definition': 'Factual accuracy and agreement with reference, when provided.',
        'anchors': '0 contradictory or fabricated; 1 mostly wrong; 2 material errors; 3 minor error; 4 fully accurate',
    },
    'faithfulness': {
        'weight': 0.25,
        'definition': 'Claims traceable to supplied context, if any; do not reward outside knowledge.',
        'anchors': '0 unsupported; 1 mostly unsupported; 2 mixed; 3 mostly grounded; 4 fully grounded',
    },
    'completeness': {
        'weight': 0.20,
        'definition': 'Covers the requested essentials without unrelated filler.',
        'anchors': '0 misses task; 1 large omissions; 2 partial; 3 nearly complete; 4 complete and concise',
    },
    'instruction_following': {
        'weight': 0.15,
        'definition': 'Follows the explicit system prompt and format constraints.',
        'anchors': '0 ignores instructions; 1 serious violations; 2 some violations; 3 minor issue; 4 follows fully',
    },
    'tone_safety': {
        'weight': 0.10,
        'definition': 'Clear, neutral tone with appropriate restraint; do not reward flattery.',
        'anchors': '0 unsafe/inappropriate; 1 severe issue; 2 noticeable issue; 3 acceptable; 4 clear and appropriate',
    },
}


def weighted_score(scores: dict[str, dict]) -> float:
    """Compute overall from validated sub-scores, not a judge-provided overall."""
    return round(sum(RUBRIC[k]['weight'] * scores[k]['score'] for k in RUBRIC), 4)


def rubric_instructions() -> str:
    return '\n'.join(f'- {k} (weight {v["weight"]:.0%}): {v["definition"]} Anchors: {v["anchors"]}.'
                     for k,v in RUBRIC.items())
