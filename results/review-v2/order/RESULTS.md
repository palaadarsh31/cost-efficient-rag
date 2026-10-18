# Controlled evaluation review

Status: incomplete
Started: 2026-10-18T05:50:54.356771+00:00; finished: 2026-10-18T05:51:28.161259+00:00

Full prompts, responses, usage and settings: audit.jsonl. Inputs: input_cases.json. Individual outcomes: trials.json.

| Condition | Valid comparisons | Mean absolute score change | Score disagreement | Fully-supported decision disagreement |
|---|---:|---:|---:|---:|
| Repeated original order | 0 | None | None | None |
| Reversed/shuffled order | 0 | None | None | None |

Descriptive fixed-answer judge sensitivity. Reordering differences must be compared with unchanged-order variability; this small sample does not establish systematic position bias or generation-order effects.

Fully supported iff faithfulness score equals 1.0; also report continuous score differences.

Recorded usage: {"actual_requests": 54, "failed_requests": 54, "prompt_tokens": 0, "completion_tokens": 0, "by_role": {"judge": {"requests": 54, "prompt_tokens": 0, "completion_tokens": 0}}}
