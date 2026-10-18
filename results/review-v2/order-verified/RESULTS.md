# Controlled evaluation review

Status: complete
Started: 2026-10-18T05:52:21.068407+00:00; finished: 2026-10-18T05:54:17.278496+00:00

Full prompts, responses, usage and settings: audit.jsonl. Inputs: input_cases.json. Individual outcomes: trials.json.

| Condition | Valid comparisons | Mean absolute score change | Score disagreement | Fully-supported decision disagreement |
|---|---:|---:|---:|---:|
| Repeated original order | 18 | 0.0 | 0.0 | 0.0 |
| Reversed/shuffled order | 30 | 0.0 | 0.0 | 0.0 |

Descriptive fixed-answer judge sensitivity. Reordering differences must be compared with unchanged-order variability; this small sample does not establish systematic position bias or generation-order effects.

Fully supported iff faithfulness score equals 1.0; also report continuous score differences.

Recorded usage: {"actual_requests": 54, "failed_requests": 0, "prompt_tokens": 153720, "completion_tokens": 10642, "by_role": {"judge": {"requests": 54, "prompt_tokens": 153720, "completion_tokens": 10642}}}
