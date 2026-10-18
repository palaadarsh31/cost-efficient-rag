# Controlled evaluation review

Status: complete
Started: 2026-10-18T05:48:57.574014+00:00; finished: 2026-10-18T05:49:16.067690+00:00

Full prompts, responses, usage and settings: audit.jsonl. Inputs: input_cases.json. Individual outcomes: trials.json.

Historical q24 faithfulness score: 0.5

| Context condition | Valid trials | Mean | Min | Max |
|---|---:|---:|---:|---:|
| text_only | 4 | 0.0 | 0.0 | 0.0 |
| full_metadata | 4 | 1.0 | 1.0 | 1.0 |

Same saved q24 question, generated answer, passage texts, stable IDs, current judge prompt and model settings. Only source metadata differs; conditions interleaved by fixed random seed.

Historical baseline and fresh trials differ in execution time and judge prompt version; use contemporary text-only trials to assess metadata effects.

Recorded usage: {"actual_requests": 8, "failed_requests": 0, "prompt_tokens": 21364, "completion_tokens": 1598, "by_role": {"judge": {"requests": 8, "prompt_tokens": 21364, "completion_tokens": 1598}}}
