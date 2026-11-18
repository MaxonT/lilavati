# Baseline results (2026-11-17/18)

Raw per-item outputs of `scripts/run_llm_baseline.py` (direct prompt, no tools, no retrieval).
`baseline_*.jsonl` = 7 golden cases (Gemini: 1 pass, temp 0; Gemma: 3 passes, temp 0.7).
`generated_*.jsonl` = templated cases, seed 2026 (Gemma 4/track, Gemini 4 Hydrogen 2/track, temp 0).
`archive_prompt_v1/` = superseded first pass (prompt mis-stated the vega unit; 2,048-token cap truncated
reasoning models). Do not cite. API keys are never written to these files.
Provider outages (HTTP 429/503/timeouts) and empty replies are counted as *failures to answer*, not wrong answers.
