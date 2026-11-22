---
name: lilavati-bench
description: Run, analyse and report Līlāvatī-Bench LLM evaluations (no-tool and tool-augmented arms) honestly and reproducibly. Use when a user or student asks to benchmark a model on finance tasks, compare models or tools vs no tools, summarise results/*.jsonl, rebuild the results manifest, or write up benchmark findings.
---

# Līlāvatī-Bench skill (for users, students and contributors)

Follow these steps in order. The matching tutorials explain the why:
`docs/tutorials/05_benchmarking_llms_with_open_models.md` (running),
`06_analysing_and_publishing_results.md` (analysis, publishing),
`07_tools_versus_mental_math.md` (tool arm). For code changes, also load `lilavati-dev`.

## 1. Keys

- Check a key without printing it:
  `[ -n "$GROQ_API_KEY" ] && echo set || echo missing` (or `GEMINI_API_KEY`).
- If missing, ask the user to `export` it in their shell. Never ask them to paste it into chat,
  never write it to a file (including `.env`, notebooks, scripts) and never put it in a URL.
- If a key was pasted into chat anyway, pass it only as an environment variable on the command
  and tell the user to rotate it.

## 2. Choose cases and models

- Golden cases (7) are saturated for strong models; use generated cases (`--generated 4 --seed 2026`
  matches the published runs) to compare models.
- List Groq models with `GET https://api.groq.com/openai/v1/models`; use only chat models.
- Use one prompt version (`lilavati.bench.llm_eval.PROMPT_VERSION`) for everything you compare.

## 3. Run

```bash
uv run python scripts/run_llm_baseline.py  --provider groq --model <id> --generated 4 --seed 2026 \
    --max-tokens 8192 --rpm 20 --out results/<name>.jsonl            # no tools
uv run python scripts/run_tool_baseline.py --provider groq --model <id> --generated 4 --seed 2026 \
    --out results/tools_<name>.jsonl                                  # tools, same cases
```

- `--resume` re-runs provider errors only. Never re-run unparsable, truncated, model_error or wrong
  answers: that re-rolls model failures and inflates scores.
- After a parser or scoring fix, `--rescore` (no API calls); never call the model again for it.
- Never edit a raw result file by hand.

## 4. Analyse

```bash
uv run python scripts/build_results_manifest.py
uv run python scripts/analyse_results.py     # results/ANALYSIS.md, results/analysis.json
```

Report, for every model: answered / n, the non-answer classes (error, model_error, truncated,
unparsable), accuracy on answered, item mean with its 95% CI, and tool grounding for tool runs.

## 5. Interpret honestly

- A difference is supported only by a paired CI that excludes 0 (same questions, matched by case
  hash) or clearly non-overlapping intervals. Otherwise say "not distinguishable at this sample size".
- Provider failures are not model errors. Truncation is not a wrong answer. Say which is which.
- Tool-arm accuracy measures tool choice and parameter extraction, not engine correctness (expected
  values come from the same engines). Engine correctness is in `docs/validation/`.
- Quote only numbers that are in `results/ANALYSIS.md` or that you computed in this session from
  stored files; cite the file. Never estimate or round a number into a stronger claim.
- Note prompt versions and superseded files (`manifest.json` `status`) whenever you cite results.

## 6. Before you finish

- `uv run pytest tests/test_results_manifest.py` passes (manifest current, no secrets in results).
- If docs quote changed numbers, regenerate them from `results/ANALYSIS.md`, not by hand.
- Do not commit or push unless the user asks.
