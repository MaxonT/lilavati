# Līlāvatī-Bench results

Every reply every model gave in this repository's benchmark runs, with the metadata needed to
reproduce and verify each number. Data under [CC BY 4.0](LICENSE.md). API keys are never written here
(`tests/test_results_manifest.py` checks).

**Numbers:** see [`ANALYSIS.md`](ANALYSIS.md), generated from these files by
`scripts/analyse_results.py`. Do not copy numbers from older docs; regenerate them.

## Layout

| Files | Arm | Cases | Prompt | Status |
| :--- | :--- | :--- | :---: | :--- |
| `baseline_*.jsonl` | no tools | 7 golden (Gemma: 3 passes at temp 0.7; others 1 pass, temp 0) | v2 | current |
| `generated_gemma-*.jsonl`, `generated_gemini-4-hydrogen.jsonl` | no tools | generated, seed 2026 (Gemma 4/track; Hydrogen 2/track: different questions) | v2 | current, with the v2 bond-key defect noted |
| `generated_groq_*.jsonl` | no tools | generated, seed 2026, 4/track | v2 | superseded by `v3_generated_groq_*` |
| `v3_generated_groq_*.jsonl` | no tools | generated, seed 2026, 4/track | v3 | current |
| `v3_tools_generated_groq_*.jsonl` | tools (t2) | same questions as `v3_generated_groq_*` | v3 + t2 | current |
| `archive_prompt_v1/` | no tools | golden | v1 | superseded: wrong vega unit, 2,048-token cap. Do not cite |
| `archive_tools_pilot_t1/` | tools (t1) | generated | v3 + t1 | superseded: see below. Do not cite |
| `logs/` | | | | printed run summaries |

Per-file settings, SHA-256 and the content hash of every question answered are in
[`manifest.json`](manifest.json). Runs from 2026-11-21 on carry their own `<file>.meta.json`;
older runs are described in [`legacy_metadata.json`](legacy_metadata.json), with `null` where a
setting was not recorded.

## Versions and known defects

- **Prompt v1 → v2:** the vega unit was mis-stated, and a 2,048-token cap truncated reasoning models.
- **Prompt v2 → v3:** the `bond_price` key said "price of the 5-year bond", but generated bonds have
  2–10-year maturities. gpt-oss-120b priced a 10-year bond as a 5-year bond and was scored wrong. The
  golden bond case is a 5-year bond, so golden runs are unaffected.
- **Tool prompt t1 → t2:** t1 asked for the final answer as text JSON; gpt-oss-120b tried to send it as
  a call to a non-existent tool named `json`, which Groq rejected (HTTP 400 `tool_use_failed`) although
  the answer was correct. t2 adds a `submit_answer` tool.
- **Case ids:** before 2026-11-21, generated ids did not include the set size, so `GEN-T2-000` meant
  different bonds in the 2-per-track and 4-per-track sets. The manifest records each file's case set;
  analysis matches answers to questions by content hash.
- **allam-2-7b has no tool arm:** Groq returns HTTP 400 "`tool calling` is not supported with this
  model". The tool runner now stops on that error instead of recording it as results.
- **Gemini runs:** `finish_reason` was not recorded, so their empty replies cannot be split into
  truncation and provider failure; they are counted as provider failures (`error`).

## How replies are classified

`error` (provider failure), `model_error` (provider rejected the model's own output), `truncated`
(token cap), `unparsable` (no valid JSON) or `answered`. Non-answers score 0 in the item mean and are
reported separately; only `error` is ever re-run. Details: [Tutorial 05](../docs/tutorials/05_benchmarking_llms_with_open_models.md).

## Reproduce

```bash
uv run python scripts/build_results_manifest.py   # verify files against their declared case sets
uv run python scripts/analyse_results.py          # regenerate ANALYSIS.md and analysis.json
```

New runs: [Tutorial 05](../docs/tutorials/05_benchmarking_llms_with_open_models.md) (no tools) and
[Tutorial 07](../docs/tutorials/07_tools_versus_mental_math.md) (tools). Analysis and citation:
[Tutorial 06](../docs/tutorials/06_analysing_and_publishing_results.md).
