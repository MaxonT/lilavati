# Tutorial 05: Benchmarking LLMs on Finance Tasks with Free and Open Models

**Who this is for:** students, researchers and contributors who want to measure how well a language
model does quantitative finance, without a paid API budget.
**Time:** about 60 minutes. **Cost:** free-tier API quota, or nothing with a local model.

## Learning objectives

By the end you will be able to:

1. Connect Līlāvatī to a hosted open-weight model (Groq), Google's Gemini API, or a local model.
2. Keep API keys out of code, files, logs and chat.
3. Run the no-tool baseline on golden and generated benchmark cases.
4. Tell apart the four ways a model can fail to give a correct answer, and explain why mixing them
   up produces misleading benchmark numbers.

## 1. Choose a model provider

Līlāvatī talks to three API styles: OpenAI-compatible (Groq, vLLM, Ollama, many others), Gemini and
Anthropic. The provider presets live in `src/lilavati/llm/providers.py`.

| Provider | What you get | Key variable |
| :--- | :--- | :--- |
| Groq | open-weight models (gpt-oss, Qwen, ALLaM) over an OpenAI-compatible endpoint | `GROQ_API_KEY` |
| Gemini API | Gemini and Gemma models | `GEMINI_API_KEY` |
| Ollama / vLLM | any local model, no key | none |

List the models your Groq account can use (the key is read from the environment):

```python
import os
import httpx

resp = httpx.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    timeout=30,
)
for m in sorted(resp.json()["data"], key=lambda m: m["id"]):
    print(m["id"], m.get("context_window"))
```

Not every listed model is a chat model: speech-to-text, text-to-speech and safety classifiers are
listed too. Use the chat models.

## 2. Keep your key private

Keys go in environment variables, nowhere else:

```bash
export GROQ_API_KEY=...            # in your shell or shell profile; never in a file in the repo
uv run python -c "import os; print('key set' if os.environ.get('GROQ_API_KEY') else 'missing')"
```

Rules this project enforces (see the [developer skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md), section 5):

- never paste a key into code, a notebook, a commit, an issue or an AI chat. If you did, rotate it
  in the provider console;
- send keys in headers, never in URLs (URLs end up in logs);
- error messages must not echo URLs or headers. The run scripts write `__ERROR__ HTTP 503`, not the request.

A repository test (`tests/test_results_manifest.py::test_no_secrets_in_results`) fails if a key
pattern ever appears in `results/`.

## 3. Look at exactly what the model sees

Every case becomes one prompt. Reading it first is good practice: a wrong instruction produces wrong
answers that look like model errors.

```python
from lilavati.bench.generator import generate_cases
from lilavati.bench.llm_eval import PROMPT_VERSION, build_prompt

case = next(c for c in generate_cases(seed=2026, per_track=4) if c.track == "track2_fixed_income")
print(case.case_id, PROMPT_VERSION)
print(build_prompt(case))
```

!!! warning "A real harness bug this caught"
    Prompt v2 defined the `bond_price` answer key as "price of the 5-year bond", but generated bonds
    have maturities from 2 to 10 years. gpt-oss-120b followed the instruction and priced a 10-year
    bond as a 5-year bond (95.03; the 5-year price is 95.0183), and was scored wrong. Prompt v3 fixed
    the wording, and runs record their prompt version so results from different prompts are never
    mixed.

## 4. Run the no-tool baseline

Golden cases (7, one per track, fixed) and generated cases (fresh numbers from a seed):

```bash
uv run python scripts/run_llm_baseline.py --provider groq --model openai/gpt-oss-20b \
    --max-tokens 8192 --rpm 20 --out results/my_golden_gpt-oss-20b.jsonl

uv run python scripts/run_llm_baseline.py --provider groq --model openai/gpt-oss-20b \
    --generated 4 --seed 2026 --max-tokens 8192 --rpm 20 \
    --out results/my_generated_gpt-oss-20b.jsonl
```

Each run writes one JSON line per answer and a `.meta.json` file with the settings, prompt version
and git commit. Each record stores a `case_hash` that identifies the exact question answered.

Local model instead (no key; start `ollama serve` and pull a model first): create a
`ModelSpec(provider=ModelProvider.OLLAMA, model_name="...", base_url="http://localhost:11434/v1")`
and use `lilavati.bench.llm_eval.build_prompt` with `UniversalLLMClient`, as the script does.

## 5. Four ways not to get a right answer

`lilavati.bench.llm_eval.classify_outcome` puts every reply in one class:

| Class | Meaning | Whose fault | Re-run? |
| :--- | :--- | :--- | :--- |
| `error` | HTTP 429/5xx, timeout, or an empty reply not caused by the token cap | provider | yes (`--resume`) |
| `model_error` | the provider rejected the model's own output (Groq `tool_use_failed`) | model | never |
| `truncated` | the reply hit the `--max-tokens` cap | budget / model | never; report it |
| `unparsable` | the model stopped but gave no valid JSON | model | never |
| `answered` | a JSON answer was parsed; its score says how many fields were right | n/a | never |

```python
from lilavati.bench.llm_eval import classify_outcome

print(classify_outcome({"raw": "__ERROR__ HTTP 503"}))                              # error
print(classify_outcome({"raw": "", "finish_reason": "length", "parsed": None}))     # truncated
print(classify_outcome({"raw": "{bad", "finish_reason": "stop", "parsed": None}))   # unparsable
```

Why it matters: an earlier version of the script re-ran unparsable answers along with provider
errors. That re-rolls the model's failures until it succeeds, and lifted one score from 0.4417 to
0.4667. `--resume` now re-runs only `error`. If a parser or scoring bug is fixed later, use
`--rescore`, which re-scores the stored replies without calling the model again.

## What can go wrong (and what it teaches about LLMs)

- **Inline reasoning breaks naive parsers.** Qwen3.6 writes `<think>...</think>` before its answer;
  braces inside the reasoning made two correct answers unparsable until the parser stripped it.
- **Reasoning models run out of budget.** Qwen3.6 spent its whole 8,192-token budget doing
  Black-Scholes arithmetic by hand. That is a cost and reliability problem, not just a wrong answer.
- **Temperature 0 is not deterministic on every provider.** Two Groq runs of the same cases at
  temperature 0 scored 0.4417 and 0.4917. Report run-to-run spread, or run several samples.
- **Small sets saturate.** Strong models score 100% on the 7 golden cases; use generated cases to
  compare models.

Current measured results for every model: [`results/ANALYSIS.md`](https://github.com/revendrat/lilavati/blob/main/results/ANALYSIS.md).

## Using this with Claude Code

The [`lilavati-bench` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-bench/SKILL.md)
walks an AI coding agent through this tutorial: checking the key is set without printing it,
choosing a model, running, and reporting outcome classes honestly. Ask, for example:
*"Run the generated-case baseline on gpt-oss-20b on Groq and summarise the outcomes."*

## Exercises

1. Run the golden and generated baselines on two Groq models. Which set separates them, and why?
2. Run the same model twice at temperature 0. How different are the item means?
3. Lower `--max-tokens` to 1024 for a reasoning model. Which outcome class grows?
4. Find one wrong answer in your output file and explain the model's mistake from its `raw` text.

## For instructors

- **Discussion:** why is "accuracy on answered items" alone misleading? What should a benchmark
  report about failures to answer?
- **Lab:** each group runs one model; the class pools results into one table and argues which
  differences are real given the confidence intervals ([Tutorial 06](06_analysing_and_publishing_results.md)).
- Free-tier quotas are per account: stagger runs or use a local model for large classes.
