# Tutorial 07: Tools versus Mental Math: Harnessing LLMs Without Trusting Their Arithmetic

**Who this is for:** anyone building or evaluating a finance agent; the core experiment of a course
on AI in finance.
**Time:** about 60 minutes. **Cost:** free-tier API quota (a few model calls per case).

## Learning objectives

1. Explain the division of labour: the model reads the task and chooses tools; deterministic,
   validated code computes the numbers.
2. Run the same benchmark cases with and without tools, and compare them on identical questions.
3. Measure **tool grounding**: whether each number in an answer actually came from a tool.
4. State precisely what this experiment does and does not show.

## 1. The idea in one example

Without tools, a model asked for a Black-Scholes call price and Greeks computes them in text. In
this repository's no-tool runs (prompt v2, 2026-11-20), Qwen3.8 27B missed all four generated Black-Scholes cases and
Qwen3.6 27B ran out of its 8,192-token budget doing the arithmetic by hand. With a tool, the model's
job shrinks to reading `spot`, `strike`, `rate`, `volatility` and `maturity` from the input and
passing them on. The number comes from `DerivativesEngine`, which is tested against QuantLib
([validation](../validation/derivatives.md)).

## 2. The tools

`lilavati.bench.tool_eval.TOOLS` registers eight tools for every case, so choosing the right one is
part of the test:

```python
from lilavati.bench.tool_eval import TOOLS

for name, (description, schema, fn) in TOOLS.items():
    print(f"{name:22} {description}")
```

A ninth tool, `submit_answer`, is the answer channel: its arguments are the answer keys, typed
(number or boolean) and described exactly as in the no-tool prompt. Its arguments are the answer;
they are not counted as a tool output when measuring grounding.

Each of the eight calculation tools is a thin wrapper over a validated engine (bond analytics, Black-Scholes price and Greeks,
order-book microprice, WACI, sustainability-linked bond covenant, Z3 balance-sheet check, cash
reconciliation). The same engines are exposed to any agent framework through the MCP servers
([MCP setup](../mcp-setup.md)).

## 3. Run one case with tools

```python
import os
from lilavati.bench.generator import generate_cases
from lilavati.bench.tool_eval import run_case
from lilavati.llm.providers import INSTITUTIONAL_MODEL_PRESETS

assert os.environ.get("GROQ_API_KEY"), "export GROQ_API_KEY first"
case = next(c for c in generate_cases(2026, 4) if c.track == "track5_derivatives")
res = run_case(case, INSTITUTIONAL_MODEL_PRESETS["groq-gpt-oss-20b"])
for call in res["tool_calls"]:
    print(call["name"], call["arguments"])
print(res["parsed"], "grounded", res["grounded_fields"], "of", res["numeric_fields"])
```

## 4. Run the paired experiment

```bash
# no tools (if you have not already, see Tutorial 05)
uv run python scripts/run_llm_baseline.py --provider groq --model qwen/qwen3.8-27b \
    --generated 4 --seed 2026 --max-tokens 8192 --rpm 20 \
    --out results/my_generated_qwen3.8.jsonl
# tools, same cases
uv run python scripts/run_tool_baseline.py --provider groq --model qwen/qwen3.8-27b \
    --generated 4 --seed 2026 --out results/my_tools_generated_qwen3.8.jsonl
```

Then compare them on the same questions with `paired_difference` ([Tutorial 06](06_analysing_and_publishing_results.md#3-compare-conditions-on-the-same-questions)).
This repository's measured results for five models are in
[`results/ANALYSIS.md`](https://github.com/revendrat/lilavati/blob/main/results/ANALYSIS.md)
("Paired comparison").

## 5. Tool grounding: did the number come from a tool?

A correct number is not proof that a tool was used, and a tool call is not proof that its output
was copied. `tool_grounding` checks each numeric answer field against every number the tools
returned in that episode:

```python
from lilavati.bench.generator import generate_cases
from lilavati.bench.tool_eval import tool_grounding

case = next(c for c in generate_cases(2026, 4) if c.track == "track2_fixed_income")
price = case.expected_values["bond_price"]
print(tool_grounding(case, {"bond_price": price}, [{"clean_price": price}]))   # (1, 1): grounded
print(tool_grounding(case, {"bond_price": price}, []))                         # (0, 1): mental math
```

Grounding is a measured version of the Tool Math Purity Ratio (TMPR) defined in
[the benchmark](../benchmarks.md). Caveat: tool outputs echo some inputs (spot, bid), so a number can
be grounded by matching an echoed input; correctness is still judged by the score.

!!! example "A harness lesson: how the answer comes back matters"
    The first version of this experiment (tool prompt t1) asked for the final answer as a plain
    JSON object, as the no-tool arm does. gpt-oss-120b called `price_bond` correctly, received the
    right price, and then tried to return its answer by **calling a tool named `json`**, which did
    not exist. Groq rejected the call (HTTP 400 `tool_use_failed`), although the rejected text held
    the correct answer. An instruction that reads naturally to a person collided with how the model
    formats tool use. Tool prompt t2 adds `submit_answer`; the pilot is kept, labelled, in
    `results/archive_tools_pilot_t1/`. Records store `tool_prompt_version` so the two are never mixed.

## 6. What this does and does not show

- **It shows** whether a model chooses the right tool, extracts the right parameters, and copies the
  result, and how often that beats computing in text, on the same questions.
- **It does not show** that the engines are correct: the expected answers come from the same
  engines the tools call. Engine correctness is established separately, against textbooks, papers
  and QuantLib (`docs/validation/`). Keep the two claims apart in any write-up.
- **New failure modes appear**, and this repository's runs (2026-11-21/22) measured them:
    - *malformed tool calls*: gpt-oss-20b on all four sustainability cases, rejected by Groq
      (`model_error`, never re-run);
    - *tool-call loops*: Qwen3.8 27B called `reconcile_cash_flow` again and again until the 8-round
      limit on all four accounting cases;
    - *units at the tool boundary*: both gpt-oss models divided an already per-vol-point vega by 100
      again (the price tool did not state its units: a harness lesson too);
    - *doing the maths anyway*: in one live run gpt-oss-20b called the price tool at volatility 0.31
      and 0.32 and estimated vega by finite difference itself instead of reading the returned Greeks.
  Grounding exposed the last two: the ungrounded field was the wrong one. Measured paired results are
  in [`results/ANALYSIS.md`](https://github.com/revendrat/lilavati/blob/main/results/ANALYSIS.md).

## Using this with Claude Code

The [`lilavati-bench` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-bench/SKILL.md)
covers both arms. The repository's [`lilavati-dev` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md)
holds the rule behind this tutorial ("zero LLM arithmetic"), so an AI coding agent contributing an
engine knows the number must come from tested code, not from the model.

## Exercises

1. Run both arms on one model. Which tracks improve most with tools? Which do not, and why?
2. Find a tool-arm answer that is wrong. Was the tool wrong, the parameters, or the copying?
3. Add a ninth tool that is useless (for example, returns the current date). Does tool selection
   get worse?
4. Change the `price_bond` tool description to omit "annual". Does the model still pass `frequency=1`?

## For instructors

- **Lecture pairing:** LLMs as parsers and planners versus calculators; why validated numerical
  libraries matter more, not less, when agents use them.
- **Debate:** "If the tool computes the answer, the benchmark is trivial." Use section 6 to
  structure both sides.
- **Project idea:** wrap a new engine (for example, a duration-matching immunisation solver), validate
  it against a textbook example, expose it as a tool, and measure the paired improvement.
