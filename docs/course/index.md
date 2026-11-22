# Course: AI in Finance: Harnessing LLMs, Correcting Their Mistakes

*Draft syllabus for a one-semester elective (graduate or final-year undergraduate) in finance,
financial engineering, data science or computer science. Built entirely on this repository; free to
adapt under Apache-2.0 (code) and CC BY 4.0 (results data). Not yet taught: feedback from instructors
who try it is very welcome (open an issue).*

## Why this course

Language models can read a term sheet, a 10-K or a covenant faster than any analyst, and they still
get arithmetic, units and thresholds wrong. This repository's own measurements show both sides
([results](https://github.com/revendrat/lilavati/blob/main/results/ANALYSIS.md)): without tools,
open-weight models scored between 21% and 94% of the answer fields on the same 20 generated
finance cases, with errors such as a carbon-intensity figure 1,000 times too small and a
missed coupon step-up. And carefully written finance code is not safe either: auditing this
repository's own engines against textbooks, papers and QuantLib found and fixed 33 bugs
([validation](../validation/index.md)).

The course teaches students to use LLMs for what they are good at (reading, planning, choosing
tools) and to make every number come from validated, deterministic code, and to show it with
evidence.

## Learning outcomes

Students who complete the course can:

1. Explain, with measured examples, how LLMs fail on quantitative finance tasks (arithmetic, units,
   thresholds, instruction-following, budget exhaustion, provider failures).
2. Implement a finance calculation and validate it against an independent reference, stating units
   and conventions.
3. Expose validated calculations as tools (MCP) and build an agent that uses them.
4. Design and run a fair LLM benchmark: fixed prompts, content-hashed cases, failure classes,
   confidence intervals, paired comparisons.
5. Publish results reproducibly: raw outputs, manifest, licence, citable snapshot.
6. Use AI coding agents responsibly on a shared codebase, with written rules (skills) and tests.

## Prerequisites

Python, introductory statistics, and one course in corporate finance or investments. No machine
learning background is required.

## Schedule (12 weeks)

| Week | Topic | Reading / lab | Deliverable |
| :--- | :--- | :--- | :--- |
| 1 | Why LLMs get finance numbers wrong | README "Why Līlāvatī"; [Tutorial 05](../tutorials/05_benchmarking_llms_with_open_models.md) §5 | Run one model on the golden cases |
| 2 | Validated engines: references, units, conventions | [Validation overview](../validation/index.md); developer skill §2 | Find the convention behind one past bug |
| 3 | Fixed income and credit: covenants and structural default risk | [Tutorial 01](../tutorials/01_build_your_first_financial_agent.md); [fixed-income validation](../validation/fixed_income.md) | Reference test for a bond function |
| 4 | Econometrics, statistical arbitrage, execution | [Tutorial 02](../tutorials/02_statistical_arbitrage_and_execution_agent.md); [econometrics validation](../validation/econometrics.md) | Explain the GARCH order bug |
| 5 | Forensic accounting and adversarial review | [Tutorial 03](../tutorials/03_forensic_accounting_and_adversarial_committee.md) | Beneish score from a real filing |
| 6 | Sustainable finance | [Tutorial 04](../tutorials/04_climate_risk_and_sustainable_portfolio.md); [sustainability validation](../validation/sustainability.md) | WACI and SLB worked example |
| 7 | Benchmarking LLMs fairly | [Tutorial 05](../tutorials/05_benchmarking_llms_with_open_models.md) | Baselines for two models |
| 8 | Evidence: statistics, error analysis, publishing | [Tutorial 06](../tutorials/06_analysing_and_publishing_results.md) | Failure taxonomy for one model |
| 9 | Tools versus mental math | [Tutorial 07](../tutorials/07_tools_versus_mental_math.md) | Paired experiment, both arms |
| 10 | Agents in existing frameworks (MCP, LangGraph, ADK, A2A) | [Integrations](../integrations.md); [MCP setup](../mcp-setup.md) | Agent that answers with tool outputs |
| 11 | AI-assisted engineering on a shared codebase | developer skill; [contributing](../contributing.md) | A reviewed pull request |
| 12 | Capstone presentations | | Capstone report |

## Assessment (suggested)

- **Labs (40%)**: weekly deliverables above, graded on correctness and on evidence (commands, outputs).
- **Error-analysis report (20%)**: classify every wrong answer of one model (Tutorial 06 taxonomy).
- **Capstone (40%)**: one of
    - add an engine from the roadmap (for example Nelson-Siegel-Svensson, key-rate durations), with
      reference tests and a validation page, exposed as a tool, with a paired benchmark;
    - extend the benchmark with a new track or harder multi-step cases, and baseline three models;
    - audit an unaudited engine (order book, Black-Litterman, HRP) and publish the validation page.

Capstones that meet the repository's contribution rules can be submitted as pull requests, so
student work can become part of the shared library.

## Using AI coding agents in the course

Students may use Claude Code or other coding agents. Two skills in the repository give the agent the
same rules students follow:

- [`lilavati-dev`](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-dev/SKILL.md):
  coding, testing and documentation rules (reference tests, units, no invented numbers, key safety).
- [`lilavati-bench`](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-bench/SKILL.md):
  running and analysing benchmarks honestly.

A useful policy: students submit the agent transcript with each lab, and are graded on whether they
caught the agent's mistakes.

## Practicalities for instructors

- **Cost:** everything runs on free tiers or local models. Free-tier quotas are per account; for
  large classes, stagger runs or use Ollama on lab machines.
- **Data:** free tiers of some providers use prompts for training. Keep any held-out exam cases off
  those tiers (generate them from a private seed).
- **Reproducibility:** pin a release of the repository for the semester and cite its snapshot.
