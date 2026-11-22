# Tutorial 06: Analysing, Publishing and Citing Benchmark Results

**Who this is for:** anyone turning benchmark runs into a claim: a report, a thesis chapter, a
paper or a pull request that changes a number in the docs.
**Time:** about 45 minutes. **Cost:** none (no API calls).

## Learning objectives

1. Compute accuracy, failure classes and confidence intervals from stored replies.
2. Compare models and conditions only on identical questions (content hashes, paired differences).
3. Find *how* a model was wrong, field by field, and turn that into a failure taxonomy.
4. Publish results so that others can verify every number: manifest, licence, citable snapshot.

## 1. Every answer is tied to its question

Case ids are labels; the **content hash** is the identity. Before 2026-11-21, `generate_cases(2026, 2)`
and `generate_cases(2026, 4)` both produced a case `GEN-T2-000`, but with different bonds. Results
matched by id would have compared different questions. Ids now carry the seed and set size, and
every new record stores `case_hash`:

```python
from lilavati.bench.datasets import case_content_hash
from lilavati.bench.generator import generate_cases

small = next(c for c in generate_cases(2026, 2) if c.track == "track2_fixed_income")
large = next(c for c in generate_cases(2026, 4) if c.track == "track2_fixed_income")
print(small.case_id, case_content_hash(small)[:12])   # GEN-S2026-N2-T2-000 ...
print(large.case_id, case_content_hash(large)[:12])   # GEN-S2026-N4-T2-000 ... (different)
```

## 2. Summarise a result file

```python
from lilavati.bench.analysis import load_records, summarise
from lilavati.bench.generator import generate_cases

records = load_records("results/v3_generated_groq_openai_gpt-oss-120b.jsonl")
s = summarise(records, generate_cases(2026, 4))
print(s["outcomes"], s["accuracy_on_answered"], s["item_mean"], s["item_mean_ci95"])
```

- **Item mean** scores every non-answer as 0. **Accuracy on answered** ignores them. Report both:
  a model that answers 6 of 20 items perfectly has 100% accuracy on answered and an item mean of 0.30.
- **The 95% CI** is a percentile bootstrap over items (`bootstrap_ci`, 5,000 resamples, fixed seed).
  With 20 items, intervals are wide; two models whose intervals overlap are not shown to differ.

## 3. Compare conditions on the same questions

To claim "tools help", pair each question's no-tool score with its tool score:

```python
from lilavati.bench.analysis import load_records, paired_difference
from lilavati.bench.generator import generate_cases

d = paired_difference(
    load_records("results/v3_generated_groq_qwen_qwen3.8-27b.jsonl"),        # no tools
    load_records("results/v3_tools_generated_groq_qwen_qwen3.8-27b.jsonl"),  # tools
    generate_cases(2026, 4),
)
print(d["n_pairs"], d["mean_diff"], d["diff_ci95"], d["improved"], d["worse"])
```

Pairing removes question difficulty from the comparison, so it needs fewer items than comparing two
unrelated averages. [Tutorial 07](07_tools_versus_mental_math.md) runs the tool arm.

## 4. Look at the mistakes, not just the score

`field_results` gives, per answer key, the value, the expected value and the relative error.
Patterns found in this repository's runs (see `results/ANALYSIS.md` for the current list):

| Failure mode | Example from the runs |
| :--- | :--- |
| Scale or unit error | WACI answered as 0.096 against 94.2 (about 1,000× too small: revenue units) |
| Threshold misjudged | emissions 78,000 against a 75,000 limit judged "target met", so no coupon step-up |
| Following a wrong instruction | a 10-year bond priced as a 5-year bond because the prompt said "5-year" |
| Budget exhausted | long-hand Black-Scholes arithmetic until the 8,192-token cap |
| Formula slip | call price right but gamma and vega about 6–7% off (golden Black-Scholes case) |

These are the cases a deterministic tool removes, and the material for an error-analysis section.

## 5. Regenerate the full analysis

```bash
uv run python scripts/build_results_manifest.py   # results/manifest.json: hashes, case sets, settings
uv run python scripts/analyse_results.py          # results/ANALYSIS.md and results/analysis.json
```

`tests/test_results_manifest.py` fails if the manifest does not match the files, so a published
number cannot drift from its evidence.

## 6. Publish and cite

What this repository does, and what to copy for your own study:

1. **Publish the raw replies**, not only summaries (`results/*.jsonl` are tracked in git). Readers
   can re-score them under a different rule.
2. **Keep superseded runs, labelled.** `results/archive_prompt_v1/` is the first pass with a wrong
   vega unit; it stays, marked "do not cite", because a visible correction is part of the record.
3. **A manifest** with SHA-256, run settings, prompt version and case hashes for every file.
   Older runs whose settings were not recorded say `null`, never a guess.
4. **A data licence.** `results/LICENSE.md` puts the data under CC BY 4.0. Model outputs also
   remain subject to each provider's terms.
5. **A citable snapshot.** Cut a GitHub release, and enable the
   [Zenodo GitHub integration](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
   to get a DOI for that release. Cite the DOI, not the moving `main` branch.

## Using this with Claude Code

Ask the agent to follow the [`lilavati-bench` skill](https://github.com/revendrat/lilavati/blob/main/.claude/skills/lilavati-bench/SKILL.md):
*"Rebuild the manifest, regenerate the analysis, and tell me which model differences are supported
by non-overlapping intervals or paired CIs."* The skill tells it never to edit raw result files and
never to quote a number that is not in `results/ANALYSIS.md`.

## Exercises

1. Compute the item mean and CI for two models on the generated set. Is the difference supported?
2. Pick a model with field misses; classify each miss into the taxonomy above.
3. Re-score a file with a looser tolerance (copy `BenchmarkRunner._score_case_accuracy` and change
   `rel_tol`). Which conclusions change? What does that say about tolerance choices?
4. Write the "Data availability" paragraph of a paper using the manifest.

## For instructors

- **Discussion:** what makes a benchmark number trustworthy? Compare a leaderboard screenshot with a
  manifest-backed result.
- **Assessment idea:** students receive a result file with a hidden flaw (a re-rolled failure, a
  mixed prompt version, a mismatched case hash) and must find it with these tools.
