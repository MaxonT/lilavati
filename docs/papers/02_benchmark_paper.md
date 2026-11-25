# Līlāvatī-Bench: A Pilot Benchmark of Numerical Fidelity and Tool Use for Financial Language-Model Agents

**Author:** Revendranath Tirumalsety ([ORCID 0000-0003-0549-4022](https://orcid.org/0000-0003-0549-4022))  
**Subject areas:** Computational finance; evaluation of language models; tool use  
**Status:** Working paper, not peer reviewed. Version of 26 November 2026.  

---

> **Reading note.** Every result in sections 5 and 6 is regenerated from stored model replies by
> `scripts/analyse_results.py` into `results/ANALYSIS.md`, or by the script named next to it; no model
> was called again to produce this text. The benchmark is a pilot: seven golden cases and 20 generated
> cases per model, one pass. Section 8 lists what it cannot show.

## Abstract

Financial question-answering benchmarks score answers, but an agent that handles money must also get
the arithmetic exactly right, keep statements reconciled and use tools without corrupting their outputs.
Līlāvatī-Bench is an open-source pilot benchmark for these properties. Its seven tracks cover accounting
identities, fixed income and credit, econometrics, financial machine learning, derivatives,
microstructure and sustainable finance; reference answers come from deterministic engines checked
against independent references, and a seeded generator produces fresh cases for five tracks. We report
nine models answering without tools and four of them with tools on the same questions. Without tools,
the golden cases did not separate the strongest models, while the generated cases did: accuracy on
answered cases ranged from 21.2 to 100 per cent, and one model missed all four Black–Scholes cases. With
tools, two models improved by 0.30 and 0.41 in mean score, with paired 95 per cent confidence intervals
that exclude zero; tools also introduced protocol failures and a unit error at the tool boundary. On
real 10-Ks, models read the figures almost without error; the failures that remained came from answer
keys, labels and the checker itself. The evidence trail (every reply, a manifest, paired comparisons) is
published so that each number can be checked.

---

## 1. Introduction

Benchmarks for language models in finance mostly test question answering over tables and text (Chen et
al., 2021; Zhu et al., 2021; Chen et al., 2022) or classification (Malo et al., 2014; Xie et al., 2023).
They do not test whether an agent's arithmetic is exact, whether its statements reconcile, or whether
it copies a tool's output faithfully. These matter because a financial figure that is slightly wrong is
not approximately right: a balance sheet out by one unit does not balance, and a covenant test either
holds or fails.

Līlāvatī-Bench tests these properties in a small, reproducible form. It contributes three things:

1. Cases in seven tracks with reference answers from deterministic engines that are themselves checked
   against independent references, and a seeded generator for fresh cases (section 3).
2. A paired design that poses the same questions with and without tools, matched by a content hash of
   each case, with tool grounding measured (section 4).
3. Measured results for nine models, with provider failures, truncation, unparsable replies and wrong
   answers reported separately, and an error analysis drawn from the recorded replies (sections 5 and 6).

Section 2 reviews related benchmarks. Sections 3 and 4 describe the cases and the metrics. Section 5
reports the results and section 6 the errors. Section 7 describes the harness, section 8 the
limitations, and section 9 concludes.

---

## 2. Related work

Table 1 compares Līlāvatī-Bench with existing financial benchmarks on four properties. The comparison
concerns what each benchmark checks, not its size or difficulty: the established benchmarks are far
larger.

**Table 1.** *Financial benchmarks and the properties they check*

| Benchmark | Venue | Primary task | Exact arithmetic checked | Accounting identities | Point-in-time gate | Source verification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| FPB (Malo et al., 2014) | JASIST 2014 | sentiment classification | no | no | no | no |
| FinQA (Chen et al., 2021) | EMNLP 2021 | numerical QA over reports | partly | no | no | no |
| TAT-QA (Zhu et al., 2021) | ACL 2021 | hybrid table and text QA | partly | no | no | no |
| ConvFinQA (Chen et al., 2022) | EMNLP 2022 | conversational numerical QA | partly | no | no | no |
| BloombergGPT evaluation (Wu et al., 2023) | arXiv 2023 | NLP tasks | no | no | no | no |
| PIXIU / FLARE (Xie et al., 2023) | NeurIPS 2023 | financial NLP and prediction | no | no | no | no |
| Līlāvatī-Bench | this paper | quantitative tasks, with and without tools | yes | yes (given numbers) | partial | partial |

*Notes: "partly" means answers are numbers scored with a tolerance but without an exact-arithmetic or
reconciliation check. Līlāvatī-Bench checks identities on the figures given in a case (Z3), not on
statements a model builds; its point-in-time and source checks exist in the harness but are not yet
measured for models (section 4.2).*

Tool use itself is studied by Toolformer (Schick et al., 2023) and ReAct (Yao et al., 2023); this
benchmark measures it in one domain, against reference values.

---

## 3. Cases

### 3.1 Tracks

The benchmark has seven tracks. Table 2 lists what the cases test today; the planned content is in the
repository's roadmap.

**Table 2.** *What each track's cases test*

| Track | Cases test today |
| :--- | :--- |
| 1. Three-statement accounting | the balance sheet balances; the cash flow statement reconciles opening to closing cash |
| 2. Fixed income and credit | bond price; Macaulay duration range; Merton distance to default and default probability |
| 3. Econometrics | the unit-root judgement on a random walk; rejecting a spurious regression in levels |
| 4. Financial machine learning | stationarity after fractional differencing; memory retained |
| 5. Derivatives | Black–Scholes price, delta, gamma and vega; the butterfly no-arbitrage check |
| 6. Microstructure and execution | spread and microprice; Almgren–Chriss holdings at the start and end |
| 7. Sustainable finance | portfolio WACI; sustainability-linked bond target, coupon and step-up |

*Notes: tracks 1, 2, 5, 6 and 7 also have seeded generated cases; tracks 3 and 4 have golden cases only.*

The engines behind the answers implement the standard models: Merton (1974) for structural credit,
Engle and Granger (1987) for cointegration, López de Prado (2018) for fractional differencing, Black and
Scholes (1973) for options, Almgren and Chriss (2000/2001) for execution. Each engine's agreement with
its reference is documented on a validation page in the repository.

### 3.2 Golden and generated cases

Each case is a typed record:

```python
class BenchmarkTestCase(BaseModel):
    case_id: str                    # e.g. LILAVATI-T1-001
    track: str
    title: str
    prompt: str                     # the question put to the model
    as_of_date: date                # point-in-time cutoff
    doc_hash: str                   # SHA-256 of the source document, where there is one
    source_citation: str
    context_data: dict[str, Any]    # the inputs
    expected_values: dict[str, Any] # reference answers from the engines
    tolerance: float                # relative numerical tolerance
```

There are seven golden cases, one per track. The generator produces cases for five tracks from a seed
and a count per track; the results reported here use seed 2026 with four cases per track (20 cases).
Generated boolean answers are balanced so that guessing does not pay. A content-free answerer, which
answers every boolean "true" and every number zero, scores 63.6 per cent on the golden cases but 15.0
per cent on the generated ones (`scripts/trivial_baseline.py`); the golden cases are therefore easy to
guess and the generated cases are the more informative set.

Case identifiers are labels; a content hash of each case's inputs is its identity. Two generator
settings once produced the same identifier for different bonds, so comparisons are paired on the hash.

---

## 4. Metrics

### 4.1 Measured

A case's score is the share of its answer fields within tolerance of the reference. A reply is
classified as answered, truncated (cut off at the token limit), unparsable, a model error (for example,
a malformed tool call rejected by the provider) or a provider error (overload, quota, timeout). Two
summaries are reported:

- **Accuracy on answered cases**: the mean score over answered cases. It measures correctness when the
  model answers.
- **Item mean**: the mean score over all cases, with every non-answer scored zero, and a 95 per cent
  confidence interval by percentile bootstrap over cases (5,000 resamples, seed 0).

For the paired comparison, the same questions are posed with and without tools and matched by content
hash; the difference in score is bootstrapped over the pairs. **Tool grounding** is the share of numeric
answer fields equal to a number a tool returned.

### 4.2 Proposed, not yet measured for models

The harness defines four further metrics, computed today only for the deterministic reference solver,
which passes them by construction because it uses the same engines as the answer key:

- **Proof-of-citation score**: the share of claims whose citation matches the source's SHA-256, the
  page and a verbatim snippet.
- **Invariant compliance rate**: the share of statement sets a model produces that satisfy the balance
  and cash identities (checked with Z3; De Moura & Bjørner, 2008).
- **Temporal purity index**: the share of observations used whose publication date precedes the
  cutoff.
- **Tool math purity ratio**: the share of calculations done by tools rather than in text. Tool
  grounding (section 4.1) is the measured counterpart.

A composite of the four is defined in the code with weights 0.30, 0.30, 0.20 and 0.20. The weights are
not calibrated and no threshold is proposed.

---

## 5. Results

### 5.1 Setup

Four Google models were run on 17 and 18 November 2026 through free-tier APIs; five open-weight models
were run through Groq on 20 to 22 November 2026, at temperature 0, with a limit of 8,192 output tokens
(2,048 for ALLaM 2 7B, whose context is 4,096 tokens). Each case was posed with a direct prompt, no
retrieval and no tools (`scripts/run_llm_baseline.py`), once per model except the Gemma golden runs
(three passes at temperature 0.7). The generated cases for the Groq models use
prompt version 3: version 2 defined the bond answer as "price of the 5-year bond" although generated
bonds run from 2 to 10 years, and gpt-oss-120b priced a 10-year bond as a 5-year one; the version-2 runs
are kept and labelled as superseded. Groq is not deterministic at temperature 0: two version-2 runs of
the same Qwen3.8 cases scored 0.4417 and 0.4917.

### 5.2 Without tools

Table 3 reports every current run.

**Table 3.** *Results without tools*

| Model | Cases | Answered | Accuracy on answered | Item mean (95% CI) | Not answered |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Gemini 4 Hydrogen | golden 7 | 7 / 7 | 100% | 1.000 (1.00–1.00) | none |
| Gemini 3.8 Flash | golden 7 | 5 / 7 | 100% | 0.714 (0.43–1.00) | 2 provider errors |
| Gemma 4 31B | golden 7, 3 passes | 21 / 21 | 100% | 1.000 (1.00–1.00) | none |
| Gemma 4 26B A4B | golden 7, 3 passes | 18 / 21 | 100% | 0.857 (0.71–1.00) | 3 provider errors |
| gpt-oss-120b | golden 7 | 7 / 7 | 100% | 1.000 (1.00–1.00) | none |
| gpt-oss-20b | golden 7 | 7 / 7 | 100% | 1.000 (1.00–1.00) | none |
| Qwen3.6 27B | golden 7 | 6 / 7 | 100% | 0.857 (0.57–1.00) | 1 truncated |
| Qwen3.8 27B | golden 7 | 7 / 7 | 90.7% | 0.907 (0.78–1.00) | none |
| ALLaM 2 7B | golden 7 | 6 / 7 | 69.2% | 0.593 (0.27–0.89) | 1 truncated |
| Gemini 4 Hydrogen | generated, 2 per track | 6 / 10 | 100% | 0.600 (0.30–0.90) | 4 provider errors |
| Gemma 4 31B | generated, 4 per track | 13 / 20 | 98.1% | 0.637 (0.42–0.84) | 7 provider errors |
| Gemma 4 26B A4B | generated, 4 per track | 9 / 20 | 100% | 0.450 (0.25–0.65) | 11 provider errors |
| gpt-oss-120b | generated, 4 per track | 19 / 20 | 98.7% | 0.938 (0.82–1.00) | 1 truncated |
| gpt-oss-20b | generated, 4 per track | 14 / 20 | 100% | 0.700 (0.50–0.90) | 6 truncated |
| Qwen3.6 27B | generated, 4 per track | 14 / 20 | 100% | 0.700 (0.50–0.90) | 6 truncated |
| Qwen3.8 27B | generated, 4 per track | 18 / 20 | 43.5% | 0.392 (0.20–0.59) | 2 unparsable |
| ALLaM 2 7B | generated, 4 per track | 20 / 20 | 21.2% | 0.212 (0.10–0.35) | none |
| Content-free answerer | golden 7 | 7 / 7 | 63.6% | | |
| Content-free answerer | generated, 4 per track | 20 / 20 | 15.0% | | |

*Notes: generated cases use seed 2026; the Gemini 4 Hydrogen generated run used two cases per track.
The Gemma golden runs used temperature 0.7 and three passes; all other runs one pass at temperature 0. Provider errors are overloads (HTTP 503), quota limits
(HTTP 429), timeouts and empty replies. Source: `results/ANALYSIS.md`.*

Three findings follow. First, when the strongest models answered, they were almost always right on
these single-step tasks: the golden cases do not separate them. Second, the generated cases do separate
the Groq models, from 21.2 per cent (ALLaM 2 7B) and 43.5 per cent (Qwen3.8 27B) to 100 per cent on
answered cases. Third, losses have distinct causes that must be reported apart: provider failures
account for most of the Google runs' losses, and truncation for six of 20 generated replies each from
gpt-oss-20b and Qwen3.6 27B. The samples are small, so these are indications, not rankings.

### 5.3 With tools, paired with the runs without

The same 20 generated cases were posed to four Groq models with eight deterministic tools and a tool for
submitting the answer (`scripts/run_tool_baseline.py`, tool prompt t2, 21 and 22 November 2026,
temperature 0); Table 4 compares them with the runs without tools. ALLaM 2 7B has no tool arm: Groq does
not support tool calling for it.

**Table 4.** *Tools minus no tools on the same 20 questions*

| Model | Answered with tools | Item mean, no tools | Item mean, tools | Difference (95% CI) | Better / same / worse | Tool grounding |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| gpt-oss-120b | 20 / 20 | 0.938 | 0.988 | +0.050 (0.00 to +0.15) | 1 / 19 / 0 | 94.4% |
| gpt-oss-20b | 16 / 20 | 0.700 | 0.787 | +0.087 (−0.21 to +0.38) | 6 / 10 / 4 | 96.4% |
| Qwen3.6 27B | 20 / 20 | 0.700 | 1.000 | +0.300 (+0.10 to +0.50) | 6 / 14 / 0 | 100% |
| Qwen3.8 27B | 16 / 20 | 0.392 | 0.800 | +0.408 (+0.11 to +0.69) | 12 / 5 / 3 | 100% |

*Notes: percentile bootstrap over the 20 paired differences (5,000 resamples, seed 0). Over 20 bootstrap
seeds the bounds varied by about ±0.01 (Qwen3.8's lower bound from 0.10 to 0.12); no conclusion
depends on the seed. Source: `results/ANALYSIS.md`.*

For the two Qwen models, tools raised the mean score on the same questions, with intervals that exclude
zero. For gpt-oss-120b the difference is not distinguishable from zero: it was already accurate. For
gpt-oss-20b the interval spans zero because tool-protocol failures offset its gains (section 6). These
expected values come from the engines the tools call, so this arm measures tool selection, parameter
extraction and copying, not the engines' correctness, which is established separately.

### 5.4 On real filings

Two pilots outside the benchmark use real filings and the filers' own XBRL data as the truth.

In the first, four Groq models read the financial statements of four 10-Ks
(`scripts/run_filing_comparison.py`, `results/filings/`). They extracted all 324 line items correctly.
Computing ratios themselves, they got 225 of 228 right; when code computed the ratios from their
extracted figures, 228 of 228 were right.

In the second, a filing-extraction agent with quote and identity checks was run on the latest 10-Ks of
five companies with the same four models (`scripts/run_extraction_repair_eval.py`,
`results/extraction/`). The first run found faults in the checker and the prompt: the quote check
rejected correct quotes that differed only in spacing or apostrophes, and an ambiguous revenue label led
two models to take Walmart's net sales instead of its total revenues. After both were fixed, every model
read all 60 figures correctly in one pass (240 of 240). At temperature 1.0, one model quoted three lines
that are not in the document, all caught, and misread one figure that no check covers (JPMorgan's
noninterest revenue for its total net revenue).

---

## 6. Error analysis

The failures in the recorded runs fall into seven classes. Each is reported separately because each
calls for a different remedy.

1. **Provider failures.** Overloads, quota limits, timeouts and empty replies cost the Google runs up to
   11 of 20 generated cases (Table 3). They say nothing about the model and are never scored as wrong.
2. **Truncation.** gpt-oss-20b and Qwen3.6 27B each ran out of tokens on six of 20 generated cases
   without tools. In the earlier prompt-version-2 runs, Qwen3.6's truncated replies were long-hand
   Black–Scholes arithmetic; with tools, neither model truncated.
3. **Wrong arithmetic.** Qwen3.8 27B missed the price, delta, gamma and vega in all four generated
   Black–Scholes cases, with median relative errors from 0.56 to 2.13. ALLaM 2 7B's misses were larger:
   median relative errors of 45.8 for the call price and 111 for vega, one to two orders of magnitude.
4. **Tool-protocol failures.** With tools, gpt-oss-20b produced malformed first tool calls on all four
   sustainability cases, which Groq rejected; Qwen3.8 27B called the cash-reconciliation tool
   repeatedly until the eight-round limit on all four accounting cases.
5. **Units at the tool boundary.** Both gpt-oss models missed one vega with tools: the pricing tool
   returned vega per volatility point, and the models divided by 100 again because the answer key reads
   "per 1 percentage point". The tool's description did not state the unit of its nested Greeks, so the
   error is partly the harness's. Tool grounding flagged it: the ungrounded field was the wrong one.
6. **Answer keys and prompts.** When every model fails the same way, the key or the prompt is the first
   suspect. Prompt version 2's bond definition (section 5.1) and an answer key that used Walmart's net
   sales where the prompt asked for total revenue were both found this way; the stored replies were
   re-scored, not re-asked.
7. **The checker itself.** In the extraction pilot, 17 figures on which the agent abstained in its
   first run all had correct values: the quote check, not the models, was wrong.

Classes 6 and 7 are harness faults. They were found only because the replies were stored and every
failure was read.

---

## 7. Harness and reproducibility

The harness runs the reference solver, the model baselines and the analysis without network access once
replies are stored:

```bash
uv sync --extra dev
uv run lilavati-bench run --format markdown                 # the reference solver, all tracks
GROQ_API_KEY=... uv run python scripts/run_llm_baseline.py --provider groq \
    --model openai/gpt-oss-120b --generated 4 --seed 2026 --max-tokens 8192 \
    --out results/v3_generated_groq_openai_gpt-oss-120b.jsonl
uv run python scripts/build_results_manifest.py --check    # every result file, hashed
uv run python scripts/analyse_results.py --check           # results/ANALYSIS.md regenerates unchanged
```

Every reply is stored as JSON lines with its run settings; a manifest records each file's SHA-256,
record count, model, case set, prompt version and the content hashes of the questions. Superseded runs
are kept and labelled. Keys are read from environment variables only. Results are published under CC BY
4.0.

---

## 8. Limitations

Six limitations bound what these results show.

1. **Size.** Seven golden and 20 generated cases per model, one pass. Confidence intervals are wide and
   the results are indications, not rankings.
2. **Saturation.** The golden cases do not separate strong models, and the generated cases are single
   steps; multi-step modelling across statements is not yet tested.
3. **Provider variance.** Groq is not deterministic at temperature 0, and free-tier limits removed many
   Google cases.
4. **Expected values from the engines.** The tool arm's answer key comes from the same engines the
   tools call, so the arm measures dispatch and copying, not engine correctness.
5. **Proposed metrics not yet measured.** Citation, identity-compliance, temporal-purity and
   tool-purity metrics are computed only for the reference solver (section 4.2).
6. **Coverage.** Two tracks have golden cases only, no closed commercial models were run, and the
   real-filing pilots cover four and five 10-Ks.

---

## 9. Conclusion

Līlāvatī-Bench is small, but each of its numbers can be traced to a stored reply. On its generated
cases, it separates models that the golden cases do not, and it shows that tools can raise accuracy on
the same questions while adding failure modes of their own. On real filings, reading figures was not the
weak point; answer keys, labels and the checker were. The next version needs more cases per track,
several samples per model, multi-step cases on real filings and measured citation and point-in-time
metrics.

**Code and data.** The harness, cases, replies and analysis are at
[https://github.com/revendrat/lilavati](https://github.com/revendrat/lilavati) (Apache License 2.0 for
the code; CC BY 4.0 for the results).

**Use of AI tools.** The software and this text were developed with AI coding assistants under the
author's direction. The author reviewed the text, the code and every reference, and takes responsibility
for the contents. <!-- AUTHOR TO CONFIRM: wording and extent of AI use (arXiv asks authors to report significant use). -->

---

## References

Almgren, R., & Chriss, N. (2000/2001). Optimal execution of portfolio transactions. *Journal of Risk*, *3*(2), 5–39. https://doi.org/10.21314/JOR.2001.041

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, *81*(3), 637–654.

Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., … Wang, W. Y. (2021). FinQA: A dataset of numerical reasoning over financial data. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 3697–3711). <!-- VERIFY: complete the author list from the paper. -->

Chen, Z., Li, S., Smiley, C., Ma, Z., Shah, S., & Wang, W. Y. (2022). ConvFinQA: Exploring the chain of numerical reasoning in conversational finance question answering. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6279–6292).

De Moura, L., & Bjørner, N. (2008). Z3: An efficient SMT solver. In *Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2008)* (pp. 337–340). Springer.

Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, *55*(2), 251–276.

López de Prado, M. (2018). *Advances in financial machine learning*. John Wiley & Sons.

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, *65*(4), 782–796.

Merton, R. C. (1974). On the pricing of corporate debt: The risk structure of interest rates. *The Journal of Finance*, *29*(2), 449–470.

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., … Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. In *Advances in Neural Information Processing Systems 36*. <!-- VERIFY: complete the author list from the paper. -->

Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., … Mann, G. (2023). *BloombergGPT: A large language model for finance* (arXiv:2303.17564). arXiv. <!-- VERIFY: complete the author list from the paper. -->

Xie, Q., Han, W., Zhang, X., Lai, Y., Peng, M., Lopez-Lira, A., & Huang, J. (2023). PIXIU: A comprehensive benchmark, instruction dataset and large language model for finance. In *Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track*.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. In *International Conference on Learning Representations (ICLR 2023)*.

Zhu, F., Lei, W., et al. (2021). TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing* (Vol. 1, pp. 3277–3287). <!-- VERIFY: complete the author list from the paper. -->
