# Līlāvatī: A Neuro-Symbolic Tool Layer with Checkable Numbers for Financial Agents

**Author:** Revendranath Tirumalsety ([ORCID 0000-0003-0549-4022](https://orcid.org/0000-0003-0549-4022))  
**Classification:** Quantitative Finance, Artificial Intelligence, Multi-Agent Systems, Formal Verification  
**Status:** Working draft, not peer reviewed. Performance numbers and case studies come from internal/simulated runs; see notice below.  

---

> **Draft notice.** Apple 10-K headline figures (net sales, total assets, operating cash flow, page numbers, file SHA-256) were checked against the filing PDF on 2026-11-17 (total assets also against SEC XBRL). Case studies in section 6 are demonstrations on sample/simulated inputs, and statements about commercial LLM behaviour (e.g. error rates) are illustrative, not measured. Latency figures in section 7 come from `scripts/benchmark_latency.py`. The 100% metrics in section 7 reflect the deterministic reference pipeline passing its own checks by construction. Verify filing-derived figures against the primary source before relying on them.

## Abstract

Generic large language model (LLM) agent frameworks exhibit alarming failure rates when deployed in institutional finance. These systems routinely commit numerical hallucinations during multi-step arithmetic, generate financial statements that violate fundamental accounting equations ($\text{Assets} \neq \text{Liabilities} + \text{Shareholders' Equity}$), leak future information through look-ahead bias, and output ungrounded investment claims lacking auditability. 

In this paper, we introduce **Līlāvatī**, an open-source, alpha-stage tool layer whose numbers can be checked, built for quantitative finance, credit risk, econometrics, algorithmic execution, and sustainable finance. Drawing conceptual inspiration from Bhāskara II's classical 1150 CE mathematical treatise *Līlāvatī*—which treated commercial ledgers and gold assaying (*Suvarṇa-gaṇita*) as exact computation—our framework institutes a strict architectural dichotomy between **unstructured semantic perception** (governed by foundation models) and **deterministic quantitative computation** (reference-tested numerical engines, with the Z3 SMT solver checking accounting identities on the given numbers).

Līlāvatī incorporates four foundational innovations:

1. **The Zero LLM Arithmetic Protocol**: Enforced via the Model Context Protocol (FastMCP), barring language models from performing internal mental math and delegating all quantitative computations to verified, sandboxed numerical routines.
2. **The Suvarṇa Invariant Engine**: Z3-based checks of three-statement identities (balance sheet and cash reconciliation) in exact decimal arithmetic on the given values, plus covenant and butterfly no-arbitrage checks. Violations are fed back to the agent as a reflection prompt.
3. **Cryptographic Provenance Fabric (`CitationRef`)**: Anchors every extracted financial metric to a SHA-256 hash of its source (the filing document, or the XBRL fact record as SEC serves it), the page where the source has pages, and a verbatim snippet.
4. **Bi-Temporal State Isolation**: Implemented natively in DuckDB, enforcing dual-time semantics (Valid Time $T_v$ vs. System Time $T_s$) to formally eliminate restatement leakage and look-ahead bias.

We demonstrate the system across complex institutional workflows: parsing real 123-page SEC Form 10-K regulatory disclosures, detecting earnings manipulation via the 8-factor Beneish M-Score, predicting corporate distress via Altman Z-Scores, evaluating Merton structural credit default, and stress-testing climate transition impacts under shadow carbon pricing. We release Līlāvatī as an open-source framework alongside its evaluation harness, establishing a new paradigm for provably sound financial artificial intelligence.

---

## 1. Introduction

The integration of artificial intelligence into quantitative asset management, credit underwriting, and corporate finance has entered a transformative era. Foundation large language models (LLMs) demonstrate remarkable proficiency in processing complex qualitative narratives, synthesizing earnings call transcripts, and parsing regulatory filings [Wu et al., 2023; Yang et al., 2023]. However, the direct application of standard LLM agent frameworks (e.g., LangChain, AutoGen, CrewAI) to institutional capital markets has revealed deep architectural vulnerabilities:

1. **Stochastic Arithmetic and Token Hallucination**: Language models process numerical tokens as probabilistic sequences rather than continuous mathematical entities. Multi-step operations such as Compound Annual Growth Rate (CAGR), Discounted Cash Flow (DCF) discounting, bond convexity, and option Greeks are error-prone when computed in text [Schick et al., 2023]. In this project's own measurements, open-weight models answering without tools scored between 21% and 94% of answer fields on the same generated finance cases, and one missed all four Black-Scholes cases (companion benchmark paper, section 5; `results/ANALYSIS.md`).
2. **Violation of Accounting Invariants**: Generic agents frequently generate or extract financial statement projections where total assets fail to equal total liabilities plus shareholders' equity, or where net cash flow does not reconcile with the cash flow statement's ending cash balance. In audited finance, an unbalanced model is not a minor approximation—it is an unacceptable, invalid model.
3. **Look-Ahead Bias and Restatement Poisoning**: Standard Retrieval-Augmented Generation (RAG) frameworks index corporate disclosures by semantic similarity without temporal gating. As a result, historical backtests routinely retrieve subsequent restatements, revisions, or future footnotes that were not publicly known at the decision epoch $t$, inducing severe look-ahead bias [López de Prado, 2018].
4. **Absence of Cryptographic Provenance**: Institutional allocators and regulatory bodies require full auditability. Standard agent outputs produce declarative quantitative claims without traceable, immutable links to the primary statutory document, physical page, or bounding text coordinates.
5. **Econometric Naivety**: General-purpose coding agents frequently apply linear regressions directly to non-stationary financial time series, generating spurious correlation and illusory alpha signals without verifying stationarity or cointegration [Engle and Granger, 1987; Johansen, 1991].

```
                     GENERIC AGENT PARADIGM (FLAWED)
       User Query ──► [ LLM Mental Math & Unconstrained RAG ] ──► Hallucinated Numbers
                                                                Unbalanced Ledgers
                                                                Look-Ahead Contamination

                 PROJECT LĪLĀVATĪ PARADIGM (CHECKABLE NUMBERS)  
                                 User Query
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │        Generative Semantic Cortex        │
                │    (Hypothesis, Planning, Parameter)     │
                └────────────────────┬─────────────────────┘
                                     │ Structured Tool Call
                                     ▼
                ┌──────────────────────────────────────────┐
                │          FastMCP Tool Bus Layer          │
                │  (Polars, DuckDB, NumPy, SciPy, QuantLib)│
                └────────────────────┬─────────────────────┘
                                     │ Deterministic Execution
                                     ▼
                ┌──────────────────────────────────────────┐
                │       Suvarṇa Z3 SMT Formal Gate         │
                │  (Assets == Liab + Equity | ΔCash Valid) │
                └────────┬─────────────────────────┬───────┘
                         │ SAT                     │ UNSAT
                         ▼                         ▼
                ┌──────────────────┐      ┌─────────────────────────┐
                │ Checked Execution│      │ Violation Reflection    │
                │ & Cited Dossier  │      │ (Autonomous Correction) │
                └──────────────────┘      └─────────────────────────┘
```

### The Historic Ethos: Suvarṇa-Gaṇita and Algorithmic Auditability

To resolve these structural flaws, we return to the historical origins of applied financial mathematics. In 1150 CE, the Indian mathematician and astronomer **Bhāskarācārya (Bhāskara II)** composed the classical mathematical treatise *Līlāvatī*, the opening volume of his magnum opus *Siddhānta Śiromaṇi* [Colebrooke, 1817; Patwardhan, 2001]. Bhāskara devoted whole sections to commercial calculation:

* *Miśraka-vyavahāra* (mixtures): interest, partnerships and the division of profits in proportion to capital shares.
* *Kraya-vikraya* (buying and selling): rules of proportion for prices and quantities.
* *Suvarṇa-gaṇita* (The Computation of Gold): An early systematic treatment of **assaying, alligation and balance verification**. Bhāskara gave exact rules showing that the total intrinsic purity and weight of gold alloys across multiple accounts remained conserved under exchange and melting operations.

Project Līlāvatī takes its name from this tradition of exact commercial computation. We replace the unconstrained, stochastic approximations of generic LLM frameworks with a **neuro-symbolic tool layer** in which numbers come from reference-tested engines, accounting identities are checked by Z3 on the given numbers, and extracted values carry hash-linked provenance.

---

## 2. Related Work & Theoretical Landscape

### 2.1 Financial Language Models and Benchmarks
Recent years have witnessed the introduction of domain-adapted financial language models, including FinBERT [Araci, 2019], BloombergGPT [Wu et al., 2023], and FinGPT [Yang et al., 2023]. While domain pre-training improves financial lexicon comprehension, these models remain auto-regressive next-token predictors susceptible to mathematical hallucination and calibration decay when tasked with numerical manipulation. 

Existing financial benchmarks, such as FinQA [Chen et al., 2021], TAT-QA [Zhu et al., 2021], and FPB [Malo et al., 2014], evaluate isolated question-answering or sentiment classification on synthetic or curated snippets. They do not test an agent's ability to maintain dynamic accounting balance sheets across time, avoid look-ahead bias during historical backtests, or satisfy real-world debt covenants.

### 2.2 Neuro-Symbolic AI and Formal Theorem Proving <!-- claims-ok: names the research field -->
Neuro-symbolic AI seeks to unite the pattern-recognition capabilities of deep neural architectures with the deductive certainty of symbolic logic [Garcez et al., 2019; Manhaeve et al., 2018]. In mathematical reasoning, systems like Lean [de Moura et al., 2015] and Isabelle [Paulson, 1994] have been paired with language models to verify formal proofs. In software verification, Satisfiability Modulo Theories (SMT) solvers, most notably Z3 [De Moura and Bjørner, 2008], provide automated decision procedures for first-order logic over linear real arithmetic, bit-vectors, and uninterpreted functions. 

Līlāvatī integrates an SMT solver into a financial agent runtime, formalizing accounting identities and capital structure constraints as first-order theories over real numbers ($\mathcal{T}_{\mathbb{R}}$).

### 2.3 Financial Machine Learning and Econometric Rigor
In *Advances in Financial Machine Learning*, López de Prado [2018] identified the principal methodological traps in quantitative finance: standard integer differencing ($d=1$) destroys long-range memory, standard $k$-fold cross-validation leaks information across autocorrelated time series, and static time-sampling yields non-Gaussian, leptokurtic returns. Līlāvatī directly embeds de Prado's core algorithms—including optimal fractional differentiation ($d^*$), triple-barrier event labeling, and purged cross-validation—as foundational execution tools.

---

## 3. System Architecture & Formal Methods

Līlāvatī is organized into a modular four-tier architecture designed around the principle of strict separation between semantic perception and mathematical computation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 4: AGENTIC WORKFLOW                        │
│   FinancialStateGraph │ Adversarial Committee │ Deep Reasoning Tree    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    TIER 3: FORMAL INVARIANT GATES                      │
│   SuvarṇaEngine (Z3 SMT) │ CovenantGate │ TemporalPurityGate (DuckDB)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  TIER 2: DETERMINISTIC MCP TOOL BUS                    │
│   Fixed Income │ Econometrics │ Financial ML │ Derivatives │ Sustainability│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                 TIER 1: CONNECTORS & PROVENANCE FABRIC                 │
│   SEC EDGAR │ Ind AS XBRL │ Market Data │ 10-K PDF Decomposer (SHA-256) │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 The Zero LLM Arithmetic Protocol
We formally state the Zero LLM Arithmetic invariant:
> **Axiom 1 (Zero LLM Arithmetic)**: *Let $\mathcal{M}$ be a generative foundation model and let $\mathcal{S}$ be the agent state space. No arithmetic operation $f: \mathbb{R}^n \to \mathbb{R}$ governing valuation, leverage, yield, duration, or accounting reconciliation may be evaluated via the auto-regressive token generation of $\mathcal{M}$. Every operation $f$ must be dispatched as a structured Remote Procedure Call (RPC) across the FastMCP bus to a deterministic compiler $\mathcal{C}$.*

By enforcing Axiom 1, the LLM is restricted to semantic parameter extraction and tool coordination, eliminating numerical rounding and precision drift.

### 3.2 The Suvarṇa Invariant Engine: SMT Formulation of Financial Equilibrium
In audited corporate reporting <!-- claims-ok: audited financial statements -->, a three-statement financial model must satisfy simultaneous linear identities over the real numbers $\mathbb{R}$. We formalize this in first-order logic.

Let $\mathbf{x} \in \mathbb{R}^k$ represent the state vector of canonical financial statement metrics:
$$\mathbf{x} = \begin{bmatrix} A_t & L_t & E_t & C_t & C_{t-1} & CFO_t & CFI_t & CFF_t & NI_t \end{bmatrix}^T$$
where $A_t$ is Total Assets, $L_t$ is Total Liabilities, $E_t$ is Total Shareholders' Equity, $C_t$ is Ending Cash, $CFO_t$ is Cash Flow from Operations, $CFI_t$ is Cash Flow from Investing, $CFF_t$ is Cash Flow from Financing, and $NI_t$ is Net Income at period $t$.

We define the primary **Balance Sheet Equilibrium Theory** $\mathcal{T}_{BS}$:
$$\phi_{BS}(\mathbf{x}) \iff |A_t - (L_t + E_t)| \le \epsilon$$
and the **Cash Flow Dynamic Reconciliation Theory** $\mathcal{T}_{CF}$:
$$\phi_{CF}(\mathbf{x}) \iff |C_t - (C_{t-1} + CFO_t + CFI_t + CFF_t)| \le \epsilon$$
where $\epsilon \ge 0$ represents a strictly parameterized tolerance (e.g., $\epsilon = 1.0$ for unit-rounding in $M disclosures).

The Suvarṇa Engine constructs a Z3 assertion context $\Sigma$:
$$\Sigma = \{ \phi_{BS}(\mathbf{x}), \phi_{CF}(\mathbf{x}), \mathbf{x} = \hat{\mathbf{x}} \}$$
where $\hat{\mathbf{x}}$ represents the vector of empirical observations extracted from the regulatory disclosure.

```python
# Formal Z3 SMT Assertion in SuvarnaEngine
solver = z3.Solver()
a, liab, e = z3.Real("assets"), z3.Real("liabilities"), z3.Real("equity")
c_curr, c_prev = z3.Real("cash_t"), z3.Real("cash_t_minus_1")
cfo, cfi, cff = z3.Real("cfo"), z3.Real("cfi"), z3.Real("cff")

# Invariant 1: Balance Sheet Equilibrium
solver.add(z3.Abs(a - (liab + e)) <= tolerance)

# Invariant 2: Dynamic Cash Flow Delta
solver.add(z3.Abs(c_curr - (c_prev + cfo + cfi + cff)) <= tolerance)

# Grounding empirical observations
for var, val in empirical_bindings.items():
    solver.add(var == z3.RealVal(str(val)))

status = solver.check()
```

If $\Sigma \models \text{SAT}$, the given numbers satisfy the identities and execution proceeds. Because every variable is bound to a concrete value, this check is equivalent to exact rational arithmetic on the decimal inputs; it verifies the numbers supplied, not a model over ranges of inputs. If $\Sigma \models \text{UNSAT}$, the implementation reports which identity failed and by how much, and diagnoses which figures could be wrong: with each figure's given value as a soft constraint and every identity as a hard one, Z3's MaxSMT optimiser (Bjørner, Phan and Fleckenstein 2015) finds the smallest sets of figures whose change makes every identity hold, which in Reiter's (1987) terms are the minimal hitting sets of the conflicts. A figure used by two failing identities is singled out with the value the others imply; when one identity fails, all its figures are listed, since the arithmetic cannot tell them apart. The diagnosis is passed to the agent in the **SMT Reflection Prompt**, which asks for each candidate to be re-checked against the source.

Two further uses go beyond concrete numbers. The smallest debt repayment meeting a net-leverage and an interest-coverage covenant together is a linear optimisation solved exactly in rationals (`z3.Optimize`), under a stated model (repayment from outside the cash balance; interest proportional to debt). Whether both covenants hold for every combination of debt, cash, EBITDA and interest within given ranges is decided by asking Z3 for a breach anywhere in the box: an unsatisfiable answer is a statement about every value in the ranges. Both are tested against closed forms (the repayment formula; the worst corner, since both ratios are monotone in each input). Over several years the inputs are linked (interest on each year's opening debt, an excess cash sweep that may step down with leverage), and the corner shortcut can fail: in the tutorial's two-year example all four corners of the EBITDA guidance comply while a mid-range first year breaches second-year coverage, because better results cut the sweep. The path check searches every path in the ranges, replays each counterexample in plain exact arithmetic, and finds the smallest upfront repayment by a counterexample-guided loop with a quantified-bisection fallback.

### 3.3 Cryptographic Provenance Model (`CitationRef`)
Every empirical metric $\hat{x}_i \in \mathbf{x}$ extracted by Līlāvatī is encapsulated in a formal `CitationRef` tuple:
$$\text{CitationRef}(\hat{x}_i) = \left\langle \mathcal{H}_{\text{SHA-256}}(\mathcal{D}), \mathcal{T}_{filing}, t_{pub}, t_{valid}, p, \text{table\_id}, \sigma_{verbatim} \right\rangle$$
where:

* $\mathcal{H}_{\text{SHA-256}}(\mathcal{D}) \in \{0, 1\}^{256}$ is the SHA-256 digest of the source $\mathcal{D}$: the filing document for PDF or HTML, the fact record (canonical JSON) for an XBRL fact.
* $\mathcal{T}_{filing} \in \{\text{10-K}, \text{10-Q}, \text{8-K}, \text{Ind-AS}\}$ denotes the statutory form type.
* $t_{pub}$ is the official timestamp of publication (System Time $T_s$).
* $t_{valid}$ is the fiscal period end date to which the data applies (Valid Time $T_v$).
* $p \in \mathbb{N}_{\ge 1}$ is the 1-based page within a PDF document; it is absent for XBRL facts, which have no pages.
* $\sigma_{verbatim}$ is the exact textual substring or table cell extracted from the source filing.

The `ProvenanceGate` rejects claims whose citation is malformed (a hash that is not 64 characters, a page below 1, an empty snippet). Given a `SourceStore` of the sources a pipeline actually read, it also resolves each citation: the hash must be a registered source; for an XBRL fact, the tag, period end, filing date and value must match the fact record; for a document, the snippet must appear on the cited page; for a structured disclosure, the value must equal the disclosed one. Without a store only the form is checked. An XBRL fact record carries no concept name, so concepts reported with identical records share a hash and the citation's tag is part of the reference.

### 3.4 Bi-Temporal Isolation and Point-in-Time Queries
To formally eliminate look-ahead bias, Līlāvatī implements a bi-temporal warehouse on DuckDB. Every relational record is bound by a two-dimensional temporal interval:
$$R = \langle \text{data}, [T_v^{start}, T_v^{end}), [T_s^{start}, T_s^{end}) \rangle$$
where Valid Time $[T_v^{start}, T_v^{end})$ specifies the real-world operational window, and System Time $[T_s^{start}, T_s^{end})$ specifies the interval during which the record was known to the market.

For any analytical query executed by an agent at simulated trading epoch $t$, the `TemporalPurityGate` injects an immutable boundary predicate:
$$\mathcal{Q}_{PIT}(t) \iff \sigma_{(T_s^{start} \le t < T_s^{end}) \land (T_v^{end} \le t)}(\text{Warehouse})$$
Queries through this interface therefore exclude revisions published after $t$ (e.g., a restatement filed six months later). This limits look-ahead leakage through the store; it does not by itself stop an agent from using information obtained elsewhere, which the benchmark's TPI is designed to measure (currently partial).

---

## 4. Quantitative Engine Formalisms

The Līlāvatī FastMCP tool registry implements seven core quantitative engines, providing deterministic computational primitives:

### 4.1 Fixed Income & Structural Credit Risk
* **Nelson-Siegel Yield Curve Modeling**: Zero-coupon yield interpolation across maturities $\tau$:
  $$y(\tau) = \beta_0 + \beta_1 \left( \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} \right) + \beta_2 \left( \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda} \right)$$
* **Merton Structural Default Model**: Equity viewed as a European call option on firm assets $V_A$ with face value of debt $D$ maturing at $T$:
  $$E = V_A \Phi(d_1) - D e^{-r T} \Phi(d_2)$$
  $$d_1 = \frac{\ln(V_A / D) + (r + \frac{1}{2}\sigma_A^2)T}{\sigma_A \sqrt{T}}, \quad d_2 = d_1 - \sigma_A \sqrt{T}$$
  Default probability under the physical measure: $\pi_{default} = \Phi(-d_2)$.

### 4.2 Financial Econometrics & Stationarity
* **Augmented Dickey-Fuller (ADF) & Johansen Cointegration**: Testing order of integration $I(d)$.
* **GARCH(1,1) Volatility Dynamics**:
  $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \alpha + \beta < 1$$
* **Ornstein-Uhlenbeck Mean Reversion**: Half-life estimation for statistical arbitrage spreads:
  $$dX_t = \theta (\mu - X_t) dt + \sigma dW_t \implies t_{1/2} = \frac{\ln 2}{\theta}$$

### 4.3 Financial Machine Learning (de Prado Standards)
* **Optimal Fractional Differentiation ($d^*$)**: Expanding $(1-B)^d$ via binomial series to achieve stationarity while maximizing memory preservation:
  $$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = 1 - d B + \frac{d(d-1)}{2!} B^2 - \dots$$
  The optimal order $d^*$ is discovered via binary search satisfying ADF $p\text{-value} \le 0.05$ with minimal correlation loss with the original series.
* **Triple-Barrier Event Labeling**: Bounding paths by upper barrier (take-profit $h_+$), lower barrier (stop-loss $h_-$), and temporal barrier (max holding horizon $T$).

### 4.4 Analytical Derivatives & Volatility Surfaces
* **Black-Scholes-Merton and Cox-Ross-Rubinstein Binomial**: European/American pricing with full first- and second-order Greeks ($\Delta, \Gamma, \mathcal{V}, \Theta, \rho, \text{Vanna}, \text{Volga}$).
* **SABR Stochastic Volatility Model**: Calibration of forward volatility $\sigma_{SABR}(K, F)$:
  $$dF_t = \sigma_t F_t^\beta dW_t^1, \quad d\sigma_t = \alpha \sigma_t dW_t^2, \quad \mathbb{E}[dW_t^1 dW_t^2] = \rho dt$$
* **No-Arbitrage Butterfly Convexity Verification**:
  $$\frac{\partial^2 C(K)}{\partial K^2} \ge 0 \iff C(K - \Delta K) - 2C(K) + C(K + \Delta K) \ge 0$$

### 4.5 Market Microstructure & Algorithmic Execution
* **Order Flow Imbalance (OFI) & Microprice**:
  $$OFI_t = \Delta q_t^b \cdot \mathbb{I}(\Delta p_t^b \ge 0) - \Delta q_t^a \cdot \mathbb{I}(\Delta p_t^a \le 0)$$
* **Almgren-Chriss Optimal Liquidation**: Minimizing expected execution shortfall and variance:
  $$\min_{\{n_k\}} \mathbb{E}[x] + \lambda \mathbb{V}[x], \quad \frac{x_j}{X} = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)}$$
* **Portfolio Optimization**: Black-Litterman Bayesian view integration and López de Prado's Hierarchical Risk Parity (HRP) via quasi-diagonalization and inverse-variance tree clustering.

### 4.6 Sustainable Finance & Climate Risk
* **Weighted Average Carbon Intensity (WACI)**:
  $$\text{WACI} = \sum_{i=1}^N w_i \left( \frac{\text{Scope 1}_i + \text{Scope 2}_i}{\text{Revenue}_i (\text{\$M})} \right)$$
* **Shadow Carbon Tax Stress Testing on Debt Covenants**:
  $$\text{EBIT}_{stressed} = \text{EBIT}_0 - [(\text{Scope 1} + \text{Scope 2}) \cdot \tau_{carbon} \cdot (1 - \gamma_{pass})]$$
  Evaluating whether post-stress Interest Coverage Ratio $\frac{\text{EBIT}_{stressed}}{\text{Interest}} \ge \text{Covenant Floor}$.

---

## 5. Multi-Agent Deliberation: The Adversarial Committee Protocol

The committee (`AdversarialCommittee.evaluate_company`, MCP tool `evaluate_investment_committee_memo`) is a rule-based protocol over deterministic tools; no language model is called. Its roles are labels for fixed rules, and the verdict follows from the tool outputs:

```mermaid
sequenceDiagram
    participant B as Bull (growth rule)
    participant F as Forensic Bear (Beneish, Altman)
    participant R as Risk Officer (identity checks)
    participant C as Verdict rules

    B->>C: Revenue growth vs target
    Note over F: Computes Beneish M-Score & Altman Z-Score
    F->>C: Manipulation flag, distress zone
    Note over R: Z3 balance sheet identity, cash reconciliation on the reported inputs
    alt An identity fails
        R->>C: REJECTED (accounting anomaly), naming the failed identity
    else All identities hold
        R->>C: Identities hold
        Note over C: Rejected on a Beneish flag or Altman distress; approved if growth meets target and Altman is safe; otherwise hold
    end
```

1. **The Bull**: compares year-on-year revenue growth with a target.
2. **The Forensic Bear Auditor**: Executes deterministic forensics:
   * **Beneish 8-Factor M-Score**:
     $$\begin{aligned}
     M = &-4.84 + 0.920 \cdot \text{DSRI} + 0.528 \cdot \text{GMI} + 0.404 \cdot \text{AQI} \\
         &+ 0.892 \cdot \text{SGI} + 0.115 \cdot \text{DEPI} - 0.172 \cdot \text{SGAI} \\
         &+ 4.679 \cdot \text{TATA} - 0.327 \cdot \text{LVGI}
     \end{aligned}$$
     If $M > -1.78$, the company is flagged as a likely manipulator (Beneish 1999).
   * **Altman Z-Score**:
     $$Z = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 0.999 X_5$$
     with $X_2$ retained earnings and $X_3$ operating income (EBIT) over total assets, and $X_4$ market equity over total liabilities, all from the inputs.
3. **The Risk Officer**: checks total assets = total liabilities + equity (Z3, on the given numbers) and opening cash + operating, investing and financing cash flows + exchange-rate effect = closing cash. A failed identity rejects the memo. These inputs are taken as reported: an earlier version derived equity from assets and liabilities and opening cash from the flows, so the checks could not fail, and used net income in place of retained earnings in the Altman score.

---

## 6. Case Studies & Demonstrations

We demonstrate Līlāvatī on example institutional problem sets, showing deterministic execution and formal invariant checking.

### 6.1 SEC Form 10-K Parsing: Apple Inc. FY2026
We ran Līlāvatī on the Apple Inc. FY2026 Form 10-K filing (123-page PDF document, `SHA-256: f9a3e5b32cb7cbf2...`) with `examples/problem_solving/04_sec_10k_filing_analyzer.py` (re-run 2026-11-22). The PDF is not redistributed in this repository: download it from SEC EDGAR to `data/sample_filings/aapl_10k_2026.pdf` to reproduce. 

The `PDF10KParser` decomposed the document, located statutory section boundaries (Item 1, 1A, 7, 8), extracted multi-year three-statement canonical models, and anchored every metric with physical page coordinates:

```
=== LĪLĀVATĪ FILING DOSSIER: APPLE INC. (FY2026) ===
Document Provenance: 123 pages | SHA-256: f9a3e5b32cb7cbf2c44dc2faadc0edc9...

[1] Income Statement (Page 35):
    • Net Sales:              $478,834M (+15.1% YoY vs $416,161M)
    • Gross Margin:           $233,602M (48.8% gross margin)
    • Operating Income:       $158,063M (33.0% EBIT margin)
    • Net Income:             $132,175M (18.0% YoY growth)

[2] Balance Sheet (Page 37):
    • Total Assets:           $404,548M
    • Total Liabilities:      $292,455M
    • Shareholders' Equity:   $112,093M
    • Suvarṇa Z3 check:       SAT (Assets == Liab + Equity for the reported numbers)

[3] Statement of Cash Flows (Page 39):
    • Operating Cash Flow:    $145,097M (FCF Conversion: 1.10x Net Income)
    • Capital Expenditures:   $9,041M (1.9% of Net Sales)
    • Free Cash Flow:         $136,056M
    • Shareholder Cash Return:$102,111M (Buybacks $86.4B + Dividends $15.7B)

[4] Deterministic Diagnostics:
    • Beneish M-Score:        -2.54 (Safe / Non-Manipulator, Threshold -1.78)
    • Altman Z-Score:         12.02 (Safe Zone, Threshold 2.99)
    • Climate Carbon Tax:     1.57 bps EBIT-margin reduction at $100/t (illustrative emissions)

[5] Committee Verdict:        APPROVED_BUY (identities checked; illustrative output)
```

All 12 extracted metrics were anchored to exact primary snippets:

* `curr_revenue`: Page 35 $\to$ `"Total net sales  478,834  416,161  391,035"`
* `curr_total_assets`: Page 37 $\to$ `"Total assets $ 404,548 $ 359,241"`
* `curr_cfo`: Page 39 $\to$ `"Cash generated by operating activities  145,097  111,482  118,254"`

The Beneish and Altman scores are computed from the extracted figures. The carbon-tax line uses illustrative inputs that are not taken from Apple's disclosures (Scope 1 of 350,000 t, Scope 2 of 1,150,000 t, interest expense of USD 3.0 billion); it demonstrates the calculation, not Apple's exposure. A pilot comparison on the same filing and three others, using the HTML statements and XBRL facts rather than the PDF, is in Tutorial 08 (`scripts/run_filing_comparison.py`, `results/filings/`): four open-weight models extracted every line item asked correctly, and one of them computed a gross margin wrongly in all three samples where code-computed ratios were correct.

### 6.2 Forensic Accounting M&A Due Diligence
In a simulated acquisition scenario (`examples/problem_solving/01_forensic_ma_due_diligence.py`, synthetic target claiming 35% revenue growth), Līlāvatī's `ForensicEngine` reported:

* Days Sales in Receivables Index ($\text{DSRI} = 1.48 > 1.0$): receivables grew faster than sales, one sign of aggressive revenue recognition.
* Beneish M-Score: $-1.34 > -1.78$, triggering an automated manipulation alert.
* The committee's rule-based decision was `REJECT ACQUISITION`. A Beneish flag is a reason to investigate, not a finding of fraud.

### 6.3 Private Credit Covenant Monitoring
In a simulated private-credit case (`examples/problem_solving/02_credit_covenant_and_distress_monitor.py`), the covenant check (plain arithmetic with an at-limit tolerance) found net leverage of 4.00x against a 3.50x maximum and interest coverage of 2.33x against a 3.00x minimum; the two-year Merton default probability was 9.93%. The net debt reduction needed to restore the leverage test follows from the covenant definition:
$$\Delta D = (\text{TotalDebt} - \text{Cash}) - 3.50 \times \text{EBITDA} = \$17.5\text{M}$$
How a sponsor equity cure counts towards this (added to EBITDA, or applied to repay debt) is set by each credit agreement.

---

## 7. Performance & Verification Benchmarks

Latency was measured with the reproducible script [`scripts/benchmark_latency.py`](https://github.com/revendrat/lilavati/blob/main/scripts/benchmark_latency.py) (30 runs after warm-up; 3 for the PDF; re-measured 2026-11-22). Environment: Python 3.11.14, macOS 15.5 arm64, Z3 5.4.1. Numbers are machine-dependent; rerun on your hardware.

| Operation | Engine | Median (ms) | p95 (ms) |
| :--- | :--- | ---: | ---: |
| Balance-sheet equation check | Z3 | 1.10 | 1.39 |
| Nelson-Siegel fit (10 points) | SciPy | 3.75 | 5.54 |
| Optimal fractional differentiation d* (1,000 obs) | NumPy / statsmodels | 53.19 | 60.92 |
| SABR calibration (9 strikes, incl. arbitrage check) | SciPy | 2.53 | 2.75 |
| Bi-temporal as-of query | DuckDB (in-memory) | 1.70 | 1.98 |
| 10-K full extraction (123 pages) | pypdf | 5,081 | 5,281 |

### Līlāvatī-Bench metrics for the reference pipeline
The deterministic reference solver scores 100% on TMPR, ICR, PoCS and TPI **by construction**: it uses the same engines the checks use, so these figures confirm the harness runs, not that any model or agent is reliable. Measured model results, including a tool-augmented arm with a measured tool-grounding rate (a version of TMPR), are in section 5 of the companion benchmark paper and in `results/ANALYSIS.md`.

---

## 8. Discussion & Future Roadmap

Project Līlāvatī establishes that the path to reliable autonomous agents in capital markets does not lie in training larger, monolithic generative models to "learn" arithmetic through token prediction. Rather, it requires **neuro-symbolic orchestration**: combining the pattern recognition, summarization, and hypothesis-generation capabilities of frontier language models with deterministic, reference-tested numerical libraries, SMT checks of accounting identities, and bi-temporal databases.

### Active Engineering Roadmap
1. **Open-Source Fine-Tuned Financial Reasoning Models**: Curating a dataset of 500,000 verified SMT reasoning traces to train specialized, compact foundation models (8B–70B) optimized for parameter extraction and tool dispatching.
2. **Streaming Order Book FastMCP Connector**: Extending the market microstructure tier to direct FIX protocol and Level 3 NASDAQ ITCH/OUCH market data feeds.
3. **Multi-Jurisdiction Regulatory Taxonomies**: Expanding the accounting crosswalk from US GAAP and Indian Ind AS to European IFRS and Japanese J-GAAP.

---

## 9. Conclusion

We have introduced **Līlāvatī**, an open-source, alpha-stage tool layer for quantitative finance. It routes arithmetic to reference-tested engines (Zero LLM Arithmetic as a design goal), checks accounting identities on the given numbers with the Suvarṇa Z3 engine, attaches hash-linked provenance with `CitationRef`, and limits look-ahead leakage through bi-temporal DuckDB queries. Section 5 of the companion benchmark paper measures how much this changes model accuracy, and which failure modes remain. 

The complete codebase, documentation, FastMCP microservices, and interactive problem-solving pipelines are open-sourced under the Apache License 2.0 at [https://github.com/revendrat/lilavati](https://github.com/revendrat/lilavati).

---

## References

* **Almgren, R., and Chriss, N.** (2000/2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39. doi:10.21314/JOR.2001.041.
* **Altman, E. I.** (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609.
* **Araci, D.** (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.
* **Beneish, M. D.** (1999). The detection of earnings manipulation. *Financial Analysts Journal*, 55(5), 24–36.
* **Bhāskarācārya (Bhāskara II)** (1150). *Līlāvatī* (*Siddhānta Śiromaṇi*). Translated by H. T. Colebrooke (1817), *Algebra with Arithmetic and Mensuration from the Sanscrit of Brahmegupta and Bhascara*, John Murray, London.
* **Bjørner, N., Phan, A.-D., and Fleckenstein, L.** (2015). νZ - An optimizing SMT solver. *Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2015)*, Lecture Notes in Computer Science 9035, 194–199. doi:10.1007/978-3-662-46681-0_14.
* **Black, F., and Litterman, R.** (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.
* **Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., ... and Wang, W. Y.** (2021). FinQA: A dataset of numerical reasoning over financial data. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 3697–3711.
* **de Moura, L., Kong, S., Avigad, J., van Doorn, F., and von Raumer, J.** (2015). The Lean theorem prover (system description). *Automated Deduction – CADE-25*, Lecture Notes in Computer Science 9195. Springer, Cham. doi:10.1007/978-3-319-21401-6_26.
* **De Moura, L., and Bjørner, N.** (2008). Z3: An efficient SMT solver. *International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS)*, 337–340. Springer, Berlin, Heidelberg.
* **Engle, R. F., and Granger, C. W.** (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
* **Garcez, A. D., Gori, M., Lamb, L. C., Serafini, L., Spranger, M., and Tran, S. N.** (2019). Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. *Journal of Applied Logics*, 6(4), 611–631.
* **Hagan, P. S., Kumar, D., Lesniewski, A. S., and Woodward, D. E.** (2002). Managing smile risk. *Wilmott Magazine*, September, 84–108.
* **Johansen, S.** (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.
* **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. John Wiley & Sons, Hoboken, NJ.
* **López de Prado, M.** (2020). *Machine Learning for Asset Managers*. Cambridge University Press, Cambridge, UK.
* **Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P.** (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.
* **Manhaeve, R., Dumancic, S., Kimmig, A., Demeester, T., and De Raedt, L.** (2018). DeepProbLog: Neural probabilistic logic programming. *Advances in Neural Information Processing Systems (NeurIPS)*, 31, 3749–3759.
* **Merton, R. C.** (1974). On the pricing of corporate debt: The risk structure of interest rates. *The Journal of Finance*, 29(2), 449–470.
* **Nelson, C. R., and Siegel, A. F.** (1987). Parsimonious modeling of yield curves. *Journal of Business*, 60(4), 473–489.
* **Paulson, L. C.** (1994). *Isabelle: A Generic Theorem Prover*. Lecture Notes in Computer Science 828. Springer, Berlin, Heidelberg.
* **Patwardhan, K. S., Naimpally, S. A., and Singh, S. L.** (2001). *Līlāvatī of Bhāskarācārya: A Treatise of Mathematics of Vedic Tradition*. Motilal Banarsidass Publishers, Delhi.
* **Reiter, R.** (1987). A theory of diagnosis from first principles. *Artificial Intelligence*, 32(1), 57–95.
* **Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., ... and Scialom, T.** (2023). Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
* **Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., ... and Mann, G.** (2023). BloombergGPT: A large language model for finance. *arXiv preprint arXiv:2303.17564*.
* **Yang, H., Liu, X. Y., and Wang, C. D.** (2023). FinGPT: Open-source financial large language models. *FinLLM Symposium at IJCAI 2023*. arXiv:2306.06031.
* **Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y.** (2023). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations (ICLR 2023)*.
* **Zhu, F., Lei, W., et al.** (2021). TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP), Volume 1: Long Papers*, 3277–3287.
