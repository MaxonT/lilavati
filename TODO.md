# Līlāvatī to-do: from validated toolkit to verifiable financial intelligence

Working plan from the strategy review of 2026-11-21. The phase summary is in [ROADMAP.md](ROADMAP.md)
(Phases 8–12); this file holds the detail. Every phase ends in a **gate**: measurable evidence that
must exist before the next phase starts, so ambition never runs ahead of proof.

**Mission:** verifiable intelligence for finance. Probabilistic cognition, deterministic computation,
verifiable evidence, reproducible conclusions. One validated engine shared by learners, practitioners,
researchers and AI engineers.

## Principles adopted from the review

- [ ] Publish the **layered guarantees** table in the README and every release note:

  | Layer | Guarantee |
  | :--- | :--- |
  | Financial calculation | Deterministic for fixed inputs, configuration and pinned dependencies, within stated tolerances |
  | Formal invariants | Checked by Z3 on the given numbers (not proofs over ranges of inputs) |
  | Data provenance | Traceable to identified sources (document hash, page, tag) |
  | Point-in-time queries | No observation later than the requested cutoff |
  | Agent execution | Reproducible plan and tool trace |
  | LLM reasoning | Evaluated and calibrated, not guaranteed deterministic |
  | Research conclusion | Evidence-supported, with uncertainty disclosed |

- MCP is the interoperability layer, not the product; Līlāvatī stays a tool layer for LangGraph,
  Google ADK and others, not a competing framework.
- Open code is not open data: every dataset states source, licence, retrieval method, version,
  limitations; where redistribution is not allowed, publish ingestion scripts instead of data.
- Every user-facing feature ships with a tutorial and, for workflows, a skill (see the dev skill).

## Corrections to the brainstorm (decided)

- **Multi-agent protocol as a library, not an orchestrator.** Plan validation against a tool
  registry, typed hand-off contracts and an evidence ledger, used inside LangGraph/ADK.
- **India-first data needs per-source licence checks.** *To verify before any doc states it:*
  redistribution terms of NSE/BSE market data, MCA filings, RBI DBIE, and third-party copyright on
  individual FRED series. Licence-clean starting points to confirm: SEC EDGAR, World Bank, ALFRED.
- **Decimal arithmetic only where it matters** (money, accounting identities; Z3 already uses exact
  rationals). Floating-point models (Black-Scholes, GARCH, optimisers) get stated tolerances instead.
- **Replay consistency is defined within a tolerance** on pinned dependencies (solvers can differ in
  the last digits across BLAS builds and platforms).
- **Education/conceptual-competency track deferred** until rubrics and human graders exist.
- **External benchmark facts must be verified before citing** (FinToolBench tool/question counts,
  QuantEval arXiv id, FinanceBench and Finance Agent Benchmark details).
- **Added:** contamination control (private held-out seed; keep it off free tiers that train on
  prompts) and project sustainability (domain reviewers, contributor ladder, funding).

## Gap analysis (brainstorm item → what exists → gap)

| Item | Exists | Gap |
| :--- | :--- | :--- |
| `VerifiedResult` contract | `SourcedValue`, `CitationRef`, `InvariantCheckResult`, `AgentOutput.completed` | One result type: value, unit, currency, method and engine version, input hashes, status VERIFIED / UNVERIFIED / FAILED / NOT_APPLICABLE |
| Evidence ledger, replay | Tracer spans; tool-arm records store every tool call and output; results manifest with content hashes | Claim-level ledger; re-executing a recorded trace and comparing within tolerance |
| Typed plans / DAG | `core/graph.py` statement DAG; `FinancialState`, `FinancialStateGraph` | Validate an LLM-proposed plan against an allowlisted tool registry |
| Point-in-time | Bi-temporal DuckDB store; TPI partial | Vintage sources (ALFRED); poisoned-corpus track (roadmap 7.15) |
| Data connectors | SEC EDGAR XBRL, 10-K PDF parser, Ind AS crosswalk, market-data normalisation | Live validation (7.5); FRED/ALFRED, World Bank, RBI connectors |
| Ontology | Ind AS crosswalk | Canonical metric ids mapped across US GAAP, IFRS, Ind AS |
| Benchmark | 7-track harness; generator for 5 tracks; no-tool and tool arms; manifest, CIs, paired tests | Real-document cases, abstention cases, multi-agent track, scale (7.13c) |
| Multi-agent | Adversarial committee; MCP servers; LangGraph/ADK/A2A examples | A measured architecture comparison |
| Academy | Tutorials 01, 05–07 at the standard; draft course; two skills | Tutorials 02–04 retrofit; tutorial code run in CI |
| Numerical precision | Z3 exact rationals for identities; tolerance rules in the dev skill | Written rounding and tolerance policy per engine |
| Model governance | Validation pages for audited engines | Remaining audits; assumption registry; intended use and limitations per engine |

## Phase 8 (about 4 weeks): earn credibility, v0.2

- [~] Finish Phase 7 blocking items (2026-11-22):
    - [x] CI runs the strict docs build, every example, tutorial code, claims linter and manifest check.
    - [x] Claims linter and sweep of README, docs, docstrings and examples (7.8, language part).
    - [x] Release workflow (Trusted Publishing, clean-install smoke test) and docs deploy workflow.
    - [x] GitHub Pages enabled (source: GitHub Actions) and `pypi` environment created, tag-restricted (7.10, 7.11).
    - [ ] Maintainer: register the PyPI trusted publisher (7.10).
    - [x] Author name and ORCID in `pyproject.toml`, `CITATION.cff` and the papers (7.12).
    - [x] Framework paper sections 6–7 traced to scripts; Table 1 venues verified and corrected (7.8).
    - [x] All 27 unique reference entries verified against primary sources; 9 corrected or removed (7.8).
- [x] Tutorials 02–04 retrofitted and executed in CI (7.24, 2026-11-22).
- [x] Paired tool/no-tool runs on four Groq models, recorded in the benchmark paper §5.3 (2026-11-22).
- [ ] Tool prompt t3: state output units in every tool description (vega per vol point caused a
      double division); re-run the tool arm with 3+ samples; analyse `rejected_output` of malformed calls.
- [ ] Zenodo DOI for the release (7.13b).
- [ ] External review: one practitioner, one software reviewer (7.17).
- **Gate:** v0.2 tagged; every number in README, docs and papers traceable to `results/manifest.json`
  or a validation page.

## Alpha release gate (v0.2)

Do not tag, publish or announce until every item below is true; then ask the maintainer to confirm.

- [x] Claims linter clean and every remaining number traced (7.8).
- [x] Tutorials 01–07 at the tutorial standard and executed in CI (7.24).
- [x] `results/manifest.json` current; benchmark paper numbers match `results/ANALYSIS.md`.
- [x] Author and ORCID (7.12); Pages enabled (7.11); `pypi` environment (7.10); commits on
      `release/v0.2.0`, PR #1 (2026-11-22).
- [ ] PR #1 merged with CI green on `main`.
- [ ] PyPI trusted publisher registered by the maintainer (7.10), then tag `v0.2.0`.
- [x] Release candidate artifacts built and verified locally (2026-11-22, `dist/`, see `dist/VERIFICATION-0.2.0.md`).
- [ ] Maintainer go-ahead for the release.

## Phase 9 (months 1–3): contracts and evidence

- [ ] `VerifiedResult` contract on top of `SourcedValue`; adopt it in all MCP servers.
- [ ] Evidence ledger built from tool traces; replay of a recorded trace with tolerance comparison.
- [ ] Benchmark v2: all 7 tracks generated, 20+ cases per track, 3+ samples per model, private seed.
- [ ] Abstention cases (missing or contradictory inputs, where the right answer is "cannot conclude").
- [ ] Real-document cases: SEC EDGAR for 2 issuers and one Ind AS annual report (7.5, 7.14).
- **Gate:** a paired tool/no-tool result whose CI excludes zero on non-saturated cases, written up
  as a short paper (deterministic execution for probabilistic financial agents).

## Phase 10 (months 3–6): narrow data commons, driven by the benchmark

- [ ] Licence review per source (see corrections); record the result in a dataset card.
- [ ] Connectors: SEC EDGAR company facts, ALFRED vintages, World Bank.
- [ ] Curated India set (about 20 companies) with provenance and licence notes per source.
- [ ] Minimal ontology: about 30 canonical metrics used by the benchmark and tutorials, mapped
      across US GAAP, IFRS and Ind AS, with consolidated/standalone, audited/unaudited, reported/derived,
      restatement links.
- [ ] Point-in-time track with a poisoned corpus (7.15).
- **Gate:** a point-in-time leak measured on a real vintage dataset.

## Phase 11 (months 6–9): multi-agent reliability as research

- [ ] Architecture ladder on LangGraph with constant model, data, questions and budget: no tools;
      tools; two agents; committee; committee + verification; + point-in-time data.
- [ ] Plan validation against the tool registry; disagreement resolved by evidence, not majority.
- [ ] Measure accuracy, unsupported claims, cost per successful task, latency, replay consistency.
- **Gate:** pilot Līlāvatī Financial AI Reliability Report v0, with sample sizes stated.

## Phase 12 (months 9–12): Academy and community

- [ ] Three learning pathways (students, practitioners, researchers/AI engineers) built from the
      tutorials and course; executable lessons with a deliberately wrong AI answer to correct.
- [ ] Pilot the course with one university; collect instructor feedback.
- [ ] Student benchmark competition on a private seed.
- **Gate:** outside contributors landing validated engines through the skills and review process.

## Enterprise-quality track (runs alongside all phases)

- [ ] Semantic versioning; API stability and deprecation policy.
- [ ] Coverage reporting; CI for docs, examples, tutorial code and `scripts/` lint.
- [ ] Dependency vulnerability scanning, SBOM, signed releases.
- [ ] Authentication option for the API server; threat model for agent tool use (prompt injection
      from retrieved documents, tool permission boundaries).
- [ ] Maintainer team with CODEOWNERS per engine; response targets; release cadence.
- [ ] Independent audit of the core engines before any "production" language.

## Decisions for the maintainer

1. Primary audience for the next 6 months: researchers (recommended), students or practitioners?
2. India as flagship geography: yes or no; who owns the licensing checks?
3. Capacity: solo or team; this decides whether Phases 10–12 run in sequence or in parallel.
4. Funding and partners: grants or a university partner would change the timeline.
