# Privacy Policy for Project Līlāvatī

**Effective Date:** November 16, 2026  
**Last Updated:** November 16, 2026  

Project Līlāvatī (`lilavati`) is an open-source, audited & verifiable agentic stack and Model Context Protocol (MCP) server suite designed for institutional finance, quantitative modeling, and autonomous multi-agent reasoning.

We are committed to absolute transparency and data privacy. This Privacy Policy documents our data handling practices for users, developers, and AI agent hosts (including OpenAI, Anthropic Claude, and custom MCP clients).

---

## 1. Zero Telemetry & No Data Collection

* **No Telemetry:** Project Līlāvatī does not contain any analytics, telemetry, crash reporting, tracking scripts, or background phone-home mechanisms.
* **No User Data Collection:** We do not collect, store, profile, or sell personal data, IP addresses, browsing patterns, financial holdings, or proprietary quantitative strategies.
* **No Cookies or Identifiers:** The Līlāvatī framework and MCP servers operate without tracking cookies, advertising identifiers, or persistent device fingerprinting.

---

## 2. Local In-Memory Execution Architecture

* **Client-Side / Local Host Execution:** All Līlāvatī FastMCP servers (`lilavati-adversarial-committee`, `lilavati-derivatives`, `lilavati-econometrics`, `lilavati-financial-ml`, `lilavati-fixed-income`, `lilavati-portfolio-execution`, `lilavati-sustainability`) execute locally in-memory on the user's computing infrastructure or within the user's isolated container.
* **Ephemeral Invariant Verification:** Accounting invariants, Z3 SMT solver routines, BSM/CRR options pricing, and time-series computations run strictly in volatile memory. No user financial inputs, balance sheets, or trade allocations are persisted to external disk or third-party servers.

---

## 3. External Network Endpoints & Public SEC EDGAR Access

* **Public Data Ingestion (`lilavati-corporate-filings`):** The `get_company_filing_statements` and `extract_single_line_item` tools connect directly to the public **U.S. Securities and Exchange Commission (SEC) EDGAR API** (`data.sec.gov`) to retrieve publicly filed 10-K and 10-Q XBRL disclosures.
* **Direct Connections Only:** SEC EDGAR requests are dispatched directly from the host running the server to SEC servers in compliance with standard SEC fair access policies. No intermediary proxy or third-party data aggregator inspects, intercepts, or retains these requests.
* **No Secret Transmission:** The Līlāvatī framework never transmits user API keys, credentials, or private secrets to third-party endpoints.

---

## 4. MCP Tool Safety & Integrity

All 24 Model Context Protocol tools declared in Project Līlāvatī adhere to strict security hints:
* **`readOnlyHint: true`:** All analytical, quantitative, and econometric tools are read-only.
* **`destructiveHint: false`:** No tool modifies local filesystem state, alters databases, or executes destructive operations.
* **`idempotentHint: true`:** Mathematical and quantitative models yield deterministic outputs for given inputs.
* **`openWorldHint`:** Explicitly set to `false` for mathematical and local invariant solvers, and `true` only for SEC EDGAR public network retrieval tools.

---

## 5. Security & Vulnerability Reporting

If you have questions regarding this privacy policy or identify a potential security issue in the repository, please open an issue on the [GitHub repository](https://github.com/revendrat/lilavati) or contact the project maintainers.

---

## 6. Open Source License

Project Līlāvatī is distributed under the **Apache License 2.0**. You are free to inspect, audit, modify, and self-host the complete source code.
