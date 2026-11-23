"""
SEC 10-K PDF Connector and Document Decomposition Engine.

Ingests unstructured/semi-structured SEC Form 10-K PDF filings, computes cryptographic
document hashes (SHA-256), locates standard statutory disclosure sections (Item 1, 1A, 7, 8),
extracts canonical Three-Statement accounting models with physical citation anchors (page numbers,
verbatim bounding text snippets), and verifies accounting invariants via the Suvarna Z3 SMT solver.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from lilavati.connectors.sec_filing import IdentityCheck
from lilavati.core.invariants import SuvarnaEngine
from lilavati.core.provenance import CitationRef, SourceStore
from lilavati.schemas.statements import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    ThreeStatementModel,
)


class SectionBoundaries(BaseModel):
    """Page index boundaries for key 10-K disclosure sections."""

    model_config = ConfigDict(frozen=True)

    business_page: int | None = Field(None, description="Item 1 Business 1-based page")
    risk_factors_page: int | None = Field(None, description="Item 1A Risk Factors 1-based page")
    mda_page: int | None = Field(None, description="Item 7 MD&A 1-based page")
    financial_statements_page: int | None = Field(
        None, description="Item 8 Financial Statements 1-based page"
    )
    income_statement_page: int | None = Field(
        None, description="Consolidated Statements of Operations page"
    )
    balance_sheet_page: int | None = Field(None, description="Consolidated Balance Sheets page")
    cash_flows_page: int | None = Field(
        None, description="Consolidated Statements of Cash Flows page"
    )


class ExtractedFiling10K(BaseModel):
    """Complete decomposed 10-K filing container with cryptographic provenance."""

    model_config = ConfigDict(frozen=True)

    company_name: str
    fiscal_period_end: date
    filing_date: date
    doc_hash: str
    page_count: int
    sections: SectionBoundaries
    model_current_year: ThreeStatementModel
    model_prior_year: ThreeStatementModel
    citations: dict[str, CitationRef]
    is_smt_verified: bool = Field(
        ..., description="current year: assets = liabilities + equity (Z3) and the cash reconciliation hold")
    smt_verification_details: str = Field(..., description="the checks as JSON")
    missing_items: list[str] = Field(
        default_factory=list,
        description=(
            "Current-year line items not found in the filing. The statement models still hold 0.0 "
            "for them (the schema requires a number), so check this list before using such a field. "
            "A check whose inputs are missing is not run, and its flag (is_balanced, cash_reconciled) is False."
        ),
    )
    checks: list[IdentityCheck] = Field(default_factory=list, description="identities on the parsed numbers")


class PDF10KParser:
    """SEC Form 10-K PDF parser: statement line items with page citations and identity checks.

    Text comes from pypdf; the row labels are Apple's (validated on its FY2026 10-K only).
    """

    def __init__(self, file_path_or_url: str, cache_dir: str = "data/sample_filings") -> None:
        self.source = file_path_or_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = self._resolve_local_path(file_path_or_url)
        self.doc_bytes = self.local_path.read_bytes()
        self.doc_hash = hashlib.sha256(self.doc_bytes).hexdigest()
        self.reader = PdfReader(str(self.local_path))
        self.page_count = len(self.reader.pages)
        self._sections: SectionBoundaries | None = None

    def register_in(self, store: SourceStore) -> str:
        """Register this filing (bytes and page text) in ``store`` so ``ProvenanceGate(store)`` can
        resolve its citations: the hash must match and each snippet must be on its cited page."""
        return store.add_document(self.doc_bytes, [self.reader.pages[i].extract_text() for i in range(self.page_count)])

    def _resolve_local_path(self, path_or_url: str) -> Path:
        """Resolves a local path or downloads remote PDF to local cache."""
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            filename = path_or_url.split("/")[-1].split("?")[0]
            if not filename.endswith(".pdf"):
                filename = "sec_10k_filing.pdf"
            target_path = self.cache_dir / filename
            if not target_path.exists() or target_path.stat().st_size == 0:
                with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                    resp = client.get(path_or_url)
                    resp.raise_for_status()
                    target_path.write_bytes(resp.content)
            return target_path
        p = Path(path_or_url)
        if not p.exists():
            raise FileNotFoundError(f"PDF filing not found at: {path_or_url}")
        return p

    def locate_sections(self) -> SectionBoundaries:
        """Scans the PDF to identify 1-based page positions of major statutory sections."""
        if self._sections is not None:
            return self._sections

        biz_p = None
        rf_p = None
        mda_p = None
        fs_p = None
        is_p = None
        bs_p = None
        cf_p = None

        # Scan up to page 70 (where Item 8 and core statements reside)
        scan_limit = min(self.page_count, 70)
        for idx in range(scan_limit):
            page = self.reader.pages[idx]
            p_num = idx + 1
            txt = page.extract_text()
            txt_lower = txt.lower()

            if p_num > 3:
                if biz_p is None and (
                    "item 1. business" in txt_lower or "item 1.   business" in txt_lower
                ):
                    biz_p = p_num
                if rf_p is None and (
                    "item 1a. risk factors" in txt_lower or "item 1a.   risk factors" in txt_lower
                ):
                    rf_p = p_num
                if mda_p is None and (
                    "item 7. management’s discussion" in txt_lower
                    or "item 7. management's discussion" in txt_lower
                    or "item 7.   management" in txt_lower
                ):
                    mda_p = p_num
                if fs_p is None and (
                    "item 8. financial statements" in txt_lower
                    or "item 8.   financial statements" in txt_lower
                ):
                    fs_p = p_num

            # Detect specific financial statements
            if (
                is_p is None
                and "consolidated statements of operations" in txt_lower
                and ("net sales" in txt_lower or "total revenue" in txt_lower)
            ):
                is_p = p_num
            if (
                bs_p is None
                and "consolidated balance sheets" in txt_lower
                and "total assets" in txt_lower
            ):
                bs_p = p_num
            if (
                cf_p is None
                and "consolidated statements of cash flows" in txt_lower
                and "operating activities" in txt_lower
            ):
                cf_p = p_num

        self._sections = SectionBoundaries(
            business_page=biz_p,
            risk_factors_page=rf_p,
            mda_page=mda_p,
            financial_statements_page=fs_p,
            income_statement_page=is_p,
            balance_sheet_page=bs_p,
            cash_flows_page=cf_p,
        )
        return self._sections

    @staticmethod
    def _parse_line_numbers(line: str) -> list[float]:
        """Extracts financial numbers from a line of text, handling parenthesis for negative numbers."""
        matches = re.finditer(
            r"(?:(?P<neg>\()\s*)?(?:\$\s*)?(?P<num>[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)(?(neg)\))",
            line,
        )
        vals: list[float] = []
        for m in matches:
            val = float(m.group("num").replace(",", ""))
            if m.group("neg"):
                val = -val
            vals.append(val)
        return vals

    _TRAILING_NUMBERS = re.compile(r"[A-Za-z][^A-Za-z]*$")

    @classmethod
    def _row_amounts(cls, lines: list[str], idx: int) -> tuple[list[float], str]:
        """Amounts of the statement row starting at ``lines[idx]``: the numbers after its last word.

        A label can wrap onto following lines (e.g. "Common stock ..., $0.00001 par value: ... shares"
        then "authorized; ... respectively  104,815  93,568"); the row is joined with up to two
        continuation lines until it ends in numbers. Taking only the trailing numbers keeps par
        values and share counts inside the label out of the amounts.
        """
        row = lines[idx].strip()
        if row.endswith(":"):  # a section header ("Cost of sales:"), not a row with amounts
            return [], row
        for extra in lines[idx + 1: idx + 3]:
            tail = cls._TRAILING_NUMBERS.search(row)
            if tail and cls._parse_line_numbers(tail.group(0)[1:]):
                break
            row = f"{row} {extra.strip()}"
        tail = cls._TRAILING_NUMBERS.search(row)
        amounts = cls._parse_line_numbers(tail.group(0)[1:]) if tail else []
        return amounts, row

    @classmethod
    def _find_in_lines(
        cls, lines: list[str], target_keywords: list[str], col_idx: int = 0
    ) -> tuple[float | None, str | None]:
        """Value in column ``col_idx`` of the first row labelled by a keyword.

        Rows whose label starts with the keyword win over rows that merely contain it:
        "Income before provision for income taxes" contains "provision for income taxes", and the
        substring match once returned pretax income as the tax expense.
        """
        for prefer_start in (True, False):
            for kw in target_keywords:  # keywords are in priority order
                k = kw.lower()
                for i, line in enumerate(lines):
                    text = line.strip().lower()
                    if (text.startswith(k) if prefer_start else k in text):
                        amounts, row = cls._row_amounts(lines, i)
                        if len(amounts) > col_idx:
                            return amounts[col_idx], row
        return None, None

    @classmethod
    def _sum_rows(cls, lines: list[str], label: str, col_idx: int = 0) -> float | None:
        """Sum of column ``col_idx`` over every row whose label starts with ``label``.

        Balance sheets list some items twice, e.g. "Term debt" under current and again under
        non-current liabilities; total debt needs both.
        """
        total, found = 0.0, False
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(label.lower()):
                amounts, _ = cls._row_amounts(lines, i)
                if len(amounts) > col_idx:
                    total += amounts[col_idx]
                    found = True
        return total if found else None

    CASH_BEGIN_LABELS = ["cash, cash equivalents, and restricted cash and cash equivalents, beginning",
                         "cash, cash equivalents and restricted cash, beginning",
                         "cash and cash equivalents, beginning", "cash and cash equivalents at beginning"]
    CASH_END_LABELS = ["cash, cash equivalents, and restricted cash and cash equivalents, ending",
                       "cash, cash equivalents and restricted cash, end",
                       "cash and cash equivalents, end", "cash and cash equivalents at end"]

    @classmethod
    def _find_wrapped(cls, lines: list[str], labels: list[str], col_idx: int = 0) -> float | None:
        """Value in column ``col_idx`` of the first row whose label, joined with the next line, starts
        with one of ``labels`` (lower case). Apple's cash balance labels wrap before "beginning"."""
        for label in labels:
            for i in range(len(lines)):
                joined = " ".join(x.strip() for x in lines[i: i + 2]).lower()
                if joined.startswith(label):
                    amounts, _ = cls._row_amounts(lines, i)
                    if len(amounts) > col_idx:
                        return amounts[col_idx]
        return None

    def _page_lines(self, page_num: int) -> list[str]:
        if page_num < 1 or page_num > self.page_count:
            return []
        return self.reader.pages[page_num - 1].extract_text().split("\n")

    def _find_metric_in_page(
        self, page_num: int, target_keywords: list[str], col_idx: int = 0
    ) -> tuple[float | None, str | None]:
        """Finds a line item by label on a page and returns (value in column, raw row)."""
        return self._find_in_lines(self._page_lines(page_num), target_keywords, col_idx)

    def extract_full_filing(
        self,
        company_name: str = "Apple Inc.",
        filing_date: date = date(2026, 11, 3),
        period_end_curr: date = date(2026, 9, 26),
        period_end_prior: date = date(2025, 9, 27),
    ) -> ExtractedFiling10K:
        """Extracts full 3-statement model for current and prior years with citations and SMT verification."""
        sections = self.locate_sections()
        if (
            not sections.income_statement_page
            or not sections.balance_sheet_page
            or not sections.cash_flows_page
        ):
            raise ValueError(
                f"Could not automatically locate all 3 financial statement pages in filing. "
                f"Found: IS={sections.income_statement_page}, BS={sections.balance_sheet_page}, CF={sections.cash_flows_page}"
            )

        citations: dict[str, CitationRef] = {}

        def record_citation(metric_key: str, p_num: int, snippet: str | None, p_end: date) -> None:
            if snippet:
                citations[metric_key] = CitationRef(
                    doc_hash=self.doc_hash,
                    filing_type="10-K",
                    filing_date=filing_date,
                    period_end=p_end,
                    page=p_num,
                    raw_snippet=snippet,
                )

        # 1. Income Statement (col_idx=0 is current year, col_idx=1 is prior year)
        is_p = sections.income_statement_page
        rev_curr, snip_rev_curr = self._find_metric_in_page(
            is_p, ["Total net sales", "Total net revenue", "Total revenue"], 0
        )
        rev_prior, snip_rev_prior = self._find_metric_in_page(
            is_p, ["Total net sales", "Total net revenue", "Total revenue"], 1
        )
        record_citation("curr_revenue", is_p, snip_rev_curr, period_end_curr)
        record_citation("prior_revenue", is_p, snip_rev_prior, period_end_prior)

        cogs_curr, snip_cogs_curr = self._find_metric_in_page(
            is_p, ["Total cost of sales", "Cost of sales", "Cost of goods sold"], 0
        )
        cogs_prior, _ = self._find_metric_in_page(
            is_p, ["Total cost of sales", "Cost of sales", "Cost of goods sold"], 1
        )

        gp_curr, snip_gp_curr = self._find_metric_in_page(is_p, ["Gross margin", "Gross profit"], 0)
        gp_prior, _ = self._find_metric_in_page(is_p, ["Gross margin", "Gross profit"], 1)
        record_citation("curr_gross_profit", is_p, snip_gp_curr, period_end_curr)

        opinc_curr, snip_opinc_curr = self._find_metric_in_page(is_p, ["Operating income"], 0)
        opinc_prior, _ = self._find_metric_in_page(is_p, ["Operating income"], 1)
        record_citation("curr_operating_income", is_p, snip_opinc_curr, period_end_curr)

        ni_curr, snip_ni_curr = self._find_metric_in_page(is_p, ["Net income"], 0)
        ni_prior, _ = self._find_metric_in_page(is_p, ["Net income"], 1)
        record_citation("curr_net_income", is_p, snip_ni_curr, period_end_curr)

        rnd_curr, _ = self._find_metric_in_page(is_p, ["Research and development"], 0)
        rnd_prior, _ = self._find_metric_in_page(is_p, ["Research and development"], 1)

        sga_curr, _ = self._find_metric_in_page(is_p, ["Selling, general and administrative"], 0)
        sga_prior, _ = self._find_metric_in_page(is_p, ["Selling, general and administrative"], 1)

        tax_curr, _ = self._find_metric_in_page(is_p, ["Provision for income taxes"], 0)
        tax_prior, _ = self._find_metric_in_page(is_p, ["Provision for income taxes"], 1)

        # 2. Balance Sheet (col_idx=0 is current year, col_idx=1 is prior year)
        bs_p = sections.balance_sheet_page
        cash_curr, snip_cash_curr = self._find_metric_in_page(
            bs_p, ["Cash and cash equivalents"], 0
        )
        cash_prior, _ = self._find_metric_in_page(bs_p, ["Cash and cash equivalents"], 1)
        record_citation("curr_cash", bs_p, snip_cash_curr, period_end_curr)

        ar_curr, _ = self._find_metric_in_page(bs_p, ["Accounts receivable, net"], 0)
        ar_prior, _ = self._find_metric_in_page(bs_p, ["Accounts receivable, net"], 1)

        inv_curr, _ = self._find_metric_in_page(bs_p, ["Inventories"], 0)
        inv_prior, _ = self._find_metric_in_page(bs_p, ["Inventories"], 1)

        cur_assets_curr, _ = self._find_metric_in_page(bs_p, ["Total current assets"], 0)
        cur_assets_prior, _ = self._find_metric_in_page(bs_p, ["Total current assets"], 1)

        ppe_curr, _ = self._find_metric_in_page(bs_p, ["Property, plant and equipment, net"], 0)
        ppe_prior, _ = self._find_metric_in_page(bs_p, ["Property, plant and equipment, net"], 1)

        assets_curr, snip_assets_curr = self._find_metric_in_page(bs_p, ["Total assets"], 0)
        assets_prior, snip_assets_prior = self._find_metric_in_page(bs_p, ["Total assets"], 1)
        record_citation("curr_total_assets", bs_p, snip_assets_curr, period_end_curr)
        record_citation("prior_total_assets", bs_p, snip_assets_prior, period_end_prior)

        ap_curr, _ = self._find_metric_in_page(bs_p, ["Accounts payable"], 0)
        ap_prior, _ = self._find_metric_in_page(bs_p, ["Accounts payable"], 1)

        cp_curr, _ = self._find_metric_in_page(bs_p, ["Commercial paper"], 0)
        cp_prior, _ = self._find_metric_in_page(bs_p, ["Commercial paper"], 1)

        term_curr = self._sum_rows(self._page_lines(bs_p), "Term debt", 0)  # current + non-current
        term_prior = self._sum_rows(self._page_lines(bs_p), "Term debt", 1)

        cur_liab_curr, _ = self._find_metric_in_page(bs_p, ["Total current liabilities"], 0)
        cur_liab_prior, _ = self._find_metric_in_page(bs_p, ["Total current liabilities"], 1)

        liab_curr, snip_liab_curr = self._find_metric_in_page(bs_p, ["Total liabilities"], 0)
        liab_prior, _ = self._find_metric_in_page(bs_p, ["Total liabilities"], 1)
        record_citation("curr_total_liabilities", bs_p, snip_liab_curr, period_end_curr)

        equity_curr, snip_equity_curr = self._find_metric_in_page(
            bs_p, ["Total shareholders’ equity"], 0
        )
        if equity_curr is None:
            equity_curr, snip_equity_curr = self._find_metric_in_page(
                bs_p, ["Total shareholders' equity"], 0
            )
        equity_prior, _ = self._find_metric_in_page(bs_p, ["Total shareholders’ equity"], 1)
        if equity_prior is None:
            equity_prior, _ = self._find_metric_in_page(bs_p, ["Total shareholders' equity"], 1)
        record_citation("curr_shareholders_equity", bs_p, snip_equity_curr, period_end_curr)

        re_curr, _ = self._find_metric_in_page(bs_p, ["Retained earnings"], 0)
        re_prior, _ = self._find_metric_in_page(bs_p, ["Retained earnings"], 1)

        common_eq_curr, _ = self._find_metric_in_page(
            bs_p, ["Common stock and additional paid-in capital"], 0
        )
        common_eq_prior, _ = self._find_metric_in_page(
            bs_p, ["Common stock and additional paid-in capital"], 1
        )

        # 3. Cash Flow Statement (col_idx=0 is current year, col_idx=1 is prior year)
        cf_p = sections.cash_flows_page
        cfo_curr, snip_cfo_curr = self._find_metric_in_page(
            cf_p,
            ["Cash generated by operating activities", "Cash provided by operating activities"],
            0,
        )
        cfo_prior, _ = self._find_metric_in_page(
            cf_p,
            ["Cash generated by operating activities", "Cash provided by operating activities"],
            1,
        )
        record_citation("curr_cfo", cf_p, snip_cfo_curr, period_end_curr)

        capex_curr, snip_capex_curr = self._find_metric_in_page(
            cf_p,
            ["Payments for acquisition of property, plant and equipment", "Capital expenditures"],
            0,
        )
        capex_prior, _ = self._find_metric_in_page(
            cf_p,
            ["Payments for acquisition of property, plant and equipment", "Capital expenditures"],
            1,
        )
        record_citation("curr_capex", cf_p, snip_capex_curr, period_end_curr)

        cfi_curr, _ = self._find_metric_in_page(
            cf_p,
            [
                "Cash generated by/(used in) investing activities",
                "Cash used in investing activities",
            ],
            0,
        )
        cfi_prior, _ = self._find_metric_in_page(
            cf_p,
            [
                "Cash generated by/(used in) investing activities",
                "Cash used in investing activities",
            ],
            1,
        )

        cff_curr, _ = self._find_metric_in_page(
            cf_p,
            [
                "Cash used in financing activities",
                "Cash generated by/(used in) financing activities",
            ],
            0,
        )
        cff_prior, _ = self._find_metric_in_page(
            cf_p,
            [
                "Cash used in financing activities",
                "Cash generated by/(used in) financing activities",
            ],
            1,
        )

        repurchases_curr, _ = self._find_metric_in_page(cf_p, ["Repurchases of common stock"], 0)
        repurchases_prior, _ = self._find_metric_in_page(cf_p, ["Repurchases of common stock"], 1)

        dividends_curr, _ = self._find_metric_in_page(cf_p, ["Payments for dividends"], 0)
        dividends_prior, _ = self._find_metric_in_page(cf_p, ["Payments for dividends"], 1)

        # Record what was not found before the fallbacks below replace it with 0.0, so that a
        # missing line item is never mistaken for a reported zero.
        found = {
            "revenue": rev_curr, "cogs": cogs_curr, "gross_profit": gp_curr,
            "operating_income": opinc_curr, "net_income": ni_curr, "sga_expense": sga_curr,
            "research_and_development": rnd_curr, "tax_expense": tax_curr,
            "interest_expense": None,  # not searched: many filers do not report it as a line
            "cash_and_equivalents": cash_curr, "accounts_receivable": ar_curr, "inventory": inv_curr,
            "net_ppe": ppe_curr, "total_assets": assets_curr, "accounts_payable": ap_curr,
            "commercial_paper": cp_curr, "term_debt": term_curr, "total_liabilities": liab_curr,
            "total_equity": equity_curr, "retained_earnings": re_curr, "common_equity": common_eq_curr,
            "cfo": cfo_curr, "capex": capex_curr,
        }
        missing_items = sorted(k for k, v in found.items() if v is None)

        # Cash flow statement balances, per year: the cash flow statement reconciles cash including
        # restricted cash, which can differ from the balance sheet's cash line.
        cf_lines = self._page_lines(cf_p)
        begin = [self._find_wrapped(cf_lines, self.CASH_BEGIN_LABELS, col) for col in (0, 1)]
        end = [self._find_wrapped(cf_lines, self.CASH_END_LABELS, col) for col in (0, 1)]
        fx = [self._find_in_lines(cf_lines, ["Effect of exchange rate"], col)[0] for col in (0, 1)]
        found["cfi"], found["cff"] = cfi_curr, cff_curr
        found["cash_flow_beginning"], found["cash_flow_ending"] = begin[0], end[0]
        missing_items = sorted(k for k, v in found.items() if v is None)

        def z(x: float | None) -> float:
            """0.0 for an item not found (listed in missing_items); a reported 0 stays 0."""
            return 0.0 if x is None else x

        # Construct Pydantic canonical models. Items not found are 0.0 and listed in missing_items;
        # nothing is substituted (no equity from assets minus liabilities, no cash flows from capex,
        # buybacks or dividends), so no check below can pass on a value derived from what it checks.
        is_model_curr = IncomeStatement(
            revenue=z(rev_curr), cogs=z(cogs_curr), gross_profit=z(gp_curr), sga_expense=z(sga_curr),
            operating_income=z(opinc_curr), interest_expense=0.0, tax_expense=z(tax_curr),
            net_income=z(ni_curr),
        )
        is_model_prior = IncomeStatement(
            revenue=z(rev_prior), cogs=z(cogs_prior), gross_profit=z(gp_prior), sga_expense=z(sga_prior),
            operating_income=z(opinc_prior), interest_expense=0.0, tax_expense=z(tax_prior),
            net_income=z(ni_prior),
        )
        bs_model_curr = BalanceSheet(
            cash_and_equivalents=z(cash_curr), accounts_receivable=z(ar_curr), inventory=z(inv_curr),
            net_ppe=z(ppe_curr), total_assets=z(assets_curr), accounts_payable=z(ap_curr),
            total_debt=z(cp_curr) + z(term_curr), total_liabilities=z(liab_curr),
            common_equity=z(common_eq_curr), retained_earnings=z(re_curr), total_equity=z(equity_curr),
        )
        bs_model_prior = BalanceSheet(
            cash_and_equivalents=z(cash_prior), accounts_receivable=z(ar_prior), inventory=z(inv_prior),
            net_ppe=z(ppe_prior), total_assets=z(assets_prior), accounts_payable=z(ap_prior),
            total_debt=z(cp_prior) + z(term_prior), total_liabilities=z(liab_prior),
            common_equity=z(common_eq_prior), retained_earnings=z(re_prior), total_equity=z(equity_prior),
        )
        flows = [(cfo_curr, cfi_curr, cff_curr), (cfo_prior, cfi_prior, cff_prior)]
        cf_models = [CashFlowStatement(cfo=z(o), cfi=z(i), cff=z(f), net_change_in_cash=z(o) + z(i) + z(f))
                     for o, i, f in flows]

        checks: list[IdentityCheck] = []
        balanced: list[bool] = []
        reconciled: list[bool] = []
        for col, (year, (ass, liab, eq)) in enumerate(
                (("current", (assets_curr, liab_curr, equity_curr)),
                 ("prior", (assets_prior, liab_prior, equity_prior)))):
            if ass is not None and liab is not None and eq is not None:
                res = SuvarnaEngine.verify_balance_sheet_equation(
                    total_assets=ass, total_liabilities=liab, shareholders_equity=eq, tolerance=0.5)
                checks.append(IdentityCheck(name=f"{year}: assets = liabilities + equity", holds=res.is_valid,
                                            difference=ass - liab - eq, detail="Z3 on the parsed numbers"))
                balanced.append(res.is_valid)
            else:
                balanced.append(False)  # not checkable: an input is missing (see missing_items)
            o, i, f = flows[col]
            if begin[col] is not None and end[col] is not None and None not in (o, i, f):
                d = z(end[col]) - (z(begin[col]) + z(o) + z(i) + z(f) + z(fx[col]))
                checks.append(IdentityCheck(
                    name=f"{year}: opening cash + cash flows = closing cash", holds=abs(d) <= 0.5, difference=d,
                    detail="cash flow statement balances; CFO + CFI + CFF + exchange-rate effect"
                           + ("" if fx[col] is not None else " (no exchange-rate line, taken as 0)")))
                reconciled.append(abs(d) <= 0.5)
            else:
                reconciled.append(False)

        tsm_curr = ThreeStatementModel(
            income_statement=is_model_curr, balance_sheet=bs_model_curr, cash_flow_statement=cf_models[0],
            is_balanced=balanced[0], cash_reconciled=reconciled[0],
        )
        tsm_prior = ThreeStatementModel(
            income_statement=is_model_prior, balance_sheet=bs_model_prior, cash_flow_statement=cf_models[1],
            is_balanced=balanced[1], cash_reconciled=reconciled[1],
        )

        return ExtractedFiling10K(
            company_name=company_name,
            fiscal_period_end=period_end_curr,
            filing_date=filing_date,
            doc_hash=self.doc_hash,
            page_count=self.page_count,
            sections=sections,
            model_current_year=tsm_curr,
            model_prior_year=tsm_prior,
            citations=citations,
            is_smt_verified=balanced[0] and reconciled[0],
            smt_verification_details=json.dumps([c.model_dump() for c in checks]),
            missing_items=missing_items,
            checks=checks,
        )

    def extract_mda_excerpt(self, max_pages: int = 3) -> str:
        """Extracts executive overview text from Item 7 MD&A."""
        sections = self.locate_sections()
        if not sections.mda_page:
            return ""
        start = sections.mda_page - 1
        end = min(start + max_pages, self.page_count)
        chunks: list[str] = []
        for p in range(start, end):
            chunks.append(self.reader.pages[p].extract_text())
        return "\n\n".join(chunks)

    def extract_risk_factor_excerpt(self, max_pages: int = 2) -> str:
        """Extracts risk factors summary from Item 1A."""
        sections = self.locate_sections()
        if not sections.risk_factors_page:
            return ""
        start = sections.risk_factors_page - 1
        end = min(start + max_pages, self.page_count)
        chunks: list[str] = []
        for p in range(start, end):
            chunks.append(self.reader.pages[p].extract_text())
        return "\n\n".join(chunks)
