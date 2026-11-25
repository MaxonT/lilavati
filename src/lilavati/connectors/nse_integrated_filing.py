"""
NSE Integrated Filing (Financial), Ind AS: the exchange's page for one filing, read into the Ind AS crosswalk.

SEBI's Integrated Filing (circular of 31 December 2024) is filed in XBRL against SEBI's Integrated Filing
Finance Taxonomy (``in-capmkt``, version 1.0 of 2025-01-31). NSE publishes each filing as a page rendered
from that XBRL (``nsearchives.nseindia.com/corporate/ixbrl/INTEGRATED_FILING_INDAS_<id>_<ddmmyyyyHHMMSS>_iXBRL_WEB.html``,
title "IndAS Details"): general information, then sections that each start with a "Particulars" row and
the start date, end date, audit status and standalone/consolidated flag of every column. The page carries
labels, not element names, so each crosswalk item is matched to its label (some labels are NSE's own,
including the misspelt "Total equity and liabilites").

This module reads a page the caller supplies; it fetches nothing (NSE's terms of use prohibit automated
collection without written consent). Values stay in the filing's rounding ("Lakhs" = 100,000 rupees),
which becomes the disclosure's ``scale``. Nothing is filled: an item the page does not show is listed in
``missing``. Two items are sums of shown lines (recorded in ``derived``): trade receivables (current +
non-current) and trade payables (current + non-current totals).
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Literal

from pydantic import BaseModel, Field

from lilavati.connectors.india_filings import IndASCompanyDisclosure
from lilavati.core.provenance import CitationRef, SourceStore

SectionKind = Literal["profit_and_loss", "balance_sheet", "cash_flow", "other"]

ROUNDING_SCALE = {"lakhs": 100_000, "crores": 10_000_000, "millions": 1_000_000, "thousands": 1_000,
                  "actual": 1, "rupees": 1}
# A section is identified by a line that only it has (the segment section also has "Revenue from operations").
SECTION_ANCHORS: dict[SectionKind, str] = {
    "profit_and_loss": "total expenses",
    "balance_sheet": "total assets",
    "cash_flow": "net cash flows from (used in) operating activities",
}
COLUMN_META = {"date of start of reporting period": "start", "date of end of reporting period": "end",
               "whether results are audited or unaudited": "audit_status",
               "nature of report standalone or consolidated": "nature"}

# crosswalk key -> (section, accepted labels)
LABELS: dict[str, tuple[SectionKind, tuple[str, ...]]] = {
    "RevenueFromOperations": ("profit_and_loss", ("Revenue from operations",)),
    "OtherIncome": ("profit_and_loss", ("Other income",)),
    "CostOfMaterialsConsumed": ("profit_and_loss", ("Cost of materials consumed",)),
    "PurchasesOfStockInTrade": ("profit_and_loss", ("Purchases of stock-in-trade",)),
    "ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade": (
        "profit_and_loss", ("Changes in inventories of finished goods, work-in-progress and stock-in-trade",)),
    "EmployeeBenefitsExpense": ("profit_and_loss", ("Employee benefit expense",)),
    "FinanceCosts": ("profit_and_loss", ("Finance costs",)),
    "DepreciationAndAmortisationExpense": ("profit_and_loss", ("Depreciation, depletion and amortisation expense",)),
    "TotalExpenses": ("profit_and_loss", ("Total expenses",)),
    "ProfitBeforeTax": ("profit_and_loss", ("Total profit before tax",)),
    "CurrentTax": ("profit_and_loss", ("Current tax",)),
    "DeferredTax": ("profit_and_loss", ("Deferred tax",)),
    "TaxExpense": ("profit_and_loss", ("Total tax expenses",)),
    "ProfitForThePeriod": ("profit_and_loss", ("Total profit (loss) for period",)),
    "PropertyPlantAndEquipment": ("balance_sheet", ("Property, plant and equipment",)),
    "Inventories": ("balance_sheet", ("Inventories",)),
    "CashAndCashEquivalents": ("balance_sheet", ("Cash and cash equivalents",)),
    "TotalAssets": ("balance_sheet", ("Total assets",)),
    "EquityShareCapital": ("balance_sheet", ("Equity share capital",)),
    "OtherEquity": ("balance_sheet", ("Other equity",)),
    "TotalEquity": ("balance_sheet", ("Total equity",)),
    "TotalLiabilities": ("balance_sheet", ("Total liabilities",)),
    "TotalEquityAndLiabilities": ("balance_sheet", ("Total equity and liabilites", "Total equity and liabilities")),
    "NonCurrentBorrowings": ("balance_sheet", ("Borrowings, non-current",)),
    "CurrentBorrowings": ("balance_sheet", ("Borrowings, current",)),
    "NetCashFromOperatingActivities": ("cash_flow", ("Net cash flows from (used in) operating activities",)),
    "NetCashFromInvestingActivities": ("cash_flow", ("Net cash flows from (used in) investing activities",)),
    "NetCashFromFinancingActivities": ("cash_flow", ("Net cash flows from (used in) financing activities",)),
    "EffectOfExchangeRateChangesOnCash": ("cash_flow", ("Effect of exchange rate changes on cash and cash equivalents",)),
    "CashAndCashEquivalentsAtBeginningOfPeriod": (
        "cash_flow", ("Cash and cash equivalents cash flow statement at beginning of period",)),
    "CashAndCashEquivalentsAtEndOfPeriod": ("cash_flow", ("Cash and cash equivalents cash flow statement at end of period",)),
}
# crosswalk key -> parts (section, label, heading the row must sit under, or None)
SUMS: dict[str, list[tuple[SectionKind, str, str | None]]] = {
    "TradeReceivables": [("balance_sheet", "Trade receivables, current", None),
                         ("balance_sheet", "Trade receivables, non-current", None)],
    "TradePayables": [("balance_sheet", "Total Trade payable", "Trade payables, current"),
                      ("balance_sheet", "Total Trade payable", "Trade payables, non-current")],
}

_NUMBER = re.compile(r"\(?-?[\d,]+(?:\.\d+)?\)?")
_INDEX = re.compile(r"\(?[A-Za-z]\)?|\(?[ivx]{1,4}\)?|\d{1,3}")
_FILED_AT = re.compile(r"_(\d{14})_")


def _norm(label: str) -> str:
    return " ".join(label.lower().split())


def _number(cell: str) -> float | None:
    s = cell.strip()
    if not _NUMBER.fullmatch(s):
        return None
    negative = s.startswith("(") and s.endswith(")")
    v = float(s.strip("()").replace(",", ""))
    return -v if negative else v


def _date(cell: str) -> date | None:
    try:
        return datetime.strptime(cell.strip(), "%d-%m-%Y").date()
    except ValueError:
        return None


class _Rows(HTMLParser):
    """Table rows as lists of cell texts; script and style content ignored."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if not self._skip and self._cell is not None:
            self._cell.append(data)


class FilingColumn(BaseModel):
    start: date | None = None
    end: date | None = None
    audit_status: str | None = Field(default=None, description="the column's \"Whether results are audited or unaudited\"")
    nature: str | None = None


class FilingRow(BaseModel):
    label: str
    heading: str = Field(default="", description="the nearest preceding row without values")
    values: list[float | None]
    text: str = Field(..., description="the row as shown on the page (cells joined by spaces)")


class FilingSection(BaseModel):
    kind: SectionKind
    columns: list[FilingColumn]
    rows: list[FilingRow]


class NseIntegratedFiling(BaseModel):
    doc_hash: str = Field(..., description="SHA-256 of the page bytes as supplied")
    company_name: str
    symbol: str
    isin: str
    scrip_code: str | None = None
    nature: str = Field(..., description="Consolidated or Standalone")
    rounding: str
    scale: int = Field(..., description="rupees per unit of the values shown (Lakhs: 100,000)")
    currency: str
    financial_year_start: date
    financial_year_end: date
    board_meeting_date: date | None = None
    filed_at: datetime | None = Field(default=None, description="submission time, from NSE's file name when given")
    general: dict[str, str]
    sections: list[FilingSection]
    page_text: str


class ExchangeDisclosure(BaseModel):
    disclosure: IndASCompanyDisclosure
    period: Literal["year", "quarter"]
    period_start: date
    period_end: date
    sources: dict[str, list[str]] = Field(..., description="crosswalk key -> the page rows it was read from")
    derived: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list, description="crosswalk keys the page does not show")


def read_nse_integrated_filing(content: bytes, source_name: str | None = None) -> NseIntegratedFiling:
    """Parse NSE's page for an Integrated Filing (Financial), Ind AS.

    ``source_name`` (optional): NSE's file name or URL; its ``ddmmyyyyHHMMSS`` stamp gives ``filed_at``.
    Raises ``ValueError`` if the page lacks the general information or the rounding is unknown.
    """
    parser = _Rows()
    parser.feed(content.decode("utf-8", errors="strict"))
    rows = [r for r in parser.rows if any(c for c in r)]
    general: dict[str, str] = {}
    sections: list[tuple[list[FilingColumn], list[FilingRow]]] = []
    columns: list[FilingColumn] | None = None
    body: list[FilingRow] = []
    heading = ""
    for cells in rows:
        if cells[0] == "Particulars":
            if columns is not None:
                sections.append((columns, body))
            columns, body, heading = [FilingColumn() for _ in cells[1:]], [], ""
            continue
        if columns is None:
            if len(cells) >= 2 and cells[0] not in general:
                general[cells[0]] = cells[1]
            continue
        k = len(columns)
        meta = [c for c in cells if _norm(c) in COLUMN_META]
        if meta:
            field = COLUMN_META[_norm(meta[0])]
            vals = cells[cells.index(meta[0]) + 1:][:k]
            for col, v in zip(columns, vals, strict=False):
                setattr(col, field, _date(v) if field in ("start", "end") else v)
            continue
        i = 0
        while i < len(cells):
            c = cells[i]
            if not c or _number(c) is not None or c == "null" or _INDEX.fullmatch(c):
                i += 1
                continue
            raw = cells[i + 1: i + 1 + k]
            values = [_number(v) for v in raw] + [None] * (k - len(raw))
            row = FilingRow(label=c, heading=heading, values=values, text=" ".join([c, *raw]).strip())
            body.append(row)
            if all(v is None for v in values):
                heading = c
            i += 1 + len(raw)
    if columns is not None:
        sections.append((columns, body))

    def need(key: str) -> str:
        if key not in general:
            raise ValueError(f"the page has no {key!r}; is it NSE's Integrated Filing (Financial) page?")
        return general[key]

    rounding = need("Level of rounding used in financial results")
    if rounding.lower() not in ROUNDING_SCALE:
        raise ValueError(f"unknown rounding {rounding!r}; known: {sorted(ROUNDING_SCALE)}")
    fy_start, fy_end = _date(need("Date of start of financial year")), _date(need("Date of end of financial year"))
    if fy_start is None or fy_end is None:
        raise ValueError("financial year dates are not dd-mm-yyyy.")
    filed_at = None
    if source_name and (m := _FILED_AT.search(source_name)):
        filed_at = datetime.strptime(m.group(1), "%d%m%Y%H%M%S")
    out_sections = []
    for cols, body_rows in sections:
        labels = {_norm(r.label) for r in body_rows}
        kind: SectionKind = next((k for k, a in SECTION_ANCHORS.items() if a in labels), "other")
        out_sections.append(FilingSection(kind=kind, columns=cols, rows=body_rows))
    return NseIntegratedFiling(
        doc_hash=hashlib.sha256(content).hexdigest(), company_name=need("Name of company"),
        symbol=need("NSE Symbol"), isin=need("ISIN"), scrip_code=general.get("Scrip Code"),
        nature=need("Nature of report standalone or consolidated"), rounding=rounding,
        scale=ROUNDING_SCALE[rounding.lower()], currency=need("Description of presentation currency"),
        financial_year_start=fy_start, financial_year_end=fy_end,
        board_meeting_date=_date(general.get("Date of board meeting when results were approved", "")),
        filed_at=filed_at, general=general, sections=out_sections,
        page_text="\n".join(" ".join(r) for r in rows))


def _section(filing: NseIntegratedFiling, kind: SectionKind) -> FilingSection | None:
    found = [s for s in filing.sections if s.kind == kind]
    if len(found) > 1:
        raise ValueError(f"the page has {len(found)} {kind} sections.")
    return found[0] if found else None


def _column(section: FilingSection, filing: NseIntegratedFiling, period: str) -> int | None:
    fy_s, fy_e = filing.financial_year_start, filing.financial_year_end
    for j, c in enumerate(section.columns):
        if c.nature and c.nature != filing.nature:
            raise ValueError(f"column {j} is {c.nature}, the filing is {filing.nature}.")
        if section.kind == "balance_sheet":
            if c.end == fy_e and period == "year":
                return j
        elif c.end == fy_e and ((period == "year" and c.start == fy_s)
                                or (period == "quarter" and c.start != fy_s)):
            return j
    return None


def _match(section: FilingSection, labels: tuple[str, ...], heading: str | None) -> list[FilingRow]:
    want = {_norm(lab) for lab in labels}
    return [r for r in section.rows if _norm(r.label) in want
            and (heading is None or _norm(r.heading) == _norm(heading))]


def to_ind_as_disclosure(filing: NseIntegratedFiling, cin: str, filing_date: date | None = None,
                         period: Literal["year", "quarter"] = "year") -> ExchangeDisclosure:
    """The filing's figures as an ``IndASCompanyDisclosure`` (values in the filing's rounding, ``scale`` set).

    ``period="year"``: the financial year to date (balance sheet at its end; the cash flow statement if
    shown). ``"quarter"``: the quarter's income statement only. ``cin`` is not on the page. ``filing_date``
    defaults to the date of ``filed_at``. A label found more than once where one is expected raises.
    """
    if filing_date is None:
        if filing.filed_at is None:
            raise ValueError("filing_date is required when the page's file name gave no submission time.")
        filing_date = filing.filed_at.date()
    facts: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    derived: dict[str, str] = {}
    missing: list[str] = []
    period_start = filing.financial_year_start
    for key, (kind, labels) in LABELS.items():
        sec = _section(filing, kind)
        j = _column(sec, filing, period) if sec else None
        if sec is None or j is None:
            missing.append(key)
            continue
        if kind == "profit_and_loss" and sec.columns[j].start:
            period_start = sec.columns[j].start  # type: ignore[assignment]
        rows = [r for r in _match(sec, labels, None) if r.values[j] is not None]
        if len(rows) > 1:
            raise ValueError(f"{key}: label {labels[0]!r} has values on {len(rows)} rows.")
        if not rows:
            missing.append(key)
            continue
        facts[key], sources[key] = float(rows[0].values[j]), [rows[0].text]  # type: ignore[arg-type]
    for key, parts in SUMS.items():
        total, used, shown = 0.0, [], []
        for kind, label, heading in parts:
            sec = _section(filing, kind)
            j = _column(sec, filing, period) if sec else None
            if sec is None or j is None:
                continue
            rows = [r for r in _match(sec, (label,), heading) if r.values[j] is not None]
            if len(rows) > 1:
                raise ValueError(f"{key}: {label!r} under {heading!r} has values on {len(rows)} rows.")
            if rows:
                total += float(rows[0].values[j])  # type: ignore[arg-type]
                used.append(rows[0].text)
                shown.append(f"{label}" + (f" ({heading})" if heading else ""))
        if used:
            facts[key], sources[key] = total, used
            derived[key] = " + ".join(shown)
        else:
            missing.append(key)
    disclosure = IndASCompanyDisclosure(
        company_name=filing.company_name, symbol=filing.symbol, cin=cin,
        fiscal_year=filing.financial_year_end.year, reporting_date=filing.financial_year_end,
        filing_date=filing_date, currency=filing.currency, scale=filing.scale, facts=facts)
    return ExchangeDisclosure(disclosure=disclosure, period=period, period_start=period_start,
                              period_end=filing.financial_year_end, sources=sources, derived=derived,
                              missing=missing)


def register_filing(store: SourceStore, filing: NseIntegratedFiling, content: bytes) -> str:
    """Register the page in ``store`` (hash of its bytes, its text as one page); returns the hash."""
    h = store.add_document(content, [filing.page_text])
    if h != filing.doc_hash:
        raise ValueError("content is not the page this filing was read from.")
    return h


def citation(filing: NseIntegratedFiling, result: ExchangeDisclosure, key: str) -> list[CitationRef]:
    """Citations for one crosswalk item: the page rows it was read from."""
    return [CitationRef(doc_hash=filing.doc_hash, filing_type="NSE Integrated Filing (Financial)",
                        filing_date=result.disclosure.filing_date, period_end=result.period_end, page=None,
                        raw_snippet=text) for text in result.sources[key]]
