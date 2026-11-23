"""
Cryptographic Provenance Engine for Financial Disclosures.

Ensures that every quantitative claim extracted from regulatory filings (10-K, 10-Q, Ind AS)
carries an immutable, verifiable reference back to the source document, page, table, and character span.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CitationRef(BaseModel):
    """Cryptographic provenance record linking a extracted financial value to its primary source."""

    model_config = ConfigDict(frozen=True)

    doc_hash: str = Field(
        ...,
        description=(
            "SHA-256 of the source: the document bytes for a PDF or HTML filing; for an XBRL fact, the "
            "fact record as served by SEC (canonical JSON), so anyone can re-fetch and check it"
        ),
        min_length=64,
        max_length=64,
    )
    filing_type: str = Field(
        ...,
        description="Type of regulatory filing (e.g., '10-K', '10-Q', 'Ind-AS-Annual', '8-K')",
    )
    filing_date: date = Field(
        ...,
        description="Official publication/filing date of the disclosure",
    )
    period_end: date = Field(
        ...,
        description="Fiscal period end date described by this report (Valid Time)",
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description="1-based page number where the claim appears; None for sources without pages (XBRL facts)",
    )
    table_id: str | None = Field(
        default=None,
        description="Unique identifier or index of the table within the document",
    )
    xbrl_tag: str | None = Field(
        default=None,
        description="Canonical or extension XBRL taxonomy tag (e.g., 'us-gaap:Revenues')",
    )
    raw_snippet: str = Field(
        ...,
        description="Verbatim text, cell excerpt, or numerical token from the original disclosure",
    )

    @classmethod
    def create_from_content(
        cls,
        raw_document_bytes: bytes,
        filing_type: str,
        filing_date: date,
        period_end: date,
        page: int | None,
        raw_snippet: str,
        table_id: str | None = None,
        xbrl_tag: str | None = None,
    ) -> CitationRef:
        """Helper to compute SHA-256 hash automatically from raw document bytes."""
        hasher = hashlib.sha256()
        hasher.update(raw_document_bytes)
        doc_hash = hasher.hexdigest()

        return cls(
            doc_hash=doc_hash,
            filing_type=filing_type,
            filing_date=filing_date,
            period_end=period_end,
            page=page,
            table_id=table_id,
            xbrl_tag=xbrl_tag,
            raw_snippet=raw_snippet.strip(),
        )


class SourcedValue(BaseModel):
    """A financial numerical value strictly bound to its cryptographic CitationRef."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(..., description="The numerical magnitude")
    unit: str = Field(
        default="USD", description="Currency or unit (e.g., 'USD', 'INR', 'ratio', 'bps')"
    )
    scale: int = Field(
        default=1, description="Multiplier (e.g., 1_000_000 for figures reported in millions)"
    )
    citation: CitationRef = Field(..., description="Mandatory cryptographic citation")

    @property
    def normalized_value(self) -> float:
        """Returns the value scaled to base units (e.g., millions * 1e6)."""
        return self.value * self.scale


def record_hash(record: dict[str, Any]) -> str:
    """SHA-256 of a structured record as canonical JSON (sorted keys, no spaces).

    Used for XBRL fact records (``sec_filing.fact_hash``): re-fetch the record and hash it the same way.
    """
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.split())


class SourceStore:
    """The sources a pipeline read, keyed by SHA-256, so that citations can be resolved against them.

    Register what was actually read, then ``check`` a citation (and optionally its value):

    - ``add_document(content, pages)``: a PDF or HTML filing; the hash is of the bytes, ``pages`` the
      text of each page (one entry for a document without pages). A citation must point to a page that
      exists, and its snippet (whitespace collapsed) must appear on that page (anywhere when ``page`` is
      None).
    - ``add_xbrl_company_facts(facts)``: SEC company facts; every fact record is registered by its
      ``record_hash``. A citation must match the record's concept (``xbrl_tag``), period end and filing
      date, and a value must equal the record's. The record carries no concept name, so concepts
      reported with identical records share a hash; the citation's tag tells them apart.
    - ``add_tagged_values(doc_hash, values)``: a structured disclosure whose hash the caller computed
      (e.g. ``india_filings.register_disclosure``); a citation's ``xbrl_tag`` must be one of the tags and
      a value must equal it.

    Values are compared in base units (``SourcedValue.normalized_value``) with a relative tolerance of
    1e-12 (float rounding only: a revenue of 478,834,000,000 one dollar off fails). A store holds only what it was given: a citation to a source it never saw fails.
    """

    def __init__(self) -> None:
        self._pages: dict[str, list[str]] = {}
        # hash -> [(taxonomy:concept, record)]: a record has no concept name, so two concepts reported
        # with the same value, period and accession (Apple's cash with and without restricted cash) share a hash
        self._facts: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._tagged: dict[str, dict[str, float]] = {}

    def add_document(self, content: bytes, pages: list[str]) -> str:
        doc_hash = hashlib.sha256(content).hexdigest()
        self._pages[doc_hash] = [_norm(p) for p in pages]
        return doc_hash

    def add_xbrl_company_facts(self, facts: dict[str, Any]) -> int:
        n = 0
        for taxonomy, concepts in facts.get("facts", {}).items():
            for concept, data in concepts.items():
                for entries in data.get("units", {}).values():
                    for e in entries:
                        self._facts.setdefault(record_hash(e), []).append((f"{taxonomy}:{concept}", e))
                        n += 1
        return n

    def add_tagged_values(self, doc_hash: str, values: dict[str, float]) -> None:
        self._tagged.setdefault(doc_hash, {}).update(values)

    def __contains__(self, doc_hash: object) -> bool:
        return doc_hash in self._pages or doc_hash in self._facts or doc_hash in self._tagged

    def check(self, citation: CitationRef, value: float | None = None) -> list[str]:
        """Problems resolving ``citation`` (empty when it resolves); ``value`` is in base units."""
        h = citation.doc_hash
        short = f"{h[:12]}..."

        def same(a: float, b: float) -> bool:
            return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))  # float rounding only

        if h in self._facts:
            entries = self._facts[h]
            tags = [t for t, _ in entries]
            match = [r for t, r in entries if t == citation.xbrl_tag]
            rec = match[0] if match else entries[0][1]
            out = []
            if citation.xbrl_tag is not None and not match:
                out.append(f"{short}: cited tag {citation.xbrl_tag}, the record is {' or '.join(tags)}")
            if str(rec.get("end")) != citation.period_end.isoformat():
                out.append(f"{short}: cited period end {citation.period_end}, the record's is {rec.get('end')}")
            if str(rec.get("filed")) != citation.filing_date.isoformat():
                out.append(f"{short}: cited filing date {citation.filing_date}, the record's is {rec.get('filed')}")
            if value is not None and not same(value, float(rec["val"])):
                out.append(f"{short}: value {value:,.2f} differs from the record's {float(rec['val']):,.2f}")
            return out
        if h in self._tagged:
            vals = self._tagged[h]
            if citation.xbrl_tag not in vals:
                return [f"{short}: tag {citation.xbrl_tag} is not in the registered disclosure"]
            if value is not None and not same(value, vals[citation.xbrl_tag]):
                return [f"{short}: value {value:,.2f} differs from the disclosure's {vals[citation.xbrl_tag]:,.2f}"]
            return []
        if h in self._pages:
            pages = self._pages[h]
            snippet = _norm(citation.raw_snippet)
            if citation.page is not None:
                if citation.page > len(pages):
                    return [f"{short}: page {citation.page} is beyond the document's {len(pages)} pages"]
                if snippet not in pages[citation.page - 1]:
                    return [f"{short}: snippet not found on page {citation.page}"]
                return []
            return [] if any(snippet in p for p in pages) else [f"{short}: snippet not found in the document"]
        return [f"{short}: source is not in the source store"]
