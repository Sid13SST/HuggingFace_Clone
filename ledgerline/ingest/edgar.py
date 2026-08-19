"""SEC EDGAR client.

No API key exists for EDGAR. What it does require is a descriptive User-Agent
carrying a contact address, and staying under 10 requests/second -- ignore
either and the SEC blocks the IP, not the request. Both are enforced here
rather than left to the caller.

Endpoints used (all public, all JSON):
  https://www.sec.gov/files/company_tickers.json      ticker -> CIK
  https://data.sec.gov/submissions/CIK##########.json filing index per issuer
  https://data.sec.gov/api/xbrl/companyfacts/...      XBRL facts per issuer
  https://www.sec.gov/Archives/edgar/data/...         the documents themselves
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from shared.config import get_settings
from shared.http import CachedClient
from shared.logging import get_logger

log = get_logger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_bare}/{document}"


class MissingUserAgent(RuntimeError):
    pass


@dataclass(frozen=True)
class Filing:
    cik: str
    accession: str
    form: str
    filed_at: date
    period_end: date | None
    primary_document: str
    description: str
    items: list[str]

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik_int=int(self.cik),
            accession_bare=self.accession.replace("-", ""),
            document=self.primary_document,
        )


def normalize_cik(cik: str | int) -> str:
    """EDGAR wants a zero-padded 10-digit CIK in URLs and a bare int in paths."""
    digits = str(cik).strip().upper().removeprefix("CIK").lstrip("0")
    if not digits.isdigit():
        raise ValueError(f"not a CIK: {cik!r}")
    return digits.zfill(10)


class EdgarClient:
    def __init__(self, *, use_cache: bool = True) -> None:
        settings = get_settings()
        if not settings.edgar_user_agent.strip():
            raise MissingUserAgent(
                "EDGAR_USER_AGENT is required and must include a contact address, "
                "e.g. 'Jane Doe jane@example.com'. The SEC blocks clients without one."
            )
        self._client = CachedClient(
            headers={
                "User-Agent": settings.edgar_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            rate_per_sec=settings.edgar_rate_limit_rps,
            cache_namespace="edgar",
            use_cache=use_cache,
        )

    async def __aenter__(self) -> EdgarClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def ticker_to_cik(self, ticker: str) -> str:
        payload = await self._client.get_json(TICKERS_URL)
        wanted = ticker.strip().upper()
        for entry in payload.values():
            if str(entry.get("ticker", "")).upper() == wanted:
                return normalize_cik(entry["cik_str"])
        raise KeyError(f"no CIK for ticker {ticker!r}")

    async def submissions(self, cik: str | int) -> dict[str, Any]:
        return await self._client.get_json(SUBMISSIONS_URL.format(cik=normalize_cik(cik)))

    async def company_facts(self, cik: str | int) -> dict[str, Any]:
        """XBRL facts.

        Worth pulling even though the pipeline parses tables directly: XBRL is
        an independent source of truth for headline figures, which makes it a
        free consistency check on the table extractor.
        """
        return await self._client.get_json(COMPANYFACTS_URL.format(cik=normalize_cik(cik)))

    async def recent_filings(
        self,
        cik: str | int,
        forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
        limit: int = 25,
    ) -> list[Filing]:
        payload = await self.submissions(cik)
        return parse_recent_filings(payload, forms=forms, limit=limit)

    async def fetch_document(self, filing: Filing) -> bytes:
        return await self._client.get_bytes(filing.url)

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_recent_filings(
    payload: dict[str, Any],
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    limit: int = 25,
) -> list[Filing]:
    """Turn the submissions blob into Filing rows.

    The `recent` block is column-oriented -- parallel arrays, not a list of
    records -- so it has to be transposed. Split out from the client so it can
    be unit-tested against a fixture instead of the live SEC.
    """
    cik = normalize_cik(payload["cik"])
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return []

    columns = (
        "accessionNumber",
        "form",
        "filingDate",
        "reportDate",
        "primaryDocument",
        "primaryDocDescription",
        "items",
    )
    rows = zip(*(recent.get(name, []) for name in columns), strict=False)

    wanted = {f.upper() for f in forms}
    filings: list[Filing] = []
    for accession, form, filed, period, document, description, items in rows:
        if wanted and str(form).upper() not in wanted:
            continue
        filings.append(
            Filing(
                cik=cik,
                accession=accession,
                form=form,
                filed_at=_parse_date(filed) or date.min,
                period_end=_parse_date(period),
                primary_document=document or "",
                description=description or "",
                # 8-K items ("2.02", "7.01") are the cheapest possible
                # zero-shot label for what a filing is actually about.
                items=[i.strip() for i in str(items or "").split(",") if i.strip()],
            )
        )
        if len(filings) >= limit:
            break
    return filings


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
