"""311 service requests via Socrata (SODA API).

Chicago, NYC, SF and a long tail of other cities publish 311 on Socrata with
the same query grammar, so one client covers all of them. An app token is
optional but lifts the anonymous rate limit sharply -- worth the two minutes
it takes to register once you are backfilling a year of reports.

Dataset defaults to Chicago's pothole requests (`v6vf-nfxy`), which is a good
first city: high volume, clean coordinates, and a status field that tells you
when the city actually closed the ticket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shared.config import get_settings
from shared.http import CachedClient
from shared.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Report:
    source: str
    external_id: str
    category: str | None
    description: str | None
    reported_at: datetime | None
    status: str | None
    lat: float | None
    lon: float | None
    raw: dict[str, Any]

    @property
    def has_location(self) -> bool:
        return self.lat is not None and self.lon is not None


class SocrataClient:
    def __init__(
        self, domain: str | None = None, dataset: str | None = None, *, use_cache: bool = True
    ) -> None:
        settings = get_settings()
        self.domain = domain or settings.socrata_domain
        self.dataset = dataset or settings.socrata_dataset

        headers = {"Accept": "application/json"}
        if settings.socrata_app_token:
            headers["X-App-Token"] = settings.socrata_app_token
        else:
            log.warning(
                "socrata.no_app_token",
                hint="anonymous rate limits are low; set SOCRATA_APP_TOKEN before backfilling",
            )

        self._client = CachedClient(
            headers=headers,
            rate_per_sec=4.0,
            cache_namespace=f"socrata-{self.domain}",
            use_cache=use_cache,
        )

    async def __aenter__(self) -> SocrataClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def query(
        self,
        *,
        where: str | None = None,
        order: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        select: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"$limit": limit, "$offset": offset}
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        if select:
            params["$select"] = select

        url = f"https://{self.domain}/resource/{self.dataset}.json"
        return await self._client.get_json(url, params=params)

    async def page_all(
        self, *, where: str | None = None, page_size: int = 1000, max_records: int = 20_000
    ) -> list[dict[str, Any]]:
        """Page through a result set.

        Ordered by `:id` rather than a timestamp on purpose: paging by a
        non-unique sort key silently drops and duplicates rows across page
        boundaries when new records land mid-backfill.
        """
        collected: list[dict[str, Any]] = []
        offset = 0
        while offset < max_records:
            batch = await self.query(
                where=where, order=":id", limit=min(page_size, max_records - offset),
                offset=offset,
            )
            if not batch:
                break
            collected.extend(batch)
            offset += len(batch)
            if len(batch) < page_size:
                break
        log.info("socrata.paged", dataset=self.dataset, records=len(collected))
        return collected

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_chicago_potholes(rows: list[dict[str, Any]]) -> list[Report]:
    """Map Chicago's 311 schema onto the internal Report shape.

    Split out per-city because every Socrata publisher names its columns
    differently; the pipeline downstream of here never sees a city-specific
    field name.
    """
    reports: list[Report] = []
    for row in rows:
        lat, lon = _coordinates(row)
        reports.append(
            Report(
                source="socrata:chicago",
                external_id=str(row.get("sr_number") or row.get(":id") or ""),
                category=row.get("sr_type") or "Pothole in Street",
                description=row.get("street_address"),
                reported_at=_parse_ts(row.get("created_date")),
                status=row.get("status"),
                lat=lat,
                lon=lon,
                raw=row,
            )
        )
    return [r for r in reports if r.external_id]


def _coordinates(row: dict[str, Any]) -> tuple[float | None, float | None]:
    location = row.get("location")
    if isinstance(location, dict) and location.get("coordinates"):
        lon, lat = location["coordinates"][:2]
        return float(lat), float(lon)
    try:
        return float(row["latitude"]), float(row["longitude"])
    except (KeyError, TypeError, ValueError):
        return None, None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
