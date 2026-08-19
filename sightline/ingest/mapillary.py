"""Mapillary street-level imagery.

Free, permissively licensed, and covers most cities well enough to find a
frame near an arbitrary 311 report. Needs a client token from
https://www.mapillary.com/dashboard/developers.

The v4 Graph API takes a bbox and returns image metadata; the pixels come from
a separate signed CDN URL requested per-image via the `thumb_*_url` fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from shared.config import get_settings
from shared.http import CachedClient
from shared.logging import get_logger

log = get_logger(__name__)

GRAPH_URL = "https://graph.mapillary.com/images"

DEFAULT_FIELDS = (
    "id",
    "captured_at",
    "compass_angle",
    "geometry",
    "width",
    "height",
    "thumb_1024_url",
    "is_pano",
)


class MissingMapillaryToken(RuntimeError):
    pass


@dataclass(frozen=True)
class StreetImage:
    external_id: str
    lat: float
    lon: float
    captured_at: datetime | None
    compass_angle: float | None
    width: int | None
    height: int | None
    thumb_url: str | None
    is_pano: bool = False


class MapillaryClient:
    def __init__(self, *, use_cache: bool = True) -> None:
        token = get_settings().mapillary_token.strip()
        if not token:
            raise MissingMapillaryToken(
                "MAPILLARY_TOKEN is required. Create one at "
                "https://www.mapillary.com/dashboard/developers"
            )
        self._client = CachedClient(
            headers={"Authorization": f"OAuth {token}"},
            rate_per_sec=8.0,
            cache_namespace="mapillary",
            use_cache=use_cache,
        )

    async def __aenter__(self) -> MapillaryClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def images_near(
        self,
        lat: float,
        lon: float,
        *,
        radius_m: float = 40.0,
        limit: int = 20,
    ) -> list[StreetImage]:
        """Frames within a rough box around a point.

        Mapillary takes a bbox, not a radius, so the caller's radius becomes a
        square and the corners are filtered out downstream by actual distance.
        Slightly over-fetching here is cheaper than missing the one frame that
        happens to face the defect.
        """
        payload = await self._client.get_json(
            GRAPH_URL,
            params={
                "fields": ",".join(DEFAULT_FIELDS),
                "bbox": ",".join(str(v) for v in bbox_around(lat, lon, radius_m)),
                "limit": limit,
            },
        )
        return [parse_image(item) for item in payload.get("data", [])]

    async def fetch_thumb(self, image: StreetImage) -> bytes:
        if not image.thumb_url:
            raise ValueError(f"image {image.external_id} has no thumb url")
        return await self._client.get_bytes(image.thumb_url)

    async def aclose(self) -> None:
        await self._client.aclose()


def bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) for a radius in metres.

    Flat-earth approximation, which is fine at the tens-of-metres scale this
    is used at and avoids a geodesy dependency for a bounding box that is
    filtered properly later.
    """
    import math

    lat_delta = radius_m / 111_320.0
    lon_delta = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)


def parse_image(item: dict[str, Any]) -> StreetImage:
    coordinates = (item.get("geometry") or {}).get("coordinates") or [None, None]
    captured = item.get("captured_at")
    return StreetImage(
        external_id=str(item["id"]),
        lon=float(coordinates[0]) if coordinates[0] is not None else 0.0,
        lat=float(coordinates[1]) if coordinates[1] is not None else 0.0,
        # captured_at arrives as epoch milliseconds, not seconds.
        captured_at=(
            datetime.fromtimestamp(int(captured) / 1000, tz=UTC) if captured else None
        ),
        compass_angle=_maybe_float(item.get("compass_angle")),
        width=_maybe_int(item.get("width")),
        height=_maybe_int(item.get("height")),
        thumb_url=item.get("thumb_1024_url"),
        is_pano=bool(item.get("is_pano", False)),
    )


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
