from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sightline.dedupe import ReportPoint, cluster, pair_score
from sightline.ingest.socrata import SocrataClient, parse_chicago_potholes
from sightline.severity import CameraIntrinsics, estimate

app = typer.Typer(help="Sightline: infrastructure triage.", no_args_is_help=True)
console = Console()


@app.command()
def reports(
    limit: Annotated[int, typer.Option(help="Max records to pull.")] = 25,
    where: Annotated[
        str | None, typer.Option(help="SoQL $where clause, e.g. \"status='Open'\"")
    ] = None,
) -> None:
    """Pull recent 311 reports from Socrata. Responses are cached on disk."""
    rows = asyncio.run(_reports(limit, where))
    table = Table(title=f"311 reports ({len(rows)})", header_style="dim")
    for column in ("id", "category", "reported", "status", "lat", "lon"):
        table.add_column(column)
    for report in rows:
        table.add_row(
            report.external_id,
            (report.category or "--")[:28],
            report.reported_at.date().isoformat() if report.reported_at else "--",
            report.status or "--",
            f"{report.lat:.5f}" if report.lat else "--",
            f"{report.lon:.5f}" if report.lon else "--",
        )
    console.print(table)

    missing = sum(1 for r in rows if not r.has_location)
    if missing:
        console.print(
            f"[yellow]{missing} of {len(rows)} reports have no coordinates[/] "
            "-- these cannot be spatially deduped and go to the text-only path."
        )


async def _reports(limit: int, where: str | None):
    async with SocrataClient() as client:
        raw = await client.query(where=where, order=":id", limit=limit)
    return parse_chicago_potholes(raw)


@app.command()
def score(
    lat1: float, lon1: float, text1: str, lat2: float, lon2: float, text2: str
) -> None:
    """Score one candidate duplicate pair, showing the component contributions."""
    a = ReportPoint(id="a", lat=lat1, lon=lon1, text=text1)
    b = ReportPoint(id="b", lat=lat2, lon=lon2, text=text2)

    from sightline.dedupe import DEFAULT_CONFIG, haversine_m, text_similarity

    distance = haversine_m(lat1, lon1, lat2, lon2)
    combined = pair_score(a, b)
    console.print(f"distance      {distance:8.1f} m")
    console.print(f"text overlap  {text_similarity(text1, text2):8.3f}")
    console.print(f"combined      {combined:8.3f}  (threshold {DEFAULT_CONFIG.threshold})")
    console.print(
        "[green]duplicate[/]" if combined >= DEFAULT_CONFIG.threshold else "[yellow]distinct[/]"
    )


@app.command("severity")
def severity_command(
    mask_area_px: Annotated[float, typer.Option(help="Segmentation mask area in pixels.")],
    depth_m: Annotated[float, typer.Option(help="Estimated depth to the defect.")],
    detector_score: Annotated[float, typer.Option(help="Detector confidence.")] = 0.9,
    width: int = 1024,
    height: int = 768,
) -> None:
    """Turn a mask and a depth into a severity band, or an abstention."""
    result = estimate(
        mask_area_px=mask_area_px,
        depth_m=depth_m,
        intrinsics=CameraIntrinsics.from_fov(width, height),
        detector_score=detector_score,
    )
    console.print(f"extent   {result.extent_cm:6.1f} cm")
    console.print(f"area     {result.area_m2:6.3f} m2")
    console.print(f"severity {result.severity}  (clause {result.standard_clause})")
    if result.confident:
        console.print(f"sla      {result.sla_days} days")
    else:
        console.print(f"[yellow]abstained[/] -- {result.note}")


@app.command("cluster")
def cluster_command() -> None:
    """Cluster the dedupe fixture set and show the resulting defect groups."""
    import json

    from shared.config import REPO_ROOT

    path = REPO_ROOT / "sightline" / "evals" / "datasets" / "dedupe_pairs.jsonl"
    points: dict[str, ReportPoint] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            example = json.loads(stripped)
            for side in ("a", "b"):
                raw = example["inputs"][side]
                points[raw["id"]] = ReportPoint(
                    id=raw["id"],
                    lat=float(raw["lat"]),
                    lon=float(raw["lon"]),
                    text=raw.get("text", ""),
                    category=raw.get("category"),
                    segment_id=raw.get("segment_id"),
                )

    groups = cluster(list(points.values()))
    merged = [g for g in groups if len(g) > 1]
    console.print(
        f"{len(points)} reports -> {len(groups)} defects "
        f"({len(merged)} clusters with more than one report)"
    )
    for group in merged:
        console.print(f"  {' + '.join(group)}")


if __name__ == "__main__":
    app()
