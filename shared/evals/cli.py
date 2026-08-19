from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from shared.evals.registry import list_suites
from shared.evals.report import render_console, render_markdown
from shared.evals.runner import run_suites, write_baseline

app = typer.Typer(help="Run the eval suites for Ledgerline and Sightline.", no_args_is_help=True)
console = Console()


@app.command("list")
def list_command() -> None:
    """Show every registered suite, its dataset, and its gates."""
    suites = list_suites()
    if not suites:
        console.print("[yellow]no suites registered[/]")
        raise typer.Exit(0)
    for suite in suites:
        console.print(f"[bold]{suite.name}[/]  [dim]{suite.dataset}[/]")
        if suite.description:
            console.print(f"  {suite.description}")
        for gate in suite.gates:
            console.print(f"  [dim]gate:[/] {gate.describe()}")


@app.command("run")
def run_command(
    projects: Annotated[
        list[str] | None,
        typer.Argument(help="Projects to run. Default: all registered."),
    ] = None,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Run just these suite names.")
    ] = None,
    write_baseline_flag: Annotated[
        bool,
        typer.Option("--write-baseline", help="Record this run as the new baseline."),
    ] = False,
    markdown: Annotated[
        Path | None, typer.Option("--markdown", help="Also write a markdown report here.")
    ] = None,
) -> None:
    """Run suites, compare against baselines, and exit non-zero on a gate failure."""
    results = run_suites(projects or None, only or None)
    if not results:
        console.print("[yellow]no suites matched[/]")
        raise typer.Exit(0)

    render_console(results, console)

    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(results), encoding="utf-8")
        console.print(f"[dim]markdown report -> {markdown}[/]")

    if write_baseline_flag:
        for result in results:
            if result.error:
                console.print(f"[yellow]skipping baseline for errored {result.suite}[/]")
                continue
            path = write_baseline(result)
            console.print(f"[dim]baseline -> {path}[/]")
        raise typer.Exit(0)

    raise typer.Exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    app()
