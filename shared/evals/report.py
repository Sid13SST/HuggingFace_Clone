from __future__ import annotations

from rich.console import Console
from rich.table import Table

from shared.evals.runner import RunResult


def render_console(results: list[RunResult], console: Console | None = None) -> None:
    console = console or Console()

    for result in results:
        title = f"{result.suite}  ({result.n_examples} examples, {result.seconds:.1f}s)"
        if result.error:
            console.print(f"[bold red]{title}[/]\n  {result.error}")
            continue

        table = Table(title=title, title_justify="left", header_style="dim")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_column("baseline", justify="right")
        table.add_column("delta", justify="right")

        deltas = result.deltas()
        for key in sorted(result.metrics):
            delta = deltas[key]
            table.add_row(
                key,
                f"{result.metrics[key]:.4f}",
                f"{result.baseline[key]:.4f}" if key in result.baseline else "--",
                _delta_cell(delta),
            )
        console.print(table)

        for gate in result.gates:
            mark = "[green]PASS[/]" if gate.passed else "[bold red]FAIL[/]"
            console.print(f"  {mark}  {gate.gate.describe()} -- {gate.reason}")
        console.print()

    failed = [r for r in results if not r.passed]
    if failed:
        console.print(
            f"[bold red]{len(failed)} of {len(results)} suites failed:[/] "
            + ", ".join(r.suite for r in failed)
        )
    else:
        console.print(f"[bold green]all {len(results)} suites passed[/]")


def render_markdown(results: list[RunResult]) -> str:
    """Markdown for the CI comment and for the top of the README.

    The README leading with this table rather than an install guide is the
    whole point of building the harness first.
    """
    lines: list[str] = ["# Eval report", ""]

    verdict = "PASS" if all(r.passed for r in results) else "FAIL"
    lines += [f"**{verdict}** -- {len(results)} suite(s)", ""]

    for result in results:
        lines.append(f"## {result.suite}")
        if result.error:
            lines += ["", f"> errored: `{result.error}`", ""]
            continue

        lines += [
            "",
            f"{result.n_examples} examples, {result.seconds:.1f}s",
            "",
            "| metric | value | baseline | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
        deltas = result.deltas()
        for key in sorted(result.metrics):
            baseline = f"{result.baseline[key]:.4f}" if key in result.baseline else "--"
            lines.append(
                f"| {key} | {result.metrics[key]:.4f} | {baseline} | "
                f"{_delta_text(deltas[key])} |"
            )

        if result.gates:
            lines += ["", "| gate | verdict | note |", "| --- | --- | --- |"]
            for gate in result.gates:
                mark = "PASS" if gate.passed else "**FAIL**"
                lines.append(f"| {gate.gate.describe()} | {mark} | {gate.reason} |")
        lines.append("")

    return "\n".join(lines)


def _delta_cell(delta: float | None) -> str:
    if delta is None:
        return "[dim]--[/]"
    if abs(delta) < 5e-5:
        return "[dim]0[/]"
    colour = "green" if delta > 0 else "red"
    return f"[{colour}]{delta:+.4f}[/]"


def _delta_text(delta: float | None) -> str:
    if delta is None:
        return "--"
    return "0" if abs(delta) < 5e-5 else f"{delta:+.4f}"
