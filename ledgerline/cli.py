from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ledgerline.ingest.edgar import EdgarClient, MissingUserAgent
from ledgerline.retrieval.bm25 import BM25Index

app = typer.Typer(help="Ledgerline: disclosure intelligence.", no_args_is_help=True)
console = Console()


@app.command()
def filings(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol, e.g. CAT")],
    forms: Annotated[str, typer.Option(help="Comma-separated form types.")] = "10-K,10-Q,8-K",
    limit: Annotated[int, typer.Option(help="Max filings to list.")] = 15,
) -> None:
    """List recent EDGAR filings for an issuer. Responses are cached on disk."""
    try:
        asyncio.run(_filings(ticker, tuple(f.strip() for f in forms.split(",")), limit))
    except MissingUserAgent as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from exc


async def _filings(ticker: str, forms: tuple[str, ...], limit: int) -> None:
    async with EdgarClient() as client:
        cik = await client.ticker_to_cik(ticker)
        rows = await client.recent_filings(cik, forms=forms, limit=limit)

    table = Table(title=f"{ticker.upper()}  CIK {cik}", header_style="dim")
    for column in ("form", "filed", "period", "items", "document"):
        table.add_column(column)
    for filing in rows:
        table.add_row(
            filing.form,
            filing.filed_at.isoformat(),
            filing.period_end.isoformat() if filing.period_end else "--",
            ",".join(filing.items) or "--",
            filing.primary_document,
        )
    console.print(table)
    if rows:
        console.print(f"[dim]{rows[0].url}[/]")


@app.command()
def search(
    question: Annotated[str, typer.Argument(help="Question to run against the fixture corpus.")],
    k: Annotated[int, typer.Option(help="Results to show.")] = 5,
    mode: Annotated[str, typer.Option(help="hybrid | bm25")] = "hybrid",
) -> None:
    """Query the committed fixture corpus.

    Hybrid shows which arm found each result, which is how you tell whether
    fusion is earning its keep or the lexical side is carrying everything.
    """
    from ledgerline.evals import corpus_by_id, load_corpus

    lookup = corpus_by_id()
    documents = [(r["id"], r["text"]) for r in load_corpus()]

    if mode == "bm25":
        results = [(cid, s, None) for cid, s in BM25Index.build(documents).search(question, k=k)]
    else:
        results = [
            (row["doc_id"], row["fused_score"], row)
            for row in _live_retriever().explain(question, k=k)
        ]

    if not results:
        console.print("[yellow]no matches[/]")
        return

    for chunk_id, score, detail in results:
        record = lookup[chunk_id]
        provenance = ""
        if detail:
            only = detail["found_only_by"]
            provenance = (
                f" · bm25 #{detail['bm25_rank'] or '--'} dense #{detail['dense_rank'] or '--'}"
                + (f" [bold]{only}-only[/]" if only else "")
            )
        console.print(
            f"[bold]{chunk_id}[/] [dim]{score:.4f} · {record.get('kind')} · "
            f"{record.get('section', '--')}{provenance}[/]"
        )
        console.print(f"  {record['text'][:200]}...\n")


def _live_retriever():
    """Retriever that can handle a question nobody has embedded yet.

    The eval path deliberately has no fallback -- a cache miss there must be
    fatal, or half a run gets scored under a different model. Interactive use
    is the opposite: encoding one ad-hoc query on the spot is the whole point.
    """
    from ledgerline.evals import EMBEDDING_CACHE_PATH, load_corpus
    from ledgerline.retrieval.hybrid import HybridRetriever
    from shared.embeddings import CachedEmbedder, StaticEmbedder

    try:
        fallback = StaticEmbedder()
    except ImportError:
        fallback = None
        console.print(
            "[yellow]model2vec not installed[/] -- only questions already in the "
            "cache will work. `pip install -e \".[ledgerline]\"` for ad-hoc queries."
        )
    embedder = CachedEmbedder.from_npz(EMBEDDING_CACHE_PATH, fallback=fallback)
    return HybridRetriever.build(
        [(r["id"], r["text"]) for r in load_corpus()], embedder
    )


@app.command()
def diagnose(
    tag: Annotated[str, typer.Option(help="Only show this slice, e.g. narrative.")] = "",
) -> None:
    """Per-question nDCG across all three retrieval stages.

    Built because a headline that moved +0.014 hid one query improving by
    +0.37 and two regressing. An aggregate tells you something changed; this
    tells you what.
    """
    import json

    from ledgerline.evals import HERE, build_index, hybrid_retriever, reranking_retriever
    from shared.evals.metrics import ndcg_at_k

    bm25, hybrid, reranked = build_index(), hybrid_retriever(), reranking_retriever()
    table = Table(title=f"retrieval diagnosis{f' · {tag}' if tag else ''}", header_style="dim")
    for column, justify in (
        ("id", "left"), ("tags", "left"), ("bm25", "right"), ("hybrid", "right"),
        ("+rerank", "right"), ("delta", "right"), ("question", "left"),
    ):
        table.add_column(column, justify=justify)

    with (HERE / "datasets" / "retrieval.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            example = json.loads(stripped)
            if tag and tag not in example["tags"]:
                continue
            question = example["inputs"]["question"]
            gold = example["expected"]["relevant"]
            b = ndcg_at_k(gold, bm25.rank(question, k=10), 10)
            h = ndcg_at_k(gold, hybrid.rank(question, k=10), 10)
            r = ndcg_at_k(gold, reranked.rank(question, k=10), 10)
            delta = r - b
            colour = "green" if delta > 0.001 else "red" if delta < -0.001 else "dim"
            table.add_row(
                example["id"],
                ",".join(example["tags"][:2]),
                f"{b:.3f}",
                f"{h:.3f}",
                f"{r:.3f}",
                f"[{colour}]{delta:+.3f}[/]",
                question[:44],
            )
    console.print(table)


@app.command("rerank-cache")
def rerank_cache(
    model: Annotated[str, typer.Option(help="fastembed cross-encoder to score with.")] = "",
) -> None:
    """Rebuild the committed cross-encoder score cache.

    Scores the full (question x document) cross product, so changing
    candidate_k re-measures rather than producing a cache miss. Run after
    editing the corpus or any golden set.
    """
    from ledgerline.evals import RERANK_CACHE_PATH, rerank_pairs
    from ledgerline.retrieval.rerank import (
        DEFAULT_RERANK_MODEL,
        CrossEncoderReranker,
        save_rerank_cache,
    )

    pairs = rerank_pairs()
    console.print(f"scoring {len(pairs)} pairs with {model or DEFAULT_RERANK_MODEL}...")
    path = save_rerank_cache(
        RERANK_CACHE_PATH, pairs, CrossEncoderReranker(model or DEFAULT_RERANK_MODEL)
    )
    console.print(f"[green]wrote[/] {path}")


@app.command()
def ingest(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol, e.g. CAT")],
    form: Annotated[str, typer.Option(help="Form type to ingest.")] = "10-K",
    limit: Annotated[int, typer.Option(help="How many filings back to take.")] = 1,
    write: Annotated[bool, typer.Option(help="Write to Postgres.")] = True,
) -> None:
    """Fetch a real filing, parse it, and index it.

    Reports the table extraction rate and why the rest were declined, because
    that number is the fraction of the filing this system cannot see -- and a
    parser change that quietly raises it is a regression even when every
    extracted table is still correct.
    """
    asyncio.run(_ingest(ticker, form, limit, write))


async def _ingest(ticker: str, form: str, limit: int, write: bool) -> None:
    from ledgerline.ingest.edgar import EdgarClient
    from ledgerline.ingest.filing import parse_html
    from ledgerline.retrieval.chunking import chunk_text

    async with EdgarClient() as client:
        cik = await client.ticker_to_cik(ticker)
        filings = await client.recent_filings(cik, forms=(form,), limit=limit)
        if not filings:
            console.print(f"[yellow]no {form} filings for {ticker.upper()}[/]")
            raise typer.Exit(1)
        documents = [(f, await client.fetch_document(f)) for f in filings]

    for filing, raw in documents:
        document_id = f"{ticker.lower()}-{filing.period_end or filing.filed_at}"
        parsed = parse_html(raw, document_id)
        chunks = chunk_text(parsed.text)

        table = Table(title=f"{ticker.upper()} {form} {filing.period_end}", header_style="dim")
        table.add_column("metric")
        table.add_column("value", justify="right")
        rate = parsed.extraction_rate
        colour = "green" if rate > 0.5 else "yellow" if rate > 0.15 else "red"
        for name, value in (
            ("narrative chars", f"{len(parsed.text):,}"),
            ("chunks", f"{len(chunks):,}"),
            ("tables extracted", f"[{colour}]{len(parsed.tables)}[/]"),
            ("tables declined", str(len(parsed.skipped))),
            ("extraction rate", f"[{colour}]{rate:.1%}[/]"),
        ):
            table.add_row(name, value)
        console.print(table)

        console.print("[dim]declined because:[/]")
        for reason, count in parsed.skip_reasons().items():
            console.print(f"  {count:4}  {reason}")

        if write:
            _write_filing(filing, document_id, ticker, cik, parsed, chunks)


def _write_filing(filing, document_id, ticker, cik, parsed, chunks) -> None:
    from ledgerline.ingest.pipeline import ChunkRow, Document, Issuer, ingest_document
    from shared.db import connection
    from shared.embeddings import StaticEmbedder

    rows = [
        ChunkRow(
            external_id=f"{document_id}-c{c.ordinal}",
            content=c.content,
            ordinal=c.ordinal,
            section=c.section,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        for c in chunks
    ]
    with connection() as conn:
        ingest_document(
            conn,
            Issuer(cik=cik, name=ticker.upper(), ticker=ticker.upper()),
            Document(
                cik=cik,
                kind="filing",
                accession=filing.accession,
                form=filing.form,
                title=f"{ticker.upper()} {filing.form} {filing.period_end}",
                source_url=filing.url,
                fiscal_period=str(filing.period_end),
            ),
            rows,
            StaticEmbedder(),
        )
        conn.commit()
    console.print(f"[green]wrote[/] {len(rows)} chunks for {document_id}")


@app.command()
def index() -> None:
    """Load the fixture corpus into Postgres, embeddings and all.

    Re-runnable: each document's chunks are replaced rather than appended, so
    running this twice leaves the same seventeen rows rather than thirty-four.
    """
    from ledgerline.evals.parity import ingest_fixture_corpus
    from shared.db import connection

    with connection() as conn:
        written = ingest_fixture_corpus(conn)
    console.print(f"[green]indexed[/] {written} chunks")


@app.command()
def parity(
    k: Annotated[int, typer.Option(help="Depth to compare rankings at.")] = 10,
    reindex: Annotated[bool, typer.Option(help="Reload the corpus first.")] = True,
) -> None:
    """Compare SQL retrieval against the offline mirror, arm by arm.

    Exits non-zero only if the *dense* arms disagree. Those read the same
    vectors and order by the same metric, so a difference there is a bug in the
    write path. The lexical arms are different algorithms -- BM25 against
    ts_rank_cd over a stemmed tsvector -- and their divergence is reported
    rather than gated, because gating it would mean tuning one of them until it
    imitated the other.
    """
    from ledgerline.evals import HERE, embedder, hybrid_retriever
    from ledgerline.evals.parity import (
        FIXTURE_CIK,
        compare_all,
        ingest_fixture_corpus,
        scored_comparison,
    )
    from ledgerline.retrieval.sql import SqlRetriever
    from shared.db import connection
    from shared.evals.dataset import load_jsonl

    examples = load_jsonl(HERE / "datasets" / "retrieval.jsonl")
    questions = [e.inputs["question"] for e in examples]

    with connection() as conn:
        if reindex:
            ingest_fixture_corpus(conn)
        offline = hybrid_retriever()
        # Scoped to the fixture issuer: a database that has had a real 10-K
        # ingested into it holds several corpora, and an unscoped retriever
        # compares the mirror against a strictly larger index. The gate
        # caught exactly that the first time a real filing landed.
        sql = SqlRetriever(conn=conn, embedder=embedder(), cik=FIXTURE_CIK)

        missing = {r["id"] for r in _corpus_ids()} - sql.indexed_ids()
        if missing:
            console.print(f"[red]{len(missing)} corpus chunks are not indexed[/]")
            raise typer.Exit(1)

        results = compare_all(questions, offline, sql, k=k)
        quality = scored_comparison(examples, offline, sql, k=k)

    table = Table(title=f"python vs postgres · {len(questions)} queries · k={k}",
                  header_style="dim")
    for column in ("arm", "exact order", "top-1", "overlap", "mean displacement"):
        table.add_column(column, justify="right" if column != "arm" else "left")
    for row in results:
        expected_exact = row.arm == "dense"
        colour = (
            "green" if row.exact_order == 1.0
            else "red" if expected_exact
            else "yellow"
        )
        table.add_row(
            row.arm,
            f"[{colour}]{row.exact_order:.3f}[/]",
            f"{row.top1_agreement:.3f}",
            f"{row.overlap_at_k:.3f}",
            f"{row.mean_displacement:.2f}",
        )
    console.print(table)

    quality_table = Table(
        title="retrieval quality on the same golden set", header_style="dim"
    )
    for column in ("path", f"ndcg@{k}", f"recall@{k}"):
        quality_table.add_column(column, justify="left" if column == "path" else "right")
    for label in ("offline", "sql"):
        quality_table.add_row(
            label,
            f"{quality[f'{label}.ndcg@{k}']:.3f}",
            f"{quality[f'{label}.recall@{k}']:.3f}",
        )
    console.print(quality_table)

    gap = quality[f"sql.ndcg@{k}"] - quality[f"offline.ndcg@{k}"]
    console.print(
        f"[dim]sql - offline nDCG@{k}: {gap:+.3f} -- ordering can diverge freely "
        "so long as quality does not.[/]"
    )

    dense = next(row for row in results if row.arm == "dense")
    if dense.exact_order < 1.0:
        console.print(
            "[bold red]dense arms disagree[/] -- same vectors and same metric "
            "should give the same order. Check the write path, not the ranker."
        )
        raise typer.Exit(1)
    console.print(
        "[dim]dense arms identical; lexical divergence is BM25 vs ts_rank_cd "
        "and is expected.[/]"
    )


def _corpus_ids() -> list[dict]:
    from ledgerline.evals import load_corpus

    return load_corpus()


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to run through the agent graph.")],
    save: Annotated[bool, typer.Option(help="Persist the run to ledgerline.runs.")] = False,
    live: Annotated[bool, typer.Option(help="Use a real model for the narrative analyst.")] = False,
) -> None:
    """Run one question through the agent graph and show how it ended.

    Prints the route, the terminal outcome and the path taken, not just the
    answer. A run that degraded is more interesting than one that succeeded,
    and the step list is where you find out which node gave up.
    """
    import time

    from ledgerline.agent.graph import LedgerlineAgent
    from ledgerline.agent.llm import AnthropicModel, ModelUnavailable
    from ledgerline.agent.state import Outcome
    from ledgerline.evals import corpus_by_id, table_store

    model = None
    if live:
        try:
            model = AnthropicModel()
        except ModelUnavailable as exc:
            console.print(f"[yellow]narrative analyst unavailable:[/] {exc}")

    agent = LedgerlineAgent.build(
        _live_retriever(), corpus_by_id(), table_store(), model=model
    )
    started = time.perf_counter()
    state = agent.run(question)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    outcome = state.get("outcome", "?")
    colour = {
        Outcome.ANSWERED.value: "green",
        Outcome.REFUSED.value: "yellow",
        Outcome.DEGRADED.value: "red",
    }.get(outcome, "white")
    console.print(
        f"[{colour}]{outcome}[/]  [dim]route={state.get('route')} "
        f"{elapsed_ms}ms  {' -> '.join(state.get('steps', []))}[/]"
    )

    if state.get("answer"):
        console.print(f"\n{state['answer']}\n")
    for reason in state.get("degraded_reasons", []):
        console.print(f"  [red]x[/] {reason}")
    for citation in state.get("citations", []):
        console.print(f"  [dim]cite {citation.get('chunk_id')} {citation.get('quote', '')[:70]}[/]")

    if save:
        from ledgerline.agent.persistence import save_run
        from shared.db import connection

        with connection() as conn:
            run_id = save_run(conn, state, latency_ms=elapsed_ms)
            conn.commit()
        console.print(f"[dim]saved run {run_id}[/]")


@app.command()
def embed(
    model: Annotated[str, typer.Option(help="model2vec model to encode with.")] = "",
) -> None:
    """Rebuild the committed embedding cache.

    Run this after editing the fixture corpus or a golden set. CI reads the
    cache and never downloads a model, so a stale cache is a loud failure
    rather than a silent one -- the lookup is keyed by content hash.
    """
    from ledgerline.evals import EMBEDDING_CACHE_PATH, texts_to_embed
    from shared.embeddings import DEFAULT_MODEL, StaticEmbedder, save_cache

    texts = texts_to_embed()
    console.print(f"encoding {len(texts)} unique texts with {model or DEFAULT_MODEL}...")
    embedder = StaticEmbedder(model or DEFAULT_MODEL)
    path = save_cache(EMBEDDING_CACHE_PATH, texts, embedder)
    console.print(f"[green]wrote[/] {path}  (dim {embedder.dim})")


@app.command("export-corpus")
def export_corpus(
    ticker: Annotated[str, typer.Argument(help="Ticker symbol, e.g. CAT")] = "CAT",
    form: Annotated[str, typer.Option(help="Form type to export.")] = "10-K",
) -> None:
    """Export a real filing into the committed eval fixtures.

    Reads the EDGAR disk cache, so this runs offline once a filing has been
    fetched. The output is checked in: CI cannot reach EDGAR, and a suite that
    silently re-downloads its corpus is a suite whose numbers are not
    reproducible.

    Table cells are written as the *raw strings the filing shows*, not as the
    parsed floats. Committing the parser's output would make the numeric golden
    set a fixed point -- the parser would be graded against itself and could
    never be shown to be wrong.
    """
    asyncio.run(_export_corpus(ticker, form))


async def _export_corpus(ticker: str, form: str) -> None:
    import json

    from ledgerline.evals.real import CORPUS_PATH, TABLES_PATH
    from ledgerline.ingest.edgar import EdgarClient
    from ledgerline.ingest.filing import parse_html
    from ledgerline.retrieval.chunking import chunk_text

    async with EdgarClient() as client:
        cik = await client.ticker_to_cik(ticker)
        filings = await client.recent_filings(cik, forms=(form,), limit=1)
        if not filings:
            console.print(f"[yellow]no {form} filings for {ticker.upper()}[/]")
            raise typer.Exit(1)
        raw = await client.fetch_document(filings[0])

    filing = filings[0]
    document_id = f"{ticker.lower()}-{filing.period_end or filing.filed_at}"
    parsed = parse_html(raw, document_id)
    chunks = chunk_text(parsed.text)

    corpus_header = [
        f"# {ticker.upper()} {form} for {filing.period_end}: chunked narrative text.",
        f"# Exported by `ledgerline export-corpus {ticker.upper()}` from accession",
        f"# {filing.accession}. SEC filings are freely redistributable.",
        "#",
        "# Tables are excluded here and carried in cat_tables.jsonl instead.",
        "# Flattening a table into prose loses its scale and column headers, which",
        "# is the failure this project argues against -- so the eval corpus must",
        "# not do it either.",
    ]
    with CORPUS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for line in corpus_header:
            handle.write(line + "\n")
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "id": f"{document_id}-c{chunk.ordinal}",
                        "kind": "filing",
                        "section": chunk.section,
                        "text": chunk.content,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(parsed.tables) + len(parsed.skipped)
    table_header = [
        f"# {ticker.upper()} {form} tables the parser could read: "
        f"{len(parsed.tables)} of {total}.",
        "#",
        "# `values` are the filing's raw cell strings. `scale_hint` and the",
        "# per-row `unit` are the parser's *inferences*, and they are the thing",
        "# under test: numeric_cat.jsonl states what the filing means, arrived at",
        "# by reading the filing rather than by running this parser.",
    ]
    with TABLES_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for line in table_header:
            handle.write(line + "\n")
        for index, table in enumerate(parsed.tables):
            rows = []
            for label in table.row_labels:
                values = []
                for column in table.columns:
                    cell = table.cell(label, column)
                    values.append(cell.raw if cell else "")
                row: dict = {"label": label, "values": values}
                if label in table.row_units:
                    row["unit"] = table.row_units[label]
                if label in table.row_scales:
                    row["scale"] = table.row_scales[label]
                rows.append(row)
            handle.write(
                json.dumps(
                    {
                        "id": f"{document_id}-t{index}",
                        "document_id": document_id,
                        "caption": table.caption,
                        "columns": table.columns,
                        "scale_hint": table.scale_hint,
                        "unit": table.unit,
                        "rows": rows,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    console.print(
        f"[green]wrote[/] {len(chunks)} chunks and {len(parsed.tables)} tables "
        f"for {document_id}"
    )


@app.command("embed-cat")
def embed_cat(
    model: Annotated[str, typer.Option(help="model2vec model to encode with.")] = "",
) -> None:
    """Rebuild the committed embedding cache for the real 10-K corpus.

    Separate from `embed` because the two corpora have separate caches: one
    npz per corpus keeps a fixture edit from invalidating the real corpus's
    5,000-odd vectors, and keeps the real corpus's size out of the fast suite.
    """
    from ledgerline.evals.real import EMBEDDING_CACHE_PATH, texts_to_embed
    from shared.embeddings import DEFAULT_MODEL, StaticEmbedder, save_cache

    texts = texts_to_embed()
    console.print(f"encoding {len(texts)} texts with {model or DEFAULT_MODEL}...")
    embedder = StaticEmbedder(model or DEFAULT_MODEL)
    path = save_cache(EMBEDDING_CACHE_PATH, texts, embedder)
    console.print(f"[green]wrote[/] {path}  (dim {embedder.dim})")


@app.command("rerank-cache-cat")
def rerank_cache_cat(
    model: Annotated[str, typer.Option(help="Cross-encoder to score with.")] = "",
) -> None:
    """Score every (question, chunk) pair of the real corpus and cache it.

    21,780 pairs. Slow once, then free forever -- CI never loads a
    cross-encoder, and a pair outside the cache is a fatal miss rather than a
    silent re-score with whatever model happens to be installed.
    """
    from ledgerline.evals.real import RERANK_CACHE_PATH, rerank_pairs
    from ledgerline.retrieval.rerank import (
        DEFAULT_RERANK_MODEL,
        CrossEncoderReranker,
        save_rerank_cache,
    )

    pairs = rerank_pairs()
    console.print(f"scoring {len(pairs):,} pairs with {model or DEFAULT_RERANK_MODEL}...")
    path = save_rerank_cache(
        RERANK_CACHE_PATH, pairs, CrossEncoderReranker(model or DEFAULT_RERANK_MODEL)
    )
    console.print(f"[green]wrote[/] {path}")


if __name__ == "__main__":
    app()
