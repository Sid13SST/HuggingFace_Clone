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
    from ledgerline.retrieval.embeddings import CachedEmbedder, StaticEmbedder
    from ledgerline.retrieval.hybrid import HybridRetriever

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
def embed(
    model: Annotated[str, typer.Option(help="model2vec model to encode with.")] = "",
) -> None:
    """Rebuild the committed embedding cache.

    Run this after editing the fixture corpus or a golden set. CI reads the
    cache and never downloads a model, so a stale cache is a loud failure
    rather than a silent one -- the lookup is keyed by content hash.
    """
    from ledgerline.evals import EMBEDDING_CACHE_PATH, texts_to_embed
    from ledgerline.retrieval.embeddings import DEFAULT_MODEL, StaticEmbedder, save_cache

    texts = texts_to_embed()
    console.print(f"encoding {len(texts)} unique texts with {model or DEFAULT_MODEL}...")
    embedder = StaticEmbedder(model or DEFAULT_MODEL)
    path = save_cache(EMBEDDING_CACHE_PATH, texts, embedder)
    console.print(f"[green]wrote[/] {path}  (dim {embedder.dim})")


if __name__ == "__main__":
    app()
