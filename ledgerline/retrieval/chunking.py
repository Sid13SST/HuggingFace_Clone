"""Chunking that keeps its receipts.

Every chunk carries the character offsets it came from. That is not
bookkeeping for its own sake: the citation verifier resolves a claim by
mapping it back to a span in the source document, and a chunk that has
forgotten where it came from can never be cited, only paraphrased.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 10-K/10-Q section headers. Splitting on these before splitting on size keeps
# "Item 1A. Risk Factors" from bleeding into "Item 2. Properties", which is the
# most common source of confidently wrong retrieval on filings.
SECTION_RE = re.compile(
    r"^\s*(Item\s+\d+[A-Z]?\.?[^\n]{0,80}|PART\s+[IVX]+[^\n]{0,40})\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    content: str
    char_start: int
    char_end: int
    section: str | None = None


def split_sections(text: str) -> list[tuple[str | None, int, int]]:
    """Return (section_label, start, end) spans covering the whole document."""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [(None, 0, len(text))]

    spans: list[tuple[str | None, int, int]] = []
    if matches[0].start() > 0:
        spans.append((None, 0, matches[0].start()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((match.group(1).strip(), match.start(), end))
    return spans


def chunk_text(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Section-aware, sentence-boundary chunking with offsets preserved.

    `target_chars` and `overlap_chars` are the two knobs worth an ablation --
    run the retrieval suite across a sweep and put the table in the README
    rather than picking the values a blog post used.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    chunks: list[Chunk] = []
    ordinal = 0

    for section, span_start, span_end in split_sections(text):
        body = text[span_start:span_end]
        for local_start, local_end in _windows(body, target_chars, overlap_chars):
            content = body[local_start:local_end].strip()
            if not content:
                continue
            # Re-anchor onto the untrimmed offsets so citations point at the
            # real document, not at our whitespace handling.
            leading = len(body[local_start:local_end]) - len(
                body[local_start:local_end].lstrip()
            )
            start = span_start + local_start + leading
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    content=content,
                    char_start=start,
                    char_end=start + len(content),
                    section=section,
                )
            )
            ordinal += 1
    return chunks


def _windows(body: str, target: int, overlap: int) -> list[tuple[int, int]]:
    if not body.strip():
        return []
    if len(body) <= target:
        return [(0, len(body))]

    boundaries = [0, *(m.end() for m in _SENTENCE_END.finditer(body)), len(body)]
    windows: list[tuple[int, int]] = []
    start = 0

    while start < len(body):
        hard_end = min(start + target, len(body))
        if hard_end >= len(body):
            windows.append((start, len(body)))
            break

        # Prefer the last sentence boundary inside the window; fall back to a
        # hard cut if a single sentence is longer than the target.
        candidates = [b for b in boundaries if start < b <= hard_end]
        end = max(candidates) if candidates else hard_end
        windows.append((start, end))

        next_start = end - overlap
        start = next_start if next_start > start else end
    return windows
