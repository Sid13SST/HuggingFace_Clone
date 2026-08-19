"""Ledgerline -- multimodal disclosure intelligence with claim-level provenance.

Answers questions across a company's filings, earnings call audio, and investor
deck, where every figure resolves back to a table cell, a page region, or a
timestamp in the call.

HuggingFace tasks in play (Task_2 screenshot, plus audio):
  automatic speech recognition, audio classification / voice activity detection,
  document question answering, table question answering, token classification,
  zero-shot classification, sentence similarity, text ranking, summarization,
  text classification (NLI, for groundedness), text generation.
"""

__all__ = ["ingest", "retrieval"]
