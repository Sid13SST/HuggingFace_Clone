"""Sightline -- vision-grounded infrastructure triage.

Turns street-level imagery and noisy 311 reports into a deduplicated,
severity-ranked repair queue, with each dispatch brief grounded in the
municipality's own published maintenance standards.

HuggingFace tasks in play (Task_1 screenshot, plus text):
  object detection, image segmentation, depth estimation, zero-shot image
  classification, zero-shot object detection, visual question answering,
  image feature extraction, image-to-text, document question answering
  (the standards PDFs), plus text classification, sentence similarity and
  time series forecasting on the 311 side.
"""

__all__ = ["dedupe", "ingest", "severity"]
