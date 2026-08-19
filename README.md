# Ledgerline & Sightline

Two production-shaped AI systems in one repo, sharing an eval harness.

- **Ledgerline** — multimodal disclosure intelligence. Answers questions across
  a company's SEC filings, earnings call audio, and investor deck, where every
  figure resolves back to a table cell, a page region, or a call timestamp.
- **Sightline** — vision-grounded infrastructure triage. Turns street-level
  imagery and noisy 311 reports into a deduplicated, severity-ranked repair
  queue grounded in the municipality's published maintenance standards.

They are separate products that share `shared/`: config, a rate-limited caching
HTTP client, and the eval harness that grades both.

---

## Current numbers

Every suite runs on committed fixtures — no network, no database, no GPU, no
API keys. `evalctl run` reproduces this table in about a second.

### Ledgerline

| suite | metric | value | what it means |
| --- | --- | ---: | --- |
| `ledgerline.retrieval` | nDCG@10 | **0.987** | hybrid + cross-encoder rerank |
| | MRR | **1.000** | a gold chunk ranks first on every question |
| | recall@10 | 1.000 | |
| | nDCG@10 · narrative | **0.949** | the slice dense retrieval could not fix |
| | nDCG@10 · transcript | 1.000 | |
| | nDCG@10 · numeric | 1.000 | |
| `ledgerline.retrieval_hybrid` | nDCG@10 | 0.877 | no-rerank control |
| `ledgerline.retrieval_bm25` | nDCG@10 | 0.862 | lexical-only control |
| `ledgerline.numeric` | exact_match | **0.917** | figures resolved to table cells |
| | exact_match · distractor-heavy | 0.800 | |
| | refusal_recall | **1.000** | refuses all three undisclosed questions |
| | refusal_precision | 0.750 | one over-refusal, explained below |
| `ledgerline.numeric_baseline` | exact_match | 0.167 | the prose approach, still measured every run |
| | refusal_recall | 0.000 | never abstains |

### Sightline

| suite | metric | value | what it means |
| --- | --- | ---: | --- |
| `sightline.dedupe` | pair_precision | 1.000 | conservative: never merges two distinct defects |
| | pair_recall | 0.667 | but misses a third of real duplicates |
| | accuracy · hard slice | 0.500 | coin-flip on the cases that matter |
| | false_merge_rate | 0.000 | |
| `sightline.detection` | mAP@50 | 0.861 | |
| | mAP@50-95 | 0.657 | |
| | ECE | **0.222** | badly overconfident — severity routing reads this score |
| `sightline.severity` | extent_abs_rel_error | 0.083 | vs hand-measured references |
| | extent_within_15pct | 1.000 | |
| | band_accuracy | 0.917 | |
| | abstain_recall | 1.000 | correctly refuses when depth or confidence is unreliable |

### The retrieval ablation, all three stages

| slice | BM25 | + dense | + rerank |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.862 | 0.877 | **0.987** |
| MRR | 0.850 | 0.856 | **1.000** |
| recall@10 | 0.967 | 0.967 | **1.000** |
| · narrative | 0.719 | 0.722 | **0.949** |
| · transcript | 0.754 | 0.833 | **1.000** |
| · numeric | 0.915 | 0.933 | **1.000** |

Three stages, three controls, all re-measured on every CI run
(`retrieval_bm25`, `retrieval_hybrid`, `retrieval`). `ledgerline diagnose`
prints the per-question version.

Reranking also lifted **recall@10**, which reordering alone cannot do: the
shortlist handed to the cross-encoder is 25 wide, so a document the cheap
stages ranked 12th can be pulled into the final ten. Retrieve wide, rerank
narrow.

### The second ablation: dense retrieval, and what it did not fix

| slice | BM25 | hybrid | delta |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.862 | **0.877** | +0.014 |
| · transcript | 0.754 | **0.833** | **+0.079** |
| · numeric | 0.915 | 0.933 | +0.018 |
| · narrative | 0.719 | 0.722 | +0.003 |

Dense retrieval was added specifically to fix the `narrative` slice. **It did
not.** The headline moved, but almost entirely on the transcript slice, and the
average hid two regressions:

```
r-008  What did the CFO say about margin pressure?     0.631 -> 1.000   +0.369
r-002  Why did gross margin decline?                   0.613 -> 0.387   -0.226
r-013  How much of the announced pricing is holding?   0.631 -> 0.500   -0.131
```

`ledgerline diagnose` prints that table, and it exists because the aggregate
was actively misleading. Reading the failures:

- **r-002** asks *why* margin declined. The dense arm ranks the CFO's "margin
  pressure is transitory" chunk first — same topic, wrong aspect. That chunk is
  about whether the pressure *persists*, not what caused it.
- **r-013** asks how much pricing is *holding*. The dense arm ranks a
  supply-cost risk factor above the CFO's actual answer.

Both are the same failure: static embeddings match topics, not propositions.
They have no compositional or discourse sense, so "why did X decline" and "will
X persist" land in the same neighbourhood.

**This is what the cross-encoder was added for, and it worked.** Reading the
query and the passage together fixes exactly the two queries dense retrieval
regressed:

```
r-002  Why did gross margin decline?                  0.387 -> 0.798
r-013  How much of the announced pricing is holding?  0.500 -> 1.000
```

The diagnosis predicted the fix, and the fix is pinned by
`test_fixes_the_queries_dense_retrieval_regressed` so it cannot silently
reverse. Hybrid retrieval on its own was worth shipping for the transcript
slice and was never going to fix narrative questions; that is now a documented
step in an ablation rather than an unexplained flat number.

### The first ablation: tables as data

| | naive extraction | table analyst | delta |
| --- | ---: | ---: | ---: |
| exact_match | 0.167 | **0.917** | **+0.750** |
| exact_match · distractor-heavy | 0.000 | **0.800** | **+0.800** |
| refusal_recall | 0.000 | **1.000** | **+1.000** |

Same golden set, same metrics, same scoring function — the only thing that
changes is whether a figure is read out of prose or resolved to a table cell.
`ledgerline.numeric_baseline` keeps measuring the prose approach on every CI
run and is deliberately ungated, so the "before" number cannot rot and nobody
is tempted to quietly strengthen the straw man.

No model is involved. The row resolver is deterministic token matching. That is
the point: the architecture moved the number, not model size, and a real
Table QA model drops in behind the same interface and gets measured the same way.

The one remaining miss is honest and worth reading. *"What share of net revenue
came from the single largest customer?"* covers the row `Net revenue`
completely and is still not a question about that cell — the answer lives in
prose. The resolver declines rather than confidently returning the wrong
figure, which costs a point of `refusal_precision` (0.750) and is the right
trade. That over-refusal disappears when the prose path lands.

**The other baselines are still bad on purpose.** The detector is
miscalibrated (ECE 0.222). The dedupe misses a third of duplicates. Each is the
*before* half of a future ablation, and every improvement has to move a number
in this table to be merged.

One honesty note: the Ledgerline retrieval numbers look high because the fixture
corpus is 17 chunks. They are a smoke test for the harness, not a claim about
retrieval quality. Real numbers come from the EDGAR-backed corpus, which is
generated rather than committed.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
# python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # macOS / Linux

pytest -q                    # 200 tests, no external dependencies
evalctl run                  # the table above
evalctl list                 # every suite, its dataset, and its gates
```

Live data and the database are optional and additive:

```bash
cp .env.example .env         # set EDGAR_USER_AGENT at minimum
docker compose up -d --wait  # postgres with pgvector + postgis, minio
make migrate                 # apply both schemas

ledgerline filings CAT                       # real EDGAR filings, cached to disk
ledgerline search "why did gross margin decline"
sightline reports --limit 20                 # real 311 data via Socrata
sightline severity --mask-area-px 9000 --depth-m 6.0
sightline cluster                            # dedupe the fixture set
```

---

## The eval harness

Built first, before either agent graph. That ordering is the point: it means
every decision after it — chunk size, reranker, model swap, dedupe threshold —
is justified by a number that moved rather than by taste.

A suite is a labelled dataset plus a function plus its gates:

```python
register_suite(Suite(
    name="ledgerline.numeric",
    project="ledgerline",
    dataset=HERE / "datasets" / "numeric.jsonl",
    run=run_numeric,
    gates=[
        Gate("exact_match", min_value=0.75, max_regression=0.01),
        Gate("refusal_recall", min_value=0.90, max_regression=0.05),
    ],
))
```

Two gate flavours, and both are needed:

- `min_value` — an absolute floor, for things that must never be bad
  (groundedness, abstention recall).
- `max_regression` — a relative guard against a stored baseline, which is what
  catches the prompt edit that quietly costs three points of nDCG.

`evalctl run` exits non-zero when a gate trips, so CI blocks the merge.
`evalctl run --write-baseline` records the current numbers as the new bar,
stamped with the commit that produced them.

Things the harness deliberately does:

- **Slices every metric by tag.** A headline nDCG that holds steady while the
  `cross-modal` slice collapses is the regression worth catching, and you only
  see it if you can slice.
- **Treats a raising suite as a failing suite**, not a crash — a model server
  being down should turn CI red, not lose the other four suites' results.
- **Fails a gate whose metric is missing.** A suite that quietly stops
  reporting a gated metric must not pass.
- **Tests itself.** `tests/test_harness.py` asserts that every gated metric is
  actually emitted by its suite, and that every committed golden set parses.

### Golden sets

Hand-labelled JSONL, one object per line, `#` comments allowed:

```json
{"id": "n-004", "inputs": {"question": "How much was drawn on the revolver?"},
 "expected": {"value": "$75.0 million", "source": "c-liquidity"},
 "tags": ["liquidity", "distractor-heavy"]}
```

Committed sets are synthetic fixtures so CI runs anywhere. The real sets are
built from live EDGAR and 311 data and are regenerated, not trusted. Every
labelled `not_disclosed` item exists to be *refused* — a pipeline that answers
them is worse than one that scores lower.

---

## Design decisions worth defending

**Tables are stored as data, never flattened into prose.** `ledgerline.tables`
and `table_cells` keep parsed, scale-normalised numerics, and a figure is
answered by resolving it to a cell. Rows carry a unit that can opt out of the
table's scale, because a table stated "in thousands" must not turn a 34.2%
margin into 34,200 or a 6,480 headcount into 6.48 million. An unparseable cell
is empty, never zero. **Measured: +0.750 exact_match.**

**Every retrievable chunk keeps its span.** Page plus bbox, character offsets,
or an audio time range. A chunk that has forgotten where it came from can never
be cited, only paraphrased — and the citation verifier has nothing to check.
`test_offsets_round_trip_to_the_source` pins this.

**Retrieval is three stages, each measured separately.** BM25 matches tokens,
static embeddings match topics, a cross-encoder matches propositions. Each has
a permanent ungated control suite so the contribution of every stage stays
visible. **Measured: +0.125 nDCG@10 end to end.**

**Hybrid retrieval fuses on rank, not score.** BM25 scores and cosine distances
are not on comparable scales; normalising them is a calibration problem you do
not need to have. `ledgerline.hybrid_search` does RRF in SQL, `reciprocal_rank_fusion`
mirrors it in Python for the offline suite.

**Severity is a measurement, not an adjective.** Monocular depth times mask area
gives an extent in centimetres, which is what the standards clause keys on. A
supervisor can ask why a defect was called urgent and get "38 cm across, clause
4.2.1" instead of "the model thought so".

**The system abstains.** Sightline refuses to size a defect past ~25 m of depth
or below a confidence floor, and `abstain_recall` is gated at 0.90. Ledgerline's
refusal subset is measured the same way. Knowing when not to answer is a feature
with a metric, not an apology.

**Spatial blocking happens before vector maths.** `sightline.duplicate_candidates`
cuts the candidate set with a GiST index first, then scores. The other order —
nearest-neighbour over every embedding, then filter by distance — looks fine on
1k rows and falls over on 100k.

---

## Layout

```
shared/            config, rate-limited caching HTTP, Postgres, eval harness
  evals/           dataset, metrics, detection metrics, registry, runner, report
ledgerline/
  ingest/edgar.py  SEC EDGAR client (rate limit + User-Agent enforced)
  retrieval/       chunking with offsets, BM25, dense, RRF fusion, reranking
  tables/          cell model with unit-aware scaling, question -> cell resolver
  evals/           retrieval + numeric + baseline suites, fixtures, golden sets
  schema.sql       documents, chunks, tables, runs, hybrid_search()
sightline/
  ingest/          Socrata 311, Mapillary imagery
  dedupe.py        spatial + lexical duplicate scoring, sweeps, clustering
  severity.py      depth x mask area -> extent -> band -> SLA, with abstention
  evals/           dedupe + severity + detection suites and golden sets
  schema.sql       PostGIS + pgvector, duplicate_candidates(), open_queue
evals/baselines/   committed baselines, each stamped with its commit
infra/postgres/    Dockerfile combining PostGIS and pgvector
```

---

## What is built and what is not

Built and tested: the eval harness, both schemas, the EDGAR/Socrata/Mapillary
clients with rate limiting and disk caching, chunking with span preservation,
BM25 + RRF, the table model and cell resolver with unit-aware scaling and
explicit declining, the dedupe engine with tunable config and threshold sweeps, the
severity estimator with abstention, detection metrics (AP, mAP, IoU, ECE), and
CI wiring that gates on the eval suites and applies both schemas twice to check
idempotence.

Not built yet, in the order they should land:

1. Populating `chunks.embedding` from the ingest path so the SQL
   `hybrid_search()` runs on real data rather than only the offline mirror.
2. The LangGraph agent graph — planner, routing, the two analysts, the
   contradiction checker — with checkpointing and a `degraded` terminal state
   from the first commit.
3. Extraction of tables out of real filing HTML, replacing the committed
   fixture — the resolver and its metrics already exist.
4. The LangGraph agent graphs, with checkpointing and a `degraded` terminal
   state from the first commit.
5. NLI-based citation verification and the refusal path, at which point
   `refusal_recall` gets an actual floor.
6. Real detector fine-tuning, ONNX export, and the reviewer-override flywheel.

Each of those is a pull request whose description is a diff in the table at the
top of this file.

---

## Notes on the data sources

- **SEC EDGAR** — no API key exists. It requires a descriptive `User-Agent`
  with a contact address and a ceiling of 10 req/s; both are enforced in the
  client rather than left to the caller, because the penalty for ignoring them
  is an IP block. Responses are cached to disk, which is also what makes eval
  runs reproducible.
- **Socrata / Open311** — an app token is optional but sharply raises the
  anonymous rate limit. Paging is ordered by `:id`, not by timestamp: paging on
  a non-unique sort key silently drops and duplicates rows when new records
  land mid-backfill.
- **Mapillary** — needs a client token. `captured_at` arrives as epoch
  milliseconds; the bbox query is a square, so corners get filtered by true
  distance downstream.

All fixture data in this repo is synthetic. Northwind Manufacturing Inc. is not
a real company, the 311 coordinates are fabricated, and the severity thresholds
are illustrative rather than any city's actual published standard.
