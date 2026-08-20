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
| `ledgerline.agent` | answered_correct | **1.000** | end-to-end, every answer it gives is right |
| | answered_rate | 0.733 | the rest it refuses rather than guessing |
| | refused_rate | 0.267 | |
| | degraded_rate | **0.000** | nothing broke |
| | fabrication_rate | **0.000** | never answers an undisclosed figure |
| | terminal_rate | 1.000 | every run ends in a named state, never an exception |
| `ledgerline.routing` | accuracy | 0.933 | keyword routing, on the set it was tuned against |
| `ledgerline.routing_heldout` | accuracy | **1.000** | on a set it was not — the number to believe |

### Sightline

| suite | metric | value | what it means |
| --- | --- | ---: | --- |
| `sightline.dedupe` | pair_precision | 1.000 | never merges two distinct defects |
| | pair_recall | **1.000** | was 0.667 behind a lexical placeholder |
| | accuracy · hard slice | **1.000** | was a 0.500 coin-flip |
| | false_merge_rate | 0.000 | recall bought without a single bad merge |
| `sightline.dedupe_lexical` | pair_recall | 0.667 | the Jaccard control, still measured every run |
| | accuracy · hard slice | 0.500 | |
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

### The dedupe ablation: a seam finally used

`similarity_fn` has been a seam in the dedupe engine since it was written, with
a Jaccard baseline behind it and a comment saying the sentence embedding would
have to beat it on the same labelled pairs before earning its place. It does.

| metric | Jaccard | + embeddings | delta |
| --- | ---: | ---: | ---: |
| pair_recall | 0.667 | **1.000** | **+0.333** |
| accuracy · hard slice | 0.500 | **1.000** | **+0.500** |
| pair_precision | 1.000 | 1.000 | 0 |
| false_merge_rate | 0.000 | 0.000 | 0 |

The failures it fixes are the ones a set-overlap score can never reach:

```
"Road damage"                       vs  "Broken pavement here"           0.000
"Sunken area around a utility cut"  vs  "Utility patch has sunk, rough"  0.100
```

Both are the same defect. Their token intersection is empty or nearly so, so no
amount of tuning Jaccard finds them — this had to be a replacement, not a
refinement.

**The swap is not drop-in, and the reason is scale.** Jaccard over 311 text
sits near zero for unrelated reports; cosine between two short strings of
municipal English rarely drops below 0.3. Every score rises, negatives
included, so the threshold calibrated against Jaccard is not a conservative
choice under cosine — it is simply the wrong operating point, and keeping it
would have looked like caution while costing recall. Threshold and similarity
function now live in the same config object for exactly this reason: separated,
they drift, and you end up running a semantic similarity against a lexical
threshold.

Two things kept the recalibration honest. The floor parameter was swept from
0.0 to 0.4 and **none beat 0.0** — the spatial and category gates reject
negatives long before text is consulted, so text really is the tiebreak the
weights claim it is. And 0.0 is what production already computes:
`sightline.duplicate_candidates` scores text as
`1 - (r.text_embedding <=> t.text_embedding)`, raw cosine, so any floor
invented in Python would have made the two disagree — the bug class the
retrieval parity work existed to find.

The classes separate completely on this set: lowest duplicate **0.5112**,
highest non-duplicate **0.4327**. The threshold is **0.47**, the midpoint of
that margin rather than either edge. The margin is 0.078 wide on twenty pairs,
so read it as a calibration rather than a discovery —
`test_the_threshold_sits_inside_the_measured_margin` fails the moment a fixture
edit narrows it past the threshold.

### A gate that could not fire

While rewiring the suite: `false_merge_rate` was gated as
`Gate("false_merge_rate", max_regression=0.05)` under a harness that assumed
higher is better. A false-merge rate climbing from 0.00 to 0.10 computed as a
0.10 *improvement* and passed.

The comment above it read *"a rising false-merge rate is the failure worth
blocking a merge over"* — and it was the one failure that gate could not
detect. It is now `max_value=0.05, higher_is_better=False`, using the bounds
added for the agent suite. Same latent problem still applies to Sightline's
ECE 0.222, which is reported and ungated; that one needs the number to come
down first.

### The agent graph, and why it has three endings

```
plan ──▶ retrieve ──┬─▶ table_analyst ─────┐
                    ├─▶ narrative_analyst ─┼─▶ reconcile ──▶ finalize ──▶ END
                    └─▶ (both, cross-modal)┘
```

LangGraph, checkpointed from the first commit, with every final state written
to `ledgerline.runs`. The design decision worth defending is not the graph
shape — it is that a run ends in one of **three** named states, not two:

| outcome | meaning | what a rising rate tells you |
| --- | --- | --- |
| `answered` | produced a figure or a claim, with citations | |
| `refused` | declined **on the merits** — evidence retrieved, read, judged insufficient | the corpus thinned or the questions got harder |
| `degraded` | could not do the job — retrieval empty, model unavailable, a node raised | something is broken |

Collapsing the last two into "no answer" is the mistake that makes an agent
impossible to operate. They have opposite fixes, and the on-call engineer needs
to know which within seconds. So `finalize` is the only node permitted to set
an outcome, degradation beats a partial answer, and an unresolved contradiction
between the table and the narrative degrades rather than picking a winner.

Every node is also forbidden from raising. Anything unexpected becomes a
`degraded_reason` and the run still terminates — `terminal_rate` is gated at
1.000 because a caller answering a user needs an outcome, not a traceback.

No language model is configured in CI, and that is not a gap in the suite but
what it currently measures: **how far the deterministic half gets alone.** On
the numeric golden set that is 73% answered at perfect accuracy, 27% refused,
nothing degraded and nothing fabricated. When the narrative analyst lands, the
*refusals* should move. If `answered_correct` or `fabrication_rate` moves
instead, the model made the system worse, and this suite is where that shows.

### Routing, and an overfitting story worth telling

Routing decides which analysts run, so it caps everything downstream. The first
version matched topic words — "margin", "risk", "concentrated" — and scored
**0.533 accuracy, κ = 0.110**: barely above chance. The failures all had one
shape. *"What was ... operating margin?"* and *"Why did gross margin decline?"*
share every content word and are different questions; what separates them is
the interrogative and what it governs. Rewritten to match question **form**, it
went to 1.000 on that set.

Which is exactly when it should be distrusted. Fifteen questions cannot
validate a classifier, and I had just revised the rules while looking at which
ones missed. So the numeric golden set became a held-out slice — every question
in it expects a numeric answer by construction, so the labels are free — and it
immediately caught the problem:

> A partitive rule (`"how much of X"` → narrative) was added to fix *"How much
> of the announced pricing is holding?"*. It scored 1.000 on the tuned set and
> broke *"How much of the revolving facility remains available?"*, which wants
> a dollar amount. The distinction is between "is holding" and "remains
> available" — two verbs. A rule that has to know which verbs are assessments
> is a rule fitted to the examples in front of it.

The rule was deleted, the question it fixed is now a documented miss, and the
tuned set sits at **0.933** while the held-out set sits at **1.000**. The gates
are set *below* both. `ledgerline.routing_heldout` exists to catch the next one
of these before it ships.

### Does the database rank the way the harness says it does?

Every gated number above is produced by the offline Python retriever reading a
JSONL fixture. Production reads Postgres. Those are two implementations of one
idea, and until something wrote embeddings into `ledgerline.chunks` the second
had never ranked a real row — so the gates were protecting a system nobody
ships.

`ledgerline parity` indexes the fixture corpus into Postgres and compares the
two, one arm at a time, because the arms have different expectations:

| arm | exact order | top-1 | overlap | expectation |
| --- | ---: | ---: | ---: | --- |
| dense | **1.000** | 1.000 | 1.000 | identical — same vectors, same metric |
| lexical | 0.067 | 0.667 | 0.807 | different — BM25 vs `ts_rank_cd` |
| fused | 0.133 | 0.867 | 0.952 | inherits the lexical gap, damped by RRF |

Only the dense row is gated. Both sides read the same committed vectors and
order by cosine, so a difference there is a defect in the write path — a
truncated vector, a normalisation applied once, a chunk stored against the
wrong text. The lexical arms are genuinely different algorithms over different
tokenisations, and forcing them to agree would mean crippling one of them.

Ordering divergence is only interesting if it costs something, so quality is
measured on the same golden set:

| path | nDCG@10 | recall@10 |
| --- | ---: | ---: |
| offline mirror | 0.877 | 0.967 |
| Postgres | **0.897** | 0.967 |

The two disagree on ordering for 87% of queries and land within +0.020 nDCG of
each other, with identical recall. That is the useful result: the divergence is
real, bounded, and does not cost accuracy.

**This is also where the exercise paid for itself.** The lexical arm was
returning *nothing* on two questions in three. `websearch_to_tsquery` ANDs
every term, so `How much did net revenue grow in fiscal 2025?` compiled to

```
'much' & 'net' & 'revenu' & 'grow' & 'fiscal' & '2025'
```

which matches a chunk only if all six stems appear in it — zero rows across
most of the golden set. RRF had been silently running on the dense arm alone.
AND is the right default for a search box, where the user is iterating and
wants to narrow; it is the wrong default for a question, where terms are
evidence to be weighed rather than requirements to be met. That is precisely
what BM25 does, which is why the offline mirror never had the bug and why
comparing the two is what surfaced it. The fix is
`ledgerline.any_lexeme_tsquery`, which ORs the stemmed lexemes and returns
`NULL` for an all-stopword query so callers skip the scan instead of failing.

A second bug fell out of the same read: the lexical CTE applied `LIMIT` with no
`ORDER BY`, so truncation kept an arbitrary slice rather than the top-ranked
one. Harmless at 17 chunks, silently wrong at any real scale.

Neither was reachable by unit tests, because both need rows. `ledgerline
parity` and `tests/test_sql_parity.py` now run in the CI job that has a
database.

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

**The remaining baseline is still bad on purpose.** The detector is
miscalibrated (ECE 0.222) — the *before* half of a future ablation, and any
improvement has to move a number in this table to be merged. Dedupe used to sit
here too; it moved, and `sightline.dedupe_lexical` now keeps measuring what it
moved from.

One honesty note: the Ledgerline retrieval numbers look high because the fixture
corpus is 17 chunks. They are a smoke test for the harness, not a claim about
retrieval quality. Real numbers come from the EDGAR-backed corpus, which is
generated rather than committed.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
# python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # macOS / Linux

pytest -q                    # 304 tests, no external dependencies
                             # (27 need a database and skip without one)
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
ledgerline index                             # fixture corpus -> postgres, with vectors
ledgerline parity                            # sql retrieval vs the offline mirror
ledgerline ask "why did gross margin decline"  --save   # agent run, persisted
sightline reports --limit 20                 # real 311 data via Socrata
sightline severity --mask-area-px 9000 --depth-m 6.0
sightline embed                              # rebuild the report-text vectors
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
shared/            config, rate-limited caching HTTP, Postgres, embeddings,
                   eval harness
  evals/           dataset, metrics, detection metrics, registry, runner, report
ledgerline/
  ingest/edgar.py  SEC EDGAR client (rate limit + User-Agent enforced)
  agent/           the graph: router, nodes, llm seam, run persistence
  ingest/pipeline  chunk -> embed -> Postgres, replace-per-document
  retrieval/       chunking with offsets, BM25, dense, RRF fusion, reranking
  retrieval/sql.py the production path: hybrid_search() and its two arms
  tables/          cell model with unit-aware scaling, question -> cell resolver
  evals/           retrieval + numeric + baseline suites, fixtures, golden sets
  evals/parity.py  Postgres vs the offline mirror, measured arm by arm
  schema.sql       documents, chunks, tables, runs, hybrid_search()
sightline/
  ingest/          Socrata 311, Mapillary imagery
  dedupe.py        spatial duplicate scoring, sweeps, clustering
  similarity.py    embedding cosine behind the similarity_fn seam
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
BM25 + RRF, three-stage retrieval with cross-encoder reranking, the LangGraph
agent graph with three terminal outcomes and checkpointed replay, the ingest path
that writes chunks and vectors into Postgres, SQL retrieval measured against the
offline mirror arm by arm, the table model and cell resolver with unit-aware
scaling and explicit declining, the dedupe engine with tunable config and
threshold sweeps, the severity estimator with abstention, detection metrics
(AP, mAP, IoU, ECE), and CI wiring that gates on the eval suites, applies both
schemas twice to check idempotence, and runs retrieval parity against a real
database.

Not built yet, in the order they should land:

1. **A narrative analyst that actually calls a model.** The seam, the cost
   accounting, the refusal handling and the degraded path all exist and are
   tested; what is missing is a warmed completion cache so CI can exercise the
   path offline. This is the change `ledgerline.agent` was built to measure.
2. Extraction of tables out of real filing HTML, replacing the committed
   fixture — the resolver and its metrics already exist.
3. NLI-based citation verification, at which point the 0.750
   `refusal_precision` over-refusal above should close.
4. Real detector fine-tuning, ONNX export, and the reviewer-override
   flywheel — and calibration, so ECE 0.222 can finally get a ceiling.
5. The LLMOps layer — tracing, prompt registry, cost and latency per commit.
   Deliberately last, but no longer vacuous: `Completion.cost_usd` and
   `ledgerline.runs` already record per-run spend and latency, so the
   dashboard has something to read.

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
