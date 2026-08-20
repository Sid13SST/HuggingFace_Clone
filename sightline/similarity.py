"""Semantic text similarity for duplicate resolution.

The dedupe engine has always had a `similarity_fn` seam with a Jaccard baseline
behind it, and the baseline's failures were the obvious ones: two reports of
the same defect share no tokens.

    "Road damage"                      vs  "Broken pavement here"       -> 0.000
    "Sunken area around a utility cut" vs  "Utility patch has sunk"     -> 0.100

Both are duplicates. No amount of tuning a set-overlap score fixes a pair with
an empty intersection, which is why this is a replacement rather than a
refinement.

**Swapping in cosine is not a drop-in change, and the reason is the score
distribution.** Jaccard over 311 free text is mostly near zero -- unrelated
reports share no content tokens at all. Cosine between two short strings from
the same domain is rarely below 0.3, because "pothole on 14th" and "streetlight
out on 9th" are both terse municipal English about a street. Every pair's score
rises, negatives included, so the threshold tuned against Jaccard is no longer
the right threshold. It has to be re-swept, and the thing to watch while
sweeping is `false_merge_rate`, not recall: a similarity function that lifts
every score can buy recall with merges that make a real defect disappear from
the queue.

That did not happen here, and the reason is worth recording. Precision stayed
at 1.000 across every floor and threshold swept, because the spatial and
category gates reject the negatives before text is consulted at all -- the two
non-duplicates that survive those gates are a divided-road pair and an extended
defect, and neither reaches threshold on distance alone. Text similarity in
this system is a tiebreak, exactly as the weights say. The recall it buys is
therefore free, which is a pleasant result and not one to assume next time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shared.embeddings import Embedder, normalize
from shared.logging import get_logger

log = get_logger(__name__)


@dataclass
class EmbeddingSimilarity:
    """Cosine between two report texts.

    `floor` rescales the output so that anything below it reads as no signal.
    It defaults to 0.0 -- no rescaling -- and that default is not a shrug.
    `sightline.duplicate_candidates` in schema.sql scores text as
    `1 - (r.text_embedding <=> t.text_embedding)`, which is raw cosine, so any
    floor invented here would make Python and Postgres compute different
    numbers from the same vectors. That is the precise bug class the retrieval
    parity work existed to find, and re-introducing it in a different service
    to save a threshold recalibration is a bad trade.

    The concern that motivated a floor was real: cosine over short municipal
    text rarely drops below 0.3, so the text term contributes a large constant
    to every pair and the weights stop meaning what they say. Sweeping it
    settled the question -- floors from 0.0 to 0.4 were measured against the
    labelled pairs and none beat 0.0, because the distance and category gates
    reject the negatives long before text similarity is consulted. The knob
    stays, at zero, for a corpus where that stops being true.
    """

    embedder: Embedder
    floor: float = 0.0
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def vector(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = normalize(self.embedder.encode([text]))[0]
        return self._cache[text]

    def __call__(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        cosine = float(np.dot(self.vector(a), self.vector(b)))
        if self.floor <= 0.0:
            return max(0.0, min(1.0, cosine))
        if self.floor >= 1.0:  # degenerate config; treat as no rescale
            return max(0.0, min(1.0, cosine))
        return max(0.0, min(1.0, (cosine - self.floor) / (1.0 - self.floor)))


#: Threshold for the semantic path, and it is *not* the lexical 0.62.
#:
#: Cosine puts every pair on a higher scale than Jaccard does, so the old
#: threshold is not a conservative choice under the new similarity -- it is
#: simply a different operating point, and keeping it would have looked like
#: caution while quietly costing four points of recall.
#:
#: On the labelled pairs the two classes separate completely: the lowest
#: duplicate scores 0.5112 and the highest non-duplicate 0.4327. 0.47 is the
#: midpoint of that margin rather than either edge, which is the least
#: overfitted point available. The margin is 0.078 wide on twenty pairs, so
#: treat it as fragile: it is a calibration, not a discovery.
SEMANTIC_THRESHOLD = 0.47


def semantic_config(embedder: Embedder, threshold: float = SEMANTIC_THRESHOLD):
    """A dedupe config whose threshold matches its similarity function.

    Constructed here rather than in `dedupe.py` because that module knows
    nothing about embeddings and should not have to.
    """
    from sightline.dedupe import DedupeConfig

    return DedupeConfig(
        threshold=threshold, similarity_fn=EmbeddingSimilarity(embedder=embedder)
    )


def texts_in(path: str | Path) -> list[str]:
    """Every report text a golden set will ask the embedder to encode.

    Same discipline as the retrieval caches: the set of texts is derived from
    the committed data rather than accumulated at runtime, so a fixture edit
    produces a loud cache miss instead of a silently re-encoded vector.
    """
    import json

    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            record = json.loads(stripped)
            for side in ("a", "b"):
                text = record.get("inputs", {}).get(side, {}).get("text")
                if text:
                    texts.append(text)
    return texts
