"""Deciding what kind of question this is.

Routing is the first thing the graph does and the cheapest thing to get wrong:
send a figure question to the narrative analyst and it will happily paraphrase
a sentence containing a different number, with a citation, and look right.

The rules key on **question form**, not topic. The first version of this file
matched topic words -- "margin", "risk", "concentrated" -- and scored 0.53 with
a kappa of 0.11, barely above chance. The failures all had the same shape:
"What was ... operating margin?" and "Why did gross margin decline?" share
every content word and are different questions. What separates them is the
interrogative and what it governs, so that is what gets matched.

**Read this before trusting the number.** The routing suite has fifteen
examples and these rules were revised while looking at which ones missed. An
accuracy of 1.000 on that set therefore measures fit, not generalisation --
fifteen questions cannot validate a classifier, and any rule set can be bent to
fit them. It is reported as a ceiling and gated below itself. The point of this
file is to be the cheap, inspectable control that an LLM router has to beat on
a set large enough to mean something; it is not the finished router.
"""

from __future__ import annotations

import re

from ledgerline.agent.state import Route

# --------------------------------------------------------------------------
# cross-modal: reconciling or assessing a position across sources
# --------------------------------------------------------------------------
# Note what is *not* here: "on the call", "in the transcript", "management
# said". Those name a source, and naming a source is not the same as comparing
# two. Including them sent every straightforward transcript question down the
# reconciliation path, which is both slower and more likely to invent a
# disagreement than to find one.
_CROSS_MODAL = (
    "consistent with",
    "reconcile",
    "contradict",
    "square with",
    "compare",
    "differ from",
    "versus",
    "match the",
)

#: Asking whether someone *holds a position* rather than what they said. A
#: belief cannot be quoted from one sentence -- it is assessed by checking a
#: stated expectation against what was said elsewhere, which is the
#: reconciliation path by definition.
_BELIEF = re.compile(
    r"\b(do|does|did)\s+(management|the\s+company|they|he|she)\s+"
    r"(believe|think|expect|anticipate|maintain)\b",
    re.I,
)

# --------------------------------------------------------------------------
# narrative overrides: forms that ask for an explanation, whatever else they
# mention
# --------------------------------------------------------------------------

#: "Why ..." as the *main* clause. Deliberately anchored: "What was the
#: effective tax rate and why did it change?" is a figure question with a
#: subordinate explanation, and routing it to prose loses the number.
_WHY_OPENER = re.compile(r"^\s*(why|what\s+drove|what\s+caused)\b", re.I)

# A partitive rule once lived here: "how much of X ..." routed to narrative, on
# the theory that "How much of the announced pricing is holding?" asks for a
# judgement rather than a figure. It fixed that one question on the retrieval
# set and immediately broke "How much of the revolving facility remains
# available?" on the numeric set, which wants a dollar amount. The distinction
# is between "is holding" and "remains available" -- two verbs -- and a rule
# that has to know which verbs are assessments is a rule fitted to the examples
# in front of it. It is gone, the retrieval-set question it fixed is now a
# documented miss, and `ledgerline.routing_heldout` exists to catch the next
# one of these before it ships.

#: Reporting what a source said. Answered by quoting, not by resolving a cell.
_REPORTED_SPEECH = re.compile(
    r"\b(what|who)\s+(did|does)\b.*\b(say|state|tell|describe|note)\b|"
    r"\b(say|said)\s+about\b",
    re.I,
)

# --------------------------------------------------------------------------
# numeric: a request for a quantity
# --------------------------------------------------------------------------

#: Interrogatives that can only introduce a quantity. "How <degree-adjective>"
#: is included because "How concentrated is the customer base?" is answered
#: with a percentage, not a paragraph.
_QUANTITY_OPENER = re.compile(
    r"^\s*(how\s+(much|many|large|small|high|low|concentrated|leveraged)|"
    r"what\s+(proportion|share|percentage|fraction|portion|number|amount)|"
    r"what\s+were\s+total)\b",
    re.I,
)

#: "What ..." only asks for a figure when it governs something measurable, so
#: the opener is paired with a measure noun rather than trusted on its own.
#: Deliberately not restricted to "what was/were/is/are": "What accrual was
#: recorded for the class action?" puts the measure noun before the copula and
#: is no less a figure question for it.
_WHAT_OPENER = re.compile(r"^\s*what\b", re.I)

_MEASURE_NOUNS = (
    "revenue",
    "margin",
    "rate",
    "expenditure",
    "capex",
    "amount",
    "balance",
    "headcount",
    "employee",
    "proportion",
    "guidance",
    "accrued",
    "accrual",
    "income",
    "expense",
    "cost",
    "cash",
    "debt",
    "backlog",
    "price",
    "total",
    "growth",
    "per share",
)

#: A figure written into the question itself.
_FIGURE_SHAPE = re.compile(r"[$€£]|\b\d+(\.\d+)?\s*(%|percent|million|billion)\b", re.I)

_NARRATIVE_CUES = (
    "explain",
    "describe",
    "outlook",
    "risk",
    "temporary",
    "characterise",
    "characterize",
)


def classify(question: str) -> Route:
    """Route a question. Never raises.

    Ordered, not scored, because the categories are not symmetric. A question
    that reconciles two sources is cross-modal even when it also names a
    figure: the comparison is the hard part and it decides which analysts run.
    """
    text = question.strip()
    lowered = text.lower()

    if _BELIEF.search(text) or any(cue in lowered for cue in _CROSS_MODAL):
        return Route.CROSS_MODAL

    # Explanation forms win over quantity cues. Getting a paragraph when you
    # wanted a number is a bad answer; getting a number when you asked why is
    # a non-answer.
    if _WHY_OPENER.search(text) or _REPORTED_SPEECH.search(text):
        return Route.NARRATIVE

    if _QUANTITY_OPENER.search(text):
        return Route.NUMERIC
    if _WHAT_OPENER.search(text) and any(noun in lowered for noun in _MEASURE_NOUNS):
        return Route.NUMERIC
    if _FIGURE_SHAPE.search(text):
        return Route.NUMERIC

    if any(cue in lowered for cue in _NARRATIVE_CUES):
        return Route.NARRATIVE

    # Narrative is the safe default rather than numeric: the narrative path
    # retrieves, reads and can decline, while the numeric path resolves a cell,
    # and returning the wrong figure confidently is the worst failure this
    # system has.
    return Route.NARRATIVE


def plan_for(route: Route) -> list[str]:
    """Which analysts run, in order. The graph's conditional edges follow this.

    Cross-modal runs both because the whole point is disagreement between a
    table and what someone said out loud, and you cannot detect that with one
    of them.
    """
    return {
        Route.NUMERIC: ["table_analyst"],
        Route.NARRATIVE: ["narrative_analyst"],
        Route.CROSS_MODAL: ["table_analyst", "narrative_analyst"],
    }[route]
