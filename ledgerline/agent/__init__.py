"""The agent graph: plan, retrieve, analyse, reconcile, finalize.

Imports are kept lazy at the module edge -- `state` and `router` are pure
Python and always available, while `graph` needs langgraph and `llm` needs the
Anthropic SDK. That split is what lets the routing suite run in an environment
that has neither.
"""

from ledgerline.agent.router import classify, plan_for
from ledgerline.agent.state import AgentState, Citation, Outcome, Route, initial_state

__all__ = [
    "AgentState",
    "Citation",
    "Outcome",
    "Route",
    "classify",
    "initial_state",
    "plan_for",
]
