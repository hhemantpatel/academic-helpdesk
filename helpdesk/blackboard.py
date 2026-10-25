"""
helpdesk/blackboard.py - Shared session state ("blackboard") helpers used by the tools.
"""

from typing import Optional


def note_sources(tool_context, *doc_ids: str, source_map: Optional[dict] = None):
    """
    Record the documents a tool actually read in the session state, so the citation guardrail
    can tell real citations from fabricated ones and the evaluation can check groundedness.
    """
    if tool_context is None:
        return
    retrieved = list(tool_context.state.get("retrieved", []))
    for d in doc_ids:
        if d and d not in retrieved:
            retrieved.append(d)
    tool_context.state["retrieved"] = retrieved
    if source_map:
        merged = dict(tool_context.state.get("source_map", {}))
        merged.update(source_map)
        tool_context.state["source_map"] = merged
