"""
helpdesk/tools/escalation.py - Human escalation log for out-of-scope or unanswerable questions.
Normative enforcement: the question is logged to unanswered.jsonl instead of guessing an answer.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from google.adk.tools.tool_context import ToolContext

from helpdesk import config


def log_unanswered(
    question: str,
    reason: str,
    agent: str = "orchestrator",
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Log a question the helpdesk cannot answer so a human can follow up. Call this when the question
    is outside academic matters (dining, travel, sports, ...) or when no official document covers it.

    Args:
        question: The student's question, verbatim.
        reason: Short reason, e.g. 'out_of_scope', 'no_relevant_docs', 'course_not_found'.
        agent: The agent logging the question, e.g. 'orchestrator', 'policy_agent'.

    Returns:
        Confirmation that the question was logged for human review.
    """
    session_id = tool_context.session.id if tool_context is not None else "no_session"
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question.strip(),
        "reason": reason.strip(),
        "agent": agent.strip(),
        "session_id": session_id,
    }
    # Read the path at call time so tests can redirect the log.
    log_file = config.UNANSWERED_LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    if tool_context is not None:
        tool_context.state["escalated"] = True

    return {
        "status": "escalated_and_logged",
        "reason": reason,
        "message": "The question has been logged for a human at the academic office to answer.",
    }
