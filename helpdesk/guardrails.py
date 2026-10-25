"""
helpdesk/guardrails.py - Subsumption / Governor guardrails for the Helpdesk MAS.

Implemented as ADK callbacks:
- before_agent_scope_guardrail: cheap keyword filter that short-circuits clearly off-topic
  questions (refusal + escalation log) before any LLM call. It is only a fast path: questions it
  does not catch are refused by the agents themselves when retrieval finds nothing relevant.
- after_model_citation_guardrail: checks the final answer's citations against the documents the
  tools actually read in this session (state["retrieved"]).
    * cites a document that was never retrieved -> fabricated, replaced by a refusal and logged
    * no tool was used at all                    -> ungrounded, replaced by a refusal and logged
    * tools were used but the model forgot to cite -> the retrieved sources are appended and
      state["citation_autofilled"] is set, so the evaluation can count it
Both are toggleable for the ablation study.
"""

import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from helpdesk.citations import extract_cited_doc_ids
from helpdesk.tools.escalation import log_unanswered

# Toggle flags for ablation experiments
ENABLE_SCOPE_GUARDRAIL = True
ENABLE_CITATION_GUARDRAIL = True

# Only unambiguous non-academic topics. Words such as "climate", "train", "match" or "market"
# also occur in course titles and academic questions, so they are deliberately not listed.
OFF_TOPIC_KEYWORDS = [
    "dining", "mess menu", "mess food", "canteen", "cafeteria", "restaurant", "food court",
    "lunch", "dinner", "breakfast",
    "cab", "taxi", "uber", "ola cab", "flight", "airport", "railway station", "bus stand",
    "weather", "forecast",
    "cricket", "football", "ipl", "world cup",
    "movie", "movies", "cinema", "netflix", "shopping", "mall",
]
COURSE_CODE = re.compile(r"\b[A-Z]{2,4}\d{4}\b")

REFUSAL_MESSAGE = (
    "I don't know. I can only answer questions about IIT Jodhpur academic programmes, rules and "
    "regulations, course slots and timetable, and draft e-mails to instructors. I have logged your "
    "question so that someone at the academic office can follow up."
)

UNGROUNDED_MESSAGE = (
    "I don't know. I could not verify an answer to this in the official IIT Jodhpur documents, so I "
    "have logged your question for the academic office to answer."
)

REFUSAL_MARKERS = (
    "i don't know", "i do not know", "i cannot find", "i could not find", "i couldn't find",
    "not in the official", "not found in", "outside my scope", "out of scope", "cannot answer",
    "no relevant", "logged your question", "has been logged",
)


def is_out_of_scope(text: str) -> bool:
    """True if the question is clearly about a non-academic topic."""
    if COURSE_CODE.search(text):
        return False
    t = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in OFF_TOPIC_KEYWORDS)


def is_refusal_text(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return any(m in t for m in REFUSAL_MARKERS)


def get_latest_user_text(context: CallbackContext) -> str:
    """Text of the user message that started this invocation."""
    content = getattr(context, "user_content", None)
    if content and content.parts:
        return " ".join(p.text for p in content.parts if getattr(p, "text", None)).strip()
    return ""


def before_agent_scope_guardrail(callback_context: CallbackContext) -> Optional[types.Content]:
    """Before-agent callback: refuse and log clearly off-topic questions without calling the LLM."""
    if not ENABLE_SCOPE_GUARDRAIL:
        return None
    user_text = get_latest_user_text(callback_context)
    if not user_text or not is_out_of_scope(user_text):
        return None

    log_unanswered(question=user_text, reason="out_of_scope_guardrail", agent="guardrail_governor")
    callback_context.state["refused"] = True
    callback_context.state["refusal_reason"] = "out_of_scope_guardrail"
    return types.Content(role="model", parts=[types.Part.from_text(text=REFUSAL_MESSAGE)])


def _response_text(llm_response: LlmResponse) -> str:
    content = llm_response.content
    if not content or not content.parts:
        return ""
    return " ".join(p.text for p in content.parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


def _has_function_call(llm_response: LlmResponse) -> bool:
    content = llm_response.content
    return bool(content and content.parts and any(getattr(p, "function_call", None) for p in content.parts))


def after_model_citation_guardrail(callback_context: CallbackContext,
                                   llm_response: LlmResponse) -> Optional[LlmResponse]:
    """After-model callback: verify that the final answer is grounded in retrieved documents."""
    if not ENABLE_CITATION_GUARDRAIL or _has_function_call(llm_response):
        return None

    text = _response_text(llm_response)
    if not text or is_refusal_text(text):
        return None

    state = callback_context.state
    retrieved = [d.upper() for d in state.get("retrieved", [])]
    if "pending_email" in state:
        return None  # e-mail draft: content comes from the student, the address from the tool

    cited = {d.upper() for d in extract_cited_doc_ids(text, state.get("source_map", {}))}
    fabricated = sorted(c for c in cited if c not in retrieved)

    if retrieved and not fabricated:
        if cited:
            return None
        # Grounded in tool output but uncited: attach the sources the tools actually read.
        state["citation_autofilled"] = True
        sources = ", ".join(f"[{d}]" for d in dict.fromkeys(retrieved))
        return LlmResponse(content=types.Content(
            role="model", parts=[types.Part.from_text(text=f"{text}\n\nSources: {sources}")]))

    reason = f"fabricated citation(s) {fabricated}" if fabricated else "answer without any retrieval"
    log_unanswered(question=get_latest_user_text(callback_context) or text[:200],
                   reason=f"ungrounded_response: {reason}", agent="citation_governor")
    state["refused"] = True
    state["refusal_reason"] = reason
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=UNGROUNDED_MESSAGE)]))

