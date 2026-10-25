"""
helpdesk/agents/single_agent.py - Single-agent baseline for comparison.
Monolithic LlmAgent with all tools and unified instruction.
"""

from google.adk.agents import LlmAgent
from helpdesk.config import DEFAULT_CHAT_MODEL
from helpdesk.models import make_model
from helpdesk.prompts import SINGLE_AGENT_INSTRUCTION
from helpdesk.tools import (
    search_documents,
    get_course_info,
    get_slot_timing,
    list_courses,
    get_instructor_email,
    draft_email,
    log_unanswered
)
from helpdesk.guardrails import before_agent_scope_guardrail, after_model_citation_guardrail


def build_single_agent(model_name: str = DEFAULT_CHAT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="SingleHelpdeskAgent",
        description="Single-agent baseline capable of answering all policy, schedule, and email requests.",
        model=make_model(model_name),
        instruction=SINGLE_AGENT_INSTRUCTION,
        tools=[
            search_documents,
            get_course_info,
            get_slot_timing,
            list_courses,
            get_instructor_email,
            draft_email,
            log_unanswered
        ],
        before_agent_callback=before_agent_scope_guardrail,
        after_model_callback=after_model_citation_guardrail
    )
