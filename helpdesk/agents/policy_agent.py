"""
helpdesk/agents/policy_agent.py - Specialized Policy Agent for Academic Regulations.
"""

from google.adk.agents import LlmAgent
from helpdesk.config import DEFAULT_CHAT_MODEL
from helpdesk.models import make_model
from helpdesk.prompts import POLICY_AGENT_INSTRUCTION
from helpdesk.tools import search_documents, log_unanswered
from helpdesk.guardrails import after_model_citation_guardrail


def build_policy_agent(model_name: str = DEFAULT_CHAT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="policy_agent",
        description="Handles questions regarding institute academic regulations, B.Tech/M.Tech/Ph.D rules, attendance, grading, makeup exams, and circulars.",
        model=make_model(model_name),
        instruction=POLICY_AGENT_INSTRUCTION,
        tools=[search_documents, log_unanswered],
        after_model_callback=after_model_citation_guardrail
    )
