"""
helpdesk/agents/email_agent.py - Specialized Email Agent for Instructor Communication.
"""

from google.adk.agents import LlmAgent
from helpdesk.config import DEFAULT_CHAT_MODEL
from helpdesk.models import make_model
from helpdesk.prompts import EMAIL_AGENT_INSTRUCTION
from helpdesk.tools import get_instructor_email, draft_email, log_unanswered


def build_email_agent(model_name: str = DEFAULT_CHAT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="email_agent",
        description="Assists students in drafting formal emails to course instructors for leave, makeup exams, or inquiries. Never sends emails automatically.",
        model=make_model(model_name),
        instruction=EMAIL_AGENT_INSTRUCTION,
        tools=[get_instructor_email, draft_email, log_unanswered]
    )
