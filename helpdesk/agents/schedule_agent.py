"""
helpdesk/agents/schedule_agent.py - Specialized Schedule Agent for Slots, Courses, Timetables.
"""

from google.adk.agents import LlmAgent
from helpdesk.config import DEFAULT_CHAT_MODEL
from helpdesk.models import make_model
from helpdesk.prompts import SCHEDULE_AGENT_INSTRUCTION
from helpdesk.tools import get_course_info, get_slot_timing, list_courses, search_documents, log_unanswered
from helpdesk.guardrails import after_model_citation_guardrail


def build_schedule_agent(model_name: str = DEFAULT_CHAT_MODEL) -> LlmAgent:
    return LlmAgent(
        name="schedule_agent",
        description="Handles questions regarding course timetable slots, lecture/lab timings, course information, instructors, classrooms, and course counts.",
        model=make_model(model_name),
        instruction=SCHEDULE_AGENT_INSTRUCTION,
        tools=[get_course_info, get_slot_timing, list_courses, search_documents, log_unanswered],
        after_model_callback=after_model_citation_guardrail
    )
