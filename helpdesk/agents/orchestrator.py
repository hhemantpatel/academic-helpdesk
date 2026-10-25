"""
helpdesk/agents/orchestrator.py - Helpdesk Orchestrator Agent (MAS Root).
Delegates to Policy, Schedule, and Email agents; enforces refusal guardrails.
"""

from google.adk.agents import LlmAgent
from helpdesk.config import DEFAULT_CHAT_MODEL
from helpdesk.models import make_model
from helpdesk.prompts import ORCHESTRATOR_INSTRUCTION
from helpdesk.tools import log_unanswered
from helpdesk.guardrails import before_agent_scope_guardrail
from helpdesk.agents.policy_agent import build_policy_agent
from helpdesk.agents.schedule_agent import build_schedule_agent
from helpdesk.agents.email_agent import build_email_agent


def build_orchestrator(model_name: str = DEFAULT_CHAT_MODEL) -> LlmAgent:
    policy_agent = build_policy_agent(model_name=model_name)
    schedule_agent = build_schedule_agent(model_name=model_name)
    email_agent = build_email_agent(model_name=model_name)

    return LlmAgent(
        name="HelpdeskOrchestrator",
        description="Root academic helpdesk orchestrator. Routes questions to policy, schedule, or email agents.",
        model=make_model(model_name),
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=[log_unanswered],
        sub_agents=[policy_agent, schedule_agent, email_agent],
        before_agent_callback=before_agent_scope_guardrail
    )
