"""
helpdesk/agent.py - Main entry point exposing root_agent for ADK web and CLI runners.
"""

import os
from helpdesk.config import DEFAULT_CHAT_MODEL
from helpdesk.agents.orchestrator import build_orchestrator
from helpdesk.agents.single_agent import build_single_agent

# Expose default root_agent for adk web / adk run
root_agent = build_orchestrator(model_name=DEFAULT_CHAT_MODEL)


def get_agent(system: str = "multi", model_name: str = DEFAULT_CHAT_MODEL):
    """
    Factory to retrieve either the multi-agent orchestrator or single-agent baseline.
    """
    if system.lower() == "single":
        return build_single_agent(model_name=model_name)
    return build_orchestrator(model_name=model_name)
