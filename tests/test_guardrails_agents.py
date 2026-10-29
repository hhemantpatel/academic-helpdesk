"""
tests/test_guardrails_agents.py - Guardrail callbacks, routers, agent assembly and scoring.
No LLM needed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk import config, guardrails  # noqa: E402
from helpdesk.agent import get_agent  # noqa: E402
from helpdesk.guardrails import (  # noqa: E402
    REFUSAL_MESSAGE, UNGROUNDED_MESSAGE, after_model_citation_guardrail, before_agent_scope_guardrail,
    is_out_of_scope, is_refusal_text,
)
from main import keyword_route  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNANSWERED_LOG_FILE", tmp_path / "unanswered.jsonl")
    monkeypatch.setattr(guardrails, "ENABLE_SCOPE_GUARDRAIL", True)
    monkeypatch.setattr(guardrails, "ENABLE_CITATION_GUARDRAIL", True)


def callback_ctx(question="q", state=None):
    content = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    return SimpleNamespace(user_content=content, state=dict(state or {}))


def model_text(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))


def test_scope_keywords():
    for q in ["Where is the best dining hall?", "Can you book a cab to the airport?",
              "What is the weather forecast in Jodhpur?", "Who won the cricket match yesterday?"]:
        assert is_out_of_scope(q), q
    # Academic questions containing words that used to be keywords must pass.
    for q in ["What slot is Introduction to Climate Change Economics in?",
              "Is there a training session on ERP course registration?",
              "What is the minimum attendance required for major exams?",
              "Write an email to the instructor of CSL7650 about lunch-hour office meeting"]:
        assert not is_out_of_scope(q), q


def test_before_agent_guardrail_short_circuits_and_logs():
    ctx = callback_ctx("What are the dining options on campus?")
    out = before_agent_scope_guardrail(ctx)
    assert out.parts[0].text == REFUSAL_MESSAGE
    assert ctx.state["refused"] is True
    assert "dining" in config.UNANSWERED_LOG_FILE.read_text()
    assert before_agent_scope_guardrail(callback_ctx("What is the slot of CSL7650?")) is None


def test_citation_guardrail_passes_grounded_answer():
    ctx = callback_ctx("slot?", {"retrieved": ["CLASS_ALLOC_2026"]})
    assert after_model_citation_guardrail(ctx, model_text("Slot E [CLASS_ALLOC_2026, p.3].")) is None


def test_citation_guardrail_blocks_fabricated_citation():
    ctx = callback_ctx("attendance?", {"retrieved": ["UG_REG_2019"]})
    out = after_model_citation_guardrail(ctx, model_text("You need 75% [ATTEND_POL_2024]."))
    assert out.content.parts[0].text == UNGROUNDED_MESSAGE
    assert "fabricated" in config.UNANSWERED_LOG_FILE.read_text()


def test_citation_guardrail_blocks_answer_without_retrieval():
    ctx = callback_ctx("attendance?", {})
    out = after_model_citation_guardrail(ctx, model_text("You need 75% attendance."))
    assert out.content.parts[0].text == UNGROUNDED_MESSAGE


def test_citation_guardrail_attaches_sources_when_model_forgets():
    ctx = callback_ctx("slot?", {"retrieved": ["CLASS_ALLOC_2026", "SLOT_SYS_2026"]})
    out = after_model_citation_guardrail(ctx, model_text("Autonomous Systems is in slot E."))
    assert out.content.parts[0].text.endswith("Sources: [CLASS_ALLOC_2026], [SLOT_SYS_2026]")
    assert ctx.state["citation_autofilled"] is True


def test_citation_guardrail_ignores_refusals_and_tool_calls():
    ctx = callback_ctx("pool?", {})
    assert after_model_citation_guardrail(ctx, model_text("I don't know; I have logged your question.")) is None
    call = LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="search_documents", args={"query": "x"}))]))
    assert after_model_citation_guardrail(ctx, call) is None


def test_refusal_detection():
    assert is_refusal_text(REFUSAL_MESSAGE) and is_refusal_text(UNGROUNDED_MESSAGE)
    assert is_refusal_text("I don’t know the answer.")
    assert not is_refusal_text("Slot E meets on Tuesday [SLOT_SYS_2026].")


def test_keyword_router():
    assert keyword_route("What is the slot timing for Slot E?") == "schedule_agent"
    assert keyword_route("Who teaches CSL7650?") == "schedule_agent"
    assert keyword_route("How many 400 level electives are offered by Humanities?") == "schedule_agent"
    assert keyword_route("Please draft an email to my professor for leave") == "email_agent"
    assert keyword_route("What happens if I fail a course?") == "policy_agent"  # 'fail' is not a course code
    assert keyword_route("What are the details of the minor degree?") == "policy_agent"


def test_agent_graph_assembly():
    single, multi = get_agent("single"), get_agent("multi")
    assert single.name == "SingleHelpdeskAgent" and len(single.tools) == 7
    assert multi.name == "HelpdeskOrchestrator"
    assert [a.name for a in multi.sub_agents] == ["policy_agent", "schedule_agent", "email_agent"]
    tool_names = {a.name: {getattr(t, "__name__", str(t)) for t in a.tools} for a in multi.sub_agents}
    assert "draft_email" in tool_names["email_agent"] and "draft_email" not in tool_names["policy_agent"]


def test_scoring_rules():
    from eval.score import score_runs
    base = {"system": "multi", "model": "m", "router": "llm", "run_idx": 1, "error": "", "latency": 1, "llm_calls": 2,
            "citation_autofilled": False}
    runs = pd.DataFrame([
        {**base, "question_id": "Q08", "answer": "It is in slot E [CLASS_ALLOC_2026, p.3].",
         "route": '["HelpdeskOrchestrator", "schedule_agent"]', "tools_called": '[["get_course_info", {}]]',
         "cited_doc_ids": '["CLASS_ALLOC_2026"]', "retrieved": '["CLASS_ALLOC_2026"]'},
        {**base, "question_id": "Q21", "answer": REFUSAL_MESSAGE, "route": '["HelpdeskOrchestrator"]',
         "tools_called": "[]", "cited_doc_ids": "[]", "retrieved": "[]"},
        {**base, "question_id": "Q16", "answer": "CSL7570 was not found; please confirm the course code.",
         "route": '["HelpdeskOrchestrator", "email_agent"]', "tools_called": '[["get_instructor_email", {}]]',
         "cited_doc_ids": "[]", "retrieved": "[]"},
    ])
    scored, summary = score_runs(runs)
    assert scored["is_correct"].tolist() == [True, True, True]
    assert scored["routing_correct"].tolist() == [True, True, True]
    s = summary["multi|m|llm"]
    assert s["refusal_precision"] == 1.0 and s["correct_source_rate"] == 1.0 and s["fabricated_rate"] == 0.0
