#!/usr/bin/env python3
"""
main.py - CLI for the IIT Jodhpur Multi-Agent Academic Helpdesk.

Usage:
  python main.py ingest [--rebuild]                 extract tables and build the Chroma index
  python main.py ask "question" [--system multi|single] [--router llm|keyword] [--model qwen3:8b] [-v]
  python main.py chat [--system multi|single] [--model qwen3:8b]
  python main.py eval [--quick] [--system ...] [--model ...] [--router ...] [--runs 3]
  python main.py web                                launch the ADK web UI on the multi-agent system

Routers (multi-agent only):
  llm      the HelpdeskOrchestrator LLM delegates with transfer_to_agent (default; the real MAS)
  keyword  deterministic keyword pre-router straight to one sub-agent (ablation / fallback)
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from helpdesk import config  # noqa: E402
from helpdesk.agent import get_agent  # noqa: E402
from helpdesk.agents.email_agent import build_email_agent  # noqa: E402
from helpdesk.agents.policy_agent import build_policy_agent  # noqa: E402
from helpdesk.agents.schedule_agent import build_schedule_agent  # noqa: E402
from helpdesk.citations import extract_cited_doc_ids, extract_cited_references  # noqa: E402
from helpdesk.guardrails import REFUSAL_MESSAGE, is_out_of_scope  # noqa: E402
from helpdesk.tools.escalation import log_unanswered  # noqa: E402

COURSE_CODE = re.compile(r"\b[A-Z]{2,4}\d{4}\b")


def full_model_name(model: str) -> str:
    return model if "/" in model else f"ollama_chat/{model}"


def ensure_single_model(model_name: str):
    """Unload other Ollama models so only one 8B model sits in RAM (16 GB laptops)."""
    target = model_name.split("/", 1)[-1]
    base = config.OLLAMA_API_BASE.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=2.0) as resp:
            loaded = json.loads(resp.read().decode()).get("models", [])
        for m in loaded:
            name = m.get("name", "")
            if name and target not in name and "embed" not in name:
                req = urllib.request.Request(f"{base}/api/generate",
                                             data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5.0)
    except Exception:
        pass  # Ollama not reachable: the run itself will report the error


def keyword_route(query: str) -> str:
    """Deterministic pre-router used for the 'keyword' router ablation."""
    q = query.lower()
    if re.search(r"\b(e-?mail|draft|write (a|an) (mail|letter))\b", q):
        return "email_agent"
    if COURSE_CODE.search(query) or re.search(
            r"\b(slot|classroom|room|timing|timetable|who teaches|instructor of)\b|how many\b.*\b(courses|electives)\b", q):
        return "schedule_agent"
    return "policy_agent"


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def execute_query(query: str, system: str = "multi", model: str = config.DEFAULT_CHAT_MODEL,
                        router: str = "llm", verbose: bool = False) -> dict:
    """Run one question through the helpdesk and return the answer plus everything needed for scoring."""
    t0 = time.time()
    model = full_model_name(model)
    route, tools_called = [], []

    if system == "single":
        agent = get_agent("single", model)
    elif router == "llm":
        agent = get_agent("multi", model)
    else:
        if is_out_of_scope(query):  # the keyword router has no orchestrator, so apply the governor here
            log_unanswered(question=query, reason="out_of_scope_guardrail", agent="guardrail_governor")
            return {"query": query, "answer": REFUSAL_MESSAGE, "route": ["guardrail_governor"],
                    "tools_called": [], "citations": [], "cited_doc_ids": [], "retrieved": [],
                    "citation_autofilled": False, "llm_calls": 0, "latency": time.time() - t0, "trace_file": ""}
        target = keyword_route(query)
        route.append("keyword_router")
        agent = {"schedule_agent": build_schedule_agent, "email_agent": build_email_agent}.get(
            target, build_policy_agent)(model_name=model)

    ensure_single_model(model)
    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="student",
                                                          session_id=uuid.uuid4().hex)
    message = types.Content(role="user", parts=[types.Part.from_text(text=query)])

    events, final_text, llm_calls = [], "", 0
    async for event in runner.run_async(user_id="student", session_id=session.id, new_message=message):
        events.append(event)
        if event.author and event.author != "user" and event.author not in route:
            route.append(event.author)
        parts = event.content.parts if event.content and event.content.parts else []
        if any(p.function_call or (p.text and not p.thought) for p in parts):
            llm_calls += 1
        for p in parts:
            if p.function_call:
                tools_called.append((p.function_call.name, dict(p.function_call.args or {})))
                if verbose:
                    print(f"[{event.author}] -> {p.function_call.name}({dict(p.function_call.args or {})})")
            elif p.function_response and verbose:
                print(f"[{event.author}] <- {p.function_response.name}")
        if event.is_final_response():
            text = " ".join(p.text for p in parts if p.text and not p.thought).strip()
            if text:
                final_text = text
                if verbose:
                    print(f"[{event.author}] final: {text[:120]}")

    state = (await runner.session_service.get_session(app_name=runner.app_name, user_id="student",
                                                      session_id=session.id)).state
    answer = strip_thinking(final_text)

    trace_file = config.TRACES_DIR / f"trace_{int(time.time() * 1000)}_{session.id[:8]}.jsonl"
    with open(trace_file, "w", encoding="utf-8") as tf:
        tf.write(json.dumps({"query": query, "system": system, "router": router, "model": model}) + "\n")
        for ev in events:
            tf.write(ev.model_dump_json(exclude_none=True) + "\n")
        tf.write(json.dumps({"final_state": dict(state)}, default=str) + "\n")

    return {
        "query": query,
        "answer": answer,
        "route": route,
        "tools_called": tools_called,
        "citations": extract_cited_references(answer),
        "cited_doc_ids": sorted(extract_cited_doc_ids(answer, state.get("source_map", {}))),
        "retrieved": list(state.get("retrieved", [])),
        "citation_autofilled": bool(state.get("citation_autofilled", False)),
        "llm_calls": llm_calls,
        "latency": time.time() - t0,
        "trace_file": str(trace_file),
    }


def cli_ask(args):
    model = full_model_name(args.model)
    print(f"\n--- [{args.system.upper()}] model={model} router={args.router} ---\nQ: {args.query}\n")
    r = asyncio.run(execute_query(args.query, args.system, model, args.router, args.verbose))
    print("=" * 60 + f"\n{r['answer']}\n" + "=" * 60)
    print(f"Route: {' -> '.join(r['route'])}")
    print(f"Tools: {[t[0] for t in r['tools_called']]}")
    print(f"Cited: {r['cited_doc_ids']}  Retrieved: {r['retrieved']}  (auto-attached: {r['citation_autofilled']})")
    print(f"Latency: {r['latency']:.1f}s, model turns: {r['llm_calls']}, trace: {r['trace_file']}")


def cli_chat(args):
    model = full_model_name(args.model)
    print(f"--- Helpdesk chat [{args.system}] ({model}); 'exit' to quit ---")
    while True:
        try:
            q = input("\nStudent > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            r = asyncio.run(execute_query(q, args.system, model, args.router, args.verbose))
            print(f"\nHelpdesk [{r['route'][-1] if r['route'] else 'system'}] >\n{r['answer']}")


def cli_ingest(args):
    from ingest.build_index import build_index
    from ingest.extract_tables import main as extract_tables
    extract_tables()
    build_index(rebuild=args.rebuild)


def cli_eval(args):
    import eval.run_eval as reval
    asyncio.run(reval.run_evaluation(quick=args.quick, system_filter=args.system, model_filter=args.model,
                                     router=args.router, num_runs=args.runs))


def cli_web(args):
    # adk web takes a directory; pointing at the package folder serves helpdesk/agent.py:root_agent
    subprocess.run(["adk", "web", str(Path(__file__).resolve().parent / "helpdesk")], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="extract tables and build the Chroma index")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cli_ingest)

    for name, fn in (("ask", cli_ask), ("chat", cli_chat)):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("query")
        p.add_argument("--system", choices=["single", "multi"], default="multi")
        p.add_argument("--router", choices=["llm", "keyword"], default="llm")
        p.add_argument("--model", default="qwen3:8b")
        p.add_argument("--verbose", "-v", action="store_true")
        p.set_defaults(func=fn)

    p = sub.add_parser("eval", help="run the evaluation grid")
    p.add_argument("--quick", action="store_true", help="first 5 questions, 1 run")
    p.add_argument("--system", choices=["single", "multi", "both"], default="both")
    p.add_argument("--model", default="both", help="qwen3:8b, llama3.1:8b, both, or any Ollama model")
    p.add_argument("--router", choices=["llm", "keyword"], default="llm")
    p.add_argument("--runs", type=int, default=3)
    p.set_defaults(func=cli_eval)

    p = sub.add_parser("web", help="launch the ADK web UI")
    p.set_defaults(func=cli_web)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
