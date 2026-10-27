"""
tests/test_tools.py - Tools, table extraction and citation parsing. No LLM needed; the retrieval
test needs Ollama + a built index and is skipped otherwise.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk import config  # noqa: E402
from helpdesk.citations import extract_cited_doc_ids, extract_cited_references, verify_groundedness  # noqa: E402
from helpdesk.tools.courses import get_course_info, get_courses_df, list_courses  # noqa: E402
from helpdesk.tools.email_draft import draft_email  # noqa: E402
from helpdesk.tools.escalation import log_unanswered  # noqa: E402
from helpdesk.tools.faculty import get_instructor_email  # noqa: E402
from helpdesk.tools.slots import get_slot_timing  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Never write test entries into the real escalation log or drafts folder."""
    monkeypatch.setattr(config, "UNANSWERED_LOG_FILE", tmp_path / "unanswered.jsonl")
    monkeypatch.setattr(config, "DRAFTS_DIR", tmp_path / "drafts")


def fake_tool_context():
    return SimpleNamespace(state={}, session=SimpleNamespace(id="test-session"))


def test_course_lookup_by_code_and_title():
    by_code = get_course_info("CSL7650")
    by_title = get_course_info("Autonomous Systems")
    assert by_code["status"] == by_title["status"] == "success"
    assert by_code["code"] == by_title["code"] == "CSL7650"
    assert (by_code["slot"], by_code["classroom"], by_code["instructor"]) == ("E", "LHC 106", "Gaurav Harit")
    assert by_code["cite_as"] == "[CLASS_ALLOC_2026, p.3]"


def test_unknown_course_is_not_invented():
    # CSL7570 is not in the AY 2026-27 Sem I allocation; it must not resolve to a look-alike.
    assert get_course_info("CSL7570")["status"] == "not_found"
    assert get_instructor_email("CSL7570")["status"] == "not_found"
    assert get_course_info("Underwater Basket Weaving")["status"] == "not_found"


def test_courses_table_is_clean():
    df = get_courses_df()
    assert len(df) > 600
    assert "CSL7570" not in set(df["code"])
    # Overflowing cells used to interleave characters ("Himanshu KumAamrit"); allow a few residues.
    garbled = df[df["instructor"].str.contains(r"\d|[a-z][A-Z][a-z]", regex=True)]
    assert len(garbled) <= 3
    eel = df[df["code"] == "EEL2010"].iloc[0]
    assert eel["instructor"] == "Himanshu Kumar"
    assert "Manish Narwaria" in eel["co_instructor"]


def test_humanities_400_level_electives_count():
    res = list_courses(department="Humanities", level=400, elective=True)
    assert res["count"] == 2
    assert {c["code"] for c in res["courses"]} == {"LAL4070", "LAL4850"}


def test_list_courses_counts_distinct_codes():
    res = list_courses(department="Computer Science")
    codes = [c["code"] for c in res["courses"]]
    assert len(codes) == len(set(codes))
    assert res["count"] >= 20


def test_list_courses_treats_string_null_as_unset():
    # Llama sends 'null' for arguments it leaves out; this used to crash on int('null').
    res = list_courses(department="Computer Science", level="null", elective="none", slot="")
    assert res["count"] == list_courses(department="Computer Science")["count"]


def test_slot_timing():
    res = get_slot_timing("E")
    assert res["status"] == "success"
    assert {(t["day"], t["start"], t["end"]) for t in res["timings"]} == {
        ("Tuesday", "11:00", "11:50"), ("Thursday", "11:00", "11:50"), ("Friday", "11:00", "11:50")}
    assert get_slot_timing("Slot A")["status"] == "success"
    assert get_slot_timing("Z9")["status"] == "not_found"


def test_tools_record_sources_in_session_state():
    ctx = fake_tool_context()
    get_course_info("CSL7650", tool_context=ctx)
    get_slot_timing("E", tool_context=ctx)
    assert ctx.state["retrieved"] == ["CLASS_ALLOC_2026", "SLOT_SYS_2026"]


def test_instructor_email_is_dummy():
    res = get_instructor_email("CSL7650")
    assert res["status"] == "success"
    assert res["instructor"] == "Gaurav Harit"
    assert res["email"] == "gaurav.harit@iitj-dummy.ac.in"


def test_email_draft_saved_not_sent_and_adds_nothing():
    ctx = fake_tool_context()
    res = draft_email(to="gaurav.harit@iitj-dummy.ac.in", course_code="csl7650", purpose="leave",
                      details="I need leave on 14 and 15 October due to fever.", instructor_name="Gaurav Harit",
                      tool_context=ctx)
    assert res["status"] == "draft_created"
    content = Path(res["saved_path"]).read_text(encoding="utf-8")
    assert Path(res["saved_path"]).parent == config.DRAFTS_DIR
    assert "NOT SENT" in content and "CSL7650" in content and "14 and 15 October" in content
    # The template must not claim things the student never said.
    assert "attached" not in content.lower() and "certificate" not in content.lower()
    assert ctx.state["pending_email"]["course"] == "CSL7650"


def test_escalation_log_goes_to_configured_file():
    res = log_unanswered("Where is the dining hall?", "out_of_scope", "orchestrator", tool_context=fake_tool_context())
    assert res["status"] == "escalated_and_logged"
    entry = json.loads(config.UNANSWERED_LOG_FILE.read_text().splitlines()[-1])
    assert entry["question"] == "Where is the dining hall?" and entry["session_id"] == "test-session"


def test_citation_parsing():
    text = "See [UG_REG_2019, p.12; ATTEND_POL_2024] and [FACULTY_DIRECTORY, CLASS_ALLOC_2026], [S1] [not a cite]."
    assert extract_cited_references(text) == ["UG_REG_2019, p.12", "ATTEND_POL_2024", "FACULTY_DIRECTORY",
                                              "CLASS_ALLOC_2026", "S1"]
    assert extract_cited_doc_ids(text, {"S1": "PG_REG_2022"}) == {
        "UG_REG_2019", "ATTEND_POL_2024", "FACULTY_DIRECTORY", "CLASS_ALLOC_2026", "PG_REG_2022"}


def test_groundedness_requires_exact_retrieved_ids():
    retrieved = {"UG_REG_2019", "ATTEND_POL_2024"}
    assert verify_groundedness("Rule [UG_REG_2019, p.12] [ATTEND_POL_2024].", retrieved)[0]
    assert not verify_groundedness("Attendance is 75%.", retrieved)[0]
    assert not verify_groundedness("Per [PG_REG_2022, p.3].", retrieved)[0]
    assert not verify_groundedness("Per [UG_REG].", retrieved)[0]  # no substring matching


def _ollama_index_ready():
    try:
        from helpdesk.tools.rag import get_chroma_collection, get_ollama_client
        get_ollama_client().list()
        return get_chroma_collection().count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_index_ready(), reason="needs Ollama and a built index")
def test_retrieval_probes():
    from helpdesk.tools.rag import search_documents

    ctx = fake_tool_context()
    res = search_documents("What is the attendance requirement to sit in the Major exam?", tool_context=ctx)
    assert res["status"] == "success"
    assert "ATTEND_POL_2024" in [r["doc_id"] for r in res["results"]]  # the 2024 circular (OCR) is found
    assert ctx.state["retrieved"] and set(ctx.state["source_map"].values()) <= set(ctx.state["retrieved"])

    res = search_documents("weightage of the major examination")
    assert res["results"][0]["doc_id"] == "EXAM_POL_2024"

    assert search_documents("Who won the cricket world cup?")["status"] == "no_relevant_docs"


def test_supersession_puts_newer_circular_first():
    from helpdesk.tools.rag import apply_supersession
    results = [{"doc_id": "UG_REG_2019", "distance": 0.20}, {"doc_id": "BTECH_CSE_CURR", "distance": 0.22},
               {"doc_id": "ATTEND_POL_2024", "distance": 0.25}]
    out = apply_supersession(results)
    assert out[0]["doc_id"] == "ATTEND_POL_2024"
    assert out[-1]["doc_id"] == "UG_REG_2019" and out[-1]["superseded_by"] == ["ATTEND_POL_2024"]
    assert "superseded_by" not in out[1]
    # Without the newer circular nothing is marked.
    assert all("note" not in r for r in apply_supersession([{"doc_id": "UG_REG_2019", "distance": 0.2}]))
