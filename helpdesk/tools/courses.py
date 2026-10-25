"""
helpdesk/tools/courses.py - Deterministic course lookup and filtering tools over courses.csv.
"""

from typing import Any, Dict, Optional

import pandas as pd
from google.adk.tools.tool_context import ToolContext
from rapidfuzz import fuzz, process

from helpdesk.config import TABLES_DIR
from helpdesk.blackboard import note_sources

SOURCE_TAG = "[CLASS_ALLOC_2026] Classroom Allocation AY 2026-27 Sem I"

# Department aliases students use -> regex over the official department name.
DEPT_ALIASES = {
    "hss": r"Liberal Arts|SoLA|Digital Humanities|Economics",
    "humanities": r"Liberal Arts|SoLA|Digital Humanities|Economics",
    "humanities and social sciences": r"Liberal Arts|SoLA|Digital Humanities|Economics",
    "liberal arts": r"Liberal Arts|SoLA",
    "sola": r"Liberal Arts|SoLA",
    "cse": r"Computer Science",
    "cs": r"Computer Science",
    "computer science": r"Computer Science",
    "aide": r"Artificial Intelligence",
    "ai": r"Artificial Intelligence",
    "ee": r"Electrical Engineering",
    "me": r"Mechanical Engineering",
    "maths": r"Mathematics",
    "math": r"Mathematics",
}

_courses_df = None


def get_courses_df() -> pd.DataFrame:
    global _courses_df
    if _courses_df is None:
        csv_path = TABLES_DIR / "courses.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing courses.csv at {csv_path}; run `python main.py ingest`")
        df = pd.read_csv(csv_path, keep_default_na=False, dtype={"code": str, "slot": str})
        for col in ("code", "title", "instructor", "co_instructor", "slot", "dept", "classroom"):
            df[col] = df[col].astype(str).str.strip()
        df["classroom"] = df["classroom"].replace("", "not listed")
        _courses_df = df
    return _courses_df


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "code": row["code"],
        "title": row["title"],
        "slot": row["slot"],
        "raw_slot": row["raw_slot"],
        "ltp": row["ltp"],
        "credit": row["credit"],
        "department": row["dept"],
        "instructor": row["instructor"],
        "co_instructor": row["co_instructor"],
        "classroom": row["classroom"],
        "registered": int(row["registered"] or 0),
        "level": int(row["level"]),
        "page": int(row["page"]),
    }


def get_course_info(code_or_name: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Get the slot, instructor, classroom, L-T-P and credits of a course offered this semester.

    Args:
        code_or_name: Course code (e.g. 'CSL7650') or course title (e.g. 'Autonomous Systems').

    Returns:
        The matching course offering(s) and the source document to cite.
    """
    df = get_courses_df()
    query = code_or_name.strip()

    match = df[df["code"].str.upper() == query.upper()]
    if match.empty:
        # Title lookup. Fuzzy matching is only a fallback for typos; it needs a high score so that
        # a course that is not offered is reported as not found instead of a look-alike.
        exact_title = df[df["title"].str.lower() == query.lower()]
        if not exact_title.empty:
            match = exact_title
        else:
            best = process.extractOne(query, df["title"].tolist(), scorer=fuzz.WRatio)
            if best and best[1] >= 90:
                match = df[df["title"] == best[0]]

    if match.empty:
        return {
            "status": "not_found",
            "message": f"Course '{code_or_name}' is not in the Classroom Allocation for AY 2026-27 Sem I. "
                       "Ask the student to check the course code.",
            "source": SOURCE_TAG,
        }

    note_sources(tool_context, "CLASS_ALLOC_2026")
    offerings = [_row_to_dict(r) for _, r in match.iterrows()]
    return {
        "status": "success",
        **offerings[0],
        "other_offerings": offerings[1:],  # tutorial sections, non-English fractals, etc.
        "source": f"{SOURCE_TAG}, p.{offerings[0]['page']}",
        "cite_as": f"[CLASS_ALLOC_2026, p.{offerings[0]['page']}]",
    }


def list_courses(
    department: Optional[str] = None,
    level: Optional[int] = None,
    elective: Optional[bool] = None,
    slot: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Filter the courses offered this semester and return an exact count. Use this for every
    "how many courses ..." question instead of counting yourself.

    Args:
        department: Department name or alias, e.g. 'Humanities' (= School of Liberal Arts / HSS),
            'Computer Science', 'Electrical Engineering', 'Mathematics'.
        level: Course level such as 100, 200, 300, 400, 600, 700, 800 (first digit of the code).
        elective: true for elective courses only, false for non-elective only. Electives are inferred
            from the Slot System rules (HSS courses at level >= 300 and courses in elective slots).
        slot: Slot letter, e.g. 'A', 'E', 'K', 'N'.

    Returns:
        The count, the matching courses (distinct course codes) and the source to cite.
    """
    df = get_courses_df()
    filtered = df

    if department:
        key = department.lower().strip()
        pattern = DEPT_ALIASES.get(key, key)
        filtered = filtered[filtered["dept"].str.contains(pattern, case=False, regex=True)]
    if level is not None:
        filtered = filtered[filtered["level"] == int(level)]
    if elective is not None:
        filtered = filtered[filtered["is_elective"] == int(bool(elective))]
    if slot:
        filtered = filtered[filtered["slot"].str.upper() == slot.upper().strip()]

    # One row per course code: tutorial sections and fractals share a code.
    unique = filtered.drop_duplicates(subset="code")
    note_sources(tool_context, "CLASS_ALLOC_2026")
    return {
        "status": "success",
        "count": int(len(unique)),
        "filters": {"department": department, "level": level, "elective": elective, "slot": slot},
        "courses": [
            {k: v for k, v in _row_to_dict(r).items() if k in ("code", "title", "slot", "credit", "department", "instructor")}
            for _, r in unique.head(50).iterrows()
        ],
        "source": SOURCE_TAG,
        "cite_as": "[CLASS_ALLOC_2026]",
    }
