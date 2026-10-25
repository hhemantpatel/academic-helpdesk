"""
helpdesk/tools/faculty.py - Instructor contact lookup tool (dummy e-mail addresses).
"""

import re
from typing import Any, Dict, Optional

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from helpdesk.config import TABLES_DIR
from helpdesk.blackboard import note_sources
from helpdesk.tools.courses import get_course_info

SOURCE_TAG = "[FACULTY_DIRECTORY] / [CLASS_ALLOC_2026] Classroom Allocation AY 2026-27 Sem I"

_faculty_df = None


def get_faculty_df() -> pd.DataFrame:
    global _faculty_df
    if _faculty_df is None:
        csv_path = TABLES_DIR / "faculty.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing faculty.csv at {csv_path}; run `python main.py ingest`")
        _faculty_df = pd.read_csv(csv_path, keep_default_na=False)
    return _faculty_df


def get_instructor_email(course_code: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Look up the instructor of a course and their (dummy) e-mail address.

    Args:
        course_code: The course code, e.g. 'CSL7650'.

    Returns:
        Instructor name, dummy e-mail, department and the source to cite. If the course is not
        offered this semester, status is 'not_found' and no instructor must be invented.
    """
    course = get_course_info(course_code, tool_context=tool_context)
    if course.get("status") != "success":
        return {
            "status": "not_found",
            "message": f"Course '{course_code}' is not in the Classroom Allocation for this semester, so its "
                       "instructor cannot be determined. Ask the student to confirm the course code.",
            "source": SOURCE_TAG,
        }

    instructor = re.split(r"[,/]", course.get("instructor", ""))[0].strip()
    if not instructor:
        return {
            "status": "not_found",
            "message": f"No instructor is listed for '{course_code}' in the Classroom Allocation.",
            "source": SOURCE_TAG,
        }

    faculty = get_faculty_df()
    row = faculty[faculty["name"] == instructor]
    if row.empty:
        return {
            "status": "not_found",
            "message": f"Instructor '{instructor}' of {course_code} has no entry in the faculty directory.",
            "source": SOURCE_TAG,
        }
    row = row.iloc[0]

    note_sources(tool_context, "FACULTY_DIRECTORY", "CLASS_ALLOC_2026")
    return {
        "status": "success",
        "course_code": course["code"],
        "course_title": course["title"],
        "instructor": row["name"],
        "email": row["email"],
        "email_is_dummy": True,
        "department": row["department"],
        "source": SOURCE_TAG,
        "cite_as": "[FACULTY_DIRECTORY], [CLASS_ALLOC_2026]",
    }
