"""
helpdesk/tools/email_draft.py - Email drafting tool for leave, makeup exams and instructor queries.
Drafts are saved to logs/drafts/ for the student to review. They are NEVER sent.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from google.adk.tools.tool_context import ToolContext

from helpdesk import config
from helpdesk.blackboard import note_sources

SIGNATURE = "Yours sincerely,\n{student_name}\nRoll No: {roll}\nIIT Jodhpur"

TEMPLATES = {
    "leave": (
        "Request for leave of absence - {course}",
        "Respected Professor {instructor},\n\n"
        "I am a student registered in {course}. I am writing to request leave of absence from the classes "
        "of this course.\n\n{details}\n\n"
        "I will cover the material taught during my absence. Kindly let me know if any supporting "
        "document is required.\n\nThank you for your consideration.\n\n",
    ),
    "makeup": (
        "Request for make-up examination - {course}",
        "Respected Professor {instructor},\n\n"
        "I am a student registered in {course}. I am writing to request permission to appear for a "
        "make-up examination.\n\n{details}\n\n"
        "I would be grateful if you could consider my request and let me know the procedure to follow.\n\n"
        "Thank you for your consideration.\n\n",
    ),
    "query": (
        "Query regarding {course}",
        "Respected Professor {instructor},\n\n"
        "I am a student registered in {course}. I am writing regarding the following:\n\n{details}\n\n"
        "I would be grateful for your guidance.\n\n",
    ),
}


def _template_key(purpose: str) -> str:
    p = purpose.lower()
    if "makeup" in p or "make-up" in p or "make up" in p:
        return "makeup"
    if "leave" in p or "absence" in p or "absent" in p:
        return "leave"
    return "query"


def draft_email(
    to: str,
    course_code: str,
    purpose: str,
    details: str,
    instructor_name: str = "",
    student_name: str = "<Your Name>",
    roll: str = "<Roll No>",
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Draft an e-mail to a course instructor and save it for the student to review. Never sends it.

    Args:
        to: Instructor e-mail address returned by get_instructor_email.
        course_code: Course code, e.g. 'CSL7650'.
        purpose: 'leave', 'makeup' or 'query'.
        details: Only what the student actually said (dates, duration, reason). Do not invent anything.
        instructor_name: Instructor name returned by get_instructor_email.
        student_name: Student name if given, otherwise leave the placeholder.
        roll: Roll number if given, otherwise leave the placeholder.

    Returns:
        Subject, body, recipient and the path of the saved draft.
    """
    now = datetime.now()
    course = course_code.strip().upper()
    subject_t, body_t = TEMPLATES[_template_key(purpose)]
    fields = {"course": course, "instructor": instructor_name.strip() or "", "details": details.strip()}
    subject = subject_t.format(**fields)
    body = body_t.format(**fields).replace("Professor ,", "Professor,") + SIGNATURE.format(
        student_name=student_name, roll=roll
    )

    config.DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = config.DRAFTS_DIR / f"draft_{course}_{now.strftime('%Y%m%d_%H%M%S_%f')}.md"
    file_path.write_text(
        f"<!-- DRAFT ONLY - NOT SENT. Generated {now.isoformat(timespec='seconds')} -->\n\n"
        f"**To:** {to}  \n**Subject:** {subject}\n\n---\n\n{body}\n",
        encoding="utf-8",
    )

    note_sources(tool_context, "FACULTY_DIRECTORY")
    if tool_context is not None:
        tool_context.state["pending_email"] = {"to": to, "course": course, "subject": subject,
                                               "draft_path": str(file_path)}

    return {
        "status": "draft_created",
        "recipient": to,
        "subject": subject,
        "body": body,
        "saved_path": str(file_path),
        "note": "Draft saved for the student to review and send themselves. It has NOT been sent.",
        "source": "[FACULTY_DIRECTORY] (dummy e-mail address)",
    }
