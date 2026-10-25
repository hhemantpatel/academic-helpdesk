"""
helpdesk/tools - Export all helpdesk tools.
"""

from helpdesk.tools.rag import search_documents
from helpdesk.tools.courses import get_course_info, list_courses
from helpdesk.tools.slots import get_slot_timing
from helpdesk.tools.faculty import get_instructor_email
from helpdesk.tools.email_draft import draft_email
from helpdesk.tools.escalation import log_unanswered

__all__ = [
    "search_documents",
    "get_course_info",
    "list_courses",
    "get_slot_timing",
    "get_instructor_email",
    "draft_email",
    "log_unanswered"
]
