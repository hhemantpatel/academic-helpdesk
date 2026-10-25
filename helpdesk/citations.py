"""
helpdesk/citations.py - Citation parsing and groundedness validation.

Citations look like [UG_REG_2019, p.12], [CLASS_ALLOC_2026] or [S1]; several may share one
bracket, e.g. [UG_REG_2019, p.12; ATTEND_POL_2024].
"""

import re
from typing import Dict, List, Optional, Set, Tuple

BRACKET = re.compile(r"\[([^\[\]]{1,200})\]")
DOC_ID = re.compile(r"^(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|S\d{1,2})$")


def format_citation(doc_id: str, page: Optional[int] = None) -> str:
    return f"[{doc_id}, p.{page}]" if page else f"[{doc_id}]"


def extract_cited_references(text: str) -> List[str]:
    """Return every citation token in the text, e.g. ['UG_REG_2019, p.12', 'S1']."""
    refs = []
    for inner in BRACKET.findall(text or ""):
        in_bracket = []
        for token in (t.strip() for t in re.split(r"[;,]", inner)):
            if DOC_ID.match(token):
                in_bracket.append(token)
            elif in_bracket and re.fullmatch(r"(p|pp|page)\.?\s*\d+(\s*-\s*\d+)?", token, re.I):
                in_bracket[-1] = f"{in_bracket[-1]}, {token}"
        refs.extend(in_bracket)
    return refs


def extract_cited_doc_ids(text: str, source_id_map: Optional[Dict[str, str]] = None) -> Set[str]:
    """Canonical doc_ids cited in the text; [S#] tags are resolved through source_id_map."""
    source_id_map = source_id_map or {}
    cited = set()
    for ref in extract_cited_references(text):
        core = ref.split(",")[0].strip()
        cited.add(source_id_map.get(core, core))
    return cited


def verify_groundedness(text: str, retrieved_doc_ids: Set[str],
                        source_id_map: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """
    (is_valid, reason). Invalid when the text cites nothing, or cites any document (or [S#] tag)
    that was not retrieved in this session.
    """
    cited = {c.upper() for c in extract_cited_doc_ids(text, source_id_map)}
    if not cited:
        return False, "no citation found"
    retrieved = {d.upper() for d in retrieved_doc_ids}
    fabricated = sorted(cited - retrieved)
    if fabricated:
        return False, f"cited but never retrieved: {fabricated}"
    return True, "grounded"
