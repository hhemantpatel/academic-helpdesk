"""
helpdesk/tools/slots.py - Slot timing lookup tool over slots.csv.
"""

from typing import Any, Dict, Optional

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from helpdesk.config import TABLES_DIR
from helpdesk.blackboard import note_sources

SOURCE_TAG = "[SLOT_SYS_2026] Slot System AY 2026-27 Sem I, p.1"

_slots_df = None


def get_slots_df() -> pd.DataFrame:
    global _slots_df
    if _slots_df is None:
        csv_path = TABLES_DIR / "slots.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing slots.csv at {csv_path}; run `python main.py ingest`")
        _slots_df = pd.read_csv(csv_path, dtype=str)
        _slots_df["slot"] = _slots_df["slot"].str.strip()
    return _slots_df


def get_slot_timing(slot: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Get the weekly meeting days and times of a timetable slot.

    Args:
        slot: The slot name, e.g. 'A', 'E', 'K', 'N', 'L1', 'M2'.

    Returns:
        Meeting days, start/end times and session type, plus the source document to cite.
    """
    df = get_slots_df()
    slot_clean = slot.strip().upper().removeprefix("SLOT ").strip()

    matches = df[df["slot"].str.upper() == slot_clean]
    if matches.empty:
        return {
            "status": "not_found",
            "message": f"Slot '{slot}' is not in the official slot grid.",
            "source": SOURCE_TAG,
        }

    note_sources(tool_context, "SLOT_SYS_2026")
    timings = matches[["day", "start", "end", "type"]].to_dict("records")
    result = {
        "status": "success",
        "slot": slot_clean,
        "timings": timings,
        "count": len(timings),
        "source": SOURCE_TAG,
        "cite_as": "[SLOT_SYS_2026, p.1]",
    }
    if (matches["type"] == "lab").any():
        result["note"] = "Lab windows are inferred from where the lab row is printed in the grid; durations are not stated in the PDF."
    return result
