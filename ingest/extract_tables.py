#!/usr/bin/env python3
"""
ingest/extract_tables.py - Deterministic table extraction (no LLM).

Produces, from the two official PDFs in data/raw/:
- data/tables/courses.csv   from the Classroom Allocation PDF (CLASS_ALLOC_2026)
- data/tables/slots.csv     transcription of the Slot System grid (SLOT_SYS_2026)
- data/tables/faculty.csv   instructors joined from courses.csv, with DUMMY emails
- data/raw/slot_system_rules.md   transcription of the Slot System notes/rules
- data/raw/faculty_directory.md   text view of faculty.csv so it can be retrieved

Nothing here invents facts: every row comes from a source PDF. The only
synthetic values are the dummy e-mail addresses (personal data replacement).
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpdesk.config import RAW_DIR, TABLES_DIR  # noqa: E402

DUMMY_EMAIL_DOMAIN = "iitj-dummy.ac.in"

# Column x-ranges (PDF points) on the 2232-pt wide landscape pages.
COLUMNS = [
    ("code", 50, 160),
    ("title", 160, 420),
    ("ltp", 420, 480),
    ("credit", 480, 540),
    ("slot", 540, 640),
    ("dept", 640, 860),
    ("instructor", 860, 1040),
    ("co_instructor", 1040, 1170),
    ("registered", 1170, 1260),
    ("classroom", 1260, 1450),
]

# Departments whose courses are Humanities & Social Sciences (HSS) courses.
HSS_DEPT_PATTERN = r"Liberal Arts|SoLA|Digital Humanities|Economics"

# Elective slots per course level, transcribed from the "Programme wise slot
# allocation" table of the Slot System PDF (Elective Course column).
ELECTIVE_SLOTS_BY_LEVEL = {
    300: {"K", "N"},            # UG3 [300,400,600]: K & N (HSS or overload)
    400: {"K", "N", "E"},       # UG4 [400,600] / Science elective: K, N, E
    600: {"K", "N", "E"},
    500: {"I", "J", "K", "N"},  # M.Sc./M.Tech. [500,600,700]: I-J, K, N
    700: {"I", "J", "K", "N"},
    800: {"F", "G", "H"},       # Ph.D. [800]: F-H
}

STAFF_ID = re.compile(r"\b\d{5}-")


def _clean_person_field(text: str) -> str:
    """Strip staff ids like '00304-', split glued names, normalise separators/spaces."""
    text = STAFF_ID.sub("", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1, \2", text)  # "BanerjeeAppala" -> "Banerjee, Appala"
    text = re.sub(r"\s*,\s*", ", ", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def infer_elective(dept: str, code: str, level: int, slot: str) -> int:
    """
    Elective flag inferred from the Slot System allocation rules.
    The Classroom Allocation PDF has no elective column, so this is a documented
    heuristic: HSS courses at level >= 300 are electives (UG HSS electives / 700-level
    Humanities electives); other courses are electives if they sit in an elective slot
    for their level.
    """
    if re.search(HSS_DEPT_PATTERN, dept) and level >= 300 and not code.startswith("LAN"):
        return 1
    return int(slot in ELECTIVE_SLOTS_BY_LEVEL.get(level, set()))


def _split_people(stream_words) -> tuple[str, str]:
    """
    Instructor and co-instructor cells overflow into each other in the PDF, so splitting
    them by x-position cuts names in half. Instead read both columns in content-stream
    order and start a new cell whenever the x position jumps backwards. Non-numeric words
    that overflowed into the "registered" column belong to the last cell.
    """
    inst_lo, inst_hi = COLUMNS[6][1], COLUMNS[7][2]
    reg_hi = COLUMNS[8][2]
    region = [
        w for w in stream_words
        if inst_lo <= (w["x0"] + w["x1"]) / 2 < inst_hi
        or (inst_hi <= (w["x0"] + w["x1"]) / 2 < reg_hi and not re.fullmatch(r"[\d+]+", w["text"]))
    ]
    cells, prev = [[]], None
    for w in region:
        if prev is not None and w["x0"] < prev["x1"] - 1:
            cells.append([])
        cells[-1].append(w["text"])
        prev = w
    texts = [_clean_person_field(" ".join(c)) for c in cells if c]
    if not texts:
        return "", ""
    return texts[0], ", ".join(t for t in texts[1:] if t)


def _group_lines(words, tol=3.5):
    lines = []
    for w in words:
        for line in lines:
            if abs(line["top"] - w["top"]) < tol:
                line["words"].append(w)
                break
        else:
            lines.append({"top": w["top"], "words": [w]})
    return lines


def extract_courses() -> pd.DataFrame:
    pdf_path = RAW_DIR / "classroom_allocation_ay26_27.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing {pdf_path}")

    courses, seen = [], set()
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            # use_text_flow keeps the PDF content-stream order, which avoids the
            # character interleaving that happens where a cell overflows into the next.
            words = page.extract_words(use_text_flow=True)
            for line in _group_lines(words):
                line_words = sorted(line["words"], key=lambda w: w["x0"])
                if not re.match(r"^[A-Z]{2,4}\d{4,5}", line_words[0]["text"]):
                    continue

                row = {}
                for col, x_min, x_max in COLUMNS:
                    row[col] = " ".join(
                        w["text"] for w in line_words if x_min <= (w["x0"] + w["x1"]) / 2 < x_max
                    ).strip()

                code, dept = row["code"], row["dept"]
                instructor, co_instructor = _split_people(line["words"])
                # Long department names wrap; their last word lands in the instructor cell.
                tail = re.match(r"^(Energy|Development), ", instructor)
                if tail:
                    dept = f"{dept} {tail.group(1)}"
                    instructor = instructor[tail.end():]
                if instructor.startswith("(SME)"):
                    instructor = instructor.replace("(SME)", "").strip()

                registered = sum(int(x) for x in re.findall(r"\d+", row["registered"]))

                raw_slot = row["slot"]
                m = re.match(r"^([A-Z0-9]+)", raw_slot)
                slot = m.group(1) if m else raw_slot

                digits = re.search(r"\d+", code).group(0)
                level = int(digits[0]) * 100
                dept_code = re.match(r"^[A-Z]+", code).group(0)

                key = (code, slot, instructor)
                if key in seen:
                    continue
                seen.add(key)

                courses.append({
                    "code": code,
                    "title": row["title"],
                    "ltp": row["ltp"],
                    "credit": row["credit"],
                    "slot": slot,
                    "raw_slot": raw_slot,
                    "dept": dept,
                    "dept_code": dept_code,
                    "level": level,
                    "is_hss": int(bool(re.search(HSS_DEPT_PATTERN, dept))),
                    "is_elective": infer_elective(dept, code, level, slot),
                    "instructor": instructor,
                    "co_instructor": co_instructor,
                    "registered": registered,
                    "classroom": row["classroom"],
                    "source_doc": f"Classroom Allocation AY 2026-27 Sem I, p.{page_idx + 1}",
                    "page": page_idx + 1,
                })

    df = pd.DataFrame(courses)
    out_csv = TABLES_DIR / "courses.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} courses to {out_csv}")
    return df


# Weekly lecture grid exactly as printed in the Slot System PDF (rows = start hour).
LECTURE_GRID = {
    "08:00": ["K", "N", "K", "N", "K"],
    "09:00": ["A", "A", "B", "A", "B"],
    "10:00": ["C", "B", "C", "C", "D"],
    "11:00": ["D", "E", "D", "E", "E"],
    "12:00": ["U", "V", "U", "V", "U"],
    "14:00": ["F", "F", "G", "F", "G"],
    "15:00": ["H", "G", "H", "H", "I"],
    "16:00": ["I", "J", "I", "J", "J"],
    "17:00": ["Dept/Tutorial", "Dept/Tutorial", "Ethics/Comm", "SysEng/PM", "N"],
    "18:00": ["Tutorials"] * 5,
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
# Lab rows are printed between hour rows; the windows below are the 3-hour blocks they
# sit in. The PDF does not state lab durations explicitly.
LAB_WINDOWS = [("L", 1, "08:00", "10:50"), ("L", 6, "11:00", "13:50"),
               ("M", 1, "14:00", "16:50"), ("M", 6, "17:00", "19:50")]


def extract_slots() -> pd.DataFrame:
    rows = []
    for d_idx, day in enumerate(DAYS):
        for start, slots in LECTURE_GRID.items():
            slot = slots[d_idx]
            kind = "tutorial" if "Tutorial" in slot else "lecture"
            end = f"{start[:2]}:50"
            rows.append({"slot": slot, "day": day, "start": start, "end": end, "type": kind})
        for prefix, first, start, end in LAB_WINDOWS:
            rows.append({"slot": f"{prefix}{first + d_idx}", "day": day, "start": start, "end": end, "type": "lab"})
    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "slots.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} slot timings to {out_csv}")
    return df


def dummy_email(name: str) -> str:
    parts = re.findall(r"[a-z]+", name.lower())
    return (".".join(parts) or "faculty") + f"@{DUMMY_EMAIL_DOMAIN}"


def extract_faculty(courses_df: pd.DataFrame) -> pd.DataFrame:
    courses_map, dept_map = {}, {}
    for _, row in courses_df.fillna("").iterrows():
        names = re.split(r"[,/]", f"{row['instructor']},{row['co_instructor']}")
        for name in (n.strip() for n in names):
            if len(name) < 3:
                continue
            dept_map.setdefault(name, row["dept"])
            courses_map.setdefault(name, [])
            if row["code"] not in courses_map[name]:
                courses_map[name].append(row["code"])

    df = pd.DataFrame([
        {"name": n, "department": dept_map[n], "email": dummy_email(n), "courses": ", ".join(courses_map[n])}
        for n in sorted(courses_map)
    ])
    out_csv = TABLES_DIR / "faculty.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} faculty members to {out_csv}")
    return df


def write_faculty_directory(faculty_df: pd.DataFrame, courses_df: pd.DataFrame):
    """Text view of faculty.csv (FACULTY_DIRECTORY) so instructor questions are retrievable."""
    titles = dict(zip(courses_df["code"], courses_df["title"]))
    lines = [
        "# Faculty Directory (derived from Classroom Allocation AY 2026-27 Sem I)",
        "Document ID: [FACULTY_DIRECTORY]",
        f"Note: names and course assignments come from CLASS_ALLOC_2026. E-mail addresses are DUMMY values "
        f"(@{DUMMY_EMAIL_DOMAIN}) that replace personal data, as permitted by the assignment.",
        "",
    ]
    for dept, group in faculty_df.groupby("department"):
        lines.append(f"## {dept}")
        for _, r in group.iterrows():
            courses = ", ".join(f"{c} {titles.get(c, '')}".strip() for c in r["courses"].split(", ") if c)
            lines.append(f"- {r['name']}. Courses: {courses}. Dummy e-mail: {r['email']}.")
        lines.append("")
    (RAW_DIR / "faculty_directory.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote faculty_directory.md")


SLOT_RULES_MD = """# IIT Jodhpur Slot System: Notes and Allocation Rules (AY 2026-27, Semester I)
Document ID: [SLOT_SYS_RULES_2026]
Source: transcription of the notes and tables on p.1 of "Slot System for B.Tech. 2nd year onwards
(AY 2026-27: Semester I)" [SLOT_SYS_2026]. Nothing has been added beyond the PDF.

## Weekly lecture slots (Monday to Friday)
- Slot K: Monday, Wednesday, Friday 08:00-08:50.
- Slot N: Tuesday, Thursday 08:00-08:50 and Friday 17:00-17:50.
- Slot A: Monday, Tuesday, Thursday 09:00-09:50.
- Slot B: Tuesday 10:00-10:50, Wednesday 09:00-09:50, Friday 09:00-09:50.
- Slot C: Monday, Wednesday, Thursday 10:00-10:50.
- Slot D: Monday 11:00-11:50, Wednesday 11:00-11:50, Friday 10:00-10:50.
- Slot E: Tuesday, Thursday, Friday 11:00-11:50.
- Slot U: Monday, Wednesday, Friday 12:00-12:50.
- Slot V: Tuesday, Thursday 12:00-12:50.
- Slot F: Monday, Tuesday, Thursday 14:00-14:50.
- Slot G: Tuesday 15:00-15:50, Wednesday 14:00-14:50, Friday 14:00-14:50.
- Slot H: Monday, Wednesday, Thursday 15:00-15:50.
- Slot I: Monday 16:00-16:50, Wednesday 16:00-16:50, Friday 15:00-15:50.
- Slot J: Tuesday, Thursday, Friday 16:00-16:50.
- 17:00-17:50: Monday and Tuesday are free slots for department activities / tutorials; Wednesday is
  Professional Ethics-I (UG) / Technical Communication (PG) / tutorials; Thursday is System Engineering
  and Project Management (PG) / tutorials; Friday is slot N.
- 18:00-18:50: tutorials on all days.

## Lab and tutorial slots
- L slots (L1-L10) are for labs and tutorials of courses in slots F-J.
- M slots (M1-M10) are for labs and tutorials of courses in slots A-E.
- In the grid, L1-L5 (Mon-Fri) are printed in the morning block, L6-L10 in the late-morning block,
  M1-M5 in the early-afternoon block and M6-M10 in the late-afternoon block. The PDF does not state
  exact lab durations.
- Maximum students per tutorial is about 40. Tutorial slots are flexible (any timing) as per the
  convenience of students and instructors and classroom availability.
- Maximum students per lab is 50.

## General notes
- While allocating slots for UG/PG compulsory courses, priority is given to graduating B.Tech students
  with 2 or fewer backlogs.
- All fractals of a course run in the same slot.
- To facilitate electives/overload for 2nd and 3rd year students, all departments will open at least
  one core/elective course in slot K or N.
- Upper limit of an HSS course is 80 students, of all elective courses 80, and of a Management Minor
  course 40.

## Programme-wise slot allocation for courses with a theory component
| Programme / course level | Compulsory course slots | Elective course slots |
|---|---|---|
| UG1 / Preparatory courses [100] | - | - |
| UG2 courses [200] | A-E (PC), U (IE), K & N (PC if strength less than 160) | - |
| UG3 courses [300, 400, 600] | A-E | K & N (HSS or students opting overload) |
| UG4 courses [400, 600] / Science elective | - | K, N, E |
| UG4 HSS courses [400, 600] | Minimum 6 HSS courses | A-D |
| M.Sc./M.Tech. [500, 600, 700] | F-I, K | I-J, K, N |
| MMT courses | A-E | - |
| MBA courses including Minor in Management | A-E, I, K, N | A-E, G, I, K, N |
| Ph.D. courses [800] | - | F-H |
| Humanities electives [700] | - | F-J |
| UG Minor/Specialization courses (other than Management) | K, N, I | F-J |

## DSA / IML labs for UG2 students
| Branch | Roll numbers | Lab | Slot |
|---|---|---|---|
| CS | 1-50 | DSA | M1 |
| CS | 51-100 | DSA | M2 |
| EE | 1-45 | DSA | M3 |
| EE | 45-90 | DSA | M4 |
| AI | 1-55 | DSA | M5 |
| ES | 1-50 | DSA | M6 |
| ME | 1-40 | IML | M1 |
| ME | 41-80 | IML | M2 |
| CI | 1-45 | IML | M3 |
| BB | 1-50 | IML | M4 |
| CH | 1-50 | IML | M5 |
| MT | 1-50 | IML | M6 |
| PY | 1-50 | IML | M7 |
| CY | 1-50 | IML | M8 |
DSA = Data Structures and Algorithms, IML = Introduction to Machine Learning.
"""


def write_slot_rules():
    (RAW_DIR / "slot_system_rules.md").write_text(SLOT_RULES_MD, encoding="utf-8")
    print("Wrote slot_system_rules.md")


def main():
    courses_df = extract_courses()
    extract_slots()
    faculty_df = extract_faculty(courses_df)
    write_faculty_directory(faculty_df, courses_df)
    write_slot_rules()
    print("Table extraction completed.")


if __name__ == "__main__":
    main()
