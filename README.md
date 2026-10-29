# Multi-Agent Academic Helpdesk (IIT Jodhpur)
CSL7650 Autonomous Systems, Assignment 2.

The helpdesk answers questions about IIT Jodhpur academic programmes, rules and regulations, course
slots and the timetable. It can draft (but never send) e-mails to course instructors. Every answer cites
the document it came from, and anything it cannot answer is refused with "I don't know" and logged to
`logs/unanswered.jsonl` for a human to follow up.

It runs locally on a 16 GB laptop (CPU-only works; a GPU or Apple Silicon is much faster):
- **Agents:** Google ADK, using Ollama 8B models (`qwen3:8b`, `llama3.1:8b`) through LiteLLM.
- **Retrieval:** RAG over 24 institute documents, embedded with `nomic-embed-text` and stored in ChromaDB.
- **Tables:** deterministic tools over tables extracted from the official Slot System and Classroom
  Allocation PDFs.

Two architectures are compared:
- **multi-agent:** an orchestrator LLM delegates each question to a Policy, Schedule or Email agent.
- **single agent:** one agent holds all seven tools.

## Setup
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt   # or: python -m venv .venv && pip install -r requirements.txt
ollama pull qwen3:8b && ollama pull llama3.1:8b && ollama pull nomic-embed-text
cp .env.example .env            # optional: change models, tau, context size
python main.py ingest --rebuild  # tables + Chroma index (~3 min on CPU; scanned circulars need `tesseract`)
```

## Run
| Command | What it does |
|---|---|
| `python main.py ask "What is the slot for the course Autonomous Systems?" -v` | One question through the multi-agent system (LLM orchestrator), printing the route and tool calls |
| `python main.py ask "..." --system single` | The same question through the single-agent baseline |
| `python main.py ask "..." --router keyword` | Multi-agent with a deterministic keyword router instead of the LLM orchestrator (ablation) |
| `python main.py chat` | Interactive session |
| `python main.py web` | ADK web UI (`adk web helpdesk`) showing agent transfers and tool calls |
| `python -m pytest -q` | Unit tests for the tools, extraction, citations, guardrails and scoring. No LLM is needed; the retrieval test needs Ollama. |
| `python eval/run_eval.py --quick` | Smoke test: 5 questions × 1 run |
| `python eval/run_eval.py` | Full grid: 30 questions × {multi, single} × {qwen3, llama3.1} × 3 runs. This takes many hours on a CPU and can be resumed. |
| `python eval/run_eval.py --ablation --model qwen3:8b` | Ablation: each guardrail switched off, and τ ∈ {0.28, 0.45} |
| `python eval/run_eval.py --score-only` | Re-score after checking answers in `eval/human_labels.csv`. This regenerates the figures, `report/tables.tex`, `numbers.tex`, `case_studies.tex` and `questions.tex`. |
| `cd report && pdflatex report.tex && pdflatex report.tex` | Build the report |

Each question takes roughly 1–4 minutes on a CPU-only i5-12450H. On an Apple M4 (Metal GPU) it took about
18 s per question for Llama and about 110 s for Qwen3, which reasons before answering. The first call also
includes loading the model.

## Example (real output)
```
$ python main.py ask "What is the slot for the course Autonomous Systems?" -v
[HelpdeskOrchestrator] -> transfer_to_agent({'agent_name': 'schedule_agent'})
[schedule_agent] -> get_course_info({'code_or_name': 'Autonomous Systems'})
The course **Autonomous Systems** (code: CSL7650) is offered in **slot E** [CLASS_ALLOC_2026, p.3].
Route: HelpdeskOrchestrator -> schedule_agent
Cited: ['CLASS_ALLOC_2026']  Retrieved: ['CLASS_ALLOC_2026']
```

## How it works
- **Tools** (`helpdesk/tools/`):
  - `search_documents`: Chroma search. Results above a cosine distance of τ = 0.35 are dropped. Each
    result includes a `cite_as` string and the document date.
  - `get_course_info`, `get_slot_timing`, `list_courses`: table lookups. `list_courses` returns exact
    counts, so the LLM never counts by itself.
  - `get_instructor_email`: instructor lookup, returning a dummy address.
  - `draft_email`: saves a draft to `logs/drafts/` and never sends it.
  - `log_unanswered`: escalation to a human.

  Every tool records the documents it read in the ADK session state (`state["retrieved"]`).
- **Guardrails** (`helpdesk/guardrails.py`):
  - **Scope governor** (a before-agent callback): refuses clearly off-topic questions before any LLM call,
    using a deliberately narrow keyword list. Off-topic questions it misses must be refused by the agents
    themselves.
  - **Citation guardrail** (an after-model callback): replaces an answer that cites a document never
    retrieved in the session, or that was produced without any tool, with a refusal, and logs it. If tools
    were used but the model forgot to cite, it attaches the retrieved sources and flags the run.
- **Course concepts behind the design:** organisation-centric MAS and role descriptions, institutions and
  norms, the BDI loop, subsumption, blackboard coordination, and argumentation for conflicting
  regulations. These are explained in `report/report.tex`.

## Data provenance
- **Documents:** `ingest/manifest.csv` lists the 24 indexed documents.
  - All are real IIT Jodhpur PDFs or web pages, except for three derived files:
    - `slot_system_rules.md`, a transcription of the notes printed on the Slot System PDF;
    - `faculty_directory.md`, generated from the Classroom Allocation PDF;
    - `btech_credit_structure.md`, a clean transcription of Table 11.2 (credit structure) of the UG regulations.
  - The two 2024 Senate circulars are scans, so they are OCR'd with Tesseract during ingestion.
- **E-mail addresses** are dummy values (`firstname.lastname@iitj-dummy.ac.in`) that replace personal
  data. Staff ID numbers are stripped.
- **`data/unverified/`** holds files that are *not* indexed. These include two Markdown files that were
  generated earlier rather than collected, and that contradict the real circulars. See its README.
- **Known limitations of the extracted tables:**
  - Elective status is inferred from the Slot System's allocation table, because the PDF has no elective
    column.
  - Lab-slot times are inferred from the grid layout.
  - 2 of the 626 allocation rows still contain merged text from overflowing PDF cells.

## Code map
```
main.py                     CLI: ingest | ask | chat | eval | web; execute_query() runs one question and records route/tools/citations/trace
helpdesk/config.py          paths, models, tau, context size (.env overrides)
helpdesk/models.py          LiteLlm model factory (sets num_ctx for Ollama)
helpdesk/prompts.py         role descriptions (objective, rights, norms) for every agent
helpdesk/agents/            orchestrator, policy, schedule, email and single-agent builders
helpdesk/agent.py           root_agent for `adk web`, get_agent("multi"|"single")
helpdesk/tools/             rag, courses, slots, faculty, email_draft, escalation
helpdesk/blackboard.py      records retrieved documents in session state
helpdesk/guardrails.py      scope governor and citation guardrail callbacks
helpdesk/citations.py       citation parsing and groundedness check
ingest/extract_tables.py    Classroom Allocation / Slot System PDFs -> data/tables/*.csv (+ derived .md docs)
ingest/build_index.py       manifest -> text (OCR for scans) -> chunks -> Chroma
eval/questions.jsonl        30 questions with gold answers, required facts, gold sources, expected agent/tool
eval/run_eval.py            evaluation grid and ablation runner (resumable, results/*.csv)
eval/score.py               accuracy, refusal, citation, routing, tool and stability metrics
eval/plots.py               figures and the generated report/*.tex files
eval/human_labels.csv       manual correctness labels that override the automatic check
tests/                      pytest suite
report/report.tex           report; all numbers come from the generated numbers.tex/tables.tex
```

## Evaluation results
The full grid (30 questions x {multi, single} x {qwen3:8b, llama3.1:8b} x 3 runs = 360 runs) and the
ablation (5 configurations x 30 questions, multi-agent, qwen3:8b) were run on an Apple M4 with 16 GB.
Raw runs and scores are in `results/`; the analysis is in `report/report.pdf`.

| System | Model | Accuracy | Refusal F1 | Correct source | Latency |
|---|---|---|---|---|---|
| multi | qwen3:8b | 94.4% | 0.95 | 100% | 113 s |
| single | qwen3:8b | 92.2% | 0.90 | 91.2% | 96 s |
| multi | llama3.1:8b | 82.2% | 0.93 | 87.3% | 18 s |
| single | llama3.1:8b | 70.0% | 0.86 | 88.2% | 20 s |

All out-of-scope questions were refused in every run. Correctness uses automatic regex labels against the
gold answers (`eval/questions.jsonl`); `eval/human_labels.csv` takes manual overrides and is empty.
Questions whose outcome changed between runs are in `results/stability.csv`.

## Team
Sahilpreet Singh (B23CS1061), Hemant Patel (B23CH1018), Kartik Gehlot (B23EE1088), Rudra Thakar (B23EE1100).
