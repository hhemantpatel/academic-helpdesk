"""
eval/score.py - Metrics for the helpdesk evaluation.

Per run:
- correct     : should_refuse -> refused; otherwise all `must_include` patterns match and the system did
                not refuse (questions marked `clarification` expect a "not found / which course?" reply).
                eval/human_labels.csv overrides the automatic label after manual checking.
- refused / logged
- citation metrics, on answered in-scope runs with gold sources only:
    cited          the answer cites at least one document by itself (auto-attached sources do not count)
    fabricated     the answer cites a document that no tool retrieved in that run
    correct_source at least one cited/attached document is a gold source
- routing (multi-agent): first specialist the orchestrator delegated to == expected agent
  (for questions that should be refused, refusing without delegating also counts as correct)
- tool use: the expected tool was called
Aggregates per (system, model, router) plus a run-to-run stability table.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "eval" / "questions.jsonl"
HUMAN_LABELS_FILE = BASE_DIR / "eval" / "human_labels.csv"
SPECIALISTS = ("policy_agent", "schedule_agent", "email_agent")
GROUP_KEYS = ["system", "model", "router"]

import sys  # noqa: E402
sys.path.insert(0, str(BASE_DIR))
from helpdesk.guardrails import is_refusal_text  # noqa: E402


def load_questions() -> Dict[str, Dict[str, Any]]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return {q["id"]: q for q in (json.loads(l) for l in f if l.strip())}


def load_human_labels() -> Dict[Tuple, bool]:
    labels = {}
    if HUMAN_LABELS_FILE.exists():
        with open(HUMAN_LABELS_FILE, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("is_correct", "").strip():
                    key = (r["question_id"], r["system"], r["model"], r["router"], int(r["run_idx"]))
                    labels[key] = r["is_correct"].strip().lower() in ("true", "1", "yes", "correct")
    return labels


def _json(value, default):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def check_must_include(answer: str, patterns: List[str]) -> bool:
    return all(re.search(p, answer, re.IGNORECASE) for p in patterns)


def score_run(row: pd.Series, q: Dict[str, Any], human: Dict[Tuple, bool]) -> Dict[str, Any]:
    answer = str(row.get("answer", "") or "")
    error = bool(str(row.get("error", "") or "").strip())
    route = _json(row.get("route"), [])
    tools = [t[0] if isinstance(t, (list, tuple)) else str(t) for t in _json(row.get("tools_called"), [])]
    cited = set(_json(row.get("cited_doc_ids"), []))
    retrieved = set(_json(row.get("retrieved"), []))
    autofilled = str(row.get("citation_autofilled", "")).lower() == "true"

    refused = (not error) and is_refusal_text(answer)
    logged = "log_unanswered" in tools or "logged" in answer.lower()

    if error:
        auto_correct = False
    elif q["should_refuse"]:
        auto_correct = refused
    else:
        auto_correct = (q.get("clarification") or not refused) and check_must_include(answer, q["must_include"])
    key = (q["id"], row["system"], row["model"], row["router"], int(row["run_idx"]))
    correct = human.get(key, auto_correct)

    gold = set(q["gold_sources"])
    citation_scope = (not q["should_refuse"]) and (not refused) and bool(gold) and not q.get("clarification")
    delegated = [a for a in route if a in SPECIALISTS]
    first_agent = delegated[0] if delegated else ("guardrail_governor" if "guardrail_governor" in route else "none")
    if q["should_refuse"]:
        routing_ok = first_agent in ("none", "guardrail_governor", q["expected_agent"])
    else:
        routing_ok = first_agent == q["expected_agent"]
    exp_tool = q.get("expected_tool", "")
    tool_ok = (not exp_tool) or exp_tool in tools or (exp_tool == "log_unanswered" and logged)

    return {
        "question_id": q["id"], "category": q["category"], "system": row["system"], "model": row["model"],
        "router": row["router"], "run_idx": int(row["run_idx"]), "should_refuse": q["should_refuse"],
        "error": error, "refused": refused, "logged": logged, "is_correct": bool(correct),
        "auto_correct": bool(auto_correct), "human_labelled": key in human,
        "citation_scope": citation_scope,
        "cited": bool(cited) and not autofilled if citation_scope else np.nan,
        "fabricated": bool(cited - retrieved) if citation_scope else np.nan,
        "correct_source": bool((cited | (retrieved if autofilled else set())) & gold) if citation_scope else np.nan,
        "autofilled": autofilled if citation_scope else np.nan,
        "first_agent": first_agent,
        "routing_correct": routing_ok if row["system"] == "multi" else np.nan,
        "tool_correct": tool_ok,
        "latency": float(row.get("latency", 0) or 0), "llm_calls": int(row.get("llm_calls", 0) or 0),
        "answer": answer,
    }


def score_runs(runs_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    questions, human = load_questions(), load_human_labels()
    scored = pd.DataFrame([score_run(r, questions[str(r["question_id"])], human)
                           for _, r in runs_df.iterrows() if str(r["question_id"]) in questions])

    summary = {}
    for keys, g in scored.groupby(GROUP_KEYS):
        sr, rf = g["should_refuse"].astype(bool), g["refused"].astype(bool)
        tp, fp, fn = int((rf & sr).sum()), int((rf & ~sr).sum()), int((~rf & sr).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        cs = g[g["citation_scope"]]
        summary["|".join(keys)] = {
            "system": keys[0], "model": keys[1], "router": keys[2],
            "runs": len(g), "errors": int(g["error"].sum()),
            "accuracy": g["is_correct"].mean(),
            "accuracy_in_scope": g.loc[~sr, "is_correct"].mean(),
            "refusal_precision": prec, "refusal_recall": rec,
            "refusal_f1": 2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
            "false_refusal_rate": fp / int((~sr).sum()) if (~sr).any() else float("nan"),
            "refusals_logged": g.loc[rf, "logged"].mean() if rf.any() else float("nan"),
            "citation_rate": cs["cited"].astype(float).mean(),
            "fabricated_rate": cs["fabricated"].astype(float).mean(),
            "correct_source_rate": cs["correct_source"].astype(float).mean(),
            "autofill_rate": cs["autofilled"].astype(float).mean(),
            "routing_accuracy": g["routing_correct"].astype(float).mean() if keys[0] == "multi" else float("nan"),
            "tool_accuracy": g["tool_correct"].mean(),
            "mean_latency": g["latency"].mean(), "median_latency": g["latency"].median(),
            "mean_llm_calls": g["llm_calls"].mean(),
        }
    return scored, summary


def compute_stability_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Per question and configuration: did correctness / refusal flip across repeated runs?"""
    records = []
    for (qid, *cfg), g in scored.groupby(["question_id"] + GROUP_KEYS):
        if len(g) < 2:
            continue
        answers = g["answer"].tolist()
        sims = [fuzz.token_sort_ratio(a, b) / 100 for i, a in enumerate(answers) for b in answers[i + 1:]]
        correct_flip = g["is_correct"].nunique() > 1
        refusal_flip = g["refused"].nunique() > 1
        records.append({
            "question_id": qid, "category": g["category"].iloc[0],
            **dict(zip(GROUP_KEYS, cfg)), "runs": len(g),
            "n_correct": int(g["is_correct"].sum()),
            "correct_flipped": correct_flip, "refusal_flipped": refusal_flip,
            "avg_similarity": round(float(np.mean(sims)), 3),
            "outcome_changed": correct_flip or refusal_flip,
        })
    return pd.DataFrame(records)
