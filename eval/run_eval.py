"""
eval/run_eval.py - Evaluation runner.

Main grid:  {multi, single} x {qwen3:8b, llama3.1:8b} x N runs (default 3) on eval/questions.jsonl,
            appended to results/runs.csv (resumable: finished runs are skipped).
Ablation:   one model, one run, configurations that switch off a guardrail or change tau,
            appended to results/ablation_runs.csv.
Afterwards everything is scored and eval/plots.py regenerates figures and report/*.tex.

  python eval/run_eval.py --quick                  # 5 questions x 1 run, smoke test
  python eval/run_eval.py                          # full grid (hours on a CPU)
  python eval/run_eval.py --model qwen3:8b --system multi --runs 3
  python eval/run_eval.py --ablation --model qwen3:8b
  python eval/run_eval.py --score-only             # rescore after editing eval/human_labels.csv
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from helpdesk import config, guardrails  # noqa: E402
from eval.score import compute_stability_table, load_questions, score_runs  # noqa: E402
from main import execute_query  # noqa: E402

RUNS_CSV = config.RESULTS_DIR / "runs.csv"
ABLATION_CSV = config.RESULTS_DIR / "ablation_runs.csv"
MODELS = {"qwen3:8b": "ollama_chat/qwen3:8b", "llama3.1:8b": "ollama_chat/llama3.1:8b"}

ABLATIONS = {
    "full": {},
    "no_scope_guardrail": {"scope": False},
    "no_citation_guardrail": {"citation": False},
    "tau_0.28": {"tau": 0.28},
    "tau_0.45": {"tau": 0.45},
}


def apply_config(cfg: dict):
    guardrails.ENABLE_SCOPE_GUARDRAIL = cfg.get("scope", True)
    guardrails.ENABLE_CITATION_GUARDRAIL = cfg.get("citation", True)
    config.DEFAULT_TAU_THRESHOLD = cfg.get("tau", 0.35)


def _done_keys(path: Path, key_cols) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return {tuple(r) for r in df[key_cols].itertuples(index=False)}


def _append(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(path, mode="a", header=not path.exists(), index=False)


async def _run_one(q, system, model, router, run_idx, extra=None) -> dict:
    record = {"question_id": q["id"], "category": q["category"], "system": system,
              "model": model.split("/", 1)[-1], "router": router, "run_idx": run_idx, **(extra or {})}
    try:
        r = await execute_query(q["question"], system=system, model=model, router=router)
        record.update({
            "answer": r["answer"], "route": json.dumps(r["route"]),
            "tools_called": json.dumps(r["tools_called"], default=str),
            "cited_doc_ids": json.dumps(r["cited_doc_ids"]), "retrieved": json.dumps(r["retrieved"]),
            "citation_autofilled": r["citation_autofilled"], "latency": round(r["latency"], 2),
            "llm_calls": r["llm_calls"], "trace_file": r["trace_file"], "error": "",
        })
    except Exception as e:  # keep going, but record the failure so it is counted, not hidden
        record.update({"answer": "", "route": "[]", "tools_called": "[]", "cited_doc_ids": "[]",
                       "retrieved": "[]", "citation_autofilled": False, "latency": 0.0, "llm_calls": 0,
                       "trace_file": "", "error": f"{type(e).__name__}: {e}"})
        print(f"   ERROR: {record['error']}")
    record["timestamp"] = time.time()
    return record


def _models(model_filter: str):
    if model_filter == "both":
        return list(MODELS.values())
    return [MODELS.get(model_filter, model_filter if "/" in model_filter else f"ollama_chat/{model_filter}")]


async def run_evaluation(quick=False, system_filter="both", model_filter="both", router="llm", num_runs=3):
    questions = list(load_questions().values())
    if quick:
        questions, num_runs = questions[:5], 1
    systems = ["multi", "single"] if system_filter == "both" else [system_filter]
    key_cols = ["question_id", "system", "model", "router", "run_idx"]
    done = _done_keys(RUNS_CSV, key_cols)
    apply_config({})

    # Loop model-major so each 8B model is loaded into RAM only once.
    jobs = [(m, s, q, i) for m in _models(model_filter) for s in systems for q in questions
            for i in range(1, num_runs + 1)]
    for n, (model, system, q, run_idx) in enumerate(jobs, start=1):
        rtr = router if system == "multi" else "none"
        key = (q["id"], system, model.split("/", 1)[-1], rtr, str(run_idx))
        if key in done:
            continue
        print(f"[{n}/{len(jobs)}] {q['id']} {system}/{rtr} {key[2]} run {run_idx}: {q['question'][:70]}")
        rec = await _run_one(q, system, model, rtr, run_idx)
        _append(RUNS_CSV, rec)
        print(f"   {rec['latency']}s -> {rec['answer'][:100]!r}")
    report()


async def run_ablation(model_filter="qwen3:8b"):
    model = _models(model_filter)[0]
    questions = list(load_questions().values())
    key_cols = ["config", "question_id"]
    done = _done_keys(ABLATION_CSV, key_cols)
    for name, cfg in ABLATIONS.items():
        apply_config(cfg)
        for q in questions:
            if (name, q["id"]) in done:
                continue
            print(f"[ablation {name}] {q['id']}: {q['question'][:70]}")
            rec = await _run_one(q, "multi", model, "llm", 1, extra={"config": name})
            _append(ABLATION_CSV, rec)
    apply_config({})
    report()


def report():
    from eval.plots import generate_outputs
    runs = pd.read_csv(RUNS_CSV, keep_default_na=False) if RUNS_CSV.exists() else pd.DataFrame()
    ablation = pd.read_csv(ABLATION_CSV, keep_default_na=False) if ABLATION_CSV.exists() else pd.DataFrame()
    if runs.empty and ablation.empty:
        print("No results yet: report tables/macros will say TBD.")
    scored, summary, stability = pd.DataFrame(), {}, pd.DataFrame()
    if not runs.empty:
        scored, summary = score_runs(runs)
        stability = compute_stability_table(scored)
        scored.to_csv(config.RESULTS_DIR / "scored_runs.csv", index=False)
        stability.to_csv(config.RESULTS_DIR / "stability.csv", index=False)
        pd.DataFrame(summary.values()).to_csv(config.RESULTS_DIR / "summary.csv", index=False)
        for s in summary.values():
            print(f"{s['system']:6s} {s['model']:12s} {s['router']:7s} acc={s['accuracy']:.3f} "
                  f"refusalF1={s['refusal_f1']:.3f} correct_src={s['correct_source_rate']:.3f} "
                  f"lat={s['mean_latency']:.0f}s errors={s['errors']}")
    ablation_summary = {}
    if not ablation.empty:
        for name, g in ablation.groupby("config"):
            _, s = score_runs(g.assign(router="llm"))
            ablation_summary[name] = next(iter(s.values()))
        pd.DataFrame([{"config": k, **v} for k, v in ablation_summary.items()]).to_csv(
            config.RESULTS_DIR / "ablation_summary.csv", index=False)
    generate_outputs(scored, summary, stability, ablation_summary, runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--system", choices=["single", "multi", "both"], default="both")
    ap.add_argument("--model", default="both", help="qwen3:8b, llama3.1:8b, both, or any Ollama model")
    ap.add_argument("--router", choices=["llm", "keyword"], default="llm")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if a.score_only:
        report()
    elif a.ablation:
        asyncio.run(run_ablation(a.model if a.model != "both" else "qwen3:8b"))
    else:
        asyncio.run(run_evaluation(a.quick, a.system, a.model, a.router, a.runs))
