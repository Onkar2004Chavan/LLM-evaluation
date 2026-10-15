import csv
import json
import time
from pathlib import Path
from typing import Any
import pandas as pd
from .schemas import DatasetRow
from .ollama_client import OllamaClient, OllamaError
from .judge import LLMJudge, JudgeError
from .metrics import METRICS

class EvaluationError(RuntimeError):
    pass

def load_dataset(path: str | Path) -> list[DatasetRow]:
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f"Dataset file not found: {path}")
    try:
        if path.suffix.lower() == ".csv":
            records = pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else data.get("data", [])
        else:
            raise EvaluationError("Supported dataset formats: .csv, .json, .jsonl")
        if not records:
            raise EvaluationError("Dataset contains no rows.")
        rows = []
        for index, record in enumerate(records, 1):
            try:
                normalized = {str(k).strip().lower(): (v if v is not None and str(v).strip() else None) for k, v in record.items()}
                if "question" not in normalized:
                    raise ValueError("required column 'question' is missing")
                rows.append(DatasetRow.model_validate(normalized))
            except Exception as exc:
                raise EvaluationError(f"Invalid dataset row {index}: {exc}") from exc
        return rows
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise EvaluationError(f"Could not read dataset: {exc}") from exc

def _passed(metric: str, score: float, threshold: float) -> bool:
    return score <= threshold if metric == "hallucination_risk" else score >= threshold

def evaluate_dataset(rows: list[DatasetRow], *, target_model: str, judge_model: str,
                     threshold: float = 0.7, client: OllamaClient | None = None,
                     model_callable=None, progress=None) -> list[dict[str, Any]]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    client = client or OllamaClient()
    judge = LLMJudge(client, judge_model)
    results = []
    for row_index, row in enumerate(rows, 1):
        started = time.perf_counter()
        try:
            answer = row.actual_answer
            generation_time = 0.0
            if not answer:
                generation_started = time.perf_counter()
                if model_callable is not None:
                    try:
                        answer = model_callable(row.question)
                    except Exception as exc:
                        raise EvaluationError(f"Target model failed on dataset row {row_index}: {exc}") from exc
                    if not isinstance(answer, str) or not answer.strip():
                        raise EvaluationError(f"Target model returned an empty or non-string answer on dataset row {row_index}.")
                    answer = answer.strip()
                else:
                    answer = client.generate(target_model, f"Answer the question as accurately and directly as possible.\n\nQuestion:\n{row.question}")
                generation_time = round(time.perf_counter() - generation_started, 3)
            result_row: dict[str, Any] = {
                "row": row_index, "question": row.question, "actual_answer": answer,
                "expected_answer": row.expected_answer, "context": row.context,
                "target_model": target_model, "judge_model": judge_model, "threshold": threshold,
                "metrics": {}, "generation_time_seconds": generation_time,
            }
            # If answer was generated here, generation timing is included in total row time.
            for metric, config in METRICS.items():
                if metric == "answer_correctness" and not row.expected_answer:
                    result_row["metrics"][metric] = {"label": config["label"], "score": None, "status": "N/A", "threshold": threshold, "reason": "No expected_answer supplied.", "evidence": [], "execution_time_seconds": 0.0}
                    continue
                if metric in {"faithfulness", "hallucination_risk"} and not row.context:
                    result_row["metrics"][metric] = {"label": config["label"], "score": None, "status": "N/A", "threshold": threshold, "reason": "No supporting context supplied.", "evidence": [], "execution_time_seconds": 0.0}
                    continue
                judged = judge.evaluate(question=row.question, actual_answer=answer,
                    expected_answer=row.expected_answer, context=row.context, metric=metric, rubric=config["rubric"])
                result_row["metrics"][metric] = {"label": config["label"], "score": judged.score,
                    "status": "PASS" if _passed(metric, judged.score, threshold) else "FAIL",
                    "threshold": threshold, "reason": judged.reason, "evidence": judged.evidence,
                    "execution_time_seconds": judged.execution_time_seconds}
            result_row["row_execution_time_seconds"] = round(time.perf_counter() - started, 3)
            results.append(result_row)
        except (OllamaError, JudgeError) as exc:
            raise EvaluationError(f"Evaluation failed on dataset row {row_index}: {exc}") from exc
        if progress:
            progress(row_index, len(rows))
    return results

def flatten_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for item in results:
        base = {k: item.get(k) for k in ["row", "question", "actual_answer", "expected_answer", "context", "target_model", "judge_model", "threshold", "row_execution_time_seconds"]}
        for metric, data in item["metrics"].items():
            record = dict(base)
            record.update({"metric": data["label"], "metric_key": metric, "score": data["score"],
                "status": data["status"], "metric_threshold": data["threshold"], "reason": data["reason"],
                "evidence": json.dumps(data["evidence"], ensure_ascii=False),
                "execution_time_seconds": data["execution_time_seconds"]})
            records.append(record)
    return pd.DataFrame(records)

def aggregate_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for metric, config in METRICS.items():
        values = [row["metrics"][metric] for row in results if row["metrics"][metric]["score"] is not None]
        summary[metric] = {"label": config["label"], "evaluated": len(values),
            "mean_score": round(sum(v["score"] for v in values) / len(values), 4) if values else None,
            "pass_count": sum(v["status"] == "PASS" for v in values),
            "fail_count": sum(v["status"] == "FAIL" for v in values)}
    return summary

def export_json(results: list[dict[str, Any]]) -> str:
    return json.dumps({"results": results, "summary": aggregate_results(results)}, ensure_ascii=False, indent=2)

def export_csv(results: list[dict[str, Any]]) -> str:
    return flatten_results(results).to_csv(index=False)
