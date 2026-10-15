from unittest.mock import patch

import pytest
import minieval


def test_builtin_dataset_is_available_and_has_reference_context():
    rows = minieval.load_builtin_dataset()
    assert len(rows) >= 10
    assert all(row.question and row.expected_answer and row.context for row in rows)
    assert all(not row.actual_answer for row in rows)


def test_library_api_defaults_to_builtin_dataset_and_runs_all_metrics():
    fake_results = [{"row": 1, "metrics": {
        "answer_correctness": {"score": 0.9, "status": "PASS"},
        "answer_relevance": {"score": 0.9, "status": "PASS"},
        "faithfulness": {"score": 0.9, "status": "PASS"},
        "hallucination_risk": {"score": 0.1, "status": "PASS"},
    }}]
    with patch.object(minieval.OllamaClient, "list_models", return_value=["target", "judge"]), \
         patch.object(minieval, "evaluate_dataset", return_value=fake_results) as run_eval, \
         patch.object(minieval, "aggregate_results", return_value={"ok": True}):
        report = minieval.evaluate(target_model="target", judge_model="judge")
    assert report["results"] == fake_results
    assert report["summary"] == {"ok": True}
    assert run_eval.call_args.kwargs["rows"] if "rows" in run_eval.call_args.kwargs else run_eval.call_args.args[0]


def test_library_api_rejects_missing_target_model():
    with pytest.raises(ValueError, match="target_model is required"):
        minieval.evaluate(target_model=" ")


def test_library_api_reports_uninstalled_model():
    with patch.object(minieval.OllamaClient, "list_models", return_value=["installed"]):
        with pytest.raises(minieval.EvaluationError, match="not installed"):
            minieval.evaluate(target_model="missing", judge_model="installed")


def test_sdk_accepts_custom_python_model_callable_and_runs_all_metrics():
    fake_results = [{"row": 1, "metrics": {
        "answer_correctness": {"score": 0.9, "status": "PASS", "label": "Answer Correctness", "threshold": 0.7, "reason": "ok", "evidence": [], "execution_time_seconds": 0.1},
        "answer_relevance": {"score": 0.9, "status": "PASS", "label": "Answer Relevance", "threshold": 0.7, "reason": "ok", "evidence": [], "execution_time_seconds": 0.1},
        "faithfulness": {"score": 0.9, "status": "PASS", "label": "Faithfulness", "threshold": 0.7, "reason": "ok", "evidence": [], "execution_time_seconds": 0.1},
        "hallucination_risk": {"score": 0.1, "status": "PASS", "label": "Hallucination Risk", "threshold": 0.7, "reason": "ok", "evidence": [], "execution_time_seconds": 0.1},
    }}]
    model = lambda question: "answer"
    with patch.object(minieval.OllamaClient, "list_models", return_value=["llama3.2:latest"]), \
         patch.object(minieval, "evaluate_dataset", return_value=fake_results) as run_eval:
        report = minieval.evaluate(model=model, judge_model="llama3.2:latest")
    assert report["results"] == fake_results
    assert len(report["summary"]) == 4
    assert run_eval.call_args.kwargs["model_callable"] is model


def test_evaluation_report_can_save_json_and_csv(tmp_path):
    report = minieval.EvaluationReport(results=[{"row": 1, "question": "Q", "metrics": {
        "answer_correctness": {"label": "Answer Correctness", "score": 1.0, "status": "PASS", "threshold": 0.7, "reason": "Correct", "evidence": [], "execution_time_seconds": 0.1},
        "answer_relevance": {"label": "Answer Relevance", "score": 1.0, "status": "PASS", "threshold": 0.7, "reason": "Relevant", "evidence": [], "execution_time_seconds": 0.1},
        "faithfulness": {"label": "Faithfulness", "score": 1.0, "status": "PASS", "threshold": 0.7, "reason": "Supported", "evidence": [], "execution_time_seconds": 0.1},
        "hallucination_risk": {"label": "Hallucination Risk", "score": 0.0, "status": "PASS", "threshold": 0.7, "reason": "Low risk", "evidence": [], "execution_time_seconds": 0.1},
    }}], summary={})
    json_path, csv_path = report.save(tmp_path / "report")
    assert json_path.exists() and '"results"' in json_path.read_text(encoding="utf-8")
    assert csv_path.exists() and "Answer Correctness" in csv_path.read_text(encoding="utf-8-sig")
