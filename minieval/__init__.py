"""MiniEval: a lightweight SDK for evaluating user-built LLMs."""
from pathlib import Path
from typing import Any, Callable
import json

from .evaluator import (
    EvaluationError, aggregate_results, evaluate_dataset, export_csv,
    export_json, flatten_results, load_dataset,
)
from .ollama_client import OllamaClient, OllamaError

__version__ = "0.3.0"


class EvaluationReport(dict):
    """Dictionary-compatible evaluation report with convenient display/export methods."""

    def print_summary(self) -> None:
        print("MiniEval Evaluation Summary")
        print("=" * 28)
        for key, item in self.get("summary", {}).items():
            mean = item.get("mean_score")
            mean_text = "N/A" if mean is None else f"{mean:.3f}"
            print(f"{item.get('label', key)}: mean={mean_text}, passed={item.get('pass_count', 0)}, failed={item.get('fail_count', 0)}, evaluated={item.get('evaluated', 0)}")
        print(f"Questions evaluated: {len(self.get('results', []))}")

    def print_results(self) -> None:
        try:
            from .evaluator import flatten_results
            frame = flatten_results(self.get("results", []))
            if frame.empty:
                print("No results available.")
            else:
                columns = [c for c in ["row", "metric", "score", "status", "reason"] if c in frame.columns]
                print(frame[columns].to_string(index=False, max_colwidth=100))
        except Exception as exc:
            raise EvaluationError(f"Could not display evaluation results: {exc}") from exc

    def save(self, path: str | Path = "minieval_results") -> tuple[Path, Path]:
        """Save JSON and CSV reports; suffixes are added if not supplied."""
        path = Path(path)
        if path.suffix.lower() in {".json", ".csv"}:
            path = path.with_suffix("")
        json_path = path.with_suffix(".json")
        csv_path = path.with_suffix(".csv")
        json_path.write_text(export_json(self.get("results", [])), encoding="utf-8")
        csv_path.write_text(export_csv(self.get("results", [])), encoding="utf-8-sig")
        return json_path, csv_path


def builtin_dataset_path() -> Path:
    """Return the path to MiniEval's bundled general evaluation dataset."""
    return Path(__file__).parent / "data" / "general_eval.csv"


def load_builtin_dataset():
    """Load the included general-purpose evaluation dataset."""
    return load_dataset(builtin_dataset_path())


def evaluate(*, model: Callable[[str], str] | None = None,
             target_model: str | None = None, judge_model: str | None = None,
             dataset: str | Path | None = None, threshold: float = 0.7,
             ollama_url: str | None = None, progress=None) -> EvaluationReport:
    """Evaluate a user's LLM with MiniEval's built-in dataset by default.

    Provide either ``model`` (a callable accepting a question and returning an answer)
    or ``target_model`` (an installed Ollama model name). Ollama is currently used for
    judging. All applicable metrics run automatically.

    Args:
        model: User's model callable, e.g. ``lambda question: my_llm.generate(question)``.
        target_model: Installed Ollama target model; alternative to ``model``.
        judge_model: Installed Ollama judge model. For an Ollama target, defaults to
            that target model; for a callable target, defaults to the first installed model.
        dataset: Optional CSV/JSON/JSONL path. Defaults to the bundled dataset.
        threshold: Pass threshold in [0, 1].
        ollama_url: Optional Ollama base URL.
        progress: Optional callback called as ``progress(done, total)``.

    Returns an ``EvaluationReport`` with ``results`` and ``summary`` keys, plus
    ``print_summary()``, ``print_results()``, and ``save()`` methods.
    """
    if model is not None and not callable(model):
        raise TypeError("model must be a callable that accepts a question and returns a string")
    if model is None and (not target_model or not target_model.strip()):
        raise ValueError("target_model is required unless model=<callable> is provided")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    client = OllamaClient(base_url=ollama_url)
    installed = client.list_models()
    if not installed:
        raise EvaluationError("No Ollama models are installed. Install a judge model manually, then retry.")

    if model is None:
        target_model = target_model.strip()
        if target_model not in installed:
            raise EvaluationError(f"Target model '{target_model}' is not installed. Installed models: {', '.join(installed)}")
        target_label = target_model
    else:
        target_label = getattr(model, "__name__", None) or model.__class__.__name__
        if target_label == "<lambda>":
            target_label = "custom_python_model"

    judge_model = (judge_model or (target_model if model is None else installed[0]) or "").strip()
    if judge_model not in installed:
        raise EvaluationError(f"Judge model '{judge_model}' is not installed. Installed models: {', '.join(installed)}")

    rows = load_dataset(dataset or builtin_dataset_path())
    results = evaluate_dataset(rows, target_model=target_label, judge_model=judge_model,
                               threshold=threshold, client=client,
                               model_callable=model, progress=progress)
    return EvaluationReport(results=results, summary=aggregate_results(results))


__all__ = [
    "evaluate", "EvaluationReport", "load_builtin_dataset", "builtin_dataset_path", "load_dataset",
    "evaluate_dataset", "aggregate_results", "export_json", "export_csv",
    "flatten_results", "EvaluationError", "OllamaClient", "OllamaError",
]
