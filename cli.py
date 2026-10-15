import argparse
import json
import sys
from pathlib import Path

from minieval import (
    EvaluationError, OllamaClient, OllamaError, aggregate_results,
    builtin_dataset_path, evaluate, export_csv, export_json, load_dataset,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="MiniEval: evaluate an Ollama LLM on a built-in dataset or your own CSV/JSON/JSONL dataset."
    )
    parser.add_argument("--dataset", help="Optional dataset path; defaults to MiniEval's built-in general evaluation dataset")
    parser.add_argument("--target-model", required=True, help="Installed Ollama model to evaluate")
    parser.add_argument("--judge-model", help="Installed Ollama judge model (defaults to target model)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Pass threshold from 0 to 1 (default: 0.7)")
    parser.add_argument("--out", default="minieval_results", help="Output filename prefix (default: minieval_results)")
    parser.add_argument("--ollama-url", help="Ollama URL; defaults to OLLAMA_BASE_URL or http://localhost:11434")
    args = parser.parse_args(argv)
    try:
        dataset_path = args.dataset or str(builtin_dataset_path())
        client = OllamaClient(base_url=args.ollama_url)
        installed = client.list_models()
        judge_model = args.judge_model or args.target_model
        if args.target_model not in installed:
            raise EvaluationError(f"Target model '{args.target_model}' is not installed. Installed models: {', '.join(installed) or '(none)'}")
        if judge_model not in installed:
            raise EvaluationError(f"Judge model '{judge_model}' is not installed. Installed models: {', '.join(installed) or '(none)'}")
        rows = load_dataset(dataset_path)
        def progress(done, total):
            print(f"Evaluating row {done}/{total}...", file=sys.stderr)
        from minieval.evaluator import evaluate_dataset
        results = evaluate_dataset(rows, target_model=args.target_model, judge_model=judge_model,
                                   threshold=args.threshold, client=client, progress=progress)
        prefix = Path(args.out)
        if prefix.parent != Path('.'):
            prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = prefix.with_suffix(".json")
        csv_path = prefix.with_suffix(".csv")
        json_path.write_text(export_json(results), encoding="utf-8")
        csv_path.write_text(export_csv(results), encoding="utf-8-sig")
        print("\nAggregate summary")
        print(json.dumps(aggregate_results(results), indent=2, ensure_ascii=False))
        print(f"\nDetailed JSON: {json_path.resolve()}\nFlat CSV: {csv_path.resolve()}")
        return 0
    except (EvaluationError, OllamaError, ValueError) as exc:
        print(f"MiniEval error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
