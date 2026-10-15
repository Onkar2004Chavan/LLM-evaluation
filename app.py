import streamlit as st
import pandas as pd
from evaluator import load_dataset, evaluate_dataset, export_json, export_csv, aggregate_results, flatten_results, EvaluationError
from ollama_client import OllamaClient, OllamaError

st.set_page_config(page_title="MiniEval", page_icon="🧪", layout="wide")
st.title("MiniEval — Dataset-based LLM Evaluation")
st.caption("Run all applicable metrics automatically. Your target model generates answers; Ollama judge evaluates them.")
client = OllamaClient()
try:
    models = client.list_models()
except OllamaError as exc:
    st.error(str(exc))
    st.info("Start Ollama, then refresh this page. No models are downloaded automatically.")
    st.stop()
if not models:
    st.warning("No Ollama models are installed. Install a model manually, then refresh.")
    st.stop()
with st.sidebar:
    st.header("Models and scoring")
    judge_model = st.selectbox("Judge model", models)
    target_options = ["Use existing actual_answer column"] + models
    target_choice = st.selectbox("Target LLM", target_options)
    threshold = st.slider("Pass threshold", 0.0, 1.0, 0.7, 0.05)
    st.caption("Correctness/relevance/faithfulness pass at score ≥ threshold. Hallucination risk passes at score ≤ threshold.")

uploaded = st.file_uploader("Upload evaluation dataset", type=["csv", "json", "jsonl"])
st.markdown("Required field: `question`. Optional fields: `expected_answer`, `context`, `actual_answer`. If actual_answer is missing, select a target model to generate answers.")
if uploaded:
    import tempfile, os
    suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue()); path = tmp.name
    try:
        rows = load_dataset(path)
        st.write(f"Loaded **{len(rows)}** dataset rows.")
        st.dataframe(pd.DataFrame([r.model_dump() for r in rows]), use_container_width=True)
        if st.button("Run complete evaluation", type="primary"):
            needs_generation = any(not row.actual_answer for row in rows)
            if needs_generation and target_choice == "Use existing actual_answer column":
                st.error("Some rows have no actual_answer. Select a target model to generate those answers.")
            else:
                target_model = target_choice if needs_generation else "provided-answers"
                bar = st.progress(0)
                status = st.empty()
                def progress(done, total):
                    bar.progress(done / total); status.write(f"Evaluating row {done}/{total}")
                try:
                    with st.spinner("Running model inference and judge evaluations..."):
                        results = evaluate_dataset(rows, target_model=target_model, judge_model=judge_model, threshold=threshold, client=client, progress=progress)
                    st.session_state["minieval_results"] = results
                except (EvaluationError, OllamaError) as exc:
                    st.error(str(exc))
    except EvaluationError as exc:
        st.error(str(exc))
    finally:
        os.unlink(path)

results = st.session_state.get("minieval_results")
if results:
    st.subheader("Aggregate results")
    summary = aggregate_results(results)
    st.dataframe(pd.DataFrame([{"Metric": v["label"], "Rows evaluated": v["evaluated"], "Mean score": v["mean_score"], "Pass": v["pass_count"], "Fail": v["fail_count"]} for v in summary.values()]), use_container_width=True)
    st.caption("N/A metrics are excluded from averages and pass/fail counts. No incompatible overall score is calculated.")
    st.subheader("Per-row results")
    st.dataframe(flatten_results(results), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download JSON report", export_json(results), file_name="minieval_results.json", mime="application/json")
    c2.download_button("Download CSV report", export_csv(results), file_name="minieval_results.csv", mime="text/csv")
