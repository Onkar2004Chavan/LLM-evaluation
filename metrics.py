METRICS = {
    "answer_correctness": {
        "label": "Answer Correctness", "direction": "higher",
        "rubric": "Compare the actual answer with the expected answer for factual and semantic correctness. Partial credit is appropriate. Do not require exact wording. Score 1 for fully correct, 0 for entirely incorrect."
    },
    "answer_relevance": {
        "label": "Answer Relevance", "direction": "higher",
        "rubric": "Judge how directly the actual answer addresses the question, whether it answers the central request, and whether it includes irrelevant or distracting content."
    },
    "faithfulness": {
        "label": "Faithfulness", "direction": "higher",
        "rubric": "Assess whether the factual claims in the actual answer are supported by the supplied context. Score high when claims are supported; lower when claims are unsupported or contradicted. Distinguish unsupported from demonstrably contradicted claims in the reason/evidence."
    },
    "hallucination_risk": {
        "label": "Hallucination Risk", "direction": "lower",
        "rubric": "Estimate the proportion and severity of actual-answer claims that are unsupported by or contradicted by the supplied context. Score 0 for no detected risk and 1 for very high risk. Unsupported does not automatically mean false; identify contradicted claims separately."
    },
}
