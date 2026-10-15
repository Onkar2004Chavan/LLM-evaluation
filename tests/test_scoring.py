from evaluator import _passed, aggregate_results

def test_normal_threshold_logic():
    assert _passed("answer_relevance", 0.7, 0.7)
    assert not _passed("answer_correctness", 0.69, 0.7)
    assert _passed("faithfulness", 0.9, 0.7)

def test_hallucination_threshold_is_inverted():
    assert _passed("hallucination_risk", 0.2, 0.7)
    assert _passed("hallucination_risk", 0.7, 0.7)
    assert not _passed("hallucination_risk", 0.71, 0.7)

def test_aggregate_excludes_unavailable():
    results = [{"metrics": {"answer_correctness": {"score": None, "status": "N/A"},
      "answer_relevance": {"score": .8, "status": "PASS"}, "faithfulness": {"score": None, "status": "N/A"},
      "hallucination_risk": {"score": None, "status": "N/A"}}}]
    summary = aggregate_results(results)
    assert summary["answer_correctness"]["evaluated"] == 0
    assert summary["answer_relevance"]["mean_score"] == .8
