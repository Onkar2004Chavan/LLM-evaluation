import pytest
from pydantic import ValidationError
from schemas import JudgeResponse, DatasetRow

def test_valid_score():
    assert JudgeResponse(score=0.8, reason="Mostly correct").score == 0.8

@pytest.mark.parametrize("score", [-0.01, 1.01, "not a number"])
def test_invalid_score(score):
    with pytest.raises(ValidationError):
        JudgeResponse(score=score, reason="bad")

def test_dataset_row_question_required():
    with pytest.raises(ValidationError):
        DatasetRow(question="   ")
