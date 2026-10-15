from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

MetricName = Literal["answer_correctness", "answer_relevance", "faithfulness", "hallucination_risk"]

class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value.strip()

class DatasetRow(BaseModel):
    question: str = Field(min_length=1)
    expected_answer: str | None = None
    context: str | None = None
    actual_answer: str | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()
