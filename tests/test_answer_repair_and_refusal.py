from types import SimpleNamespace

from app.rag.context.context_builder import ContextBuilder
from app.rag.validation.answer_validator import AnswerValidator


def _context(text):
    result = SimpleNamespace(
        chunk_id="c1", document_id="d1", text=text, score=0.9,
        retrieval_method="hybrid", metadata={}
    )
    return ContextBuilder().build("question", [result])


def test_validator_detects_missing_aws_account_fact():
    ctx = _context("Phoenix AI uses Qdrant as its vector database.")
    assert AnswerValidator.is_unanswerable_from_context(
        "What is Phoenix AI's exact production AWS account number?", ctx
    )


def test_validator_does_not_mark_supported_aws_fact_unanswerable():
    ctx = _context("Production AWS account number: 123456789012")
    assert not AnswerValidator.is_unanswerable_from_context(
        "What is Phoenix AI's exact production AWS account number?", ctx
    )
