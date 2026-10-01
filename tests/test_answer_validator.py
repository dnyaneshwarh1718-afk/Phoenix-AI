from app.rag.context.context_builder import (
    ContextBuilder,
)

from app.rag.validation.answer_validator import (
    AnswerValidator,
)


def main():

    print("=" * 70)
    print("PHOENIX AI - ANSWER VALIDATION TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Test evidence
    # --------------------------------------------------

    answer = (
        "The Orchestrator Agent controls "
        "the specialized agents."
    )

    print(
        f"\nAnswer:\n{answer}"
    )

    # --------------------------------------------------
    # Minimal mock context
    # --------------------------------------------------

    from app.rag.retrieval.rrf import RRFResult

    result = RRFResult(
        chunk_id="test-chunk-001",
        document_id="test-document-001",
        chunk_index=0,
        text=(
            "The Orchestrator Agent is the central "
            "controller of Phoenix AI. It receives "
            "user requests and delegates work to "
            "specialized agents."
        ),
        score=0.032,
        retrieval_method="rrf",
        metadata={},
    )

    context = ContextBuilder().build(
        query="What component controls the specialized agents?",
        results=[result],
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    validator = AnswerValidator(
        min_confidence=0.50
    )

    validation = validator.validate(
        answer=answer,
        context=context,
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print(
        f"\nValid: {validation.is_valid}"
    )

    print(
        f"Confidence: "
        f"{validation.confidence:.4f}"
    )

    print(
        f"Reason: "
        f"{validation.reason}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANSWER VALIDATION TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()