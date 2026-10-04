from dataclasses import dataclass
import re

from app.rag.context.context_builder import RetrievalContext


@dataclass
class AnswerValidationResult:
    """
    Result produced by the answer validation layer.
    """

    is_valid: bool
    confidence: float
    reason: str


class AnswerValidator:
    """
    Validates whether a generated answer is sufficiently
    supported by the retrieved RAG context.

    This is a lightweight deterministic validator.

    It does NOT use another LLM yet.

    Pipeline:

        Generated Answer
              +
        Retrieved Context
              ↓
        AnswerValidator
              ↓
        Validation Result
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
    ) -> None:

        self.min_confidence = min_confidence

    def validate(
        self,
        answer: str,
        context: RetrievalContext,
    ) -> AnswerValidationResult:

        # --------------------------------------------------
        # Basic validation
        # --------------------------------------------------

        if not answer or not answer.strip():

            return AnswerValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="Generated answer is empty.",
            )

        if not context or not context.items:

            return AnswerValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="No retrieved evidence is available.",
            )

        answer_text = answer.strip()

        # --------------------------------------------------
        # Extract meaningful answer tokens
        # --------------------------------------------------

        answer_tokens = self._tokenize(
            answer_text
        )

        if not answer_tokens:

            return AnswerValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="Answer contains no meaningful tokens.",
            )

        # --------------------------------------------------
        # Build evidence text
        # --------------------------------------------------

        evidence_text = " ".join(
            item.text
            for item in context.items
        )

        evidence_tokens = self._tokenize(
            evidence_text
        )

        if not evidence_tokens:

            return AnswerValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="Retrieved evidence contains no meaningful text.",
            )

        evidence_token_set = set(
            evidence_tokens
        )

        # --------------------------------------------------
        # Token overlap
        # --------------------------------------------------

        matched_tokens = [
            token
            for token in answer_tokens
            if token in evidence_token_set
        ]

        overlap_ratio = (
            len(matched_tokens)
            / len(answer_tokens)
        )

        # --------------------------------------------------
        # Important-word overlap
        #
        # Ignore extremely common words because they
        # provide little evidence of factual grounding.
        # --------------------------------------------------

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "for",
            "with",
            "that",
            "this",
            "it",
            "as",
            "by",
            "from",
        }

        important_tokens = [
            token
            for token in answer_tokens
            if token not in stop_words
        ]

        if important_tokens:

            important_matches = [
                token
                for token in important_tokens
                if token in evidence_token_set
            ]

            important_overlap = (
                len(important_matches)
                / len(important_tokens)
            )

        else:

            important_overlap = overlap_ratio

        # --------------------------------------------------
        # Confidence
        #
        # Give greater weight to meaningful terms.
        # --------------------------------------------------

        confidence = (
            0.30 * overlap_ratio
            + 0.70 * important_overlap
        )

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        # --------------------------------------------------
        # Final decision
        # --------------------------------------------------

        if confidence >= self.min_confidence:

            return AnswerValidationResult(
                is_valid=True,
                confidence=confidence,
                reason=(
                    "Answer contains sufficient "
                    "overlap with retrieved evidence."
                ),
            )

        return AnswerValidationResult(
            is_valid=False,
            confidence=confidence,
            reason=(
                "Answer is insufficiently supported "
                "by retrieved evidence."
            ),
        )

    @classmethod
    def is_unanswerable_from_context(cls, query: str, context: RetrievalContext) -> bool:
        """Detect a small class of clearly unsupported fact requests.

        This is intentionally conservative: it is used to fail closed for
        requests such as an exact AWS account number when the retrieved
        evidence contains no account identifier at all.
        """
        q = (query or "").lower()
        evidence = " ".join(item.text for item in (context.items if context else []))
        e = evidence.lower()
        if "account number" in q or ("aws" in q and "account" in q):
            if "account number" not in e and not re.search(r"\b\d{8,16}\b", evidence):
                return True
        return False

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )