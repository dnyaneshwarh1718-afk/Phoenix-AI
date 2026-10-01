from dataclasses import dataclass, field
from typing import Any


@dataclass
class RRFResult:
    """
    Standard result returned by Reciprocal Rank Fusion.

    RRF combines ranked results from multiple retrieval
    systems such as:

        Dense Retrieval
        +
        BM25 Retrieval
        ↓
        Reciprocal Rank Fusion
        ↓
        RRFResult
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    retrieval_method: str = "rrf"


def reciprocal_rank_fusion(
    result_lists: list[list[Any]],
    k: int = 60,
    limit: int = 10,
) -> list[RRFResult]:
    """
    Combine multiple ranked retrieval result lists
    using Reciprocal Rank Fusion.

    RRF formula:

        RRF(d) = Σ 1 / (k + rank)

    where:

        k    = ranking constant
        rank = 1-based rank of the document
        d    = retrieved chunk

    Parameters
    ----------
    result_lists:
        List of ranked result lists.

        Example:

            [
                dense_results,
                bm25_results
            ]

    k:
        RRF ranking constant.
        Standard value is 60.

    limit:
        Maximum number of fused results.

    Returns
    -------
    list[RRFResult]
        Results sorted by descending RRF score.
    """

    if not result_lists:
        return []

    if k <= 0:
        raise ValueError(
            "RRF parameter 'k' must be greater than 0."
        )

    if limit <= 0:
        return []

    scores: dict[str, float] = {}
    result_objects: dict[str, Any] = {}

    # --------------------------------------------------
    # Process every retrieval system
    # --------------------------------------------------

    for results in result_lists:

        if not results:
            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):

            # ------------------------------------------
            # Extract chunk ID
            # ------------------------------------------

            chunk_id = getattr(
                result,
                "chunk_id",
                None,
            )

            # Some retrievers may return a
            # DocumentChunk directly.
            if chunk_id is None:

                chunk = getattr(
                    result,
                    "chunk",
                    None,
                )

                if chunk is not None:

                    chunk_id = getattr(
                        chunk,
                        "chunk_id",
                        None,
                    )

            if chunk_id is None:
                continue

            chunk_id = str(chunk_id)

            # ------------------------------------------
            # Calculate RRF contribution
            # ------------------------------------------

            contribution = 1.0 / (
                k + rank
            )

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + contribution
            )

            # Keep the first complete result object.
            #
            # Dense and BM25 should point to the same
            # DocumentChunk, so either object contains
            # enough information to construct the
            # final result.
            if chunk_id not in result_objects:

                result_objects[
                    chunk_id
                ] = result

    # --------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # --------------------------------------------------
    # Build final RRF results
    # --------------------------------------------------

    final_results: list[RRFResult] = []

    for chunk_id, rrf_score in ranked[:limit]:

        result = result_objects[
            chunk_id
        ]

        # --------------------------------------------------
        # Extract fields from DenseRetrievalResult /
        # BM25RetrievalResult
        # --------------------------------------------------

        # BM25 result stores DocumentChunk in `.chunk`.
        chunk = getattr(
            result,
            "chunk",
            None,
        )

        # Dense result stores fields directly.
        if chunk is not None:

            result_chunk_id = getattr(
                chunk,
                "chunk_id",
                chunk_id,
            )

            document_id = getattr(
                chunk,
                "document_id",
                "",
            )

            chunk_index = getattr(
                chunk,
                "chunk_index",
                -1,
            )

            text = getattr(
                chunk,
                "text",
                "",
            )

            metadata = getattr(
                chunk,
                "metadata",
                {},
            )

        else:

            result_chunk_id = getattr(
                result,
                "chunk_id",
                chunk_id,
            )

            document_id = getattr(
                result,
                "document_id",
                "",
            )

            chunk_index = getattr(
                result,
                "chunk_index",
                -1,
            )

            text = getattr(
                result,
                "text",
                "",
            )

            metadata = getattr(
                result,
                "metadata",
                {},
            )

        # --------------------------------------------------
        # Defensive normalization
        # --------------------------------------------------

        if metadata is None:
            metadata = {}

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        final_results.append(
            RRFResult(
                chunk_id=str(
                    result_chunk_id
                ),
                document_id=str(
                    document_id
                ),
                chunk_index=int(
                    chunk_index
                ),
                text=str(
                    text
                ),
                score=float(
                    rrf_score
                ),
                metadata=metadata,
                retrieval_method="rrf",
            )
        )

    return final_results