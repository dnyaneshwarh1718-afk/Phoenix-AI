import time

from app.rag.discovery.local_drive_finder import (
    LocalDriveDocumentFinder,
    MatchStatus,
)


def test_real_r_drive_finder():

    finder = LocalDriveDocumentFinder(
        primary_root=r"R:\\",
    )

    queries = [
        "SQL INTERVIEW QUESTIONS.docx",
        "SQL INTERVIEW QUESTIONS",
        "document_that_definitely_does_not_exist_12345",
    ]

    print("\n")
    print("=" * 70)
    print("PHOENIX AI - REAL LOCAL DRIVE FINDER TEST")
    print("=" * 70)

    for query in queries:

        print(f"\nQuery: {query}")

        start = time.perf_counter()

        result = finder.find(query)

        elapsed = time.perf_counter() - start

        print(f"Status:        {result.status.value}")
        print(f"Match Type:    {result.match_type.value}")

        if result.path:
            print(f"Path:          {result.path}")

        if result.score is not None:
            print(f"Score:         {result.score:.4f}")

        print(f"Searched Root: {result.searched_root}")
        print(f"Time:          {elapsed:.4f} seconds")
        print(f"Message:       {result.message}")

    print("\n")
    print("=" * 70)