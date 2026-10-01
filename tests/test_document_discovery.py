from app.rag.discovery.document_discovery import (
    DocumentDiscovery,
)


def main():

    print("=" * 70)
    print("PHOENIX AI - DOCUMENT DISCOVERY TEST")
    print("=" * 70)

    # ==========================================================
    # DISCOVERY ENGINE
    # ==========================================================

    discovery = DocumentDiscovery()

    root_path = r"R:\\"

    print(
        f"\nSearch Root: {root_path}"
    )

    # ==========================================================
    # FULL DISCOVERY
    # ==========================================================

    print("\n" + "=" * 70)
    print("RECURSIVE DOCUMENT DISCOVERY")
    print("=" * 70)

    documents = discovery.discover(
        root_path
    )

    print(
        f"\nDocuments Found: "
        f"{len(documents)}"
    )

    # Don't print thousands of files.
    # Show only the first 20.

    for index, document in enumerate(
        documents[:20],
        start=1,
    ):

        print(
            f"\n[{index}]"
        )

        print(
            f"File: "
            f"{document.file_name}"
        )

        print(
            f"Type: "
            f"{document.file_type}"
        )

        print(
            f"Path: "
            f"{document.path}"
        )

    if len(documents) > 20:

        print(
            f"\n... "
            f"{len(documents) - 20} "
            f"additional documents"
        )

    # ==========================================================
    # FIND SPECIFIC DOCUMENT
    # ==========================================================

    target = (
        "practical_statistics"
    )

    print("\n" + "=" * 70)
    print(
        f"SEARCHING FOR: {target}"
    )
    print("=" * 70)

    matches = discovery.find_by_filename(
        root_path=root_path,
        file_name=target,
    )

    print(
        f"\nMatches Found: "
        f"{len(matches)}"
    )

    for index, document in enumerate(
        matches,
        start=1,
    ):

        print(
            f"\nMatch {index}"
        )

        print(
            f"File: "
            f"{document.file_name}"
        )

        print(
            f"Type: "
            f"{document.file_type}"
        )

        print(
            f"Path: "
            f"{document.path}"
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    print("\n" + "=" * 70)

    if matches:

        print(
            "DOCUMENT DISCOVERY STATUS: PASSED"
        )

    else:

        print(
            "DOCUMENT DISCOVERY STATUS: "
            "NO MATCH FOUND"
        )

    print("=" * 70)

    print(
        "\nDOCUMENT DISCOVERY TEST COMPLETE"
    )


if __name__ == "__main__":
    main()