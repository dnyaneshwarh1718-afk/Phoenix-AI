from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "evaluation" / "corpus"


def test_evaluation_corpus_contains_all_supported_formats():
    expected = {
        ".txt", ".docx", ".xlsx", ".pptx", ".pdf"
    }
    actual = {p.suffix.lower() for p in CORPUS.iterdir() if p.is_file()}
    assert expected <= actual


def test_evaluation_corpus_contains_ground_truth_facts():
    text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in CORPUS.glob("*.txt")
    )
    assert "Qdrant" in text
    assert "qwen3:4b-instruct" in text
    assert "RRF" in text


def test_evaluation_corpus_has_no_empty_files():
    files = [p for p in CORPUS.iterdir() if p.is_file()]
    assert files
    assert all(p.stat().st_size > 0 for p in files)
