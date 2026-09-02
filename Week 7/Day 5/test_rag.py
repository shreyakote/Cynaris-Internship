from pathlib import Path

from rag_system import load_documents, check_documents


def test_documents_folder_exists():
    """Check that the documents folder exists."""
    folder = Path("documents")

    assert folder.exists(), (
        "documents folder does not exist"
    )


def test_pdf_documents_exist():
    """Check that PDF documents are available."""
    files = list(
        Path("documents").glob("*.pdf")
    )

    assert len(files) >= 1, (
        "No PDF documents found"
    )


def test_document_loading():
    """Check that PDF documents can be loaded."""
    documents = load_documents()

    assert documents is not None
    assert len(documents) >= 1


def test_document_check():
    """Check that the document validation works."""
    files = check_documents()

    assert len(files) >= 1