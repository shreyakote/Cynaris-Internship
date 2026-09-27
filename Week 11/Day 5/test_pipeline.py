
from unittest.mock import MagicMock, patch

import rag_pipeline


def test_answer_question_returns_expected_fields():
    """Check that the RAG function returns a question, answer, and contexts."""
    fake_document = MagicMock()
    fake_document.page_content = "RAG retrieves context before generating an answer."

    fake_response = MagicMock()
    fake_response.content = "RAG uses retrieved context to generate answers."

    with (
        patch.object(
            rag_pipeline,
            "build_vector_store",
        ) as mock_build_store,
        patch(
            "rag_pipeline.ChatOllama"
        ) as mock_chat,
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [fake_document]

        mock_build_store.return_value.as_retriever.return_value = (
            mock_retriever
        )
        mock_chat.return_value.invoke.return_value = fake_response

        result = rag_pipeline.answer_question("What is RAG?")

    assert result["question"] == "What is RAG?"
    assert result["answer"] == fake_response.content
    assert result["contexts"] == [fake_document.page_content]


def test_answer_question_returns_context_list():
    """Check that retrieved context is returned as a list."""
    fake_document = MagicMock()
    fake_document.page_content = "MLOps manages machine learning systems."

    fake_response = MagicMock()
    fake_response.content = "MLOps supports ML development and operations."

    with (
        patch.object(
            rag_pipeline,
            "build_vector_store",
        ) as mock_build_store,
        patch(
            "rag_pipeline.ChatOllama"
        ) as mock_chat,
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [fake_document]

        mock_build_store.return_value.as_retriever.return_value = (
            mock_retriever
        )
        mock_chat.return_value.invoke.return_value = fake_response

        result = rag_pipeline.answer_question("What is MLOps?")

    assert isinstance(result["contexts"], list)
    assert len(result["contexts"]) == 1