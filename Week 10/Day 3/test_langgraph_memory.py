from langgraph_memory import chat, get_conversation


def test_technical_classification():
    result = chat(
        "How do I fix a Python error?",
        "test-technical",
    )

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_general_classification():
    result = chat(
        "What is the capital of France?",
        "test-general",
    )

    assert result["classification"] == "general"
    assert result["route"] == "general"


def test_api_classification():
    result = chat(
        "How does an API work?",
        "test-api",
    )

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_support_classification():
    result = chat(
        "I have a payment issue.",
        "test-support",
    )

    assert result["classification"] == "support"
    assert result["route"] == "support"


def test_five_conversation_inputs():
    thread_id = "test-five-inputs"

    inputs = [
        "I want to learn Python.",
        "How does an API work?",
        "What is LangGraph?",
        "Tell me a joke.",
        "I have a payment issue.",
    ]

    for user_input in inputs:
        chat(user_input, thread_id)

    conversation = get_conversation(thread_id)

    assert len(conversation) == 10


def test_persistent_conversation():
    thread_id = "test-persistent-memory"

    first = chat(
        "My name is Shreya.",
        thread_id,
    )

    second = chat(
        "I am learning Python.",
        thread_id,
    )

    conversation = get_conversation(thread_id)

    assert first["response"]
    assert second["response"]

    assert "User: My name is Shreya." in conversation
    assert "User: I am learning Python." in conversation