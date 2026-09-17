from langgraph.types import Command

from customer_support_agent import graph, process_customer_message


def test_account_request():
    result = process_customer_message(
        "I forgot my password",
        "test-account",
    )

    assert result["category"] == "account"
    assert result["route"] == "account"
    assert "account" in result["response"].lower()


def test_technical_request():
    result = process_customer_message(
        "The website is showing an error",
        "test-technical",
    )

    assert result["category"] == "technical"
    assert result["route"] == "technical"


def test_general_request():
    result = process_customer_message(
        "What are your support hours?",
        "test-general",
    )

    assert result["category"] == "general"
    assert result["route"] == "general"


def test_conversation_memory():
    thread_id = "test-memory"

    first = process_customer_message(
        "I forgot my password",
        thread_id,
    )

    second = process_customer_message(
        "I also need help with my account",
        thread_id,
    )

    assert len(first["conversation"]) >= 2
    assert len(second["conversation"]) >= 4


def test_human_review_and_resume():
    thread_id = "test-payment"

    paused = process_customer_message(
        "I was charged twice for my payment",
        thread_id,
    )

    assert paused["category"] == "payment"
    assert "__interrupt__" in paused

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    resumed = graph.invoke(
        Command(resume="Your duplicate payment complaint has been received."),
        config,
    )

    assert resumed["response"] == "Your duplicate payment complaint has been received."
