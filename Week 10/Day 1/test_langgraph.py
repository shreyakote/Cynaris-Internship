from langgraph.types import Command

from langgraph_agent import graph, run_graph


def test_technical_route():
    result = run_graph(
        "How do I fix a Python error?",
        "test-technical",
    )

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_general_route():
    result = run_graph(
        "What is the capital of France?",
        "test-general",
    )

    assert result["classification"] == "general"
    assert result["route"] == "general"


def test_api_route():
    result = run_graph(
        "How does an API work?",
        "test-api",
    )

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_langgraph_route():
    result = run_graph(
        "Explain LangGraph state.",
        "test-langgraph",
    )

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_support_route():
    result = run_graph(
        "I have a payment issue.",
        "test-support",
    )

    assert result["classification"] == "support"
    assert result["route"] == "support"


def test_human_in_the_loop():
    thread_id = "test-human-loop"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # First run pauses at the human-in-the-loop interrupt.
    result = graph.invoke(
        {
            "input": "I have a payment issue."
        },
        config,
    )

    assert "__interrupt__" in result
    assert len(result["__interrupt__"]) > 0

    # Resume the paused graph with human input.
    resumed = graph.invoke(
        Command(
            resume="Please ask the customer for their payment ID."
        ),
        config,
    )

    assert resumed["response"] == (
        "Please ask the customer for their payment ID."
    )