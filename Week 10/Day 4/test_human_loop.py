from langgraph.types import Command

from langgraph_human_loop import graph, run_graph


def test_technical_route():
    result = run_graph("Explain Python programming", "test-technical")

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_general_route():
    result = run_graph("What is artificial intelligence?", "test-general")

    assert result["classification"] == "general"
    assert result["route"] == "general"


def test_api_route():
    result = run_graph("How does an API work?", "test-api")

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_langgraph_route():
    result = run_graph("Tell me about LangGraph", "test-langgraph")

    assert result["classification"] == "technical"
    assert result["route"] == "technical"


def test_human_in_the_loop():
    thread_id = "test-human-loop"

    paused = run_graph(
        "I have a payment issue",
        thread_id,
    )

    assert "__interrupt__" in paused

    config = {"configurable": {"thread_id": thread_id}}

    resumed = graph.invoke(
        Command(resume="Your issue has been received."),
        config,
    )

    assert resumed["response"] == "Your issue has been received."
