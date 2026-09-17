from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class AgentState(TypedDict, total=False):
    input: str
    classification: str
    route: str
    response: str
    human_input: str


def classify_node(state: AgentState):
    user_input = state["input"].lower()

    # Remove common punctuation so words are detected correctly.
    words = set(
        user_input.replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .split()
    )

    technical_words = {
        "python",
        "code",
        "api",
        "bug",
        "error",
        "docker",
        "database",
        "langgraph",
        "machine",
        "learning",
    }

    support_words = {
        "payment",
        "refund",
        "login",
        "account",
        "password",
        "support",
    }

    if words.intersection(technical_words):
        classification = "technical"

    elif words.intersection(support_words):
        classification = "support"

    else:
        classification = "general"

    return {
        "classification": classification
    }


def route_node(state: AgentState):
    classification = state["classification"]

    return {
        "route": classification
    }


def respond_node(state: AgentState):
    route = state["route"]
    user_input = state["input"]

    if route == "technical":

        response = (
            f"Technical request detected: '{user_input}'. "
            "This request should be handled by the technical response path."
        )

        return {
            "response": response
        }

    elif route == "support":

        human_input = interrupt(
            {
                "type": "human_review",
                "message": (
                    "A support request needs human review. "
                    "Enter the response that should be sent to the user."
                ),
            }
        )

        return {
            "human_input": str(human_input),
            "response": str(human_input),
        }

    else:

        response = (
            f"General request detected: '{user_input}'. "
            "This request should be handled by the general response path."
        )

        return {
            "response": response
        }


def route_condition(state: AgentState):
    return state["route"]


def build_graph():

    builder = StateGraph(AgentState)

    # Three required nodes
    builder.add_node("classify", classify_node)
    builder.add_node("route", route_node)
    builder.add_node("respond", respond_node)

    # Start -> Classify
    builder.add_edge(
        START,
        "classify"
    )

    # Classify -> Route
    builder.add_edge(
        "classify",
        "route"
    )

    # Conditional routing
    builder.add_conditional_edges(
        "route",
        route_condition,
        {
            "technical": "respond",
            "general": "respond",
            "support": "respond",
        },
    )

    # Respond -> End
    builder.add_edge(
        "respond",
        END
    )

    # Required for human-in-the-loop interrupt/resume
    memory = InMemorySaver()

    return builder.compile(
        checkpointer=memory
    )


graph = build_graph()


def run_graph(
    user_input: str,
    thread_id: str = "default",
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return graph.invoke(
        {
            "input": user_input
        },
        config,
    )


def resume_graph(
    human_response: str,
    thread_id: str = "default",
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return graph.invoke(
        Command(
            resume=human_response
        ),
        config,
    )


if __name__ == "__main__":

    test_inputs = [
        "How do I fix a Python error?",
        "Explain LangGraph state.",
        "What is the capital of France?",
        "Tell me a joke.",
        "How does an API work?",
    ]

    print()
    print("--- LangGraph Test ---")
    print()

    for user_input in test_inputs:

        result = run_graph(user_input)

        print(f"Input: {user_input}")
        print(f"Classification: {result['classification']}")
        print(f"Route: {result['route']}")
        print(f"Response: {result['response']}")
        print("-" * 60)