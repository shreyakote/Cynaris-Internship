from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class ConversationState(TypedDict, total=False):
    input: str
    classification: str
    route: str
    conversation: list[str]
    response: str


def classify_node(state: ConversationState):
    user_input = state["input"].lower()

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


def route_node(state: ConversationState):
    return {
        "route": state["classification"]
    }


def respond_node(state: ConversationState):
    user_input = state["input"]
    route = state["route"]

    old_conversation = state.get("conversation", [])

    if route == "technical":
        response = (
            f"Technical request detected: '{user_input}'. "
            "This is handled by the technical response path."
        )

    elif route == "support":
        response = (
            f"Support request detected: '{user_input}'. "
            "This is handled by the support response path."
        )

    else:
        response = (
            f"General request detected: '{user_input}'. "
            "This is handled by the general response path."
        )

    updated_conversation = old_conversation + [
        f"User: {user_input}",
        f"Assistant: {response}",
    ]

    return {
        "conversation": updated_conversation,
        "response": response,
    }


def route_condition(state: ConversationState):
    return state["route"]


def build_graph():
    builder = StateGraph(ConversationState)

    builder.add_node("classify", classify_node)
    builder.add_node("route", route_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")

    builder.add_conditional_edges(
        "route",
        route_condition,
        {
            "technical": "respond",
            "support": "respond",
            "general": "respond",
        },
    )

    builder.add_edge("respond", END)

    # Checkpoint memory makes conversation state persistent
    # for the same thread_id.
    memory = InMemorySaver()

    return builder.compile(checkpointer=memory)


graph = build_graph()


def chat(user_input: str, thread_id: str = "conversation-1"):
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


def get_conversation(thread_id: str = "conversation-1"):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = graph.get_state(config)

    return state.values.get("conversation", [])


if __name__ == "__main__":

    thread_id = "demo-conversation"

    messages = [
        "Hello, I want to learn Python.",
        "How does an API work?",
        "What is LangGraph?",
        "Tell me a simple joke.",
        "I have a payment issue.",
    ]

    print()
    print("--- LangGraph Persistent Conversation Demo ---")
    print()

    for message in messages:
        result = chat(message, thread_id)

        print(f"User: {message}")
        print(f"Classification: {result['classification']}")
        print(f"Route: {result['route']}")
        print(f"Assistant: {result['response']}")
        print()

    print("--- Stored Conversation ---")
    print()

    for message in get_conversation(thread_id):
        print(message)