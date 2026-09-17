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

    words = set(
        user_input.replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .split()
    )

    technical_words = {
        "api",
        "python",
        "code",
        "coding",
        "error",
        "bug",
        "database",
        "server",
        "programming",
        "langgraph",
    }

    support_words = {
        "payment",
        "account",
        "password",
        "login",
        "refund",
        "problem",
        "issue",
        "complaint",
    }

    if words & technical_words:
        classification = "technical"
    elif words & support_words:
        classification = "support"
    else:
        classification = "general"

    return {"classification": classification}


def route_node(state: AgentState):
    return {"route": state["classification"]}


def respond_node(state: AgentState):
    route = state["route"]
    user_input = state["input"]

    if route == "technical":
        response = f"Technical response: I received your request about '{user_input}'."

    elif route == "support":
        human_response = interrupt(
            {
                "message": "Human assistance is required.",
                "user_input": user_input,
                "question": "Please provide the human response.",
            }
        )

        return {
            "human_input": str(human_response),
            "response": str(human_response),
        }

    else:
        response = f"General response: I received your request about '{user_input}'."

    return {"response": response}


def route_condition(state: AgentState):
    return state["route"]


def build_graph():
    builder = StateGraph(AgentState)

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
            "general": "respond",
            "support": "respond",
        },
    )

    builder.add_edge("respond", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()


def run_graph(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(
        {"input": user_input},
        config,
    )


def resume_graph(human_response: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}

    return graph.invoke(
        Command(resume=human_response),
        config,
    )


if __name__ == "__main__":
    print("\n--- 5 INPUT TEST ---")

    inputs = [
        "Explain Python programming",
        "What is artificial intelligence?",
        "How does an API work?",
        "Tell me about LangGraph",
        "What is machine learning?",
    ]

    for user_input in inputs:
        result = run_graph(user_input)

        print(f"\nInput: {user_input}")
        print(f"Classification: {result['classification']}")
        print(f"Route: {result['route']}")
        print(f"Response: {result['response']}")

    print("\n--- HUMAN-IN-THE-LOOP TEST ---")

    thread_id = "human-demo"

    paused = run_graph(
        "I have a payment issue",
        thread_id,
    )

    print("\nGraph paused for human input.")
    print(paused["__interrupt__"])

    human_response = input("\nEnter human response: ")

    resumed = resume_graph(
        human_response,
        thread_id,
    )

    print("\nGraph resumed.")
    print("Final response:", resumed["response"])
