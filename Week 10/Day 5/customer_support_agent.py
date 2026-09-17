from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class SupportState(TypedDict, total=False):
    customer_message: str
    category: str
    route: str
    response: str
    conversation: list[str]


def classify_customer(state: SupportState):
    """Classify the customer's request."""
    message = state["customer_message"].lower()

    payment_words = {
        "payment",
        "refund",
        "transaction",
        "charged",
        "billing",
    }

    account_words = {
        "account",
        "password",
        "login",
        "sign in",
        "profile",
    }

    technical_words = {
        "error",
        "bug",
        "api",
        "website",
        "application",
        "app",
    }

    if any(word in message for word in payment_words):
        category = "payment"
    elif any(word in message for word in account_words):
        category = "account"
    elif any(word in message for word in technical_words):
        category = "technical"
    else:
        category = "general"

    return {"category": category}


def route_customer(state: SupportState):
    """Route the customer request based on its category."""
    return {"route": state["category"]}


def respond_to_customer(state: SupportState):
    """Generate a response and keep the conversation state."""
    category = state["route"]
    message = state["customer_message"]

    conversation = state.get("conversation", [])
    conversation = conversation + [f"Customer: {message}"]

    if category == "payment":
        response = (
            "Your payment-related request has been received. "
            "A support agent will review the transaction."
        )

    elif category == "account":
        response = (
            "Your account request has been received. "
            "Please follow the account recovery process."
        )

    elif category == "technical":
        response = (
            "Your technical issue has been received. "
            "Please provide the error details if further investigation is needed."
        )

    else:
        response = (
            "Thank you for contacting customer support. "
            "Please provide more details about your request."
        )

    conversation.append(f"Agent: {response}")

    return {
        "response": response,
        "conversation": conversation,
    }


def human_review(state: SupportState):
    """Pause the workflow when human support is required."""
    human_response = interrupt(
        {
            "message": "Human support review required.",
            "customer_message": state["customer_message"],
            "category": state["category"],
        }
    )

    conversation = state.get("conversation", [])
    conversation = conversation + [f"Human agent: {human_response}"]

    return {
        "response": str(human_response),
        "conversation": conversation,
    }


def build_support_graph():
    """Build and compile the stateful customer support graph."""
    builder = StateGraph(SupportState)

    builder.add_node("classify", classify_customer)
    builder.add_node("route", route_customer)
    builder.add_node("respond", respond_to_customer)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")

    builder.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "payment": "human_review",
            "account": "respond",
            "technical": "respond",
            "general": "respond",
        },
    )

    builder.add_edge("respond", END)
    builder.add_edge("human_review", END)

    return builder.compile(checkpointer=InMemorySaver())


graph = build_support_graph()


def process_customer_message(
    message: str,
    thread_id: str = "customer-1",
):
    """Process a customer message using a conversation thread."""
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    return graph.invoke(
        {
            "customer_message": message,
        },
        config,
    )


def resume_customer_conversation(
    human_response: str,
    thread_id: str = "customer-1",
):
    """Resume a paused customer conversation."""
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    return graph.invoke(
        Command(resume=human_response),
        config,
    )


if __name__ == "__main__":
    print("\n--- STATEFUL CUSTOMER SUPPORT AGENT ---")

    thread_id = "demo-customer"

    messages = [
        "I forgot my password",
        "The website is showing an error",
        "What are your support hours?",
    ]

    for message in messages:
        result = process_customer_message(message, thread_id)

        print(f"\nCustomer: {message}")
        print(f"Category: {result['category']}")
        print(f"Response: {result['response']}")

    print("\n--- HUMAN SUPPORT REVIEW ---")

    payment_result = process_customer_message(
        "I was charged twice for my payment",
        "payment-customer",
    )

    print("\nPayment request requires human review.")
    print(payment_result["__interrupt__"])

    human_response = input("\nEnter human support response: ")

    resumed = resume_customer_conversation(
        human_response,
        "payment-customer",
    )

    print("\nConversation resumed.")
    print("Final response:", resumed["response"])
