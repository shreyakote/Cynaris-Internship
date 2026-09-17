import ast
import operator

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def web_search_stub(query: str) -> str:
    """Simulate a web search."""

    return (
        f"Web search result for '{query}': "
        "RAG stands for Retrieval-Augmented Generation. "
        "It retrieves relevant information and provides that context "
        "to a language model before generating an answer."
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operation = operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator.")

            return operation(
                evaluate(node.left),
                evaluate(node.right),
            )

        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op,
            (ast.UAdd, ast.USub),
        ):
            value = evaluate(node.operand)

            if isinstance(node.op, ast.UAdd):
                return value

            return -value

        raise ValueError("Invalid expression.")

    tree = ast.parse(expression, mode="eval")
    result = evaluate(tree)

    return f"Result: {result}"


def create_hybrid_agent():
    """Create a LangChain agent with two tools."""

    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=[
            web_search_stub,
            calculator,
        ],
        system_prompt=(
            "You are a helpful assistant. "
            "Use the calculator tool for arithmetic. "
            "Use the web_search_stub tool for web search questions."
        ),
    )

    return agent


def get_final_answer(result):
    """Get the final answer from the agent result."""

    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            content = message.content

            if isinstance(content, str) and content.strip():
                return content

    return "No final answer returned."


def run_three_tasks():
    """Run three agent tasks."""

    agent = create_hybrid_agent()

    tasks = [
        "Use the calculator tool to calculate 25 * 4 + 10.",
        "Use the web search stub to explain what RAG is.",
        "Use the calculator tool to calculate 100 / 4.",
    ]

    results = []

    for number, task in enumerate(tasks, start=1):
        print(f"\nTask {number}: {task}")

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task,
                    }
                ]
            }
        )

        answer = get_final_answer(result)

        print(f"Answer: {answer}")

        results.append(
            f"Task {number}: {task}\n"
            f"Answer: {answer}\n"
        )

    with open(
        "agent_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("LANGCHAIN AGENT RESULTS\n")
        file.write("=======================\n\n")
        file.write("\n".join(results))

    return results


if __name__ == "__main__":
    print("Testing LangChain agent with 2 tools and 3 tasks...")

    run_three_tasks()

    print("\nAgent testing completed.")