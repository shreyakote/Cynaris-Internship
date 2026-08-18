from langchain_core.tools import tool
import ast
import operator


# ==================================================
# TOOL 1 - WEB SEARCH STUB
# ==================================================

@tool
def web_search_stub(query: str) -> str:
    """
    Simulates a web search.
    This is a stub and does not access the real internet.
    """

    return (
        f"Simulated web search result for: '{query}'\n"
        "This is a web search stub. "
        "No real search API is connected."
    )


# ==================================================
# TOOL 2 - CALCULATOR
# ==================================================

@tool
def calculator(expression: str) -> str:
    """
    Performs basic mathematical calculations.
    Example: 25 * 8
    """

    try:

        # Allowed mathematical operations
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod
        }

        # Evaluate expression safely
        def evaluate(node):

            # Expression
            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            # Numbers
            if isinstance(node, ast.Constant):

                if isinstance(
                    node.value,
                    (int, float)
                ):
                    return node.value

                raise ValueError(
                    "Only numbers are allowed."
                )

            # Binary operations
            if isinstance(node, ast.BinOp):

                left = evaluate(node.left)

                right = evaluate(node.right)

                operation = allowed_operators.get(
                    type(node.op)
                )

                if operation is None:

                    raise ValueError(
                        "Unsupported operator."
                    )

                return operation(
                    left,
                    right
                )

            # Negative numbers
            if isinstance(node, ast.UnaryOp):

                operand = evaluate(
                    node.operand
                )

                if isinstance(
                    node.op,
                    ast.USub
                ):
                    return -operand

                if isinstance(
                    node.op,
                    ast.UAdd
                ):
                    return operand

            raise ValueError(
                "Invalid mathematical expression."
            )

        # Parse expression
        tree = ast.parse(
            expression,
            mode="eval"
        )

        result = evaluate(tree)

        return str(result)

    except Exception as e:

        return f"Calculation error: {e}"


# ==================================================
# AVAILABLE TOOLS
# ==================================================

tools = [
    web_search_stub,
    calculator
]


# ==================================================
# SIMPLE AGENT FUNCTION
# ==================================================

def run_agent(task):

    print("\nTask:", task)

    task_lower = task.lower()

    # ------------------------------------------------
    # Calculator Task
    # ------------------------------------------------

    if (
        "calculate" in task_lower
        or "solve" in task_lower
        or any(
            symbol in task
            for symbol in ["+", "-", "*", "/"]
        )
    ):

        expression = task

        # Remove common words
        words_to_remove = [
            "calculate",
            "Calculate",
            "what is",
            "What is",
            "solve",
            "Solve"
        ]

        for word in words_to_remove:

            expression = expression.replace(
                word,
                ""
            )

        expression = expression.strip()

        result = calculator.invoke(
            expression
        )

        print("Tool Used: Calculator")
        print("Expression:", expression)
        print("Result:", result)

        return result

    # ------------------------------------------------
    # Web Search Stub
    # ------------------------------------------------

    else:

        result = web_search_stub.invoke(
            task
        )

        print("Tool Used: Web Search Stub")
        print("Result:", result)

        return result


# ==================================================
# TEST 3 AGENT TASKS
# ==================================================

tasks = [

    "What is LangChain?",

    "Calculate 25 * 8",

    "Search for information about RAG"

]


# ==================================================
# RUN AGENT
# ==================================================

print("=" * 60)
print("WEEK 6 DAY 5 - SIMPLE AGENT")
print("=" * 60)

print("\nAvailable Tools:")

for tool in tools:

    print("-", tool.name)


for i, task in enumerate(
    tasks,
    start=1
):

    print(f"\nTask {i}")
    print("-" * 40)

    run_agent(task)


print("\n" + "=" * 60)
print("3 AGENT TASKS COMPLETED")
print("=" * 60)