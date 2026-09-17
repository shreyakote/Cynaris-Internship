import ast
import operator


# --------------------------------------------------
# Safe Calculator
# --------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def calculator(expression):
    """Safely calculate a basic mathematical expression."""

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
        return str(result)
    except (ValueError, TypeError, ZeroDivisionError, SyntaxError) as error:
        return f"Calculation error: {error}"


def _evaluate(node):
    """Evaluate an allowed mathematical expression."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operation = _ALLOWED_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Operator is not allowed.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value

        raise ValueError("Unary operator is not allowed.")

    raise ValueError("Invalid mathematical expression.")


# --------------------------------------------------
# Web Search Stub
# --------------------------------------------------

def web_search(query):
    """
    Simple web-search stub for the research pipeline.

    The function returns a predictable response so the
    research agent can be tested without external APIs.
    """

    return (
        f"Web search stub result for: {query}. "
        "External web search can be connected here later."
    )


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":
    print("Calculator:", calculator("10 + 5 * 2"))
    print("Web Search:", web_search("RAG"))