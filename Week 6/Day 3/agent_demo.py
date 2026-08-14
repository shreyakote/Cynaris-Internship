from langchain_ollama import OllamaLLM
from langchain_core.tools import tool


# --------------------------------------------------
# TOOL 1: WEB SEARCH STUB
# --------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for information. This is a demo web search stub."""

    results = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "ollama": "Ollama allows users to run large language models locally.",
        "rag": "RAG stands for Retrieval-Augmented Generation. It combines document retrieval with language generation."
    }

    query_lower = query.lower()

    for keyword, result in results.items():
        if keyword in query_lower:
            return result

    return f"No specific result found for: {query}"


# --------------------------------------------------
# TOOL 2: CALCULATOR
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        allowed_characters = "0123456789+-*/(). "

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(expression)

        return str(result)

    except Exception:
        return "Could not calculate the expression."


# --------------------------------------------------
# DISPLAY TOOLS
# --------------------------------------------------

print("=" * 60)
print("LANGCHAIN TOOLS & AGENT DEMO")
print("=" * 60)

print("\nAvailable Tools:")

print("1. Web Search Stub")
print("2. Calculator")


# --------------------------------------------------
# TEST TOOL 1
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 1 - WEB SEARCH")
print("=" * 60)

result = web_search.invoke({
    "query": "What is LangChain?"
})

print("Result:")
print(result)


# --------------------------------------------------
# TEST TOOL 2
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 2 - CALCULATOR")
print("=" * 60)

result = calculator.invoke({
    "expression": "125 * 8 + 50"
})

print("Result:")
print(result)


# --------------------------------------------------
# TEST WEB SEARCH AGAIN
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 3 - WEB SEARCH")
print("=" * 60)

result = web_search.invoke({
    "query": "What is RAG?"
})

print("Result:")
print(result)


# --------------------------------------------------
# AGENT INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("AGENT TASKS COMPLETED")
print("=" * 60)

print("Task 1: Search information about LangChain")
print("Task 2: Calculate 125 * 8 + 50")
print("Task 3: Search information about RAG")

print("\nBoth tools executed successfully.")