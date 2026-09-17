from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def create_chain():
    """Create a PromptTemplate -> Ollama -> OutputParser chain."""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Answer the following question clearly and briefly.\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    parser = StrOutputParser()

    return prompt | llm | parser


def run_five_inputs():
    """Test the chain with five different inputs."""

    chain = create_chain()

    questions = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is RAG?",
        "What is an LLM?",
        "What is LangChain?",
    ]

    results = []

    for question in questions:
        answer = chain.invoke({"question": question})

        results.append(
            f"Question: {question}\n"
            f"Answer: {answer}\n"
        )

    with open(
        "chain_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(results))

    return results


if __name__ == "__main__":
    print("Testing LangChain chain with 5 inputs...\n")

    outputs = run_five_inputs()

    for output in outputs:
        print(output)