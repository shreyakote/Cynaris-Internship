from langchain_classic.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama


def run_memory_demo():
    """Test ConversationBufferMemory across five conversation turns."""

    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    memory = ConversationBufferMemory()

    conversations = [
        "My name is Shreya.",
        "What is my name?",
        "I am studying Information Science.",
        "What am I studying?",
        "What is my name and what am I studying?",
    ]

    results = []

    for question in conversations:
        history = memory.load_memory_variables({})["history"]

        prompt = (
            "You are a helpful assistant. Use the conversation history "
            "to answer the user's question.\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User: {question}\n"
            "Assistant:"
        )

        response = llm.invoke(prompt)
        answer = response.content

        memory.save_context(
            {"input": question},
            {"output": answer},
        )

        results.append(
            f"User: {question}\n"
            f"Assistant: {answer}\n"
        )

    final_history = memory.load_memory_variables({})["history"]

    with open(
        "memory_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("CONVERSATION RESULTS\n")
        file.write("====================\n\n")
        file.write("\n".join(results))
        file.write("\n\nFINAL MEMORY HISTORY\n")
        file.write("====================\n\n")
        file.write(final_history)

    return results, final_history


if __name__ == "__main__":
    print("Testing ConversationBufferMemory with 5 turns...\n")

    outputs, history = run_memory_demo()

    for output in outputs:
        print(output)

    print("\nConversation history was successfully stored.")