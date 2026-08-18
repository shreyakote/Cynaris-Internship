from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# --------------------------------------------------
# 1. Create Ollama LLM
# --------------------------------------------------

llm = OllamaLLM(
    model="llama3.2:3b"
)


# --------------------------------------------------
# 2. Create Conversation Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Use the conversation history to answer the user's questions.

Remember information provided earlier in the conversation.
"""
    ),

    MessagesPlaceholder(
        variable_name="history"
    ),

    (
        "human",
        "{input}"
    )
])


# --------------------------------------------------
# 3. Conversation History
# --------------------------------------------------

history = []


# --------------------------------------------------
# 4. Chat Function
# --------------------------------------------------

def chat(user_input):

    global history

    # Create messages using previous history
    messages = prompt.format_messages(
        history=history,
        input=user_input
    )

    # Send to Ollama
    response = llm.invoke(messages)

    # Save user message
    history.append(
        HumanMessage(
            content=user_input
        )
    )

    # Save AI response
    history.append(
        AIMessage(
            content=response
        )
    )

    return response


# --------------------------------------------------
# 5. Five Conversation Turns
# --------------------------------------------------

questions = [
    "My name is Shreya.",
    "What is my name?",
    "I am learning LangChain.",
    "What am I learning?",
    "What is my name and what am I learning?"
]


# --------------------------------------------------
# 6. Run Conversation
# --------------------------------------------------

print("=" * 60)
print("WEEK 6 DAY 5 - CONVERSATION MEMORY")
print("=" * 60)

for i, question in enumerate(questions, start=1):

    print(f"\nTurn {i}")
    print("-" * 40)

    print("User:", question)

    answer = chat(question)

    print("AI:", answer)


# --------------------------------------------------
# 7. Display Conversation History
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONVERSATION HISTORY")
print("=" * 60)

for message in history:

    if isinstance(message, HumanMessage):

        print("Human:", message.content)

    elif isinstance(message, AIMessage):

        print("AI:", message.content)


print("\n" + "=" * 60)
print("5-TURN MEMORY TEST COMPLETED")
print("=" * 60)