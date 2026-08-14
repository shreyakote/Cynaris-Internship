from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------
# OLLAMA LLM
# --------------------------------------------------

llm = OllamaLLM(model="llama3.2:3b")


# --------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------

conversation_history = []


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful AI assistant.

Previous conversation:
{history}

Current user message:
{input}

Answer the user using the previous conversation when relevant.
"""
)


# --------------------------------------------------
# FUNCTION TO FORMAT HISTORY
# --------------------------------------------------

def get_history():
    if not conversation_history:
        return "No previous conversation."

    history_text = ""

    for user_message, ai_response in conversation_history:
        history_text += f"User: {user_message}\n"
        history_text += f"AI: {ai_response}\n"

    return history_text


# --------------------------------------------------
# FIVE CONVERSATION TURNS
# --------------------------------------------------

questions = [
    "My name is Shreya.",
    "I am studying Information Science Engineering.",
    "What is my name?",
    "What am I studying?",
    "What do you remember about me?"
]


print("=" * 60)
print("CONVERSATION MEMORY DEMO")
print("=" * 60)


for i, question in enumerate(questions, start=1):

    print(f"\n--- Turn {i} ---")
    print("User:", question)

    history = get_history()

    formatted_prompt = prompt.format(
        history=history,
        input=question
    )

    response = llm.invoke(formatted_prompt)

    print("AI:", response)

    # Save conversation
    conversation_history.append(
        (question, response)
    )


# --------------------------------------------------
# DISPLAY FINAL HISTORY
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL CONVERSATION HISTORY")
print("=" * 60)

print(get_history())