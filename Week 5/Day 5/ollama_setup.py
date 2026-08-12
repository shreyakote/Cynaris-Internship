import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


SYSTEM_PROMPT = """
You are a helpful technical assistant.

Give clear and simple answers.
Keep your answers concise.
Use examples when they help explain the concept.
If you are unsure about something, clearly say so.
"""


def ask_ollama(question):
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": question,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


prompts = [
    "What is artificial intelligence?",
    "Explain machine learning in simple words.",
    "What is a vector database?",
    "What is Retrieval Augmented Generation?",
    "What are the advantages of running an LLM locally?"
]


print("=" * 60)
print("OLLAMA LOCAL LLM INFERENCE")
print("=" * 60)

print("\nModel:", MODEL)

for i, prompt in enumerate(prompts, start=1):

    print("\n" + "-" * 60)
    print(f"PROMPT {i}")
    print("-" * 60)

    print("Question:", prompt)

    answer = ask_ollama(prompt)

    print("\nAnswer:")
    print(answer)