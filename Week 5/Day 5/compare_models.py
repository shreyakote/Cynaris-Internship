import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]


SYSTEM_PROMPT = """
You are a helpful technical assistant.

Answer the question clearly and accurately.
Use simple language.
Keep the answer reasonably concise.
"""


QUESTIONS = [
    "Explain artificial intelligence in simple words.",
    "What is Retrieval Augmented Generation?",
    "What are the advantages of running an LLM locally?"
]


def ask_model(model, question):

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": question,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    response.raise_for_status()

    return response.json()["response"]


print("=" * 70)
print("OLLAMA MODEL COMPARISON")
print("=" * 70)


for question_number, question in enumerate(QUESTIONS, start=1):

    print("\n" + "=" * 70)
    print(f"QUESTION {question_number}")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    for model in MODELS:

        print("\n" + "-" * 70)
        print("MODEL:", model)
        print("-" * 70)

        answer = ask_model(model, question)

        print(answer)