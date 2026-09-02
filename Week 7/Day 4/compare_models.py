import ollama
import time


models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]


questions = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning in simple words.",
    "What is RAG and why is it useful?"
]


system_prompt = """
You are a helpful AI assistant.

Answer clearly, accurately and in simple words.
For technical questions, explain the important concepts
and provide a simple example when appropriate.
"""


print("=" * 70)
print("LLAMAINDEX + OLLAMA LOCAL MODEL COMPARISON")
print("=" * 70)


for model in models:

    print("\n")
    print("=" * 70)
    print("MODEL:", model)
    print("=" * 70)

    for number, question in enumerate(questions, start=1):

        print("\n" + "-" * 70)

        print(f"QUESTION {number}:")
        print(question)

        start_time = time.perf_counter()

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        latency = time.perf_counter() - start_time

        answer = response["message"]["content"]

        print("\nANSWER:")
        print(answer)

        print(f"\nLatency: {latency:.2f} seconds")


print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)