import ollama
import time


# ============================================================
# CONFIGURATION
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant for an Information Science
engineering student.

Answer questions clearly and in simple words.

For technical questions:
1. Give a short definition.
2. Explain the main points.
3. Give a simple example when useful.

Do not use unnecessary complex terminology.
"""


MODEL = "llama3.2:3b"


# ============================================================
# FUNCTION TO CALL OLLAMA
# ============================================================

def ask_ollama(question):

    start_time = time.perf_counter()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    latency = time.perf_counter() - start_time

    answer = response["message"]["content"]

    return answer, latency


# ============================================================
# 5 TEST PROMPTS
# ============================================================

prompts = [

    "What is Artificial Intelligence?",

    "Explain Machine Learning in simple words.",

    "What is a Large Language Model?",

    "What is RAG and why is it useful?",

    "Explain the difference between AI and Machine Learning."
]


# ============================================================
# RUN TESTS
# ============================================================

print("=" * 60)
print("OLLAMA LOCAL LLM INFERENCE")
print("=" * 60)

print("Model:", MODEL)

print("\nSystem Prompt:")
print(SYSTEM_PROMPT)

print("\n" + "=" * 60)
print("TESTING 5 PROMPTS")
print("=" * 60)


for number, prompt in enumerate(prompts, start=1):

    print("\n" + "-" * 60)

    print(f"PROMPT {number}")
    print("-" * 60)

    print(prompt)

    answer, latency = ask_ollama(prompt)

    print("\nRESPONSE:")
    print(answer)

    print(f"\nLatency: {latency:.2f} seconds")


print("\n" + "=" * 60)
print("LOCAL INFERENCE COMPLETED")
print("=" * 60)