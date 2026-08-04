import requests

URL = "http://localhost:11434/api/generate"

system_prompt = """
You are an AI tutor.
Always answer in simple English.
Keep answers under 120 words.
Give one practical example whenever possible.
"""

questions = [
    "Explain Machine Learning.",
    "What is Prompt Engineering?",
    "Difference between AI and ML?",
    "What is Python?",
    "Explain Neural Networks."
]

for i, question in enumerate(questions, start=1):

    payload = {
        "model": "llama3.2:3b",
        "system": system_prompt,
        "prompt": question,
        "stream": False
    }

    response = requests.post(URL, json=payload)

    print("="*70)
    print(f"Question {i}: {question}")
    print("-"*70)

    if response.status_code == 200:
        print(response.json()["response"])
    else:
        print("Error:", response.text)

    print()