import requests
import json

url = "http://localhost:11434/api/generate"

system_prompt = (
    "You are an AI tutor. "
    "Answer clearly, briefly, and with simple examples."
)

prompts = [
    "What is Machine Learning?",
    "Explain Python in simple words.",
    "Write a short poem about Artificial Intelligence.",
    "What is the difference between AI and ML?",
    "Give three uses of Generative AI."
]

for i, prompt in enumerate(prompts, start=1):
    payload = {
        "model": "llama3.2:3b",
        "prompt": f"System: {system_prompt}\nUser: {prompt}",
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("=" * 60)
        print(f"Prompt {i}: {prompt}")
        print("-" * 60)
        print(result["response"])
        print()
    else:
        print(f"Error: {response.status_code}")