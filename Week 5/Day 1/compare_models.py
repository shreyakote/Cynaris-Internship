import requests

url = "http://localhost:11434/api/generate"

questions = [
    "What is Machine Learning?",
    "Explain Neural Networks.",
    "Write Python code to find factorial using recursion."
]

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

for question in questions:
    print("=" * 80)
    print("QUESTION:")
    print(question)

    for model in models:
        payload = {
            "model": model,
            "prompt": question,
            "stream": False
        }

        response = requests.post(url, json=payload)

        print("\n" + "-" * 80)
        print("MODEL:", model)

        if response.status_code == 200:
            print(response.json()["response"])
        else:
            print("Error:", response.status_code)

    print()
    