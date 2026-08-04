import requests

URL = "http://localhost:11434/api/generate"

prompt = "Explain Generative AI."

systems = [

    "Answer in one sentence.",

    "Answer in detail with examples.",

    "Answer as if teaching a 10-year-old."
]

for system in systems:

    print("="*80)
    print("SYSTEM PROMPT")
    print(system)
    print("-"*80)

    response = requests.post(
        URL,
        json={
            "model":"llama3.2:3b",
            "system":system,
            "prompt":prompt,
            "stream":False
        }
    )

    print(response.json()["response"])
    print()