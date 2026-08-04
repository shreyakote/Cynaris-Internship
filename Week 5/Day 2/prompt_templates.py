import requests

URL = "http://localhost:11434/api/generate"

templates = {
    "Teacher":
        "You are a school teacher. Explain everything simply.",

    "Programmer":
        "You are a senior Python developer. Give code examples.",

    "Interviewer":
        "You are an interviewer. Ask follow-up questions."
}

question = "Explain Artificial Intelligence."

for role, system_prompt in templates.items():

    print("="*70)
    print(role.upper())
    print("="*70)

    response = requests.post(
        URL,
        json={
            "model":"llama3.2:3b",
            "system":system_prompt,
            "prompt":question,
            "stream":False
        }
    )

    print(response.json()["response"])
    print()