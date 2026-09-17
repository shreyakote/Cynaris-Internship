from crewai import Agent, LLM, Task


llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
)


agent = Agent(
    role="Simple AI Assistant",
    goal="Answer a simple question clearly.",
    backstory="You are a helpful AI assistant.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


task = Task(
    description="Explain Retrieval-Augmented Generation in exactly two sentences.",
    expected_output="Exactly two sentences explaining RAG.",
    agent=agent,
)


result = agent.execute_task(task)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(result)