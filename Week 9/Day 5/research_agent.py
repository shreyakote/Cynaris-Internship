from crewai import Agent, Crew, LLM, Process, Task


# Local Ollama
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
)


# --------------------------------------------------
# Agents
# --------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal="Find the key facts about the given topic.",
    backstory="You are a technical researcher.",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)

writer = Agent(
    role="Writer",
    goal="Create a short technical report from the research.",
    backstory="You are a technical report writer.",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)

reviewer = Agent(
    role="Reviewer",
    goal="Check the report and return the improved final version.",
    backstory="You are a technical report reviewer.",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)


# --------------------------------------------------
# Tasks
# --------------------------------------------------

research_task = Task(
    description=(
        "Research {topic}. Give only 5 important facts. "
        "Keep the response concise."
    ),
    expected_output="Five concise facts about the topic.",
    agent=researcher,
)

writing_task = Task(
    description=(
        "Using the research, write a short report about {topic}. "
        "Use a title, introduction, key points and conclusion. "
        "Keep it concise."
    ),
    expected_output="A short structured research report.",
    agent=writer,
    context=[research_task],
)

review_task = Task(
    description=(
        "Review the report about {topic}. "
        "Correct important errors and return the final concise report."
    ),
    expected_output="A corrected final research report.",
    agent=reviewer,
    context=[writing_task],
)


# --------------------------------------------------
# Crew
# --------------------------------------------------

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=False,
)


# --------------------------------------------------
# Generate Report
# --------------------------------------------------

def generate_report(topic):
    result = crew.kickoff(
        inputs={"topic": topic}
    )

    report = str(result)

    with open(
        "research_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    return report


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    topic = "Retrieval-Augmented Generation (RAG)"

    print("=" * 60)
    print("AUTOMATED RESEARCH REPORT AGENT")
    print("=" * 60)
    print(f"Topic: {topic}")
    print("\nRunning Researcher -> Writer -> Reviewer...\n")

    report = generate_report(topic)

    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)

    print("\nReport saved to research_report.txt")