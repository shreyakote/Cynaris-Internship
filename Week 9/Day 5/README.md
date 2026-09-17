# Week 9 Day 5 — Automated Research Report Agent

## Objective

Build an automated multi-agent research report system using CrewAI.

The system uses three agents:

1. Researcher
2. Writer
3. Reviewer

The agents work sequentially to research a topic, write a report,
and review the final report.

## Technology Stack

- Python 3.11
- CrewAI
- Ollama
- Llama 3.2 3B
- LangChain
- LangGraph
- MLflow
- Ragas
- Pytest
- Ruff

## Workflow

```text
Topic
  |
  v
Researcher
  |
  v
Research Notes
  |
  v
Writer
  |
  v
Research Report
  |
  v
Reviewer
  |
  v
Final Research Report