# Week 9 Day 4 — CrewAI + LangChain Hybrid Agent Systems

## Objective

Build a hybrid agent system using LangChain and Ollama.

The implementation covers:

1. LangChain chain using PromptTemplate, Ollama LLM, and OutputParser.
2. ConversationBufferMemory for maintaining conversation history.
3. LangChain agent with two tools:
   - Web Search Stub
   - Calculator
4. Testing using pytest.
5. Code quality checking using Ruff.

## Technology Stack

- Python 3.11
- LangChain
- LangChain Ollama
- LangChain Classic
- LangGraph
- Ollama
- llama3.2:3b
- Pytest
- Ruff

## Project Structure

```text
Day 4/
├── agent_demo.py
├── agent_output.txt
├── chain_output.txt
├── langchain_chain.py
├── memory_demo.py
├── memory_output.txt
├── test_hybrid.py
├── requirements.txt
├── README.md
└── .gitignore