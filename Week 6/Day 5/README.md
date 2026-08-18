# Week 6 Day 5 - Document Chatbot with LangChain

## Project Overview

This project demonstrates the development of a basic chatbot workflow using LangChain and Ollama.

The project includes:

- LangChain chain
- Conversation memory
- Simple agent
- Web search stub
- Calculator tool

## Technologies Used

- Python
- LangChain
- Ollama
- Llama 3.2 3B
- LangChain Core
- LangChain Ollama

---

# Task 1 - LangChain Chain

The first task implements the following pipeline:

PromptTemplate → Ollama LLM → OutputParser

The chain was tested using five inputs:

1. What is Artificial Intelligence?
2. What is LangChain?
3. What is Machine Learning?
4. What is RAG?
5. What is a vector database?

File:

`chain_demo.py`

Result:

5 inputs successfully tested.

---

# Task 2 - Conversation Memory

Conversation history was implemented using LangChain message history.

The system was tested with five conversation turns.

Example:

User:

My name is Shreya.

Assistant remembers the name.

Later:

User:

What is my name?

The assistant uses the previous conversation history to answer.

File:

`memory_demo.py`

Result:

5 conversation turns successfully tested.

---

# Task 3 - Simple Agent

A simple agent-style workflow was implemented with two tools.

## Tool 1 - Web Search Stub

The web search stub simulates a web search.

It does not connect to a real search engine.

## Tool 2 - Calculator

The calculator performs basic mathematical calculations.

Example:

25 * 8 = 200

File:

`agent_demo.py`

The agent was tested with three tasks:

1. What is LangChain?
2. Calculate 25 * 8
3. Search for information about RAG

Result:

3 agent tasks successfully tested.

---

# Project Architecture

## Chain

User Input

↓

PromptTemplate

↓

Ollama LLM

↓

OutputParser

↓

Response


## Conversation Memory

User Input

↓

Conversation History

↓

Ollama LLM

↓

Response

↓

Updated Conversation History


## Agent

User Task

↓

Simple Agent

↓

Tool Selection

├── Web Search Stub

└── Calculator

↓

Result

---

# Files

```text
Document Chatbot/
│
├── chain_demo.py
├── memory_demo.py
├── agent_demo.py
├── requirements.txt
└── README.md