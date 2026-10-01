# 3M Capstone: Production AI System — Full Pipeline

## 1. Project Overview

This capstone integrates the AI/ML 3M stack into an end-to-end production-oriented AI workflow.

The system combines:

- CrewAI for multi-agent research workflows
- LangGraph for stateful and conditional agent workflows
- RAG for knowledge-grounded question answering
- ChromaDB for vector storage and retrieval
- Ragas for RAG evaluation
- MLflow for experiment tracking and model management
- Docker for application containerisation
- GitHub Actions for CI/CD
- Production monitoring for reliability and model quality

## 2. End-to-End Architecture

```text
User Query
    |
    v
CrewAI / LangGraph
    |
    v
RAG Pipeline
    |
    +----> Document Loading
    |
    +----> Chunking
    |
    +----> Embedding
    |
    v
ChromaDB
    |
    v
Retriever
    |
    v
LLM
    |
    v
Generated Answer
    |
    v
Ragas Evaluation
    |
    +----> Faithfulness
    +----> Answer Relevancy
    +----> Context Precision
    +----> Context Recall
    |
    v
MLflow
    |
    +----> Experiments
    +----> Parameters
    +----> Metrics
    +----> Model Versions
    |
    v
Dockerised Application
    |
    v
GitHub Actions
    |
    +----> Lint
    +----> Test
    +----> Build
    +----> Push
    |
    v
Production Deployment
    |
    v
Monitoring