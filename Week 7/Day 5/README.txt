WEEK 7 PROJECT
MULTI-DOCUMENT RAG SYSTEM

Technologies Used
-----------------
• LlamaIndex
• Ollama
• ChromaDB
• MLflow
• Python

LLM Model
---------
llama3.2:3b

Embedding Model
---------------
nomic-embed-text:latest

Workflow
--------
PDF Documents
      ↓
LlamaIndex Loader
      ↓
Ollama Embeddings
      ↓
ChromaDB
      ↓
VectorStoreIndex
      ↓
Query Engine
      ↓
Answer + Source Documents

Features
--------
• Multi-document retrieval
• Local LLM inference
• ChromaDB vector storage
• MLflow experiment tracking
• Automated testing with pytest