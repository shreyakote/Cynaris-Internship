import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

print("Starting ChromaDB...")

client = chromadb.Client()

collection = client.get_or_create_collection("student_notes")

documents = [
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks.",
    "Python is widely used in AI.",
    "ChromaDB is a vector database.",
    "Ollama runs local language models."
]

ids = ["1", "2", "3", "4", "5"]

collection.add(
    ids=ids,
    documents=documents
)

print("Documents added successfully!")

results = collection.query(
    query_texts=["What is Machine Learning?"],
    n_results=3
)

print("\nTop Results:")
for doc in results["documents"][0]:
    print("-", doc)

print("\nWeek 5 Day 3 completed successfully!")