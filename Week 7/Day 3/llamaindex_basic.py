from pathlib import Path
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FOLDER = Path("documents")

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# CREATE DOCUMENTS FOLDER
# ============================================================

DOCUMENT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# CREATE SAMPLE TEXT DOCUMENTS IF NOT PRESENT
# ============================================================

sample_documents = {

    "document1.txt": """
ARTIFICIAL INTELLIGENCE

Artificial Intelligence (AI) is a branch of computer science
that focuses on creating systems that can perform tasks that
normally require human intelligence.

AI systems can perform reasoning, learning, problem solving,
decision making and language understanding.

Applications of AI include healthcare, education, banking,
robotics, transportation and cybersecurity.
""",

    "document2.txt": """
MACHINE LEARNING

Machine Learning (ML) is a branch of Artificial Intelligence
that allows computers to learn patterns from data without being
explicitly programmed for every task.

The main types of machine learning are supervised learning,
unsupervised learning and reinforcement learning.

Machine learning is used for prediction, classification,
recommendation systems and data analysis.
""",

    "document3.txt": """
DEEP LEARNING

Deep Learning is a subset of Machine Learning that uses
artificial neural networks with multiple layers.

Deep learning is commonly used for image recognition,
speech recognition, natural language processing and
complex pattern recognition.

Deep learning normally requires large datasets and
significant computational resources.
""",

    "document4.txt": """
NATURAL LANGUAGE PROCESSING

Natural Language Processing (NLP) is a field of Artificial
Intelligence that enables computers to understand, process
and generate human language.

NLP applications include chatbots, machine translation,
sentiment analysis, text classification and question answering.

NLP is widely used in modern AI applications.
""",

    "document5.txt": """
COMPUTER VISION

Computer Vision is a field of Artificial Intelligence that
enables computers to understand and analyze images and videos.

Applications of computer vision include object detection,
face detection, medical image analysis, autonomous vehicles
and security systems.

Computer vision commonly uses machine learning and
deep learning techniques.
"""
}


print("=" * 60)
print("LLAMAINDEX DOCUMENT INDEXING AND QUERYING")
print("=" * 60)


# ============================================================
# CREATE TEXT FILES
# ============================================================

print("\nChecking documents folder...")

for filename, content in sample_documents.items():

    file_path = DOCUMENT_FOLDER / filename

    if not file_path.exists():

        file_path.write_text(
            content.strip(),
            encoding="utf-8"
        )

        print(f"Created: {filename}")

    else:

        print(f"Already exists: {filename}")


# ============================================================
# FIND TEXT FILES
# ============================================================

txt_files = list(DOCUMENT_FOLDER.glob("*.txt"))

print(f"\nText files found: {len(txt_files)}")

if len(txt_files) == 0:

    print("ERROR: No .txt files found!")
    exit()


# ============================================================
# CONFIGURE OLLAMA
# ============================================================

print("\nConfiguring Ollama...")

Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=120.0,
    context_window=4096
)

Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)

print("LLM model:", LLM_MODEL)
print("Embedding model:", EMBEDDING_MODEL)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

print("\nLoading documents...")

documents = SimpleDirectoryReader(
    input_dir=str(DOCUMENT_FOLDER),
    required_exts=[".txt"]
).load_data()

print(f"Documents loaded: {len(documents)}")


# ============================================================
# CREATE VECTOR STORE INDEX
# ============================================================

print("\nCreating VectorStoreIndex...")
print("Generating embeddings using Ollama...")

start_indexing = time.perf_counter()

index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

indexing_time = time.perf_counter() - start_indexing

print(f"\nIndexing completed in {indexing_time:.2f} seconds")


# ============================================================
# CREATE QUERY ENGINE
# ============================================================

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nQueryEngine created successfully!")


# ============================================================
# 10 QUESTIONS
# ============================================================

questions = [

    "What is artificial intelligence?",

    "What are the applications of artificial intelligence?",

    "What is machine learning?",

    "What are the types of machine learning?",

    "What is deep learning?",

    "What are the applications of deep learning?",

    "What is natural language processing?",

    "What are the applications of NLP?",

    "What is computer vision?",

    "What are the applications of computer vision?"
]


# ============================================================
# RUN 10 QUERIES
# ============================================================

print("\n")
print("=" * 60)
print("RUNNING 10 QUERIES")
print("=" * 60)


latencies = []

for number, question in enumerate(questions, start=1):

    print("\n" + "-" * 60)

    print(f"Q{number}. {question}")

    start_time = time.perf_counter()

    response = query_engine.query(question)

    latency = time.perf_counter() - start_time

    latencies.append(latency)

    print("\nAnswer:")
    print(response)

    print(f"\nLatency: {latency:.2f} seconds")

    print("\nSource documents:")

    sources = set()

    for node in response.source_nodes:

        file_name = node.metadata.get(
            "file_name",
            "Unknown"
        )

        sources.add(file_name)

    if sources:

        for source in sources:
            print("-", source)

    else:

        print("- Source not identified")


# ============================================================
# CALCULATE AVERAGE LATENCY
# ============================================================

average_latency = sum(latencies) / len(latencies)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Text files indexed : {len(documents)}")
print(f"Queries executed  : {len(questions)}")
print(f"Average latency   : {average_latency:.2f} seconds")
print(f"Indexing time     : {indexing_time:.2f} seconds")

print("\nSUCCESS!")
print("LlamaIndex document indexing and querying completed.")
print("Ollama embeddings were used successfully.")