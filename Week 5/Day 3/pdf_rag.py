import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("=" * 60)
print("PDF RAG DEMO")
print("=" * 60)

# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print("PDF Loaded Successfully!")
print(f"Pages in PDF: {len(documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"\nTotal Chunks Created: {len(chunks)}")

print("\nFirst 3 Chunks:\n")

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"Chunk {i}")
    print("-" * 50)
    print(chunk.page_content)
    print()

print("=" * 60)
print("PDF Processing Completed Successfully!")
print("=" * 60)