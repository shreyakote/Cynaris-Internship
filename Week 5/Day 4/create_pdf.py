from reportlab.pdfgen import canvas

pdf = canvas.Canvas("sample.pdf")

text = [
    "ChromaDB and Retrieval Augmented Generation",
    "",
    "ChromaDB is a vector database used to store and retrieve",
    "information using embeddings.",
    "",
    "Embeddings represent text as numerical vectors.",
    "Similar documents can be found using vector similarity.",
    "",
    "Retrieval Augmented Generation, or RAG, combines",
    "information retrieval with a large language model.",
    "",
    "A RAG system first retrieves relevant documents.",
    "The retrieved information is then provided to the",
    "language model as context.",
    "",
    "Ollama allows large language models to run locally.",
    "Llama models can generate answers based on retrieved",
    "information."
]

y = 750

for line in text:
    pdf.drawString(70, y, line)
    y -= 25

pdf.save()

print("sample.pdf created successfully!")