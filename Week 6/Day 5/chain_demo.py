from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Create PromptTemplate
# --------------------------------------------------

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

Answer the following question clearly and briefly.

Question: {question}

Answer:
"""
)


# --------------------------------------------------
# 2. Create Ollama LLM
# --------------------------------------------------

llm = OllamaLLM(
    model="llama3.2:3b"
)


# --------------------------------------------------
# 3. Create Output Parser
# --------------------------------------------------

output_parser = StrOutputParser()


# --------------------------------------------------
# 4. Build LangChain
# PromptTemplate → Ollama → OutputParser
# --------------------------------------------------

chain = prompt | llm | output_parser


# --------------------------------------------------
# 5. Five Test Inputs
# --------------------------------------------------

questions = [
    "What is Artificial Intelligence?",
    "What is LangChain?",
    "What is Machine Learning?",
    "What is RAG?",
    "What is a vector database?"
]


# --------------------------------------------------
# 6. Run Tests
# --------------------------------------------------

print("=" * 60)
print("WEEK 6 DAY 5 - LANGCHAIN CHAIN")
print("=" * 60)

for i, question in enumerate(questions, start=1):

    print(f"\nTest {i}")
    print("-" * 40)

    print("Question:", question)

    response = chain.invoke({
        "question": question
    })

    print("Answer:", response)


print("\n" + "=" * 60)
print("5 INPUT TEST COMPLETED")
print("=" * 60)