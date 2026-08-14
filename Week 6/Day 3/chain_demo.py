from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# 1. Create PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words in 3 sentences."
)

# 2. Create Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# 3. Create Output Parser
parser = StrOutputParser()

# 4. Build LangChain
chain = prompt | llm | parser


# 5. Test with 5 inputs
topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "LangChain",
    "RAG",
    "AI Agents"
]

print("=" * 60)
print("LANGCHAIN CHAIN DEMO")
print("=" * 60)

for topic in topics:
    print(f"\nInput: {topic}")
    response = chain.invoke({"topic": topic})
    print("Output:")
    print(response)