from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferMemory

from agent_demo import calculator, web_search_stub


def test_prompt_template():
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Question: {question}",
    )

    result = prompt.format(question="What is AI?")

    assert "What is AI?" in result


def test_output_parser():
    parser = StrOutputParser()

    result = parser.invoke("Hello")

    assert result == "Hello"


def test_conversation_memory():
    memory = ConversationBufferMemory()

    for number in range(5):
        memory.save_context(
            {"input": f"Question {number + 1}"},
            {"output": f"Answer {number + 1}"},
        )

    history = memory.load_memory_variables({})["history"]

    assert "Question 1" in history
    assert "Question 5" in history


def test_calculator_tool():
    result = calculator.invoke("25 * 4 + 10")

    assert "110" in result


def test_web_search_stub():
    result = web_search_stub.invoke("RAG")

    assert "RAG" in result