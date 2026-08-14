# ============================================================
# WEEK 6 - DAY 2
# LANGCHAIN CHAIN, MEMORY AND AGENT
# ============================================================

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import OllamaLLM, ChatOllama

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.tools import tool

from langchain.agents import create_agent


# ============================================================
# 1. INITIALIZE OLLAMA
# ============================================================

print("\n" + "=" * 60)
print("INITIALIZING OLLAMA")
print("=" * 60)

# Normal Ollama model
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0
)

# Chat model for Agent
chat_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Ollama models initialized successfully.")


# ============================================================
# 2. LANGCHAIN CHAIN
# PromptTemplate -> Ollama LLM -> OutputParser
# ============================================================

print("\n" + "=" * 60)
print("1. LANGCHAIN CHAIN")
print("=" * 60)


prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

Answer the question clearly and briefly.

Question: {question}

Answer:
"""
)


output_parser = StrOutputParser()


# Build chain
chain = prompt | llm | output_parser


# Five test inputs
test_inputs = [
    "What is artificial intelligence?",
    "What is LangChain?",
    "What is Ollama?",
    "What is RAG?",
    "What is ChromaDB?"
]


print("\nTesting chain with 5 inputs:\n")


for i, question in enumerate(test_inputs, 1):

    try:

        response = chain.invoke({
            "question": question
        })

        print(f"Input {i}: {question}")
        print(f"Output: {response}")

    except Exception as e:

        print(f"Input {i}: {question}")
        print(f"Error: {e}")

    print("-" * 60)


# ============================================================
# 3. CONVERSATION MEMORY
# Maintain conversation history across 5 turns
# ============================================================

print("\n" + "=" * 60)
print("2. CONVERSATION MEMORY")
print("=" * 60)


chat_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a helpful assistant.

Remember information from the conversation
and use it when answering later questions.
"""
    ),

    MessagesPlaceholder(
        variable_name="history"
    ),

    (
        "human",
        "{input}"
    )

])


# Memory chain
memory_chain = chat_prompt | llm | output_parser


# Store conversation history
conversation_history = []


# Five conversation turns
conversation_turns = [

    "My name is Shreya.",

    "I am studying Information Science.",

    "I am currently in my 7th semester.",

    "What is my name?",

    "What course am I studying?"

]


print("\nConversation across 5 turns:\n")


for i, user_input in enumerate(
    conversation_turns,
    1
):

    try:

        response = memory_chain.invoke({

            "history": conversation_history,

            "input": user_input

        })


        print(f"Turn {i}")
        print(f"User: {user_input}")
        print(f"AI: {response}")


        # Store user message
        conversation_history.append(
            HumanMessage(
                content=user_input
            )
        )


        # Store AI message
        conversation_history.append(
            AIMessage(
                content=response
            )
        )


    except Exception as e:

        print(f"Turn {i}")
        print(f"User: {user_input}")
        print(f"Error: {e}")


    print("-" * 60)


# ============================================================
# DISPLAY CONVERSATION HISTORY
# ============================================================

print("\nConversation history maintained successfully.")

print("\nStored Conversation History:")


for message in conversation_history:

    print(
        f"{message.type}: {message.content}"
    )


# ============================================================
# 4. SIMPLE AGENT
# Two Tools:
# 1. Web Search Stub
# 2. Calculator
# ============================================================

print("\n" + "=" * 60)
print("3. SIMPLE AGENT")
print("=" * 60)


# ============================================================
# TOOL 1 - WEB SEARCH STUB
# ============================================================

@tool
def web_search(query: str) -> str:
    """
    Simple web search stub.
    This does not perform a real internet search.
    """

    fake_results = {

        "langchain":
            "LangChain is an open-source framework "
            "for building applications powered by "
            "language models.",

        "ollama":
            "Ollama is a tool that allows users "
            "to run large language models locally.",

        "chromadb":
            "ChromaDB is an open-source vector database "
            "commonly used for AI and RAG applications."

    }


    query_lower = query.lower()


    for key, result in fake_results.items():

        if key in query_lower:

            return result


    return (
        f"Web search stub: "
        f"No specific result found for '{query}'."
    )


# ============================================================
# TOOL 2 - CALCULATOR
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    """

    try:

        allowed_characters = (
            "0123456789+-*/(). "
        )


        if not all(
            char in allowed_characters
            for char in expression
        ):

            return "Invalid mathematical expression."


        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )


        return f"Result: {result}"


    except Exception as e:

        return f"Calculation error: {e}"


# ============================================================
# ADD TOOLS
# ============================================================

tools = [
    web_search,
    calculator
]


print("\nTwo tools created successfully.")

print("Tool 1: web_search")
print("Tool 2: calculator")


# ============================================================
# 5. CREATE AGENT
# ============================================================

print("\nCreating agent...")


agent = create_agent(

    # IMPORTANT:
    # ChatOllama is used here because
    # the agent requires tool calling support.
    model=chat_llm,

    tools=tools,

    system_prompt="""

You are a helpful AI assistant.

You have access to two tools.

Tool 1:
web_search

Use web_search when the user asks
for information about a topic.

Tool 2:
calculator

Use calculator when the user asks
for a mathematical calculation.

Use the appropriate tool when necessary.

"""
)


print("Agent created successfully.")


# ============================================================
# 6. RUN 3 AGENT TASKS
# ============================================================

agent_tasks = [

    "What is LangChain?",

    "Calculate 125 * 8 + 50.",

    "Search for information about ChromaDB."

]


print("\nRunning 3 agent tasks:\n")


for i, task in enumerate(
    agent_tasks,
    1
):

    print(f"Agent Task {i}:")
    print(task)


    try:

        result = agent.invoke({

            "messages": [

                {
                    "role": "user",
                    "content": task
                }

            ]

        })


        messages = result.get(
            "messages",
            []
        )


        if messages:

            final_message = messages[-1]

            print("\nAgent Answer:")

            print(
                final_message.content
            )

        else:

            print(
                "\nNo response received "
                "from agent."
            )


    except Exception as e:

        print("\nAgent Error:")

        print(e)


    print("-" * 60)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

print(
    "ALL LANGCHAIN PRACTICAL TASKS COMPLETED"
)

print("=" * 60)