
#%%
from typing import Sequence, Annotated, Union, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage, ToolCall # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import groq
from groq import Groq
from langchain_groq import ChatGroq
import os
from langchain_mistralai import MistralAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import math
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")


#%%
# Initialize the LLM


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: Union[int, float], b: Union[int, float]) -> float:
    """Adds two numbers together."""
    return float(a) + float(b)

@tool
def subtract(a: Union[int, float], b: Union[int, float]) -> float:
    """Subtracts the second number from the first."""
    return float(a) - float(b)

@tool
def multiply(a: Union[int, float], b: Union[int, float]) -> float:
    """Multiplies two numbers together."""
    return float(a) * float(b)

@tool
def divide(a: Union[int, float], b: Union[int, float]) -> float:
    """Divides the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return float(a) / float(b)

@tool
def floor_value(x: Union[int, float]) -> int:
    """Returns the rounded down integer value of x."""
    return math.floor(x)


tools = [add, subtract, multiply, divide, floor_value]
tool_map = {t.__name__: t for t in tools}


llm = ChatGroq(model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=1)

model = llm.bind_tools(tools)

#%%
# Define the model call function that will be used in the agent

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import MessagesState

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [model.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
app = builder.compile()


#%%

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


#%%
inputs = {"messages": [("user", "Add 3 and 4. Multiply the output by 2. Divide the output by 5")]}
print_stream(app.stream(inputs, stream_mode="values"))

#%%


# messages = [HumanMessage(content="A store has 50 items. 20 were sold in the morning. Then the store received a new shipment of 35 items. If the store owner wants to arrange the items into 5 equal groups, how many items will be in each group? ")]

messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5 ")]
messages = app.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

# %%
