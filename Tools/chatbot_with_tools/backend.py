from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import operator
from typing import Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage , BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode , tools_condition
import requests
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import sqlite3
import json
load_dotenv()

Api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key= Api_key
)



# Tools 
wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

@tool
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """Performs basic arithmetic operations of two numbers.
    Supported operations are: add, subtract, multiply, divide.
    """
    try :
        if operation == "add":
            result = first_number + second_number
        elif operation == "subtract":       
            result = first_number - second_number
        elif operation == "multiply":
            result = first_number * second_number
        elif operation == "divide":
            result = first_number / second_number
        else:
            return {"error": "Unsupported operation. Please use add, subtract, multiply, or divide."}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}    
    
@tool
def get_stock_price(symbol: str) -> dict:
    """fetches the current stock price for a given symbol.[eg: AAPL, GOOGL]
    using alpha vantage API in the url below
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=LS27XNFJSTDKIX16"
    r = requests.get(url)
    data = r.json()
    return data


# list of tools 
tools = [search_tool, calculator, get_stock_price]

model_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

# graph 
def chat_node(state : ChatState) -> ChatState:
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {
        "messages" : [response]
    }

tool_node = ToolNode(tools)
conn = sqlite3.connect('chatbot_checkpoints.db', check_same_thread=False)
checkpointer = SqliteSaver(conn= conn)
graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition) 
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)



def retrive_all_thread():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_thread)

