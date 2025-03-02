from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState
from langgraph.graph.state import StateGraph
from langchain.graph.message import add_messages
from langggraph.prebuild import ToolNode
from langchain_core.tools 


class AgentState(TypedDict):
    messages: Annotated[MessagesState, add_messages_to_state]




