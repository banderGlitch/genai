from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""
    

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent



agent=make_alternative_graph()

# my graph own graph

def make_my_alternative_graph_new():
    """Make a tool-calling agent"""
    graph_workflow = StateGraph(State)
    
    @tool
    def add(a: float, b: float):
        """Adds two numbers.
        Args:
            a: The first number.
            b: The second number.
        Returns:
            The sum of the two numbers.
        """
    
        return a + b
    
    @tool
    def subtract(a: float, b: float):
        """Subtracts two numbers.
        Args:
            a: The first number.
            b: The second number.
        Returns:
            The difference of the two numbers.
        """
        return a - b
    
    @tool
    def multiply(a: float, b: float):
        """Multiplies two numbers.
        Args:
            a: The first number.
            b: The second number.
        Returns:
            The product of the two numbers.
        """
        return a * b
    
    @tool
    def divide(a: float, b: float):
        """Divides two numbers.
        Args:
            a: The first number.
            b: The second number.
        Returns:
            The quotient of the two numbers.
        """
        return a / b
    
    @tool
    def power(a: float, b: float):
        """Raises a number to the power of another number.
        Args:
            a: The base number.
            b: The exponent.
        Returns:
            The result of raising a to the power of b.
        """
        return a ** b
    
    tool_node = ToolNode([add, subtract, multiply, divide, power])
    ## binded my models with tools
    
    model_with_tools = model.bind_tools([add, subtract, multiply, divide, power],parallel_tool_calls=False)
    
    
    def call_model(state):
        sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic operations on the given set of inputs")
        
        return {"messages": [model_with_tools.invoke( [sys_msg] + state["messages"])]}
    
    graph_workflow = StateGraph(State)
    graph_workflow.add_node("call_model", call_model)
    graph_workflow.add_node("tools", tool_node)
    #Define the edges
    graph_workflow.add_edge(START, "call_model")
    graph_workflow.add_conditional_edges("call_model",
    #If the latest message(result) from call_model is a tool call, then we will go to the tool node
    #If the latest messsage(result) from call_model is not a tool call -> tools_condition routes to END
    tools_condition,
    )
    graph_workflow.add_edge("tools", "call_model")

    new_agent = graph_workflow.compile()
    return new_agent

agent_new=make_my_alternative_graph_new()



