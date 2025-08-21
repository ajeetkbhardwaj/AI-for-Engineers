# client.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

import logging
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

async def main():
    # Initialize the model
    model = ChatOpenAI(
        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="mistralai/mistral-7b-instruct:free"
    )

    # Initialize the MCP client to connect to our running server
    client = MultiServerMCPClient(
        {
            "SimpleCalculator": {
                "transport": "streamable_http", # FIX 1: Use underscore
                "url": "http://127.0.0.1:8000/mcp"  # FIX 2: Use correct standard loopback IP
            }
        }
    )

    print("Fetching tools from MCP server...")
    tools = await client.get_tools()
    print(f"Successfully fetched {len(tools)} tools.")
    
    agent = model.bind_tools(tools)
    tool_node = ToolNode(tools)
    
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
        
    async def call_model(state: AgentState):
        messages = state["messages"]
        response = await agent.ainvoke(messages)
        return {"messages": [response]}
    
    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "call_model")
     
    graph = builder.compile()

    # FIX 3: Loop through examples and invoke the graph for each one
    examples = [ 
        "what's 45 + 67?",
        "compute 123 - 58",
        "multiply 42 by 19",
        "divide 144 by 12",
        "what is 5 factorial?",
        "what is (8 + 2) * (5 - 3)?" 
    ]

    for i, query in enumerate(examples):
        print(f"\n--- Running Example {i+1} ---")
        print(f"Query: {query}")
        # The input to the graph should be a list of messages.
        # LangGraph is smart enough to convert a simple string in a tuple to a HumanMessage.
        inputs = {"messages": [("user", query)]}
        async for event in graph.astream(inputs):
             for v in event.values():
                if "messages" in v:
                    print("Agent:", v["messages"][-1].content)
        print("------------------------")


if __name__ == "__main__":
    asyncio.run(main())