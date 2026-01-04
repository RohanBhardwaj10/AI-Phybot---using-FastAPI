from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os, json
from tools import tools

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo",)
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

MEMORY_FILE = "chat_memory.json"

def save_memory(memory):
    serializable_messages = []
    for msg in memory["messages"]:
        if isinstance(msg, HumanMessage):
            serializable_messages.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serializable_messages.append({"type": "ai", "content": msg.content})
        else:
            serializable_messages.append({"type": "unknown", "content": str(msg)})

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"messages": serializable_messages}, f, indent=4)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = []
            for msg in data.get("messages", []):
                if msg["type"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))
            return {"messages": messages}
        except (json.JSONDecodeError, KeyError, TypeError):
            print("Memory file corrupted or empty — resetting conversation memory.")
            return {"messages": []}
    return {"messages": []}

chatbot = graph.compile()

def run_agent(user_input: str):
    """
    Run the LangGraph agent with persistent memory.
    Automatically decides which tool to use.
    """
    memory = load_memory()
    state = ChatState(messages=memory["messages"] + [HumanMessage(content=user_input)])
    result = chatbot.invoke(state)
    updated_memory = {"messages": result["messages"]}
    save_memory(updated_memory)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("🚀 AI PhyBot (LangGraph Agent) Ready!\n")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye! Your chat memory is saved.")
            break
        response = run_agent(user_query)
        print(f"🤖 PhyBot: {response}\n")

