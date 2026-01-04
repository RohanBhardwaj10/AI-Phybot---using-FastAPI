from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
import math
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever=vector_store.as_retriever(search_kwargs={"k":4})

def retrieve_knowledge(query: str) -> str:
    """
    Retrieve relevant physics content from the FAISS index.
    """
    result=retriever.invoke(query)
    if not result:
        return "No relevant information found."
    return "\n\n".join([doc.page_content for doc in result])

retrieval_tool = Tool(
    name="phy_retriever",
    func=retrieve_knowledge,
    description="Search and retrieve relevant Physics content from loaded textbooks."
)

def physics_calculator(expression: str) -> str:
    """
    Safely evaluate math/physics expressions.
    Example: '2 * 9.8 * 5' → '98.0'
    """

    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    result = eval(expression, {"__builtins__": {}}, allowed_names)
    return str(result)

calculator_tool=Tool(
    name="phy_calculator",
    func=physics_calculator,
    description="Perform physics/math calculations using Python's math library."
)
search_tool_instance = DuckDuckGoSearchRun()

def web_search(query: str) -> str:
    """
    Search the web for real-time Physics information or updates.
    Example: 'latest physics discoveries 2025'
    """
    try:
        return search_tool_instance.run(query)
    except Exception as e:
        return f"Web search failed: {e}"

web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Search the web for the latest physics-related information."
)
tools=[calculator_tool,web_search_tool,retrieval_tool]