from typing import List

from dotenv import load_dotenv

from Reflexion_Agent.schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

tavily_tool = TavilySearch(max_results = 5)

def run_queries(search_queries : List[str] , **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query" : query} for query in search_queries])

execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            func=run_queries,
            name="run_queries_initial",  # Give it a unique string name
            args_schema=AnswerQuestion,  # Bind your Pydantic schema here
        ),
        StructuredTool.from_function(
            func=run_queries,
            name="run_queries_revision",  # Give it a unique string name
            args_schema=ReviseAnswer,  # Bind your Pydantic schema here
        ),
    ]
)