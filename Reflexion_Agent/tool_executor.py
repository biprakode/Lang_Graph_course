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
        StructuredTool.from_function(run_queries , AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries , ReviseAnswer.__name__),
    ]
)