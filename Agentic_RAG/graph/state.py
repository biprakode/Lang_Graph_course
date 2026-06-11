from typing import TypedDict, List

from langchain_core.outputs import generation


class GraphState(TypedDict):
    """
    Represents states of the graph

    Attributes:
        question : question
        generation : LLM generation
        web_search : where to web search
        documents : list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]