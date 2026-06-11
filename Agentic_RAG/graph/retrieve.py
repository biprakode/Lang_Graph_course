from typing import Dict, Any

from Agentic_RAG.graph.state import GraphState
from Agentic_RAG.ingestion import retriever


def retrieve(state : GraphState) -> Dict[str, Any]:
    print("Retrieving")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents , "question": question}
