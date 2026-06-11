


from Agentic_RAG.graph.chains import retrieval_grader
from Agentic_RAG.graph.chains.retrieval_grader import GradeDocuments
from Agentic_RAG.ingestion import retriever


def test_retrieval_grader_chain_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    doct_txt = docs[0].page_content

    res : GradeDocuments = retrieval_grader.retrieval_chain.invoke({
        "question" : question , "document" : doct_txt
    })

    assert res.binary_score == "yes"

def test_retrieval_grader_chain_no():
    question = "How to make pizza??"
    docs = retriever.invoke(question)
    doct_txt = docs[0].page_content

    res : GradeDocuments = retrieval_grader.retrieval_chain.invoke({
        "question" : question , "document" : doct_txt
    })

    assert res.binary_score == "no"