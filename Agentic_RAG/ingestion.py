import os
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import vector_stores

import chromadb
client = chromadb.PersistentClient(path="./agentic_rag_db")
collection = client.get_or_create_collection(name="agentic-RAG")

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEndpointEmbeddings
embeddings_model = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)


from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 250 , chunk_overlap = 0)

doc_splits = text_splitter.split_documents(docs_list)

vector_store = Chroma.from_documents(documents=doc_splits , client=client , embedding = embeddings_model)
retriever = vector_store.as_retriever()