"""
RAG Pipeline: Handles document loading, embedding, and vector similarity search.
"""
from typing import List
# FIX: Use the updated standalone text splitters package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIRECTORY = "./chroma_db"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)

def ingest_documents(texts: List[str], metadatas: List[dict]):
    """Chunks and stores enterprise documents into ChromaDB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts, metadatas=metadatas)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    vectorstore.persist()
    return len(docs)

def similarity_search(query: str, k: int = 3) -> List[dict]:
    """Retrieves top-k relevant context chunks for a query."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Internal Policy"),
            "relevance_score": round(float(score), 4)
        })
    return formatted_results