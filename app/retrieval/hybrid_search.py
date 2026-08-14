from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_hybrid_retriever(chunks, vectordb):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever], weights=[0.4, 0.6]
    )

    return ensemble