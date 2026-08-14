from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import build_vectorstore
from app.retrieval.hybrid_search import get_hybrid_retriever
from app.agents.nodes import set_retriever
from app.agents.graph import build_graph

docs = load_document("data/sample.pdf")
chunks = chunk_documents(docs)
vectordb = build_vectorstore(chunks)
retriever = get_hybrid_retriever(chunks, vectordb)

set_retriever(retriever)

graph = build_graph()

result = graph.invoke({"query": "How many days of annual leave am I entitled to?", "retry_count": 0})

print("\n--- FINAL RESULT ---")
print("Query type:", result["query_type"])
print("Answer:", result["final_answer"])
print("Valid:", result["is_valid"])
print("Confidence:", result["confidence_score"])