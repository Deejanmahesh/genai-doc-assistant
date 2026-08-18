from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import build_vectorstore
from app.retrieval.hybrid_search import get_hybrid_retriever
from app.agents.nodes import set_retriever
from app.agents.graph import build_graph
from app.evaluation.metrics import evaluate_responses
import json

docs = load_document("data/sample.pdf")
chunks = chunk_documents(docs)
vectordb = build_vectorstore(chunks)
retriever = get_hybrid_retriever(chunks, vectordb)
set_retriever(retriever)
graph = build_graph()

test_questions = [
    "How many days of annual leave am I entitled to?",
    "How many days of sick leave are given per year?",
    "How many days per week can I work from home?",
]

answers = []
contexts_list = []

for q in test_questions:
    result = graph.invoke({"query": q, "retry_count": 0})
    answers.append(result["final_answer"])
    contexts_list.append(result["retrieved_docs"])

print("\nRunning evaluation...\n")
eval_result = evaluate_responses(test_questions, answers, contexts_list)

print("\n--- EVALUATION RESULTS ---")
print(f"Average Faithfulness: {eval_result['avg_faithfulness']}")
print(f"Average Answer Relevancy: {eval_result['avg_answer_relevancy']}")

with open("evaluation_results.json", "w") as f:
    json.dump(eval_result, f, indent=2)

print("\nSaved to evaluation_results.json")