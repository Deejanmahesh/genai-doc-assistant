from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import GraphState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

hybrid_retriever = None

def set_retriever(retriever):
    global hybrid_retriever
    hybrid_retriever = retriever


router_prompt = ChatPromptTemplate.from_template("""Classify the following query into exactly one word: factual, analytical, or summarization.
Return ONLY the single word, nothing else.

Query: {query}""")

def router_node(state: GraphState) -> GraphState:
    chain = router_prompt | llm
    result = chain.invoke({"query": state["query"]})
    query_type = result.content.strip().lower()

    if query_type not in ["factual", "analytical", "summarization"]:
        query_type = "factual"
    return {**state, "query_type": query_type}


def retriever_node(state: GraphState) -> GraphState:
    if hybrid_retriever is None:
        raise ValueError("Retriever not set. Call set_retriever() first.")
    docs = hybrid_retriever.invoke(state["query"])
    retrieved_texts = [d.page_content for d in docs]
    return {**state, "retrieved_docs": retrieved_texts}


generator_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions based only on the given context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:""")

def generator_node(state: GraphState) -> GraphState:
    context = "\n\n".join(state.get("retrieved_docs", []))
    chain = generator_prompt | llm
    result = chain.invoke({"context": context, "query": state["query"]})
    retry_count = state.get("retry_count", 0)
    return {**state, "draft_answer": result.content, "retry_count": retry_count}


validator_prompt = ChatPromptTemplate.from_template("""Check if the ANSWER is fully supported by the CONTEXT below.
Respond with only "VALID" if the answer is grounded in the context,
or "INVALID" if the answer contains information not present in the context.

Context:
{context}

Answer:
{answer}

Verdict:""")

def validator_node(state: GraphState) -> GraphState:
    context = "\n\n".join(state.get("retrieved_docs", []))
    chain = validator_prompt | llm
    result = chain.invoke({"context": context, "answer": state["draft_answer"]})
    verdict = result.content.strip().upper()
    is_valid = "VALID" in verdict and "INVALID" not in verdict

    retry_count = state.get("retry_count", 0)

    if retry_count >= 2:
        is_valid = True

    confidence_score = 0.9 if is_valid else 0.4

    return {
        **state,
        "final_answer": state["draft_answer"],
        "is_valid": is_valid,
        "confidence_score": confidence_score,
        "retry_count": retry_count + 1
    }