import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

faithfulness_prompt = ChatPromptTemplate.from_template("""
You are evaluating an AI-generated answer against the source context.

Context:
{context}

Answer:
{answer}

Rate how faithful the answer is to the context on a scale of 0.0 to 1.0.
1.0 = fully grounded in context, no hallucination.
0.0 = completely unsupported by context.

Respond with ONLY a number between 0.0 and 1.0.
""")

relevancy_prompt = ChatPromptTemplate.from_template("""
Question: {question}
Answer: {answer}

Rate how relevant the answer is to the question on a scale of 0.0 to 1.0.
1.0 = directly and completely answers the question.
0.0 = does not address the question at all.

Respond with ONLY a number between 0.0 and 1.0.
""")


def score_faithfulness(context: str, answer: str) -> float:
    chain = faithfulness_prompt | judge_llm
    result = chain.invoke({"context": context, "answer": answer})
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0


def score_relevancy(question: str, answer: str) -> float:
    chain = relevancy_prompt | judge_llm
    result = chain.invoke({"question": question, "answer": answer})
    try:
        return float(result.content.strip())
    except ValueError:
        return 0.0


def evaluate_responses(questions, answers, contexts_list, ground_truths=None):
    """
    Custom lightweight evaluation using LLM-as-judge.
    Returns per-question scores and averages.
    """
    results = []

    for i, question in enumerate(questions):
        answer = answers[i]
        context = "\n\n".join(contexts_list[i])

        faithfulness = score_faithfulness(context, answer)
        relevancy = score_relevancy(question, answer)

        results.append({
            "question": question,
            "answer": answer,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy
        })

    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    return {
        "per_question": results,
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_answer_relevancy": round(avg_relevancy, 2)
    }