# 📄 GenAI Document Intelligence Assistant

A multi-agent RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions about them, powered by **LangGraph**, **FastAPI**, and **Groq**.

Instead of a simple "retrieve and generate" pipeline, this project uses an **agentic workflow** — a router classifies each query, a hybrid retriever pulls the most relevant chunks, a generator drafts an answer, and a validator checks the answer is actually grounded in the source document before returning it to the user.

---

## 🚀 Live Demo

- **Frontend (Streamlit):** [link here after deployment]
- **Backend API docs (Swagger):** [link here after deployment]

---

## 🧠 Why this project

Most beginner RAG projects are a single linear pipeline: retrieve → generate. This project goes further by treating retrieval and generation as a **multi-agent workflow with self-correction**:

- If the generated answer isn't grounded in the retrieved context, the **validator node sends it back to the generator for a retry** (up to 2 attempts) instead of silently returning a hallucinated answer.
- Retrieval combines **keyword search (BM25)** and **semantic search (embeddings)** so the system doesn't miss exact terms (like policy numbers) or paraphrased questions.
- Answer quality is measured with a **custom LLM-as-judge evaluation layer**, not just eyeballed.

---

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI → LangGraph Workflow → Groq LLM
                                       │
                    ┌──────────┬───────┴───────┬───────────┐
                    │          │                │           │
                 Router → Retriever ←→ Vector DB → Generator → Validator
                                  (Chroma + BM25)              │
                                                         (retry loop if invalid)
                                                                │
                                                          Final Answer
```

**Flow:**
1. User uploads a document via the Streamlit UI
2. FastAPI ingests it — loads, chunks, embeds, and indexes it in a Chroma vector store
3. On each question, the LangGraph workflow runs:
   - **Router** — classifies the query (factual / analytical / summarization)
   - **Retriever** — hybrid search (BM25 + semantic) pulls the top relevant chunks
   - **Generator** — Groq LLM drafts an answer using only the retrieved context
   - **Validator** — checks the answer is grounded in the context; if not, loops back to the generator
4. The final answer, along with its query type and confidence score, is returned to the UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (multi-agent state machine) |
| LLM Framework | LangChain |
| LLM | Groq (`openai/gpt-oss-120b`) — free tier |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector Store | Chroma |
| Hybrid Retrieval | BM25 (keyword) + dense vector search (semantic), ensembled |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Evaluation | Custom LLM-as-judge (faithfulness + answer relevancy scoring) |

---

## ✨ Key Features

- **Multi-agent workflow** with conditional routing and a retry loop for self-correction
- **Hybrid retrieval** (BM25 + semantic search) for better recall on both exact terms and paraphrased queries
- **Hallucination detection** — every answer is checked against retrieved context before being returned
- **Custom evaluation pipeline** — benchmarks faithfulness and answer relevancy across test queries
- **Fully free stack** — no paid API keys required (Groq free tier + local embeddings)
- **REST API** with interactive Swagger docs (`/docs`)
- **Chat-style UI** built with Streamlit, showing query classification and confidence per answer

---

## 📊 Evaluation Results

Benchmarked on a sample company policy document with a custom LLM-as-judge evaluator:

| Metric | Score |
|---|---|
| Faithfulness | 0.67 |
| Answer Relevancy | 1.00 |

*(Run `python test_evaluation.py` to reproduce, or regenerate with your own document and question set.)*

---

## 📂 Project Structure

```
genai-doc-assistant/
├── app/
│   ├── agents/
│   │   ├── state.py         # LangGraph state schema
│   │   ├── nodes.py         # Router, Retriever, Generator, Validator nodes
│   │   └── graph.py         # Workflow wiring + conditional edges
│   ├── ingestion/
│   │   ├── loader.py        # PDF / DOCX / CSV loaders
│   │   └── chunker.py       # Text splitting
│   ├── retrieval/
│   │   ├── vectorstore.py   # Chroma + HuggingFace embeddings
│   │   └── hybrid_search.py # BM25 + semantic ensemble retriever
│   ├── api/
│   │   └── routes.py        # /upload and /query endpoints
│   ├── evaluation/
│   │   └── metrics.py       # Custom LLM-as-judge evaluator
│   └── main.py               # FastAPI app entry point
├── streamlit_app.py           # Chat UI
├── test_graph.py               # End-to-end pipeline test
├── test_evaluation.py           # Evaluation test
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Deejanmahesh/Enterprise-Document-Intelligence-Multi-Agent-Assistant.git
cd Enterprise-Document-Intelligence-Multi-Agent-Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the backend
```bash
uvicorn app.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`

### 6. Run the frontend (in a separate terminal)
```bash
streamlit run streamlit_app.py
```
UI available at `http://localhost:8501`

---

## 🧪 Testing

```bash
# Test the full pipeline end-to-end
python test_graph.py

# Run the evaluation suite
python test_evaluation.py
```

---

## 🔮 Future Improvements

- Session-based / per-user vector stores (currently a single global index)
- Persist and reload the vector store across app restarts
- Streaming responses (SSE) for token-by-token answer generation
- Docker containerization for one-command deployment
- LangSmith tracing for observability into agent decisions

---

## 📝 Notes

- Embeddings run locally via `sentence-transformers` — no OpenAI API key needed.
- LLM inference uses Groq's free tier (`openai/gpt-oss-120b`), rate-limited but sufficient for demo/dev use.
- This is a portfolio project built to demonstrate multi-agent RAG architecture, hybrid retrieval, and evaluation practices — not a production-hardened system.

---

## 👤 Author

**Deejan Mahesh**
[LinkedIn](#) · [GitHub](https://github.com/Deejanmahesh)