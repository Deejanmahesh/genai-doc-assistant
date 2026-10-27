import streamlit as st
import requests

st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="📄",
    layout="centered"
)

API_BASE_URL = "https://genai-doc-assistant-eerm.onrender.com/api/v1"

st.title("📄 GenAI Document Intelligence Assistant")
st.caption("Multi-agent RAG system powered by LangGraph + FastAPI + Groq")

# ---------- Helper: detect out-of-scope answers ----------
OUT_OF_SCOPE_PHRASES = [
    "don't have enough information",
    "isn't covered in the uploaded document",
]

def is_out_of_scope(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in OUT_OF_SCOPE_PHRASES)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False

# ---------- Sidebar: Upload ----------
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or CSV file",
        type=["pdf", "docx", "csv"]
    )

    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Indexed: {data['filename']} ({data['chunks']} chunks)")
                        st.session_state.doc_indexed = True
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is FastAPI running?")

    st.divider()
    st.markdown("**Status:** " + ("🟢 Document indexed" if st.session_state.doc_indexed else "🔴 No document indexed"))

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload a document
    2. Ask questions about it
    3. AI routes → retrieves → generates → validates the answer

    💡 Ask only questions related to your uploaded document — the assistant will let you know if something is out of scope.
    """)

# ---------- Chat interface ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("out_of_scope"):
            st.warning(f"⚠️ {msg['content']}")
        else:
            st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            st.caption(
                f"Query type: `{meta['query_type']}` | "
                f"Valid: {'✅' if meta['is_valid'] else '⚠️'} | "
                f"Confidence: {meta['confidence']}"
            )

if query := st.chat_input("Ask a question about your document..."):
    if not st.session_state.doc_indexed:
        st.warning("⚠️ Please upload and index a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"query": query}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        out_of_scope = is_out_of_scope(answer)

                        if out_of_scope:
                            st.warning(f"⚠️ {answer}")
                        else:
                            st.write(answer)

                        st.caption(
                            f"Query type: `{data['query_type']}` | "
                            f"Valid: {'✅' if data['is_valid'] else '⚠️'} | "
                            f"Confidence: {data['confidence']}"
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "meta": data,
                            "out_of_scope": out_of_scope
                        })
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is FastAPI running on port 8000?")