import streamlit as st
import requests

st.set_page_config(
    page_title="GenAI Document Assistant",
    page_icon="📄",
    layout="centered"
)

API_BASE_URL= "http://127.0.0.1:8000/api/v1"
st.title("📄 GenAI Document Intelligence Assistant")
st.caption("Multi-agent RAG system powered by LangGraph + FastAPI + Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False

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
    """)

# ---------- Chat interface ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
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
                        st.write(answer)
                        st.caption(
                            f"Query type: `{data['query_type']}` | "
                            f"Valid: {'✅' if data['is_valid'] else '⚠️'} | "
                            f"Confidence: {data['confidence']}"
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "meta": data
                        })
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is FastAPI running on port 8000?")

