from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(chunks, persist_dir="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vector_db
