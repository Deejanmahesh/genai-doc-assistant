from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# Import time la oru thadava load aagum
_embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(chunks, persist_dir="./chroma_db"):
    vector_db = Chroma.from_documents(chunks, _embeddings, persist_directory=persist_dir)
    return vector_db
