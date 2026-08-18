from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

def build_vectorstore(chunks, persist_dir="./chroma_db"):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vectordb