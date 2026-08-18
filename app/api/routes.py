from fastapi import APIRouter,UploadFile,File,HTTPException
from pydantic import BaseModel
import os
import shutil

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import build_vectorstore
from app.retrieval.hybrid_search import get_hybrid_retriever
from app.agents.nodes import set_retriever
from app.agents.graph import build_graph

router =APIRouter()

graph = build_graph()

is_indexed = {"status": False}

class QueryRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("data",exist_ok=True)
    file_path =f"data/{file.filename}"

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    try:
        docs = load_document(file_path)
        chunks = chunk_documents(docs)
        vector_db = build_vectorstore(chunks)
        retriever = get_hybrid_retriever(chunks, vector_db)

        set_retriever(retriever)
        is_indexed["status"] = True

    except Exception as e:
        raise HTTPException (status_code=500,detail=str(e))

    return {"status":"indexed","filename":file.filename,"chunks":len(chunks)}


@router.post("/query")
async def query_documents(request:QueryRequest):
    if not is_indexed["status"]:
        raise HTTPException(status_code= 400,detail="No document indexed yet. Please upload a document first via /upload.")

    result=graph.invoke({"query":request.query,"retry_count":0})

    return {
        "query": request.query,
        "query_type": result["query_type"],
        "answer": result["final_answer"],
        "is_valid": result["is_valid"],
        "confidence": result["confidence_score"]
    }


