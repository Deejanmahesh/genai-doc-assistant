from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,CSVLoader

def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".CSV"):
        loader = CSVLoader(file_path)
    else:
        raise ValueError("unsuporrted files types")
    return loader.load()