from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# This script loads PDF files from a specified directory
def load_pdfs(pdf_path: str):
    loader = DirectoryLoader(pdf_path, 
                             glob="*.pdf", 
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

PDF_PATH = "policy_store/"
documents = load_pdfs(PDF_PATH)
print(f"Loaded {len(documents)} documents from {PDF_PATH}")

# Split the documents into smaller chunks
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

chunks = split_documents(documents)
print(f"Split into {len(chunks)} chunks of text.")

# create embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
def get_embedding_model(model_name):
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return hf_embeddings

embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

FAISS_DB_PATH = "vector_store/faiss_index"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(FAISS_DB_PATH)



