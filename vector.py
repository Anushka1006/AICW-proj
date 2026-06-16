from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
db_path="./chroma_langchain_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("Deleted existing database")


pdf_directory = "PDF"
documents = []

for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages from PDF files!")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)
print("DOCUMENT SPLITTED")

embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
print("EMBEDDINGS LOADED")

vector_store = Chroma.from_documents(
    documents = chunks,
    collection_name = "pdf-rag",
    embedding = embeddings,
    persist_directory = "./chroma_langchain_db"
)

print("SAVED!")
