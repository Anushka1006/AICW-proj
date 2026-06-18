import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

db_path = "./chroma_langchain_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("Deleted existing Chroma DB!")

pdf_folder = r"./PDF"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

documents = []
for pdf_path in pdf_files:
    fast_loader = PyPDFLoader(pdf_path)
    docs = fast_loader.load()

    total_text_length = sum(len(doc.page_content.strip()) for doc in docs)

    # FUnstructured OCR if the fast extraction yields no real text
    if total_text_length < 50:
        print(f"Scanned PDF detected ({pdf_path}). Running Unstructured OCR...")
        ocr_loader = UnstructuredPDFLoader(
            pdf_path, 
            strategy="hi_res"  
        )
        docs = ocr_loader.load()
    else:
        print(f"Digital PDF detected ({pdf_path}). Processing with PyPDFLoader...")



    for doc in docs:
        doc.page_content = f"title: none | text: {doc.page_content}"
        
    documents.extend(docs)
    print(f"Loaded: {pdf_path}")

print("All PDFs LOADED!")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print("DOCUMENT SPLITTED")

# Use API instead of local loading to prevent RAM crashes
embeddings = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=HF_TOKEN
)
print("Gemma API EMBEDDINGS INITIALIZED")

vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name="pdf-rag",
    embedding=embeddings,
    persist_directory=db_path
)

print(f"SAVED! Collection count is: {vector_store._collection.count()}")