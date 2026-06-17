from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
import os
import shutil
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables from .env file
load_dotenv()

#from langchain_core.chat_history import BaseChatMessageHistory
#from langchain_community.chat_message_histories import ChatMessageHistory
#from langchain_core.runnables.history import RunnableWithMessageHistory

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
db_path = "./chroma_langchain_db"

if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("Deleted existing Chroma DB!")
pdf_folder = r"./PDF"

# Collect all PDFs inside the folder
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

documents = []
for pdf_path in pdf_files:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    documents.extend(docs)
    print(f"Loaded: {pdf_path}")

print("All PDFs LOADED!")

# ---

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)
print("DOCUMENT SPLITTED")

# 2. Correct parameter is 'token', not 'use_auth_token'
model = SentenceTransformer("google/embeddinggemma-300m", token=HF_TOKEN)

# 3. Pass the token directly to the embeddings configuration
embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    model_kwargs={"token": HF_TOKEN}
)
print("Gemma EMBEDDINGS LOADED")

vector_store = Chroma.from_documents(
    documents = chunks,
    collection_name = "pdf-rag",
    embedding = embeddings,
    persist_directory = "./chroma_langchain_db"
)

print("SAVED!")