from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import login


load_dotenv()
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)


embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="google/embeddinggemma-300m",
    huggingfacehub_api_token=HF_TOKEN
embeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m"
)

vector_store = Chroma(
    collection_name="pdf-rag",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings
)



question = "What is a vector?"

docs = vector_store.similarity_search(
    question,
    k=3
)

context = "\n\n".join([d.page_content for d in docs])

prompt = f"""
Answer the question using only the provided context.

Context: {context}

Question: {question}
"""

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    groq_api_key = API_KEY
)

response = llm.invoke(prompt)
print(response.content)