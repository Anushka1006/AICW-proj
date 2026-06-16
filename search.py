from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

embeddings = HuggingFaceEmbeddings(model_name = "google/embeddinggemma-300m")

vector_store = Chroma(
    collection_name="pdf-rag",
    persist_directory = "./chroma_langchain_db",
    embedding_function = embeddings,
)
print(f"Collection count: {vector_store._collection.count()}")

while True:
    user_prompt = input("Prompt (type 'q' to quit):")
    if user_prompt.strip() == "q":
        break

    results = vector_store.similarity_search(
        user_prompt,
        k = 1
    )

    for i, doc in enumerate(results, 1):
        print(f"Result: {i}")
        print(doc.page_content)
        print(" ")