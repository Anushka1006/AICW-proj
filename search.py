import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=HF_TOKEN
)

vector_store = Chroma(
    collection_name="pdf-rag",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
)
print(f"Collection count: {vector_store._collection.count()}")

while True:
    user_prompt = input("Prompt (type 'q' to quit): ")
    if user_prompt.strip().lower() == "q":
        break
    if not user_prompt.strip():
        continue
    formatted_query = f"task: search result | query: {user_prompt}"
    results = vector_store.similarity_search(formatted_query, k=1)

    print("\nResults:")
    for i, doc in enumerate(results, 1):
        clean_text = doc.page_content.replace("title: none | text: ", "")
        print(f"[{i}] {clean_text}\n")