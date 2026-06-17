from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
import os
from dotenv import load_dotenv

# Automatically loads your existing GROQ_API_KEY from your .env file
load_dotenv()

print("Loading PDF...")
loader = PyPDFLoader("pdf/Nature.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print(f"Split PDF into {len(chunks)} chunks. Starting Groq summarization...")

# Switch the LLM to Groq using Meta's fast Llama 3 model
# This automatically reads GROQ_API_KEY from your .env file
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant"
)


chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.invoke(chunks)

print("\n--- FINAL SUMMARY (via Groq) ---")
print(summary["output_text"])
