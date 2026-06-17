import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
API_KEY = os.getenv("API_KEY")
PDF_PATH = "./PDF/Nature.pdf"

print("\n========== SUMMARY TOOL ==========")
print("1. Short Summary\n2. Detailed Summary\n3. Chapter-wise Summary")
choice = input("\nEnter choice (1-3): ").strip()

if choice == "1":
    prompt_template = "Summarize the following text briefly into a crisp response of 150-200 words:\n\n{text}"
elif choice == "2":
    prompt_template = "Write a highly detailed summary focusing on all important concepts, facts, and conclusions:\n\n{text}"
elif choice == "3":
    prompt_template = "Organize the following text into a clear, chapter-wise/section summary with main topics and points:\n\n{text}"
else:
    print("Invalid choice."); exit()

print("\nLoading and splitting PDF...")
if not os.path.exists(PDF_PATH):
    print(f"Error: File not found at {PDF_PATH}"); exit()

loader = PyPDFLoader(PDF_PATH)
chunks = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300).split_documents(loader.load())

# Initialize LLM using your .env API_KEY configuration
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=API_KEY
)

# Build custom pipeline using LCEL instead of load_summarize_chain
prompt = PromptTemplate.from_template(prompt_template)
chain = prompt | llm | StrOutputParser()

print(f"Generating summary via Groq from {len(chunks)} chunks...")

# Combine chunk contents to form a comprehensive input context
full_text = "\n\n".join([doc.page_content for doc in chunks])
summary_output = chain.invoke({"text": full_text})

print(f"\n{'='*60}\nSUMMARY OUTPUT\n{'='*60}\n{summary_output}\n{'='*60}")