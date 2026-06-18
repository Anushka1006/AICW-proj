import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv("API_KEY")
pdf_folder = r"./PDF"

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
if not os.path.exists(pdf_folder):
    print(f"Error: Folder not found at {pdf_folder}"); exit()

pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
if not pdf_files:
    print(f"No PDF files found in {pdf_folder}."); exit()

llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=API_KEY)
prompt = PromptTemplate.from_template(prompt_template)
chain = prompt | llm | StrOutputParser()
for pdf_file in pdf_files:
    full_path = os.path.join(pdf_folder, pdf_file)
    print(f"\nLoading and splitting: {pdf_file}...")
    
    try:
        loader = PyPDFLoader(full_path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300).split_documents(loader.load())
        
        full_text = "\n\n".join([doc.page_content for doc in chunks])
        print(f"Generating summary via Groq from {len(chunks)} chunks...")
        summary_output = chain.invoke({"text": full_text})
        
        print(f"\n{'='*60}\nSUMMARY OUTPUT FOR: {pdf_file}\n{'='*60}\n{summary_output}\n{'='*60}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")