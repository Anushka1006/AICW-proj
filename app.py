import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import streamlit as st
import numpy as np
import random 
import time 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DB_PATH = "./chroma_langchain_db"
PDF_FOLDER = "./PDF"
os.makedirs(PDF_FOLDER, exist_ok=True)

#UI 
st.set_page_config(page_title="AI Document Hub", layout="wide")
st.title("PDF Chatbot")
st.sidebar.header("Operations")
st.title("Multi-Functional PDF AI Tool")
st.sidebar.header("Settings & Operations")

#Operation
operation = st.sidebar.selectbox(
    "Choose Operation", 
    ["Vector Database Setup", " Q & A","Generate Summary", "Keyword Highlighter"]
)


#VECTOR DATABASE SETUP
if operation == "Vector Database Setup":
    st.header("Initialize Vector Database")
    st.write("Upload PDFs to store them into the Chroma DB Vector Store.")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Build / Refresh Vector DB"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents..."):
                # Save uploaded files locally to PDF_FOLDER
                for uploaded_file in uploaded_files:
                    with open(os.path.join(PDF_FOLDER, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load documents
                documents = []
                pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
                
                for path in pdf_files:
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    for doc in docs:
                        doc.page_content = f"title: none | text: {doc.page_content}"
                    documents.extend(docs)
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # Embedded via Hugging Face API
                embeddings = HuggingFaceEndpointEmbeddings(
                    repo_id="google/embeddinggemma-300m",
                    huggingfacehub_api_token=HF_TOKEN
                )
                
                # Vector Store Initialization
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    collection_name="pdf-rag",
                    embedding=embeddings,
                    persist_directory=DB_PATH
                )
                st.success(f"Successfully indexed! Total collection count: {vector_store._collection.count()}")

#Q & A
#Q & A
elif operation == " Q & A":
    st.header("Q&A Bot")

    # ── Session state initialization ──────────
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # ── Sidebar: list uploaded files + Filtering ──
    with st.sidebar:
        st.subheader("📄 Filter Documents")
        existing_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")] if os.path.exists(PDF_FOLDER) else []
        
        # Add a selection box to choose which file to query
        filter_option = st.selectbox("Search Scope", ["All Documents"] + existing_pdfs)
        
        st.divider()
        st.subheader("Manage Files")
        if existing_pdfs:
            for filename in existing_pdfs:
                col1, col2 = st.columns([4, 1])
                col1.write(f"📎 {filename}")
                if col2.button("🗑️", key=f"delete_{filename}"):
                    pdf_path = os.path.join(PDF_FOLDER, filename)
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    st.rerun()

    # ── Helper to build/load store with Metadata ──
    def get_vector_store(force_rebuild=False):
        embeddings = HuggingFaceEndpointEmbeddings(
            repo_id="google/embeddinggemma-300m",
            huggingfacehub_api_token=HF_TOKEN
        )
        
        existing_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")] if os.path.exists(PDF_FOLDER) else []
        if not existing_pdfs: return None
            
        vector_store = Chroma(collection_name="pdf-rag", embedding_function=embeddings, persist_directory=DB_PATH)
        
        if force_rebuild or vector_store._collection.count() == 0:
            try: vector_store.delete_collection()
            except: pass
            
            documents = []
            for filename in existing_pdfs:
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
                docs = loader.load()
                for doc in docs:
                    # IMPORTANT: Tagging the metadata for filtering
                    doc.metadata["source"] = filename 
                    doc.page_content = f"title: none | text: {doc.page_content}"
                documents.extend(docs)

            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
            vector_store = Chroma.from_documents(chunks, collection_name="pdf-rag", embedding=embeddings, persist_directory=DB_PATH)
            
        return vector_store

    # ── Answer generation with Filter ──
    def get_answer(question, vector_store, filter_file):
        search_kwargs = {"k": 6}
        # Apply metadata filter if a specific file is selected
        if filter_file != "All Documents":
            search_kwargs["filter"] = {"source": filter_file}
            
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = PromptTemplate.from_template("Use the following context to answer: {context}\n\nQuestion: {question}\n\nAnswer:")
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=API_KEY)
        return (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    # ── Chat Execution ──
    for message in st.session_state.qa_messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        vector_store = get_vector_store()
        if not vector_store:
            st.warning("Please upload a PDF first.")
        else:
            st.session_state.qa_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                answer = get_answer(prompt, vector_store, filter_option)
                st.markdown(answer)
            st.session_state.qa_messages.append({"role": "assistant", "content": answer})
            st.rerun()

# Generate summary
elif operation == "Generate Summary":
    st.header("LLM-Powered Document Summarization")
    
    # Pick a PDF present in the folder
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")] if os.path.exists(PDF_FOLDER) else []
    
    if not pdf_files:
        st.info("No PDFs found in the directory. Please upload files via the 'Vector Database Setup' tab first.")
    else:
        selected_pdf = st.selectbox("Select PDF to summarize", pdf_files)
        summary_type = st.radio("Summary Type", ["Short Summary", "Detailed Summary", "Chapter-wise Summary"])
        
        if st.button("Generate Summary"):
            with st.spinner("Analyzing document chunks via Groq..."):
                if summary_type == "Short Summary":
                    prompt_template = "Summarize the following text briefly into a crisp response of 150-200 words:\n\n{text}"
                elif summary_type == "Detailed Summary":
                    prompt_template = "Write a highly detailed summary focusing on all important concepts, facts, and conclusions:\n\n{text}"
                else:
                    prompt_template = "Organize the following text into a clear, chapter-wise/section summary with main topics and points:\n\n{text}"
                
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, selected_pdf))
                chunks = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300).split_documents(loader.load())
                
                llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=API_KEY)
                prompt = PromptTemplate.from_template(prompt_template)
                chain = prompt | llm | StrOutputParser()
                
                full_text = "\n\n".join([doc.page_content for doc in chunks])
                summary_output = chain.invoke({"text": full_text})
                
                st.subheader("Resulting Summary")
                st.info(summary_output)


# KEYWORD HIGHLIGHTER
elif operation == "Keyword Highlighter":
    st.header(" PDF Keyword Highlighter")
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")] if os.path.exists(PDF_FOLDER) else []
    
    if not pdf_files:
        st.info("No PDFs found. Please drop files via the 'Vector Database Setup' tab first.")
    else:
        selected_pdf = st.selectbox("Select PDF to scan", pdf_files)
        keyword = st.text_input("Enter the keyword to highlight:").strip()
        
        if st.button("Highlight PDF") and keyword:
            with st.spinner("Scanning file pages..."):
                pdf_path = os.path.join(PDF_FOLDER, selected_pdf)
                doc = fitz.open(pdf_path)
                file_has_keyword = False
                
                for page in doc:
                    matches = page.search_for(keyword)
                    if matches:
                        file_has_keyword = True
                        for match in matches:
                            highlight = page.add_highlight_annot(match)
                            highlight.update()
                
                if file_has_keyword:
                    out_name = f"highlighted_{selected_pdf}"
                    doc.save(out_name)
                    doc.close()
                    
                    st.success(f"Keyword matches found!")
                    with open(out_name, "rb") as f:
                        st.download_button(label="Download Highlighted PDF", data=f, file_name=out_name, mime="application/pdf")
                else:
                    doc.close()
                    st.error("Keyword not found in this PDF.")
                    st.error("Keyword not found in this PDF.")
