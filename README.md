# 📚 PDF Question-Answering Chatbot using RAG

An intelligent PDF Question-Answering Chatbot built using Retrieval-Augmented Generation (RAG). The application allows users to upload multiple PDF documents, create a vector database from the uploaded content, and interact with the documents through natural language queries.

The system leverages LangChain, ChromaDB, Hugging Face Embeddings, and Groq LLMs to provide accurate, context-aware responses from uploaded documents.

---

##  Features

### 📄 Multi-PDF Upload & Vector Database Creation

* Upload one or more PDF documents.
* Automatically extract and process document content.
* Generate embeddings using Google's Gemma Embedding Model.
* Store embeddings in a ChromaDB vector database for efficient retrieval.

### 💬 Question & Answering

* Ask questions in natural language.
* Retrieve the most relevant document chunks using semantic search.
* Generate accurate answers using Groq-hosted Large Language Models.

### 📝 Smart Document Summarization

Generate summaries in multiple formats:

1. **Short Summary**

   * Concise overview of the document.

2. **Detailed Summary**

   * Comprehensive explanation of important concepts and information.

3. **Chapter-wise Summary**

   * Structured summary organized by sections or chapters.

### 🔍 Keyword Highlighter

* Highlight important keywords within uploaded documents.
* Quickly identify key concepts and topics.
* Improve document navigation and understanding.

### 📂 Document Management

* View all uploaded documents.
* Maintain a list of processed PDFs.
* Delete selected documents from the vector database when no longer required.

---

## 🖥️ User Interface

The application provides an intuitive Streamlit-based interface.

### Sidebar Components

#### Vector Database Setup

* Upload PDF files.
* Create and update the vector database.

#### Question & Answer Section

* Ask questions related to uploaded documents.
* Receive context-aware responses generated using RAG.

#### Summary Generator

* Generate:

  * Short Summary
  * Detailed Summary
  * Chapter-wise Summary

#### Keyword Highlighter

* Extract and highlight important keywords from documents.

#### Uploaded Documents

* Displays a list of all uploaded and indexed PDF files.

#### Manage Documents

* Delete selected PDFs from the vector database.
* Keep the knowledge base organized and up to date.

---

## 🏗️ Tech Stack

### Frontend

* Streamlit

### LLM & AI Frameworks

* LangChain
* Groq LLM

### Embedding Model

* Google Gemma Embedding Model
* Hugging Face

### Vector Database

* ChromaDB

### Document Processing

* PyPDFLoader
* Recursive Character Text Splitter

---

## ⚙️ Workflow

1. Upload one or more PDF documents.
2. Extract text from uploaded files.
3. Split documents into manageable chunks.
4. Generate embeddings using Gemma Embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks based on user queries.
7. Generate responses using Groq LLM.
8. Provide summaries, keyword highlights, and document insights.

---

## 📁 Project Structure

```text
project/
│
├── app.py                 # Streamlit application
├── db.py                  # Vector database creation
├── chroma_db/             # Chroma vector database
├── uploaded_files/        # Uploaded PDFs
├── utils/                 # Helper functions
├── requirements.txt
├── .env
└── README.md
```

---

## 🔧 Installation

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### Run Application

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

* Academic Research
* Technical Documentation Search
* Study Material Summarization
* Corporate Knowledge Management
* Report Analysis
* Multi-Document Question Answering

---

## Future Enhancements

* Conversation Memory
* Citation-Based Answers
* OCR Support for Scanned PDFs
* Export Summaries to PDF
* Multi-Language Support
* Advanced Search Filters

---

## Contributors

Developed as a Retrieval-Augmented Generation (RAG) based intelligent document assistant using LangChain, Groq, Hugging Face, and ChromaDB.