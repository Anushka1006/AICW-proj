from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# -----------------------------
# Load PDF
# -----------------------------
print("Loading PDF...")

loader = PyPDFLoader("pdf/Nature.pdf")
docs = loader.load()

# -----------------------------
# Split PDF into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"Split PDF into {len(chunks)} chunks.")

# -----------------------------
# Load LLM
# -----------------------------
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant"
)

# -----------------------------
# User Menu
# -----------------------------
print("\n========== PDF SUMMARIZER ==========")
print("1. Short Summary")
print("2. Detailed Summary")
print("3. Chapter-wise Summary")
print("4. Exit")

choice = input("\nEnter your choice (1-4): ")

# -----------------------------
# Prompt Selection
# -----------------------------
if choice == "1":

    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Summarize the following text briefly.

        TEXT:
        {text}

        SHORT SUMMARY:
        """
    )

    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Combine the following summaries into one concise summary
        of about 150-200 words.

        {text}

        FINAL SHORT SUMMARY:
        """
    )

elif choice == "2":

    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Read the following text carefully.

        {text}

        Write a detailed summary including all important concepts,
        facts, explanations and conclusions.
        """
    )

    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Combine the summaries below into one comprehensive summary.

        Include:
        - Important concepts
        - Key facts
        - Examples (if present)
        - Conclusions

        {text}

        FINAL DETAILED SUMMARY:
        """
    )

elif choice == "3":

    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Read the following text.

        {text}

        Identify the major section or chapter discussed and summarize it.
        """
    )

    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Organize the summaries chapter-wise.

        For every chapter include:

        Chapter/Section Name
        Main Topics
        Important Points
        Conclusion

        {text}

        FINAL CHAPTER-WISE SUMMARY:
        """
    )

elif choice == "4":
    print("Thank you!")
    exit()

else:
    print("Invalid choice.")
    exit()

# -----------------------------
# Create Summarization Chain
# -----------------------------
print("\nGenerating summary...\n")

chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt
)

summary = chain.invoke(chunks)

# -----------------------------
# Print Summary
# -----------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(summary["output_text"])
print("=" * 60)