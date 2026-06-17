import os
from pydantic import BaseModel, Field
from typing import List

# Import your existing vector store configuration
# (Adjust this import based on where your 'vector_store' is initialized in db.py or search.py)
from db import vector_store 
from langchain_openai import ChatOpenAI

# 1. Define the Pydantic schema for structured output
class Question(BaseModel):
    id: int
    question_type: str = Field(description="MCQ, True/False, or Short Answer")
    question_text: str = Field(description="The text of the question")
    options: List[str] = Field(default=[], description="List of options if MCQ, otherwise empty")
    correct_answer: str = Field(description="The correct answer text or choice")

class Quiz(BaseModel):
    title: str = Field(description="A catchy title for the quiz based on the topic")
    questions: List[Question]

def generate_quiz():
    print("🤖 Retrieving document context from Chroma...")
    
    # 2. Retrieve a large chunk of diverse content using MMR
    docs = vector_store.max_marginal_relevance_search(
        "Overall important core concepts, key definitions, and summary points.",
        k=6,
        fetch_k=15
    )
    
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # 3. Build the generation prompt
    prompt = f"""
    You are an expert teacher. Using ONLY the context provided below, generate a comprehensive quiz consisting of:
    - 5 Multiple Choice Questions (MCQ)
    - 3 True/False Questions
    - 2 Short Answer Questions

    Context:
    {context}
    """

    # 4. Initialize LLM with structured output tracking
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    structured_llm = llm.with_structured_output(Quiz)
    
    print("🧠 LLM is generating your structured quiz...")
    quiz = structured_llm.invoke(prompt)
    
    # 5. Beautifully print the output
    print(f"\n✨ QUIZ: {quiz.title} ✨\n" + "="*40)
    
    for q in quiz.questions:
        print(f"\nQ{q.id} [{q.question_type}]: {q.question_text}")
        if q.options:
            for idx, opt in enumerate(q.options):
                print(f"   {chr(65 + idx)}. {opt}")
        print(f"👉 Answer: {q.correct_answer}")

if __name__ == "__main__":
    generate_quiz()