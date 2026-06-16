from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    groq_api_key = API_KEY
)

response = llm.invoke("Hello, how are you?")
print(response.content)