from langchain_openai import ChatOpenAI  # ✅ instead of langchain_community
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

response = llm.invoke("Summarize this review: I hated the delivery experience. It came two weeks late and broken.")
print(response)