import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-flash")

# Simple test prompt
response = model.generate_content("Hello, can you explain what RAG means?")
print("\nGemini Response:\n", response.text)
