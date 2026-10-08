import os
from pathlib import Path
from dotenv import load_dotenv

from pinecone import AsyncPinecone       
from google import genai                  

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing")

# Gemini client Configuration
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Pinecone Configuration
pc = AsyncPinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = os.getenv("INDEX_NAME")
CLOUD = os.getenv("CLOUD")
REGION = os.getenv("REGION")