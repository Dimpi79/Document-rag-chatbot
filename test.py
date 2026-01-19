import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env")

pc = Pinecone(api_key=api_key)

# list indexes (new SDK returns an object, so we print raw first)
indexes = pc.list_indexes()
print("RAW:", indexes)

# safer way: try to access .names() if available
try:
    print("Index names:", indexes.names())
except Exception:
    # fallback: just print directly
    print("Index names (fallback):", indexes)

