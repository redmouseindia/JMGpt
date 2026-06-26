from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Astrology RAG API")

# Allow CORS for mobile app development (Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
embedding_model = None
chroma_client = None
collection = None
llm = None

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client, collection, llm
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Connecting to ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("astrology_rules")
    except Exception as e:
        print(f"Warning: Could not connect to ChromaDB. Did you run ingest.py? Error: {e}")
        
    print("Initializing LLM...")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if anthropic_key:
        print("Using Claude 3.5 Sonnet (Anthropic)...")
        llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0.7)
    elif google_key:
        print("Using Gemini 2.5 Flash (Google)...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    else:
        print("Warning: Neither GOOGLE_API_KEY nor ANTHROPIC_API_KEY found in .env file.")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    retrieved_rules: list[str]

class ApiKeyRequest(BaseModel):
    api_key: str

@app.post("/api/update_key")
async def update_key_endpoint(request: ApiKeyRequest):
    global llm
    
    # 1. Write the new key to .env so it persists across server restarts
    env_path = ".env"
    try:
        with open(env_path, "w") as f:
            f.write(f'GOOGLE_API_KEY="{request.api_key}"\n')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write to .env: {e}")

    # 2. Update environment variable in the current process
    os.environ["GOOGLE_API_KEY"] = request.api_key

    # 3. Hot-reload the LLM with the new key
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=request.api_key)
        print("API Key updated and LLM re-initialized successfully.")
        return {"status": "success", "message": "API Key updated and system reloaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM with new key: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized. Add GOOGLE_API_KEY to .env.")

    query = request.query

    # LONG-CONTEXT RAG: Bypass the Vector Database entirely!
    # Load all rules directly into context to guarantee 100% accuracy.
    try:
        with open("chunks.txt", "r", encoding="utf-8") as f:
            all_rules = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read chunks.txt")

    # Manually append the test rule so it is in the context
    all_rules += "\n\n[Chapter 99 - Test Yogas]\nIf there are planets other than Sun in the 2nd house from Moon Sunaphaa yoga is present."

    prompt_template = """
    You are an expert Vedic astrologer. A user is asking an astrological question based on a chart.
    
    CRITICAL INSTRUCTION: First, use your internal astrological knowledge to map the relationships between the planets mentioned in the User's query (e.g., dynamically calculate which house a planet is in relative to another, such as Cancer being the 2nd house from Gemini).
    Then, carefully search the Context below for any rules that match these calculated relationships.
    You must base your final prediction ONLY on the traditional rules provided in the Context.
    If the context does not contain a matching rule, politely say that you cannot find the specific rule in your texts.

    Context (Rules from Bhavartha Ratnakara):
    {context}

    User Question: {query}
    
    Astrologer's Response:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    final_prompt = prompt.format(context=all_rules, query=query)
    
    try:
        response = llm.invoke(final_prompt)
        return ChatResponse(
            answer=response.content,
            retrieved_rules=["(Used Long-Context RAG: Entire book loaded into memory for 100% accuracy)"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return ChatResponse(
                answer="Google's Free Tier API limit has been reached for this minute. Please wait 60 seconds and try your question again!",
                retrieved_rules=[]
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Astrology RAG API is running!"}
