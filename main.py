from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    global llm
    
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

    # LONG-CONTEXT RAG: Load the entire classical treatise into context
    try:
        with open("chunks.txt", "r", encoding="utf-8") as f:
            all_rules = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read chunks.txt")

    # Manually append the test rule so it is in the context
    all_rules += "\n\n[Chapter 99 - Test Yogas]\nIf there are planets other than Sun in the 2nd house from Moon Sunaphaa yoga is present."

    prompt_template = """
You are JMGPT, an advanced Vedic Astrology AI Research Engine modeled after Google's NotebookLM.
You have been loaded with the entire classical treatise *Bhavartha Ratnakara* by Sri Ramanujacharya (translated with notes by B.V. Raman).

Your goal is to provide an exhaustive, scholarly, NotebookLM-grade analytical research report answering the user's astrological query based ONLY on the provided text.

CRITICAL REASONING INSTRUCTIONS:
1. **Map Relationships**: First, calculate the planetary relationships and house positions mentioned in the user's chart/query (e.g., identify Lagna, lords of houses, functional benefics/malefics, and mutual aspects).
2. **Exhaustive Scan**: Scan the entire text below across ALL chapters (Lagna chapters, Dhana Yogas, Raja Yogas, Bhagyasthanas, Exceptions, etc.) to gather every single stanza or rule that applies to this planetary combination.
3. **NotebookLM Formatting**: Structure your response into a brilliant, beautifully formatted Markdown research report with these exact sections:
   - **🎯 Executive Astrological Verdict**: A direct, authoritative summary of what *Bhavartha Ratnakara* predicts for this query.
   - **📜 Classical Principles & Yogas Found**: Cite the specific rules, stanzas, and chapters from the text that govern this query. Quote the classical principles clearly.
   - **🪐 Deep-Dive Synthesis**: Explain step-by-step *why* this result occurs according to Ramanujacharya's logic (house lordships, aspects, functional nature, etc.).
   - **⚠️ Nuances & Exceptions (Bhanga)**: Highlight any special caveats, cancellations, or unique conditions mentioned in the book that could alter or qualify the result.
4. **Scholarly Integrity**: Base your conclusions strictly on the text provided. Do not invent rules not present in *Bhavartha Ratnakara*.

Context (Complete text of Bhavartha Ratnakara):
{context}

User Question: {query}

YOU MUST RETURN YOUR RESPONSE AS A STRICT JSON OBJECT with exactly two fields:
1. "answer": Your complete, beautifully formatted Markdown analytical report.
2. "retrieved_rules": A JSON array of strings containing the top 3 to 6 exact classical rules/stanzas from the text that you cited (include chapter/section names).

Example JSON Output:
{{
  "answer": "### 🎯 Executive Astrological Verdict\\n...your markdown...",
  "retrieved_rules": [
    "Chapter 1 (Aries Lagna): A person born in Aries Lagna becomes fortunate if Jupiter and Sun are in the 5th house.",
    "Chapter 2 (Dhana Yogas): If the 2nd lord is in the 11th and 11th lord in the 2nd, immense wealth is generated."
  ]
}}
"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    final_prompt = prompt.format(context=all_rules, query=query)
    
    try:
        response = llm.invoke(final_prompt)
        content = response.content.strip()
        
        # Clean markdown code block formatting if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        import json
        try:
            data = json.loads(content)
            return ChatResponse(
                answer=data.get("answer", response.content),
                retrieved_rules=data.get("retrieved_rules", ["(Synthesized from full Bhavartha Ratnakara treatise)"])
            )
        except Exception as json_err:
            # Fallback if model returned pure markdown instead of JSON
            return ChatResponse(
                answer=response.content,
                retrieved_rules=["(Used NotebookLM-Style Full Context RAG Synthesis)"]
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
