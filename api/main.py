import os
import uuid
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.api.schemas import ChatRequest, ChatResponse, DocumentResponse
from backend.core.pdf_extractor import extract_text
from backend.core.preprocessor import preprocess_text
from backend.core.summarizer import get_t5_summarizer
from backend.core.keyword_extractor import extract_all_keywords
from backend.core.analytics_engine import compute_document_analytics
from backend.core.rag_engine import setup_rag_pipeline, generate_answer

# Load env variables
load_dotenv()

app = FastAPI(
    title="ResearchLens AI Backend API",
    version="1.0.0",
    description="FastAPI backend for NLP document processing and RAG."
)

# CORS Middleware (Allow all for development, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a real prod setup, lock this to the Hugging Face Space URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for sessions (ephemeral, wiped on restart)
# Structure: { session_id: { "vectorstore": Chroma, "filename": str, "pages": [] } }
SESSION_STORE = {}

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/v1/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    session_id = str(uuid.uuid4())
    
    # Save uploaded file to temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    try:
        # 1. Extract Text
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        extraction = extract_text(pdf_bytes)
        raw_text = extraction['text']
        pages_list = extraction['pages']
        
        # 2. Preprocess Text
        prep = preprocess_text(raw_text)
        cleaned_text = prep['cleaned_text']
        
        # 3. Analytics
        analytics_data = compute_document_analytics(cleaned_text)
        
        # 4. Keywords
        keywords_data = extract_all_keywords(cleaned_text, top_n=10)
        
        # 5. Summarization (Using T5 for efficiency on cloud)
        summarizer = get_t5_summarizer()
        summary_short = summarizer.summarize(cleaned_text, max_length=150, min_length=50)
        summary_med = summarizer.summarize(cleaned_text, max_length=250, min_length=100)
        summary_long = summarizer.summarize(cleaned_text, max_length=400, min_length=200)
        
        # 6. Initialize RAG
        vectorstore, chunk_count = setup_rag_pipeline(pages_list)
        
        # Store in session
        SESSION_STORE[session_id] = {
            "vectorstore": vectorstore,
            "filename": file.filename,
            "pages": pages_list
        }
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "page_count": extraction['page_count'],
            "summary": {
                "short": summary_short,
                "medium": summary_med,
                "detailed": summary_long
            },
            "keywords_keybert": keywords_data.get('keybert', []),
            "keywords_tfidf": keywords_data.get('tfidf', []),
            "analytics": {
                "word_count": analytics_data.get("word_count", 0),
                "reading_time_minutes": analytics_data.get("reading_time_minutes", 0),
                "vocabulary_size": analytics_data.get("vocabulary_size", 0),
                "most_frequent_terms": analytics_data.get("most_frequent_terms", [])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    session_data = SESSION_STORE.get(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload the document again.")
        
    vectorstore = session_data["vectorstore"]
    
    try:
        # Convert Pydantic ChatMessage back to dicts expected by generate_answer
        history_dicts = [{"role": m.role, "content": m.content} for m in request.chat_history]
        
        answer, docs = generate_answer(vectorstore, request.question, history_dicts, request.mode)
        
        # Format sources
        sources = []
        for doc in docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:200] + "..." # Snippet
            })
            
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")
