from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: List[ChatMessage] = []
    mode: str = "Expert"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class AnalyticsResponse(BaseModel):
    word_count: int
    reading_time_minutes: int
    vocabulary_size: int
    most_frequent_terms: List[List[Any]] # e.g. [["word", 10], ...]

class SummaryResponse(BaseModel):
    short: str
    medium: str
    detailed: str

class DocumentResponse(BaseModel):
    session_id: str
    filename: str
    page_count: int
    summary: SummaryResponse
    keywords_keybert: List[Dict[str, Any]]
    keywords_tfidf: List[Dict[str, Any]]
    analytics: AnalyticsResponse
