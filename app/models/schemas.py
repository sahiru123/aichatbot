# app/models/schemas.py
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr

class LoginRequest(BaseModel):
    email: EmailStr

class LoginResponse(BaseModel):
    token: str

class FetchDocumentsRequest(BaseModel):
    reprocess_existing: bool = False

class SearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5

class SearchResponse(BaseModel):
    answer: str
    sources: List[Dict]
    search_info: Dict

class ProcessedFile(BaseModel):
    name: str
    drive: str
    chunks_processed: int

class FailedFile(BaseModel):
    name: str
    drive: str
    reason: str

class FetchResponse(BaseModel):
    status: str
    processed_files: List[ProcessedFile]
    failed_files: List[FailedFile]
    total_processed: int
    total_failed: int
    skipped_files: int = 0

class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ConversationResponse(BaseModel):
    conversation_id: str
    messages: List[ConversationMessage]
    metadata: Dict = {}

class SearchResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: List[Dict]
    search_info: Dict