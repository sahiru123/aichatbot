# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import redis
from datetime import datetime
import jwt
import tempfile

from pinecone import Pinecone, PodSpec
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from app.models.database import Base, User, Session as DBSession
from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    FetchResponse,
    FetchDocumentsRequest,
    UserCreate,
    LoginRequest,
    LoginResponse
)
from app.config import settings
from app.services.session_manager import SessionManager
from app.services.conversation import ConversationService
from app.services.document_processor import DocumentProcessor
from app.services.sharepoint import SharePointService
from app.auth.sharepoint import SharePointAuth
from app.utils.logging import setup_logging
from app.database import get_db, engine

# Create database tables
Base.metadata.create_all(bind=engine)

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(title="Document Search API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize Redis
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True
)

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Create index if it doesn't exist
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=PodSpec(environment=settings.PINECONE_ENV)
    )

index = pc.Index(settings.PINECONE_INDEX_NAME)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=settings.EMBEDDING_DEPLOYMENT_NAME,
    openai_api_key=settings.AZURE_OPENAI_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version="2024-05-01-preview",
    chunk_size=1,
    max_retries=3
)

# Initialize language model
llm = AzureChatOpenAI(
    azure_deployment=settings.DEPLOYMENT_NAME,
    openai_api_key=settings.AZURE_OPENAI_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version="2024-05-01-preview",
    temperature=0.1,
    max_tokens=1000,
    top_p=0.95,
    request_timeout=30,
)

# Initialize vector store
vectorstore = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Initialize services
sharepoint_auth = SharePointAuth()
sharepoint_service = SharePointService(sharepoint_auth)
document_processor = DocumentProcessor(vectorstore)

# Session manager will be initialized per-request
def get_session_manager(db: Session = Depends(get_db)):
    return SessionManager(db, settings.SECRET_KEY)

# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session_manager: SessionManager = Depends(get_session_manager)
) -> User:
    try:
        token = credentials.credentials
        logger.info(f"Processing authentication for token starting with: {token[:10]}...")
        
        try:
            # Decode token
            payload = jwt.decode(token, session_manager.secret_key, algorithms=['HS256'])
            
            # Check session in database
            session = session_manager.db.query(DBSession).filter(
                DBSession.id == payload['session_id'],
                DBSession.expires_at > datetime.utcnow()
            ).first()
            
            if not session:
                raise ValueError("Invalid or expired session")
            
            return session.user
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token format: {str(e)}")
            
    except ValueError as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )

# Initialize conversation service per-request
def get_conversation_service(
    db: Session = Depends(get_db)
) -> ConversationService:
    return ConversationService(db, redis_client)

@app.post("/register", response_model=LoginResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    session_manager: SessionManager = Depends(get_session_manager)
):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user = User(email=user_data.email)
    db.add(user)
    db.commit()
    
    # Create session
    token = session_manager.create_session(user)
    
    return LoginResponse(token=token)

@app.post("/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db),
    session_manager: SessionManager = Depends(get_session_manager)
):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        user = User(email=login_data.email)
        db.add(user)
        db.commit()
    
    token = session_manager.create_session(user)
    return LoginResponse(token=token)

@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    try:
        # Get or create conversation
        conversation_id = request.conversation_id or conversation_service.create_conversation(current_user)
        
        # Add user's query to conversation history
        conversation_service.add_message(conversation_id, "user", request.query, current_user)
        
        # Get conversation history
        conversation_context = conversation_service.get_conversation_context(conversation_id, current_user)
        
        # Create system prompt with conversation awareness
        system_template = """You are an expert at analyzing course materials and providing accurate, detailed answers. 
Use the following context to answer the question. Consider the conversation history for better context understanding.

Instructions:
1. Carefully analyze the provided context and conversation history
2. Focus on finding specific, relevant information
3. If exact information is found, provide it with detailed explanation
4. Include direct quotes when relevant, using markdown formatting
5. If information isn't found in the context, clearly state that
6. Maintain conversation coherence with previous interactions
7. Resolve references to previous questions/answers in the conversation

Previous Conversation:
{chat_history}

Current Context:
{summaries}

Question: {question}

Response Guidelines:
- Be specific and detailed
- Use markdown formatting for clarity
- Include source references
- Maintain conversational coherence
- If information is not in the context, say "The specific information about [topic] is not found in the provided materials."
"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        # Format conversation history
        chat_history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_context
        ])

        # Setup retriever with conversation context
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": request.top_k,
                "filter": None
            }
        )

        # Create chain with conversation history
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True,
                "document_separator": "\n\n"
            },
            return_source_documents=True
        )

        # Get response with conversation context
        response = chain({
            "question": request.query,
            "chat_history": chat_history
        })
        
        # Add assistant's response to conversation history
        conversation_service.add_message(conversation_id, "assistant", response["answer"], current_user)
        
        if not response["source_documents"]:
            return SearchResponse(
                answer="I could not find any relevant information in the available documents to answer your question.",
                conversation_id=conversation_id,
                sources=[],
                search_info={
                    "total_documents_searched": 0,
                    "search_query": request.query
                }
            )

        return SearchResponse(
            answer=response["answer"],
            conversation_id=conversation_id,
            sources=[
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "page": doc.metadata.get("page", 1),
                    "text": doc.page_content[:300],
                    "source_type": doc.metadata.get("source", "unknown"),
                    "file_location": f"{doc.metadata.get('drive', 'unknown')}/{doc.metadata.get('title', 'unknown')}"
                }
                for doc in response["source_documents"]
            ],
            search_info={
                "total_documents_searched": len(response["source_documents"]),
                "search_query": request.query,
                "most_relevant_source": response["source_documents"][0].metadata.get("title", "Unknown") if response["source_documents"] else None
            }
        )

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Retrieve conversation history."""
    try:
        conversation = conversation_service.get_conversation(conversation_id, current_user)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch-documents", response_model=FetchResponse)
async def fetch_documents(
    request: FetchDocumentsRequest,
    current_user: User = Depends(get_current_user)
):
    """Fetch and process documents from SharePoint."""
    logger.info("Starting document fetch process")
    try:
        drives = sharepoint_service.get_all_drives(settings.SHAREPOINT_SITE_ID)
        processed_files = []
        failed_files = []
        skipped_files = 0
        total_files = 0

        # Count total files first
        for drive in drives:
            files = sharepoint_service.get_sharepoint_files(drive.get("id"))
            total_files += len(files)

        current_file = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            for drive in drives:
                drive_id = drive.get("id")
                drive_name = drive.get("name")
                
                files = sharepoint_service.get_sharepoint_files(drive_id)
                
                for file in files:
                    current_file += 1
                    file_name = file.get("name")
                    file_id = file.get("id")
                    
                    logger.info(f"Processing file {current_file}/{total_files}: {file_name}")
                    
                    try:
                        if not request.reprocess_existing:
                            existing_docs = vectorstore.similarity_search(
                                "Check for duplicates",
                                filter={"file_id": file_id},
                                k=1
                            )
                            if existing_docs:
                                logger.info(f"Skipping existing file: {file_name}")
                                skipped_files += 1
                                continue

                        file_path = sharepoint_service.download_file(
                            drive_id, file_id, temp_dir, file_name
                        )
                        if not file_path:
                            raise Exception("Download failed")

                        metadata = {
                            "title": file_name,
                            "drive": drive_name,
                            "file_id": file_id,
                            "source": "sharepoint",
                            "user_id": current_user.id
                        }
                        
                        chunks = document_processor.process_document(file_path, metadata)
                        
                        processed_files.append({
                            "name": file_name,
                            "drive": drive_name,
                            "chunks_processed": len(chunks)
                        })
                        
                        logger.info(f"Successfully processed {len(chunks)} chunks from {file_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_name}: {str(e)}")
                        failed_files.append({
                            "name": file_name,
                            "drive": drive_name,
                            "reason": str(e)
                        })

        return FetchResponse(
            status="completed",
            processed_files=processed_files,
            failed_files=failed_files,
            total_processed=len(processed_files),
            total_failed=len(failed_files),
            skipped_files=skipped_files
        )

    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)