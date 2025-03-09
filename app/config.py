import os
from typing import Any, Dict, Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Document Search API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = os.getenv("BACKEND_CORS_ORIGINS", "*").split(",")
    
    # Azure OpenAI Settings
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
    EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    DEPLOYMENT_NAME: str = os.getenv("DEPLOYMENT_NAME")
    AZURE_OPENAI_EMBEDDING_KEY: str = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
    
    # OpenAI Model Settings
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    TOP_P: float = float(os.getenv("TOP_P", "0.95"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Pinecone Settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV: str = os.getenv("PINECONE_ENV")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "1536"))
    PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")
    
    # SharePoint Settings
    SHAREPOINT_SITE_ID: str = os.getenv("SHAREPOINT_SITE_ID")
    SHAREPOINT_CLIENT_ID: str = os.getenv("SHAREPOINT_CLIENT_ID")
    SHAREPOINT_CLIENT_SECRET: str = os.getenv("SHAREPOINT_CLIENT_SECRET")
    SHAREPOINT_TENANT_ID: str = os.getenv("SHAREPOINT_TENANT_ID")
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "3600"))
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    # Document Processing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "20"))
    SUPPORTED_FORMATS: List[str] = os.getenv("SUPPORTED_FORMATS", ".pdf,.docx").split(",")
    
    # Conversation Settings
    MAX_CONTEXT_MESSAGES: int = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))
    MAX_CONVERSATION_AGE_DAYS: int = int(os.getenv("MAX_CONVERSATION_AGE_DAYS", "30"))
    
    # Performance Settings
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    MAX_REQUEST_SIZE_MB: int = int(os.getenv("MAX_REQUEST_SIZE_MB", "10"))

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Create settings instance
settings = get_settings()

# Validate required settings
def validate_settings(settings: Settings) -> None:
    """Validate that all required settings are present."""
    required_settings = [
        "SECRET_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "PINECONE_API_KEY",
        "DATABASE_URL"
    ]
    
    missing_settings = [
        setting for setting in required_settings 
        if not getattr(settings, setting)
    ]
    
    if missing_settings:
        raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

# Validate settings on startup
validate_settings(settings)