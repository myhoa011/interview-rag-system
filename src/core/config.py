from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "models/embedding-001"
    EMBEDDING_DIMENSION: int = 764
    
    # Vector Search Configuration
    VECTOR_TOP_K: int = 5
    VECTOR_SIMILARITY_THRESHOLD: float = 0.7
    
    # RAG Configuration
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.2
    
    # PostgreSQL Configuration
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "rag_db"
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 