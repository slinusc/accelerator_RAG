from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # ELOG Configuration
    elog_url: str = "https://elog-gfa.psi.ch/SwissFEL+commissioning/"
    elog_username: Optional[str] = None
    elog_password: Optional[str] = None
    elog_poll_interval: int = 300  # seconds
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:32b-q4_K_M"
    ollama_timeout: int = 300
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "swissfel_rag"
    qdrant_vector_size: int = 384  # sentence-transformers/all-MiniLM-L6-v2
    
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Embedding Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda" if available
    
    # RAG Configuration
    max_context_length: int = 4096
    max_retrieved_docs: int = 10
    similarity_threshold: float = 0.7
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Agent Configuration
    agent_timeout: int = 60
    max_reasoning_steps: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()