"""LLM Factory for managing different LLM providers (free alternatives prioritized)."""

from typing import Any, Optional
from loguru import logger

try:
    from llama_index.llms.gemini import Gemini
    from llama_index.embeddings.gemini import GeminiEmbedding
except ImportError:
    logger.warning("Gemini dependencies not installed")
    Gemini = None
    GeminiEmbedding = None

try:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    logger.warning("Ollama dependencies not installed")
    Ollama = None
    OllamaEmbedding = None

try:
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    logger.warning("OpenAI dependencies not installed")
    OpenAI = None
    OpenAIEmbedding = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    logger.warning("HuggingFace dependencies not installed")
    HuggingFaceEmbedding = None

from .config import settings


class LLMFactory:
    """Factory for creating LLM and embedding instances based on configuration."""
    
    @staticmethod
    def create_llm() -> Any:
        """Create LLM instance based on configured provider."""
        provider = settings.llm_provider.lower()
        
        if provider == "gemini":
            return LLMFactory._create_gemini_llm()
        elif provider == "ollama":
            return LLMFactory._create_ollama_llm()
        elif provider == "openai":
            return LLMFactory._create_openai_llm()
        else:
            logger.warning(f"Unknown LLM provider: {provider}, falling back to Gemini")
            return LLMFactory._create_gemini_llm()
    
    @staticmethod
    def create_embedding_model() -> Any:
        """Create embedding model based on configured provider."""
        provider = settings.embedding_provider.lower()
        
        if provider == "huggingface":
            return LLMFactory._create_huggingface_embedding()
        elif provider == "gemini":
            return LLMFactory._create_gemini_embedding()
        elif provider == "ollama":
            return LLMFactory._create_ollama_embedding()
        elif provider == "openai":
            return LLMFactory._create_openai_embedding()
        else:
            logger.warning(f"Unknown embedding provider: {provider}, falling back to HuggingFace")
            return LLMFactory._create_huggingface_embedding()
    
    @staticmethod
    def _create_gemini_llm() -> Any:
        """Create Gemini LLM instance."""
        if not Gemini:
            raise ImportError("Gemini dependencies not installed. Run: pip install llama-index-llms-gemini google-generativeai")
        
        if not settings.gemini_api_key:
            logger.error("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
            logger.info("Get free Gemini API key at: https://ai.google.dev/")
            raise ValueError("Gemini API key required")
        
        logger.info(f"Using Gemini LLM: {settings.gemini_model}")
        return Gemini(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0.1,
            max_tokens=2000
        )
    
    @staticmethod
    def _create_ollama_llm() -> Any:
        """Create Ollama LLM instance."""
        if not Ollama:
            raise ImportError("Ollama dependencies not installed. Run: pip install llama-index-llms-ollama")
        
        logger.info(f"Using Ollama LLM: {settings.ollama_model} at {settings.ollama_base_url}")
        logger.info("Make sure Ollama is running locally with: ollama serve")
        
        return Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
            temperature=0.1,
            context_window=8000
        )
    
    @staticmethod
    def _create_openai_llm() -> Any:
        """Create OpenAI LLM instance."""
        if not OpenAI:
            raise ImportError("OpenAI dependencies not installed. Run: pip install llama-index-llms-openai openai")
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required but not provided")
        
        logger.info(f"Using OpenAI LLM: {settings.openai_model}")
        return OpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=2000
        )
    
    @staticmethod
    def _create_huggingface_embedding() -> Any:
        """Create HuggingFace embedding model."""
        if not HuggingFaceEmbedding:
            raise ImportError("HuggingFace dependencies not installed. Run: pip install llama-index-embeddings-huggingface")
        
        logger.info(f"Using HuggingFace embedding: {settings.embedding_model}")
        return HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            trust_remote_code=True
        )
    
    @staticmethod
    def _create_gemini_embedding() -> Any:
        """Create Gemini embedding model."""
        if not GeminiEmbedding:
            raise ImportError("Gemini embedding dependencies not installed. Run: pip install llama-index-embeddings-gemini")
        
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key required for embeddings")
        
        logger.info("Using Gemini embedding model")
        return GeminiEmbedding(
            api_key=settings.gemini_api_key,
            model_name="models/embedding-001"
        )
    
    @staticmethod
    def _create_ollama_embedding() -> Any:
        """Create Ollama embedding model."""
        if not OllamaEmbedding:
            raise ImportError("Ollama embedding dependencies not installed. Run: pip install llama-index-embeddings-ollama")
        
        # Use a dedicated embedding model for Ollama
        embedding_model = "mxbai-embed-large"  # Good free embedding model
        logger.info(f"Using Ollama embedding: {embedding_model} at {settings.ollama_base_url}")
        logger.info(f"Make sure model is available: ollama pull {embedding_model}")
        
        return OllamaEmbedding(
            model_name=embedding_model,
            base_url=settings.ollama_base_url
        )
    
    @staticmethod
    def _create_openai_embedding() -> Any:
        """Create OpenAI embedding model."""
        if not OpenAIEmbedding:
            raise ImportError("OpenAI embedding dependencies not installed. Run: pip install llama-index-embeddings-openai")
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required for embeddings")
        
        logger.info(f"Using OpenAI embedding: {settings.embedding_model}")
        return OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
    
    @staticmethod
    def get_provider_info() -> dict:
        """Get information about current providers and their status."""
        info = {
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "providers_available": {
                "gemini": Gemini is not None,
                "ollama": Ollama is not None,
                "openai": OpenAI is not None,
                "huggingface": HuggingFaceEmbedding is not None
            },
            "api_keys_configured": {
                "gemini": bool(settings.gemini_api_key),
                "openai": bool(settings.openai_api_key)
            }
        }
        return info