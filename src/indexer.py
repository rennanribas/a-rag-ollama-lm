"""LlamaIndex integration for vector storage and retrieval."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
# LLM and embedding models will be configured via LLM factory
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
import chromadb

from .config import settings
from .llm_factory import LLMFactory
from .crawler import CrawledDocument


class IncrementalIndexer:
    """Manages incremental indexing with LlamaIndex and ChromaDB."""
    
    def __init__(self, collection_name: str = "ai_rag_docs"):
        self.collection_name = collection_name
        self.index: Optional[VectorStoreIndex] = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.storage_context: Optional[StorageContext] = None
        self.document_hashes: Dict[str, str] = {}
        
        # Configure node parser (LLM and embedding models are set by RAGAgent)
        Settings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Note: LLM and embedding models are configured by RAGAgent via LLMFactory
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize ChromaDB and vector store."""
        try:
            # Initialize LLM and embedding models first
            from llama_index.core import Settings
            Settings.llm = LLMFactory.create_llm()
            Settings.embed_model = LLMFactory.create_embedding_model()
            logger.info(f"Configured LlamaIndex with {settings.llm_provider} LLM and {settings.embedding_provider} embeddings")
            
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_directory)
            )
            
            # Get or create collection
            try:
                chroma_collection = chroma_client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                chroma_collection = chroma_client.create_collection(self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize vector store
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Try to load existing index
            storage_dir = settings.chroma_persist_directory / "storage"
            
            if storage_dir.exists() and any(storage_dir.iterdir()):
                try:
                    self.storage_context = StorageContext.from_defaults(
                        persist_dir=str(storage_dir),
                        vector_store=self.vector_store
                    )
                    self.index = load_index_from_storage(self.storage_context)
                    logger.info("Loaded existing index from storage")
                    
                    # Load document hashes
                    self._load_document_hashes()
                    
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}")
                    self._create_new_index()
            else:
                self._create_new_index()
                
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise
    
    def _create_new_index(self):
        """Create a new empty index."""
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create empty index
        self.index = VectorStoreIndex(
            nodes=[],
            storage_context=self.storage_context
        )
        
        logger.info("Created new empty index")
    
    def _get_document_hash(self, doc: CrawledDocument) -> str:
        """Generate a hash for document content and metadata."""
        content = f"{doc.url}|{doc.title}|{doc.content}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_document_hashes(self):
        """Load document hashes from storage."""
        hash_file = settings.chroma_persist_directory / "document_hashes.json"
        
        if hash_file.exists():
            import json
            try:
                with open(hash_file, "r") as f:
                    self.document_hashes = json.load(f)
                logger.info(f"Loaded {len(self.document_hashes)} document hashes")
            except Exception as e:
                logger.warning(f"Failed to load document hashes: {e}")
                self.document_hashes = {}
    
    def _save_document_hashes(self):
        """Save document hashes to storage."""
        hash_file = settings.chroma_persist_directory / "document_hashes.json"
        
        try:
            import json
            with open(hash_file, "w") as f:
                json.dump(self.document_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document hashes: {e}")
    
    def add_documents(self, crawled_docs: List[CrawledDocument], force_update: bool = False) -> int:
        """Add or update documents in the index incrementally."""
        if not self.index:
            raise RuntimeError("Index not initialized")
        
        new_documents = []
        updated_count = 0
        
        for crawled_doc in crawled_docs:
            doc_hash = self._get_document_hash(crawled_doc)
            existing_hash = self.document_hashes.get(crawled_doc.url)
            
            # Check if document needs to be added/updated
            if force_update or existing_hash != doc_hash:
                # Create LlamaIndex document
                document = Document(
                    text=crawled_doc.content,
                    metadata={
                        "url": crawled_doc.url,
                        "title": crawled_doc.title,
                        "crawled_at": crawled_doc.crawled_at,
                        "doc_type": crawled_doc.doc_type,
                        "last_modified": crawled_doc.last_modified,
                        **crawled_doc.metadata
                    },
                    doc_id=crawled_doc.url  # Use URL as document ID
                )
                
                new_documents.append(document)
                self.document_hashes[crawled_doc.url] = doc_hash
                updated_count += 1
                
                logger.info(f"Queued for indexing: {crawled_doc.title}")
        
        if new_documents:
            try:
                # Remove existing documents with same IDs (for updates)
                for doc in new_documents:
                    try:
                        self.index.delete_ref_doc(doc.doc_id, delete_from_docstore=True)
                    except Exception:
                        pass  # Document might not exist yet
                
                # Add new/updated documents
                for doc in new_documents:
                    self.index.insert(doc)
                
                # Persist changes
                self.persist()
                
                logger.info(f"Successfully indexed {len(new_documents)} documents")
                
            except Exception as e:
                logger.error(f"Failed to index documents: {e}")
                raise
        
        return updated_count
    
    def query(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict:
        """Query the index and return relevant documents."""
        if not self.index:
            raise RuntimeError("Index not initialized")
        
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="tree_summarize"
            )
            
            # Execute query
            response = query_engine.query(query_text)
            
            # Extract source documents
            source_docs = []
            if hasattr(response, 'source_nodes'):
                logger.info(f"Found {len(response.source_nodes)} source nodes")
                for i, node in enumerate(response.source_nodes):
                    logger.info(f"Node {i}: score={node.score}, title={node.metadata.get('title', 'Unknown')}")
                    if node.score >= similarity_threshold:
                        source_docs.append({
                            "url": node.metadata.get("url", "Unknown"),
                            "title": node.metadata.get("title", "Unknown"),
                            "content": node.text,
                            "score": node.score,
                            "metadata": node.metadata
                        })
            else:
                logger.warning("No source_nodes found in response")
            
            logger.info(f"Returning {len(source_docs)} documents after filtering")
            
            return {
                "response": str(response),
                "source_documents": source_docs,
                "query": query_text
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_retriever(self, top_k: int = 10):
        """Get a retriever for the index."""
        if not self.index:
            raise RuntimeError("Index not initialized")
        
        return self.index.as_retriever(similarity_top_k=top_k)
    
    def persist(self):
        """Persist the index and metadata to storage."""
        if not self.index or not self.storage_context:
            return
        
        try:
            storage_dir = settings.chroma_persist_directory / "storage"
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            self.storage_context.persist(persist_dir=str(storage_dir))
            self._save_document_hashes()
            
            logger.info("Index persisted successfully")
            
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get indexing statistics."""
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            # Get document store stats
            doc_store = self.storage_context.docstore if self.storage_context else None
            
            stats = {
                "total_documents": len(self.document_hashes),
                "collection_name": self.collection_name,
                "storage_directory": str(settings.chroma_persist_directory)
            }
            
            if doc_store and hasattr(doc_store, 'docs'):
                stats["stored_documents"] = len(doc_store.docs)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def clear_index(self):
        """Clear all documents from the index."""
        try:
            # Clear ChromaDB collection
            if self.vector_store and hasattr(self.vector_store, '_collection'):
                # Get all document IDs first
                collection = self.vector_store._collection
                result = collection.get()
                if result['ids']:
                    collection.delete(ids=result['ids'])
            
            # Recreate storage
            self._initialize_storage()
            self.document_hashes = {}
            
            logger.info("Index cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise