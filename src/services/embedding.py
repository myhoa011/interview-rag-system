from typing import List, Dict, Any, Optional
import time
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LangchainDocument

from ..core.config import settings


class EmbeddingService:
    """Service to generate embeddings using Gemini API"""
    
    def __init__(self):
        # Initialize Langchain Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="retrieval_document"
        )
        
        # Text splitter for chunking
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.6,
            min_chunk_size=300
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a text
        
        Args:
            text: Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        start_time = time.time()
        
        try:
            # Use Langchain wrapper for Google embeddings
            result = await self.embeddings.aembed_query(text)
            
            # Log execution time
            elapsed = time.time() - start_time
            print(f"Generated embedding in {elapsed:.2f}s")
            
            return result
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of text inputs
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        start_time = time.time()
        
        try:
            # Use Langchain wrapper for Google embeddings
            results = await self.embeddings.aembed_documents(texts)
            
            # Log execution time
            elapsed = time.time() - start_time
            print(f"Generated {len(texts)} embeddings in {elapsed:.2f}s")
            
            return results
        except Exception as e:
            print(f"Error generating embeddings batch: {str(e)}")
            raise
    
    async def chunk_and_embed_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks and generate embeddings
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List[Dict[str, Any]]: List of chunks with embeddings
        """
        if metadata is None:
            metadata = {}
        
        # Extract the parent document ID (create new if not exists)
        parent_document_id = metadata.get("document_id", str(uuid.uuid4()))
        
        # Create a clean copy of metadata without document_id to avoid duplication
        clean_metadata = metadata.copy()
        if "document_id" in clean_metadata:
            del clean_metadata["document_id"]
        
        # Create Langchain document
        doc = LangchainDocument(page_content=content, metadata=clean_metadata)
        
        # Split document into chunks using semantic chunking
        documents = await self.text_splitter.atransform_documents([doc])
        
        # Generate embeddings for chunks
        chunk_texts = [doc.page_content for doc in documents]
        embeddings = await self.generate_embeddings_batch(chunk_texts)
        
        # Combine chunks and embeddings
        results = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Add chunk metadata
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                "chunk_id": str(uuid.uuid4()),        # Use UUID for unique chunk identification
                "chunk_index": i,                     # Keep index for ordering
                "parent_document_id": parent_document_id  # Link to original document
            })
            
            results.append({
                "content": doc.page_content,
                "metadata": chunk_metadata,
                "embedding": embedding
            })
        
        return results 