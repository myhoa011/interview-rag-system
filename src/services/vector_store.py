from typing import List, Dict, Any, Optional
import uuid
import time
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
# Removed PGVector imports - using SQLModel only

from ..db.database import get_session
from ..models import Document, DocumentCreate, DocumentResponse
from ..core.config import settings
from .embedding import EmbeddingService


class VectorStoreService:
    """Service to interact with the PostgreSQL vector store"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        # Connection string not needed anymore - using SQLModel session directly
    
    async def upsert_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """
        Add or update a document in the vector store
        
        Args:
            content: Document content
            metadata: Document metadata
            document_id: Document ID (creates new if not provided)
            
        Returns:
            str: ID of the upserted document
        """
        if metadata is None:
            metadata = {}
            
        # Create document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
            is_update = False  # No document_id provided, so this is an insert
        else:
            # Check if document with this ID already exists
            exists = await self._document_exists(document_id)
            is_update = exists
            
        # Add document ID to metadata
        metadata["document_id"] = document_id
        
        # If updating, delete existing chunks first
        if is_update:
            print(f"Updating existing document: {document_id}")
            await self.delete_document(document_id)
        else:
            print(f"Inserting new document: {document_id}")
        
        # Split document and create embeddings
        chunked_docs = await self.embedding_service.chunk_and_embed_document(
            content=content,
            metadata=metadata
        )
        
        # Save to database using SQLModel only
        async for session in get_session():
            for chunk in chunked_docs:
                chunk_id = chunk["metadata"].get("chunk_id", str(uuid.uuid4()))
                
                # Prepare data for database
                doc = Document(
                    id=uuid.UUID(chunk_id),
                    content=chunk["content"],
                    metadata_=chunk["metadata"],
                    embedding=chunk["embedding"]
                )
                
                session.add(doc)
            
            await session.commit()
        
        return document_id
    
    async def _document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists by ID
        
        Args:
            document_id: ID of the document
            
        Returns:
            bool: True if document exists
        """
        async for session in get_session():
            # Check if any chunks with this parent_document_id exist
            query = select(Document).where(
                Document.metadata_["parent_document_id"].astext == document_id
            )
            
            result = await session.execute(query)
            documents = result.scalars().all()
            
            return len(documents) > 0
    
    async def upsert_documents(
        self, 
        documents: List[DocumentCreate]
    ) -> List[str]:
        """
        Add or update multiple documents in the vector store
        
        Args:
            documents: List of DocumentCreate objects
                      
        Returns:
            List[str]: List of document IDs that were upserted
        """
        document_ids = []
        
        for doc in documents:
            content = doc.content
            metadata = doc.metadata_ if doc.metadata_ else {}
            
            # Upsert each document
            document_id = await self.upsert_document(
                content=content,
                metadata=metadata
            )
            
            document_ids.append(document_id)
            
        return document_ids
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            async for session in get_session():
                # Find all chunks of the document
                query = select(Document).where(
                    Document.metadata_["parent_document_id"].astext == document_id
                )
                
                result = await session.execute(query)
                documents = result.scalars().all()
                
                # Delete all found chunks
                for doc in documents:
                    await session.delete(doc)
                
                await session.commit()
                
                return len(documents) > 0
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[DocumentResponse]:
        """
        Get a document by ID
        
        Args:
            document_id: ID of the document
            
        Returns:
            Optional[DocumentResponse]: Document if found, None otherwise
        """
        try:
            async for session in get_session():
                # Find all chunks of the document
                query = select(Document).where(
                    Document.metadata_["parent_document_id"].astext == document_id
                )
                
                result = await session.execute(query)
                documents = result.scalars().all()
                
                if not documents:
                    return None
                
                # Sort chunks by chunk_index
                chunks = sorted(documents, key=lambda doc: doc.metadata_.get("chunk_index", 0))
                
                # Combine into complete document
                full_content = "".join([doc.content for doc in chunks])
                
                # Get base metadata (excluding chunk-specific fields)
                metadata = chunks[0].metadata_.copy()
                for field in ["chunk_id", "chunk_index", "parent_document_id"]:
                    if field in metadata:
                        del metadata[field]
                
                # Create DocumentResponse
                return DocumentResponse(
                    id=uuid.UUID(document_id),
                    content=full_content,
                    metadata=metadata,
                    created_at=chunks[0].created_at
                )
                
        except Exception as e:
            print(f"Error getting document: {e}")
            return None
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[DocumentResponse]:
        """
        Get a list of documents (without duplicates)
        
        Args:
            limit: Maximum number of documents
            offset: Starting position
            
        Returns:
            List[DocumentResponse]: List of documents
        """
        async for session in get_session():
            # Query unique document IDs
            query = select(Document.metadata_["parent_document_id"].astext.distinct())
            result = await session.execute(query)
            document_ids = result.scalars().all()
            
            # Apply pagination
            paginated_ids = document_ids[offset:offset+limit]
            
            # Get details for each document
            documents = []
            for doc_id in paginated_ids:
                document = await self.get_document(doc_id)
                if document:
                    documents.append(document)
            
            return documents
    
    async def query_similar_documents(
        self, 
        query_text: str, 
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Query documents similar to the input query using SQLModel with vector similarity
        
        Args:
            query_text: Text query
            top_k: Maximum number of results (defaults to config)
            similarity_threshold: Minimum similarity threshold (defaults to config)
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks
        """
        # Use default config if values not provided
        if top_k is None:
            top_k = settings.VECTOR_TOP_K
            
        if similarity_threshold is None:
            similarity_threshold = settings.VECTOR_SIMILARITY_THRESHOLD
        
        # Measure execution time
        start_time = time.time()
        
        # Create embedding for query
        query_embedding = await self.embedding_service.generate_embedding(query_text)
        
        # Search using SQLModel with vector similarity
        async for session in get_session():
            from sqlalchemy import text
            
            # Convert embedding list to pgvector string format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Use pgvector's cosine distance operator with string formatting
            query = text(f"""
                SELECT id, content, metadata, embedding,
                       1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM documents 
                WHERE 1 - (embedding <=> '{embedding_str}'::vector) >= :similarity_threshold
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :top_k
            """)
            
            result = await session.execute(
                query,
                {
                    "similarity_threshold": similarity_threshold,
                    "top_k": top_k
                }
            )
            
            # Format results
            documents = []
            for row in result:
                documents.append({
                    "content": row.content,
                    "metadata": row.metadata,
                    "similarity": float(row.similarity)
                })
        
        # Measure execution time
        elapsed_time = time.time() - start_time
        print(f"Query executed in {elapsed_time:.4f} seconds")
        
        return documents 