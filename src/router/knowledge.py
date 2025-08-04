from typing import List
from fastapi import APIRouter, HTTPException, status

from ..services import VectorStoreService
from ..models import DocumentCreate, DocumentResponse

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])

# Create an instance of VectorStoreService
vector_store_service = VectorStoreService()


@router.post("/update", response_model=List[str])
async def update_knowledge(documents: List[DocumentCreate]):
    """
    Update knowledge base with provided documents
    
    - **documents**: List of documents to add or update in the knowledge base
    - Returns list of document IDs that were processed
    """
    try:
        document_ids = await vector_store_service.upsert_documents(documents)
        return document_ids
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating knowledge base: {str(e)}"
        )


@router.delete("/{document_id}", response_model=bool)
async def delete_knowledge(document_id: str):
    """
    Delete a document from the knowledge base
    
    - **document_id**: ID of the document to delete
    - Returns true if deletion was successful
    """
    try:
        success = await vector_store_service.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        return success
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/", response_model=List[DocumentResponse])
async def list_knowledge(limit: int = 100, offset: int = 0):
    """
    Get list of documents in the knowledge base
    
    - **limit**: Maximum number of documents to return (default: 100)
    - **offset**: Number of documents to skip (default: 0)
    - Returns list of documents with metadata
    """
    try:
        documents = await vector_store_service.list_documents(limit, offset)
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


 