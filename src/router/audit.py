from fastapi import APIRouter, HTTPException, status

from ..services import AuditService
from ..models import ChatAuditResponse

router = APIRouter(prefix="/audit", tags=["Audit"])

# Create an instance of AuditService
audit_service = AuditService()


@router.get("/{chat_id}", response_model=ChatAuditResponse)
async def get_audit(chat_id: str):
    """
    Get audit record for a specific chat
    
    - **chat_id**: ID of the chat to retrieve audit for
    - Returns the audit record with all details
    """
    try:
        audit = await audit_service.get_audit_log(chat_id)
        if not audit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit record for chat {chat_id} not found"
            )
        return audit
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving audit record: {str(e)}"
        )




 