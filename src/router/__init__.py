from fastapi import APIRouter

from .knowledge import router as knowledge_router
from .chat import router as chat_router
from .audit import router as audit_router

# Create main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(knowledge_router)
api_router.include_router(chat_router)
api_router.include_router(audit_router)

__all__ = ["api_router"] 