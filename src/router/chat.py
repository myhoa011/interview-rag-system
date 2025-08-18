from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ..services import ChatService


class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


router = APIRouter(tags=["Chat"])

# Create an instance of ChatService
chat_service = ChatService()


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with RAG system with streaming response
    
    - **query**: User question
    - Returns a streaming response with AI generated answers
    """
    async def generate():
        async for chunk in chat_service.stream_response(request.query, request.thread_id):
            if chunk["done"]:
                # Just end the stream, no completion message
                break
            else:
                # Only stream actual AI response text, skip stage updates
                if chunk.get("stage") == "generation":
                    yield chunk["chunk"]
                # Skip all other stages (prompt_refine, retrieval, reasoning, compacting)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
 