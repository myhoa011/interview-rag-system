from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import Field
from sqlmodel import SQLModel, Column, Field
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
import sqlalchemy as sa


class ChatAuditBase(SQLModel):
    question: str
    response: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB))
    latency_ms: Optional[int] = None
    feedback: Optional[str] = None


class ChatAudit(ChatAuditBase, table=True):
    __tablename__ = "chat_audit"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID, primary_key=True, server_default=sa.text("gen_random_uuid()"))
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(sa.TIMESTAMP, server_default=sa.text("NOW()"))
    )


class ChatAuditCreate(ChatAuditBase):
    """Schema for creating a new chat audit record"""
    pass


class ChatAuditRead(ChatAuditBase):
    """Schema for reading a chat audit record"""
    id: UUID
    created_at: datetime


class ChatAuditResponse(ChatAuditRead):
    """Schema for API response with chat audit data"""
    pass 