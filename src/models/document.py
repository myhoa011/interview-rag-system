from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import Field
from sqlmodel import SQLModel, Column, Field
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


class DocumentBase(SQLModel):
    content: str
    metadata_: Dict[str, Any] = Field(
    default_factory=dict,
    alias="metadata",
    sa_column=Column(JSONB, name="metadata")
)


class Document(DocumentBase, table=True):
    __tablename__ = "documents"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID, primary_key=True, server_default=sa.text("gen_random_uuid()"))
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(768))
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(sa.TIMESTAMP, server_default=sa.text("NOW()"))
    )


class DocumentCreate(DocumentBase):
    """Schema for creating a new document"""
    pass


class DocumentRead(DocumentBase):
    """Schema for reading a document"""
    id: UUID
    created_at: datetime


class DocumentResponse(DocumentRead):
    """Schema for API response with document data"""
    pass


class DocumentUpdate(SQLModel):
    """Schema for updating a document"""
    content: Optional[str] = None
    metadata_: Optional[Dict[str, Any]] = Field(default=None, alias="metadata")
