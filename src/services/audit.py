import time
from typing import Dict, Any, List, Optional
from uuid import UUID
from sqlmodel import select
from datetime import datetime, timedelta

from ..db.database import get_session
from ..models import ChatAudit, ChatAuditResponse


class AuditService:
    """Service to handle audit logging for AI interactions"""
    
    async def create_audit_log(
        self,
        question: str,
        response: str,
        retrieved_docs: List[Dict[str, Any]],
        latency_ms: int,
        chat_id: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> str:
        """
        Create a new audit log entry
        
        Args:
            question: User query
            response: AI response
            retrieved_docs: Retrieved documents used for context
            latency_ms: Response time in milliseconds
            chat_id: Optional chat ID (generates new if not provided)
            feedback: Optional user feedback
            
        Returns:
            str: ID of the created audit log
        """
        # Generate chat ID if not provided
        if not chat_id:
            chat_id = str(uuid.uuid4())
            
        try:
            # Create audit record
            audit = ChatAudit(
                id=UUID(chat_id),
                question=question,
                response=response,
                retrieved_docs=retrieved_docs,
                latency_ms=latency_ms,
                feedback=feedback
            )
            
            # Save to database
            async for session in get_session():
                session.add(audit)
                await session.commit()
                
            return chat_id
            
        except Exception as e:
            print(f"Error creating audit log: {e}")
            # Still return chat_id even if there was an error
            return chat_id
    
    async def get_audit_log(self, chat_id: str) -> Optional[ChatAuditResponse]:
        """
        Get an audit log by ID
        
        Args:
            chat_id: ID of the audit log to retrieve
            
        Returns:
            Optional[ChatAuditResponse]: Audit log if found
        """
        try:
            async for session in get_session():
                # Query audit log
                query = select(ChatAudit).where(ChatAudit.id == UUID(chat_id))
                result = await session.execute(query)
                audit = result.scalars().first()
                
                if not audit:
                    return None
                    
                # Convert to response model
                return ChatAuditResponse(
                    id=audit.id,
                    question=audit.question,
                    response=audit.response,
                    retrieved_docs=audit.retrieved_docs,
                    latency_ms=audit.latency_ms,
                    feedback=audit.feedback,
                    created_at=audit.created_at
                )
                
        except Exception as e:
            print(f"Error retrieving audit log: {e}")
            return None
    
    async def add_feedback(self, chat_id: str, feedback: str) -> bool:
        """
        Add feedback to an existing audit log
        
        Args:
            chat_id: ID of the audit log
            feedback: User feedback
            
        Returns:
            bool: True if successful
        """
        try:
            async for session in get_session():
                # Find audit log
                query = select(ChatAudit).where(ChatAudit.id == UUID(chat_id))
                result = await session.execute(query)
                audit = result.scalars().first()
                
                if not audit:
                    return False
                    
                # Update feedback
                audit.feedback = feedback
                session.add(audit)
                await session.commit()
                
                return True
                
        except Exception as e:
            print(f"Error adding feedback: {e}")
            return False
    
    async def list_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ChatAuditResponse]:
        """
        List audit logs with optional date filtering
        
        Args:
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List[ChatAuditResponse]: List of audit logs
        """
        try:
            async for session in get_session():
                # Build query
                query = select(ChatAudit)
                
                # Apply date filters if provided
                if start_date:
                    query = query.where(ChatAudit.created_at >= start_date)
                if end_date:
                    query = query.where(ChatAudit.created_at <= end_date)
                
                # Apply sorting and pagination
                query = query.order_by(ChatAudit.created_at.desc()).offset(offset).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                audits = result.scalars().all()
                
                # Convert to response models
                return [
                    ChatAuditResponse(
                        id=audit.id,
                        question=audit.question,
                        response=audit.response,
                        retrieved_docs=audit.retrieved_docs,
                        latency_ms=audit.latency_ms,
                        feedback=audit.feedback,
                        created_at=audit.created_at
                    )
                    for audit in audits
                ]
                
        except Exception as e:
            print(f"Error listing audit logs: {e}")
            return []
    
    async def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance metrics for recent interactions
        
        Args:
            days: Number of days to include in metrics
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            async for session in get_session():
                # Calculate date range
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=days)
                
                # Get audit logs in date range
                query = select(ChatAudit).where(
                    ChatAudit.created_at >= start_date,
                    ChatAudit.created_at <= end_date
                )
                result = await session.execute(query)
                audits = result.scalars().all()
                
                # Calculate metrics
                total_count = len(audits)
                if total_count == 0:
                    return {
                        "total_interactions": 0,
                        "avg_latency_ms": 0,
                        "min_latency_ms": 0,
                        "max_latency_ms": 0,
                        "period_days": days
                    }
                
                latencies = [audit.latency_ms for audit in audits]
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies) if latencies else 0
                max_latency = max(latencies) if latencies else 0
                
                return {
                    "total_interactions": total_count,
                    "avg_latency_ms": avg_latency,
                    "min_latency_ms": min_latency,
                    "max_latency_ms": max_latency,
                    "period_days": days
                }
                
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {
                "error": str(e),
                "total_interactions": 0,
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "period_days": days
            } 