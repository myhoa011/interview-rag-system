import time
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator, Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from ..core.config import settings
from .vector_store import VectorStoreService
from .audit import AuditService


class ChatState(TypedDict):
    """State schema for the LangGraph workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    expanded_prompts: List[str]
    retrieved_documents: List[Dict[str, Any]]
    reasoning_result: str
    final_response: str
    compacted_memory: Dict[str, Any]
    chat_id: str
    latency_ms: int


class ChatService:
    """LangGraph-based chat service with multi-node reasoning workflow"""
    
    def __init__(self):
        # Initialize services
        self.vector_store_service = VectorStoreService()
        self.audit_service = AuditService()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_TOKENS,
            convert_system_message_to_human=True,
            streaming=True  # Enable streaming
        )
        
        # Create workflow
        self.workflow = self._create_workflow()
        
        # Compile with memory
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("prompt_refine", self._prompt_refine_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("generation", self._generation_node)
        workflow.add_node("compacting", self._compacting_node)
        
        # Add edges
        workflow.add_edge(START, "prompt_refine")
        workflow.add_edge("prompt_refine", "retrieval")
        workflow.add_edge("retrieval", "reasoning")
        workflow.add_edge("reasoning", "generation")
        workflow.add_edge("generation", "compacting")
        workflow.add_edge("compacting", END)
        
        return workflow
    
    async def _prompt_refine_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Node 1: Agent Refine prompt -> expand into multiple refined prompts
        Expand the original query into multiple refined prompts
        """
        original_query = state["original_query"]
        
        system_prompt = """
        You are an expert query expansion agent. Analyze the user question and determine if it needs expansion for better information retrieval.

        For SIMPLE questions (like "What is AI?", "Define machine learning"):
        - Generate 1-2 focused variations with synonyms or related terms
        - Example: "What is AI?" → "artificial intelligence definition", "AI meaning and applications"

        For COMPLEX questions (multiple concepts, specific scenarios):
        - Break down into 2-4 meaningful sub-questions
        - Focus on different aspects or perspectives
        - Example: "How does AI impact healthcare costs?" → "AI healthcare applications", "healthcare cost reduction technology", "medical AI implementation costs"

        Guidelines:
        - Quality over quantity - only generate variations that add meaningful search value
        - Avoid redundant or overly similar prompts
        - Keep each prompt focused and searchable

        Return only the expanded prompts, one per line, without numbering or bullets.
        """
    
        human_prompt = f"Original question: {original_query}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = await self.llm.ainvoke(messages)
        
        # Parse expanded prompts
        expanded_prompts = [
            prompt.strip() 
            for prompt in response.content.split('\n') 
            if prompt.strip()
        ]
        
        # Include original query as well
        expanded_prompts.insert(0, original_query)
        
        return {
            "expanded_prompts": expanded_prompts,
            "messages": [AIMessage(content=f"Generated {len(expanded_prompts)} expanded prompts")]
        }
    
    async def _retrieval_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Node 2: Retrieval for multiple expanded prompts
        Retrieve documents for each expanded prompt
        """
        expanded_prompts = state["expanded_prompts"]
        all_documents = []
        
        for prompt in expanded_prompts:
            # Retrieve documents for each expanded prompt
            docs = await self.vector_store_service.query_similar_documents(
                query_text=prompt,
            top_k=settings.VECTOR_TOP_K,
            similarity_threshold=settings.VECTOR_SIMILARITY_THRESHOLD
        )
        
            # Add prompt context to each document
            for doc in docs:
                doc["source_prompt"] = prompt
            
            all_documents.extend(docs)
        
        # Remove duplicates based on document ID
        unique_docs = {}
        for doc in all_documents:
            doc_id = doc.get("id", doc.get("content", "")[:50])
            if doc_id not in unique_docs or doc["similarity"] > unique_docs[doc_id]["similarity"]:
                unique_docs[doc_id] = doc
        
        retrieved_documents = list(unique_docs.values())
        
        # Check if we have sufficient relevant documents
        high_quality_docs = [doc for doc in retrieved_documents if doc.get("similarity", 0) >= 0.7]
        
        return {
            "retrieved_documents": retrieved_documents,
            "high_quality_docs_count": len(high_quality_docs),
            "messages": [AIMessage(content=f"Retrieved {len(retrieved_documents)} unique documents ({len(high_quality_docs)} high-quality)")]
        }
    
    async def _reasoning_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Node 2.5: Reasoning
        Analyze and reason about the retrieved information
        """
        original_query = state["original_query"]
        expanded_prompts = state["expanded_prompts"]
        retrieved_documents = state["retrieved_documents"]
        
        # Format documents for reasoning
        docs_text = ""
        for i, doc in enumerate(retrieved_documents):
            docs_text += f"Document {i+1} [Similarity: {doc.get('similarity', 0):.2f}] [Source: {doc.get('source_prompt', 'N/A')}]:\n"
            docs_text += f"{doc['content']}\n\n"
        
        system_prompt = """
        You are an expert reasoning agent. Analyze the retrieved documents and provide a structured reasoning about how they relate to the user's question.

        Provide a structured reasoning that:
        1. Identifies key themes and patterns across documents
        2. Highlights the most relevant information for answering the question
        3. Notes any contradictions or gaps in the information
        4. Synthesizes insights from multiple documents
        5. Assesses the confidence level of the available information
        6. CRITICALLY IMPORTANT: Determines if there is sufficient relevant information to answer the question

        If the documents have low similarity scores (< 0.7) or don't contain relevant information to answer the question, clearly state: "INSUFFICIENT_INFORMATION - The available documents do not contain enough relevant information to answer this question."

        Format your reasoning clearly with sections.
        """
        
        human_prompt = f"""
        Original Question: {original_query}

        Expanded Prompts Used:
        {chr(10).join(f"- {prompt}" for prompt in expanded_prompts)}

        Retrieved Documents:
        {docs_text}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = await self.llm.ainvoke(messages)
        
        return {
            "reasoning_result": response.content,
            "messages": [AIMessage(content="Completed reasoning analysis")]
        }
    
    async def _generation_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Node 3-1: Generation
        Generate the final response based on reasoning
        """
        original_query = state["original_query"]
        reasoning_result = state["reasoning_result"]
        
        system_prompt = """
        You are a helpful AI assistant. Based on the reasoning analysis provided, generate a comprehensive and accurate answer to the user's question.

        Generate a response that:
        1. Directly answers the user's question
        2. Is based on the reasoning analysis
        3. Is clear, concise, and well-structured
        4. Acknowledges any limitations or uncertainties
        5. Provides actionable information when possible

        If the reasoning analysis indicates insufficient information, clearly state this limitation.
        """
        
        human_prompt = f"""
        User Question: {original_query}

        Reasoning Analysis:
        {reasoning_result}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        response = await self.llm.ainvoke(messages)
        
        return {
            "final_response": response.content,
            "messages": [AIMessage(content=response.content)]
        }
    
    async def _generation_node_streaming(self, state: ChatState) -> AsyncGenerator[str, None]:
        """
        Streaming version of generation node for real-time response
        """
        original_query = state["original_query"]
        reasoning_result = state["reasoning_result"]
        
        system_prompt = """
        You are a helpful AI assistant. Based on the reasoning analysis provided, generate a comprehensive and accurate answer to the user's question.

        CRITICAL INSTRUCTION: If the reasoning analysis contains "INSUFFICIENT_INFORMATION", you MUST respond with:
        "I don't have information about this topic in my current dataset. Please ask questions related to AI, Machine Learning, Deep Learning, or other topics covered in my knowledge base."

        Otherwise, generate a response that:
        1. Directly answers the user's question
        2. Is based on the reasoning analysis
        3. Is clear, concise, and well-structured
        4. Acknowledges any limitations or uncertainties
        5. Provides actionable information when possible
        6. Only uses information from the provided documents

        Never make up information that is not in the provided documents.
        """
        
        human_prompt = f"""
        User Question: {original_query}

        Reasoning Analysis:
        {reasoning_result}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Stream from LLM
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def _compacting_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Node 3-2: Compacting retrieval information -> save to state memory
        Compact and summarize information for future reference
        """
        original_query = state["original_query"]
        expanded_prompts = state["expanded_prompts"]
        retrieved_documents = state["retrieved_documents"]
        reasoning_result = state["reasoning_result"]
        final_response = state["final_response"]
        
        # Create compact memory representation
        compacted_memory = {
            "query": original_query,
            "key_topics": expanded_prompts,
            "document_count": len(retrieved_documents),
            "key_insights": reasoning_result[:500] + "..." if len(reasoning_result) > 500 else reasoning_result,
            "response_summary": final_response[:200] + "..." if len(final_response) > 200 else final_response,
            "timestamp": time.time(),
            "confidence_indicators": {
                "documents_found": len(retrieved_documents) > 0,
                "reasoning_depth": len(reasoning_result) > 100,
                "response_completeness": len(final_response) > 50
            }
        }
        
        return {
            "compacted_memory": compacted_memory,
            "messages": [AIMessage(content="Information compacted and saved to memory")]
        }
    
    async def generate_response(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response using the LangGraph workflow
        
        Args:
            query: User query
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Dict[str, Any]: Response with audit information
        """
        start_time = time.time()
        chat_id = str(uuid.uuid4())
        
        if thread_id is None:
            thread_id = chat_id
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "chat_id": chat_id,
            "expanded_prompts": [],
            "retrieved_documents": [],
            "reasoning_result": "",
            "final_response": "",
            "compacted_memory": {},
            "latency_ms": 0
        }
        
        # Run workflow
        result = await self.app.ainvoke(initial_state, config)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log interaction
        await self.audit_service.create_audit_log(
            question=query,
            response=result["final_response"],
            retrieved_docs=result["retrieved_documents"],
            latency_ms=latency_ms,
            chat_id=chat_id
        )
        
        return {
            "chat_id": chat_id,
            "response": result["final_response"],
            "latency_ms": latency_ms,
            "reasoning": result["reasoning_result"],
            "expanded_prompts": result["expanded_prompts"],
            "document_count": len(result["retrieved_documents"]),
            "compacted_memory": result["compacted_memory"]
        }
    
    async def stream_response(self, query: str, thread_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response using the LangGraph workflow
        
        Args:
            query: User query
            thread_id: Optional thread ID for conversation continuity
            
        Yields:
            Dict[str, Any]: Streaming response chunks
        """
        start_time = time.time()
        chat_id = str(uuid.uuid4())
        
        if thread_id is None:
            thread_id = chat_id
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "chat_id": chat_id,
            "expanded_prompts": [],
            "retrieved_documents": [],
            "reasoning_result": "",
            "final_response": "",
            "compacted_memory": {},
            "latency_ms": 0
        }
        
        # Execute workflow up to reasoning step
        workflow_state = initial_state
        
        # Run prompt_refine
        prompt_result = await self._prompt_refine_node(workflow_state)
        workflow_state.update(prompt_result)
        
        # Run retrieval  
        retrieval_result = await self._retrieval_node(workflow_state)
        workflow_state.update(retrieval_result)
        
        # Check if we have any relevant documents
        if not workflow_state["retrieved_documents"]:
            # No documents found at all - stream the "don't know" response
            no_info_response = "I don't have information about this topic in my current dataset. Please ask questions related to AI, Machine Learning, Deep Learning, or other topics covered in my knowledge base."
            
            # Stream in chunks instead of character by character
            chunk_size = 10
            for i in range(0, len(no_info_response), chunk_size):
                chunk = no_info_response[i:i + chunk_size]
                yield {
                    "chat_id": chat_id,
                    "chunk": chunk,
                    "done": False,
                    "stage": "generation"
                }
            
            # Log interaction
            await self.audit_service.create_audit_log(
                question=query,
                response=no_info_response,
                retrieved_docs=[],
                latency_ms=int((time.time() - start_time) * 1000),
                chat_id=chat_id
            )
            
            # Send completion signal
            yield {
                "chat_id": chat_id,
                "chunk": "",
                "done": True,
                "latency_ms": int((time.time() - start_time) * 1000),
                "stage": "completed"
            }
            return
        
        # Run reasoning
        reasoning_result = await self._reasoning_node(workflow_state)
        workflow_state.update(reasoning_result)
        
        # Stream generation directly from LLM
        full_response = ""
        async for chunk in self._generation_node_streaming(workflow_state):
            full_response += chunk
            yield {
                "chat_id": chat_id,
                "chunk": chunk,
                "done": False,
                "stage": "generation",
                "reasoning": workflow_state.get("reasoning_result", "")
            }
        
        # Update state with final response
        workflow_state["final_response"] = full_response
        workflow_state["messages"].append(AIMessage(content=full_response))
        
        # Run compacting
        final_result = await self._compacting_node(workflow_state)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log interaction
        await self.audit_service.create_audit_log(
            question=query,
            response=full_response,
            retrieved_docs=workflow_state.get("retrieved_documents", []),
            latency_ms=latency_ms,
            chat_id=chat_id
        )
        
        # Send final done message
        yield {
            "chat_id": chat_id,
            "chunk": "",
            "done": True,
            "latency_ms": latency_ms,
            "stage": "completed"
        } 