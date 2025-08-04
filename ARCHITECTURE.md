# Architecture Decisions Document

## Overview

This document explains the architectural decisions made during the development of the RAG (Retrieval-Augmented Generation) knowledge base system.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                     FastAPI Application                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Router    │  │  Services   │  │   Models    │              │
│  │    Layer    │  │   Layer     │  │    Layer    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                    LangGraph Workflow                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Prompt    │→ │ Retrieval   │→ │  Reasoning  │              │
│  │  Refine     │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
|                                           ↓                     │
|                  ┌─────────────┐  ┌─────────────┐               │
|                  │ Compacting  │← │ Generation  │               │
|                  │             │  │             │               │
|                  └─────────────┘  └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                    Data & External APIs                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ PostgreSQL  │  │   Gemini    │  │   Google    │              │
│  │ + pgvector  │  │     API     │  │ Embeddings  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Technology Stack

### Database: PostgreSQL + pgvector
Uses PostgreSQL with pgvector extension for unified storage of both structured data and vector embeddings. This provides ACID compliance for audit logs while supporting efficient similarity search through pgvector's indexing capabilities.

### LLM: Google Gemini
Integrates with Google Gemini through LangChain for text generation and reasoning. Provides good balance of performance and cost with native streaming support for real-time responses.

### Workflow Engine: LangGraph
Implements multi-step reasoning pipeline using LangGraph's state management. Enables structured workflow with clear separation of concerns and built-in observability for debugging.

### Embeddings: Google Embeddings
Uses Google's embedding model for consistent vector representations. Maintains provider consistency with the LLM and integrates seamlessly through LangChain.

### API Framework: FastAPI
Built on FastAPI for modern async/await support, automatic type validation through Pydantic, and built-in OpenAPI documentation generation.

## 🔄 Workflow

#### Node 1: Prompt Refinement (`_prompt_refine_node`)
**Purpose**: Intelligently expands user queries based on complexity
- **Input**: Original user question
- **Process**: 
  - Analyzes question complexity
  - Simple questions: 1-2 focused variations with synonyms
  - Complex questions: 2-4 meaningful sub-questions
- **Output**: Adaptive list of expanded prompts (1-4 prompts)
- **Examples**: 
  - Simple: "What is AI?" → ["artificial intelligence definition", "AI meaning and applications"]
  - Complex: "AI healthcare costs?" → ["AI healthcare applications", "healthcare cost reduction technology", "medical AI implementation costs"]

#### Node 2: Document Retrieval (`_retrieval_node`)
**Purpose**: Searches for relevant documents using vector similarity
- **Input**: Expanded prompts from Node 1
- **Process**: 
  - Queries vector database for each expanded prompt
  - Applies similarity threshold (default: 0.7)
  - Removes duplicates and ranks by relevance
- **Output**: List of relevant document chunks with metadata
- **Performance**: ~0.5 seconds for vector search

#### Node 3: Reasoning Analysis (`_reasoning_node`)
**Purpose**: Analyzes retrieved documents and assesses information quality
- **Input**: Original query + retrieved documents
- **Process**:
  - Examines document relevance and quality
  - Identifies patterns and contradictions
  - Assesses confidence level of available information
  - Determines if sufficient information exists to answer the query
- **Output**: Structured reasoning analysis
- **Key Feature**: Can flag "INSUFFICIENT_INFORMATION" for out-of-scope queries

#### Node 4: Response Generation (`_generation_node_streaming`)
**Purpose**: Generates the final response based on reasoning analysis
- **Input**: Original query + reasoning analysis
- **Process**:
  - Streams response token-by-token from Gemini
  - Returns "don't know" message if insufficient information
  - Synthesizes information from multiple documents
- **Output**: Streaming text response
- **Performance**: 3-5 seconds for generation

#### Node 5: Memory Compacting (`_compacting_node`)
**Purpose**: Extracts and stores key insights for future reference
- **Input**: Full conversation context
- **Process**: Identifies important concepts and relationships
- **Output**: Structured memory data (stored but not currently used)
- **Note**: Placeholder for future conversation memory features

### Workflow Flow
```
User Query → Prompt Refine → Retrieval → Reasoning → Generation → Compacting
     ↓             ↓            ↓          ↓           ↓           ↓
  "What is AI?" → 1-2 prompts → docs → Analysis → Response → Memory
  "AI costs?"   → 2-4 prompts → docs → Analysis → Response → Memory
```

### Streaming Implementation

**Decision**: Stream only the final generation step

**Rationale**:
- **User Experience**: Provides immediate feedback during longest step
- **Simplicity**: Easier to implement than full pipeline streaming
- **Debugging**: Intermediate steps can be logged without affecting UX


## ⚠️ Known Limitations & Future Improvements

### Performance Limitations

**Current Issues**:
- 5-10 second response time vs 500ms target
- Sequential processing in workflow
- External API dependencies

**Potential Improvements**:
- Use local LLM (Ollama, vLLM)
- Simplify workflow (skip reasoning step)
- Cache embeddings
- Use faster embedding models
- Implement response caching
- Parallel processing where possible
- Async workflow execution

### Scalability Considerations

**Current Limitations**:
- Single instance deployment
- No load balancing
- No caching layer

**Future Enhancements**:
- Horizontal scaling with load balancer
- Redis caching for embeddings
- Database read replicas
- CDN for static content

### Error Handling

**Current State**:
- Basic error handling for common cases
- Graceful degradation for unknown topics

**Improvements Needed**:
- Comprehensive error recovery
- Circuit breaker patterns
- Better user error messages
- Monitoring and alerting


## 📝 Lessons Learned

### What Worked Well

1. **LangGraph**: Provided clear workflow structure and debugging capabilities
2. **PostgreSQL + pgvector**: Simplified data architecture significantly
3. **FastAPI**: Excellent developer experience and automatic documentation
4. **Docker Compose**: Made development environment setup trivial

### What Could Be Improved

1. **Performance**: External API dependencies created significant latency
2. **Streaming**: Could implement more granular streaming for better UX
3. **Error Handling**: More robust error recovery and user feedback
4. **Testing**: More comprehensive test coverage for edge cases

### Key Takeaways

1. **Simplicity vs Features**: Chose feature completeness over performance optimization
2. **External Dependencies**: Convenient but create latency and reliability concerns
3. **Workflow Complexity**: Multi-step processing improves quality but hurts performance
4. **Documentation**: Comprehensive documentation is crucial for complex systems