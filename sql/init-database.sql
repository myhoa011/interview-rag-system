-- Enable the vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Table: documents — stores embedded documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),  -- Adjust to match your embedding dimension (e.g., OpenAI = 1536)
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table: chat_audit — logs each user question and AI response
CREATE TABLE IF NOT EXISTS chat_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    response TEXT NOT NULL,
    retrieved_docs JSONB,      -- Can store list of document IDs or raw text chunks
    latency_ms INTEGER,        -- Inference time in milliseconds
    feedback TEXT,             -- Optional user or evaluator feedback
    created_at TIMESTAMP DEFAULT NOW()
);