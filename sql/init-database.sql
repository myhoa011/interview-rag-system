-- Enable the vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table: documents — stores embedded documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content, ''))
    ) STORED,
    metadata JSONB,
    embedding vector(768),  -- Adjust to match your embedding dimension (e.g., OpenAI = 1536)
    created_at TIMESTAMP DEFAULT NOW()
);

-- Full-text index
CREATE INDEX IF NOT EXISTS idx_documents_tsv
    ON documents USING GIN (content_tsv);

-- Vector index (cosine)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_cosine
    ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

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