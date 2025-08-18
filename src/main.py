from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.database import init_db, validate_embedding_dimension
from .router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    print("Initializing database...")
    await init_db()
    print("Database initialized successfully!")
    await validate_embedding_dimension()
    yield


app = FastAPI(
    title="RAG API",
    description="FastAPI backend for RAG system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers from the router module
app.include_router(api_router)


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the RAG API!",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}