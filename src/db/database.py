from typing import AsyncGenerator

from pgvector.asyncpg import register_vector
from sqlalchemy import text
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic_core import MultiHostUrl

from ..core.config import settings

POSTGRES_CONNECTION_STRING = str(
    MultiHostUrl.build(
        scheme="postgresql+asyncpg",
        host=settings.POSTGRES_HOST,
        username=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        port=settings.POSTGRES_PORT,
        path=settings.POSTGRES_DB,
    )
)

# Create async engine
engine = create_async_engine(
    POSTGRES_CONNECTION_STRING,
    echo=True,
    future=True,
)

# Create async session maker
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session() as session:
        yield session


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        dbapi_connection = await conn.get_raw_connection()

        await register_vector(dbapi_connection.driver_connection)

async def validate_embedding_dimension():
    """
    Validate EMBEDDING_DIMENSION against PostgreSQL float[] column length.
    Raises RuntimeError if mismatch.
    """
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT attname, atttypmod
                FROM pg_attribute
                JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
                WHERE relname='documents' AND attname='embedding';
            """)
        )
        row = result.first()
        if row is None:
            print("Warning: documents table empty, cannot validate embedding dimension.")
            return

        db_dim = row[1] - 4
        if db_dim != settings.EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"EMBEDDING_DIMENSION mismatch: config={settings.EMBEDDING_DIMENSION}, DB column={db_dim}"
            )
        print(f"EMBEDDING_DIMENSION check passed: {db_dim} dimensions.")