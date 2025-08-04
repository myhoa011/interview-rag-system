from typing import AsyncGenerator
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
