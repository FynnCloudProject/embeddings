from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.embedding import embedding_model
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model.load()
    yield


app = FastAPI(
    title="FynnCloud Embeddings",
    description="Multimodal embedding service (text + image) powered by jina-clip-v2.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
