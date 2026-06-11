from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    # roda quando a API sobe — cria tabelas se não existirem
    create_tables()
    yield
    # roda quando a API desce — cleanup se necessário


app = FastAPI(
    title="Music Genre Classifier",
    description="API que classifica gênero musical a partir de arquivos de áudio",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)