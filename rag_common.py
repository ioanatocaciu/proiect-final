import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SOURCES = PROJECT_ROOT / "data_sources"
COLLECTION_NAME = "proiect_final_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RESULT_COUNT = 10


class UserQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, description="Intrebarea utilizatorului")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("Intrebarea nu poate fi goala.")
        return value


class RAGSearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, description="Intrebarea pentru cautarea RAG")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("Intrebarea nu poate fi goala.")
        return value


class RAGSearchOutput(BaseModel):
    context: str = Field(min_length=1, description="Contextul extras din documente")


def get_chroma_collection() -> Any:
    import chromadb

    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size trebuie sa fie pozitiv, iar overlap mai mic decat chunk_size.")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size].strip())
        if start + chunk_size >= len(text):
            break
    return [chunk for chunk in chunks if chunk]


def read_document_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix.casefold() == ".pdf":
        from pypdf import PdfReader

        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(PdfReader(path).pages, start=1)
        ]
    if path.suffix.casefold() == ".txt":
        return [(1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Format de document nesuportat: {path.name}")
