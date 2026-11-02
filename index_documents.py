from pathlib import Path
from typing import Any

from rag_common import DATA_SOURCES, chunk_text, get_chroma_collection, read_document_pages


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def index_documents(
    collection: Any,
    data_directory: Path = DATA_SOURCES,
) -> int:
    if not data_directory.is_dir():
        raise FileNotFoundError(f"Folderul cu documente nu exista: {data_directory}")

    files = sorted(
        path
        for path in data_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"Nu au fost gasite fisiere .txt sau .pdf in {data_directory}"
        )

    filenames = [path.name for path in files]
    if len(filenames) != len(set(filenames)):
        raise ValueError(
            "Fisierele din data_sources trebuie sa aiba nume unice "
            "deoarece source contine numele fisierului."
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for path in files:
        for page_number, page_text in read_document_pages(path):
            for chunk_number, chunk in enumerate(chunk_text(page_text), start=1):
                ids.append(f"{path.name}:{page_number}:{chunk_number}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": path.name,
                        "page": page_number,
                        "chunk_number": chunk_number,
                    }
                )

    if not documents:
        raise ValueError(f"Nu s-a extras text din documentele din {data_directory}.")

    for filename in filenames:
        collection.delete(where={"source": filename})
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def main() -> None:
    collection = get_chroma_collection()
    count = index_documents(collection)
    print(f"Au fost indexate {count} fragmente in colectia '{collection.name}'.")


if __name__ == "__main__":
    main()
