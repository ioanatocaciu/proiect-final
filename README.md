# Proiect final: RAG cu ChromaDB

- `index_documents.py` citeste toate fisierele `.txt` si `.pdf` din `data_sources`, le imparte in fragmente suprapuse si le indexeaza in ChromaDB.
- `rag_agent.py` valideaza intrebarea si combina cautarea semantica din ChromaDB cu
  BM25 pentru rezultate lexicale, apoi trimite contextul validat agentului OpenAI
  sau OpenRouter pentru raspuns.

Fiecare fragment are metadatele `source` (numele fisierului), `page` (pagina; pentru un fisier text este pagina 1) si `chunk_number`. PDF-urile sunt citite pagina cu pagina folosind pypdf. Fragmentele sunt de cel mult 1000 de caractere, cu 150 de caractere de suprapunere.

serverul ChromaDB intr-un terminal:

```powershell
uv run chroma run --path chroma_data --port 8000
```

Intr-un al doilea terminal, din folderul proiectului, indexeaza documentele:

```powershell
uv run python index_documents.py
```

Apoi agentul:

```powershell
uv run python rag_agent.py
```

Testele automate ruleaza cu:

```powershell
uv run pytest
```
