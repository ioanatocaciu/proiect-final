# Proiect final: RAG cu ChromaDB

Proiectul contine doua programe:

- `index_documents.py` citeste toate fisierele `.txt` si `.pdf` din `data_sources`, le imparte in fragmente suprapuse si le indexeaza in ChromaDB.
- `rag_agent.py` valideaza intrebarea, cauta 10 fragmente relevante prin tool-ul `rag_search` si trimite contextul validat agentului OpenAI pentru raspuns.

Fiecare fragment are metadatele `source` (numele fisierului), `page` (pagina; pentru un fisier text este pagina 1) si `chunk_number`. PDF-urile sunt citite pagina cu pagina folosind pypdf. Fragmentele sunt de cel mult 1000 de caractere, cu 150 de caractere de suprapunere.

## Instalare si rulare

Necesita Python 3.10+ si `uv`.

```powershell
uv sync
Copy-Item .env.example .env
```

Adauga cheia OpenAI in `.env` ca `OPENAI_API_KEY`. Poti schimba modelul prin `OPENAI_MODEL`.

Porneste serverul ChromaDB intr-un terminal:

```powershell
uv run chroma run --path chroma_data --port 8000
```

Intr-un al doilea terminal, din folderul proiectului, indexeaza documentele:

```powershell
uv run python index_documents.py
```

Apoi porneste agentul:

```powershell
uv run python rag_agent.py
```

Exemple de intrebari pentru documentele incluse:

- „Ce surse de energie regenerabila sunt mentionate?”
- „Cum ar trebui pregatite recipientele pentru reciclare?”
- „Ce obiceiuri pot sustine o rutina de somn?”

Poti adauga propriile fisiere `.txt` si `.pdf` in `data_sources` si relansa indexarea. La reindexare, fragmentele anterioare ale fiecarui fisier sunt inlocuite.

Testele automate ruleaza cu:

```powershell
uv run pytest
```
