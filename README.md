

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
