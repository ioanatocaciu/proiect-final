import pytest
from pydantic import ValidationError

from index_documents import index_documents
from rag_agent import RAGAssistant, RAGSearchTool
from rag_common import RAGSearchInput, RAGSearchOutput, UserQuestion, chunk_text


class FakeCollection:
    name = "test_collection"

    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.deleted_sources = []
        self.upserted = None
        self.query_arguments = None

    def delete(self, where):
        self.deleted_sources.append(where["source"])

    def get(self, include):
        assert include == ["documents", "metadatas"]
        return {
            "ids": ["sample.txt:1:2"],
            "documents": ["Relevant document text"],
            "metadatas": [
                {"source": "sample.txt", "page": 1, "chunk_number": 2}
            ],
        }

    def upsert(self, ids, documents, metadatas):
        self.upserted = (ids, documents, metadatas)

    def query(self, **kwargs):
        self.query_arguments = kwargs
        return {
            "ids": [["sample.txt:1:2"]],
            "documents": [["Relevant document text"]],
            "metadatas": [[
                {"source": "sample.txt", "page": 1, "chunk_number": 2}
            ]],
            "distances": [[0.12]],
        }


def test_chunk_text_overlaps_and_stops_at_end():
    assert chunk_text("abcdefgh", chunk_size=5, overlap=2) == ["abcde", "defgh"]


def test_question_schemas_reject_blank_input():
    with pytest.raises(ValidationError):
        UserQuestion(question="   ")
    with pytest.raises(ValidationError):
        RAGSearchInput(question="")


def test_rag_output_rejects_empty_context():
    with pytest.raises(ValidationError):
        RAGSearchOutput(context="")


def test_indexer_upserts_chunks_with_source_page_and_chunk_number(tmp_path):
    (tmp_path / "sample.txt").write_text("First page of text.", encoding="utf-8")
    collection = FakeCollection()

    count = index_documents(collection, tmp_path)

    assert count == 1
    assert collection.deleted_sources == ["sample.txt"]
    ids, documents, metadatas = collection.upserted
    assert ids == ["sample.txt:1:1"]
    assert documents == ["First page of text."]
    assert metadatas == [
        {"source": "sample.txt", "page": 1, "chunk_number": 1}
    ]


def test_rag_tool_queries_ten_results_and_returns_validated_context():
    collection = FakeCollection()
    tool = RAGSearchTool(collection)

    output = tool.run(RAGSearchInput(question="What is in the document?"))

    assert collection.query_arguments["n_results"] == 10
    assert output.context == (
        "[Sursa: sample.txt; pagina: 1; fragment: 2; distanta: 0.1200]\n"
        "Relevant document text"
    )


def test_rag_tool_combines_semantic_and_bm25_results():
    class HybridCollection(FakeCollection):
        def get(self, include):
            return {
                "ids": [
                    "semantic.txt:1:1",
                    "keyword.txt:1:1",
                    "unrelated.txt:1:1",
                ],
                "documents": [
                    "A semantically related passage",
                    "The exact keyword appears here",
                    "Unrelated content",
                ],
                "metadatas": [
                    {"source": "semantic.txt", "page": 1, "chunk_number": 1},
                    {"source": "keyword.txt", "page": 1, "chunk_number": 1},
                    {"source": "unrelated.txt", "page": 1, "chunk_number": 1},
                ],
            }

        def query(self, **kwargs):
            self.query_arguments = kwargs
            return {
                "ids": [["semantic.txt:1:1"]],
                "documents": [["A semantically related passage"]],
                "metadatas": [[
                    {"source": "semantic.txt", "page": 1, "chunk_number": 1}
                ]],
                "distances": [[0.1]],
            }

    output = RAGSearchTool(HybridCollection()).run(
        RAGSearchInput(question="exact keyword")
    )

    assert "keyword.txt" in output.context
    assert "semantic.txt" in output.context
    assert "unrelated.txt" not in output.context


def test_assistant_uses_openrouter_configuration(monkeypatch):
    import rag_agent

    client_options = {}

    def fake_openai(**kwargs):
        client_options.update(kwargs)
        return object()

    monkeypatch.setattr(rag_agent, "OpenAI", fake_openai)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_ROUTER_MODEL_NAME", "test/router-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assistant = RAGAssistant(RAGSearchTool(FakeCollection()))

    assert client_options == {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
    }
    assert assistant.model == "test/router-model"


def test_agent_passes_rag_tool_context_to_final_llm_answer():
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                class ToolCall:
                    id = "tool-1"
                    function = type(
                        "Function",
                        (),
                        {
                            "name": "rag_search",
                            "arguments": '{"question":"What is in the document?"}',
                        },
                    )()

                message = type(
                    "Message",
                    (),
                    {"content": None, "tool_calls": [ToolCall()]},
                )()
                return type(
                    "Response",
                    (),
                    {"choices": [type("Choice", (), {"message": message})()]},
                )()

            message = type(
                "Message",
                (),
                {"content": "The document says this. (sample.txt, page 1)", "tool_calls": None},
            )()
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": message})()]},
            )()

    completions = FakeCompletions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()

    class FakeRAGTool:
        def run(self, tool_input):
            assert tool_input.question == "What is in the document?"
            return RAGSearchOutput(context="Evidence from sample.txt page 1")

    assistant = RAGAssistant(FakeRAGTool(), client=client, model="test-model")
    answer = assistant.answer(" What is in the document? ")

    assert answer.startswith("The document says")
    assert completions.calls[1]["messages"][-1]["content"] == (
        "Evidence from sample.txt page 1"
    )
