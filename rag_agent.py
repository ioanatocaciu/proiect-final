import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from rank_bm25 import BM25Okapi

from rag_common import (
    RESULT_COUNT,
    RAGSearchInput,
    RAGSearchOutput,
    UserQuestion,
    get_chroma_collection,
)


TOOL_NAME = "rag_search"
RAG_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Cauta informatii relevante din documentele indexate si returneaza "
            "contextul cu numele sursei, pagina si numarul fragmentului."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Intrebarea utilizatorului",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


class RAGSearchTool:
    def __init__(self, collection: Any) -> None:
        self.collection = collection
        corpus = collection.get(include=["documents", "metadatas"])
        self.corpus_ids = corpus.get("ids") or []
        self.corpus_documents = corpus.get("documents") or []
        self.corpus_metadatas = corpus.get("metadatas") or []
        tokenized_documents = [
            re.findall(r"\b\w+\b", document.casefold())
            for document in self.corpus_documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents) if tokenized_documents else None

    def run(self, tool_input: RAGSearchInput) -> RAGSearchOutput:
        semantic_results = self.collection.query(
            query_texts=[tool_input.question],
            n_results=RESULT_COUNT,
            include=["documents", "metadatas", "distances"],
        )
        semantic_ids = (semantic_results.get("ids") or [[]])[0]
        semantic_documents = (semantic_results.get("documents") or [[]])[0]
        semantic_metadatas = (semantic_results.get("metadatas") or [[]])[0]
        semantic_distances = (semantic_results.get("distances") or [[]])[0]

        ranked_results: dict[str, dict[str, Any]] = {}

        for rank, (document_id, document, metadata, distance) in enumerate(
            zip(
                semantic_ids,
                semantic_documents,
                semantic_metadatas,
                semantic_distances,
            ),
            start=1,
        ):
            result = ranked_results.setdefault(
                document_id,
                {
                    "document": document,
                    "metadata": metadata or {},
                    "distance": distance,
                    "score": 0.0,
                },
            )
            result["score"] += 1 / rank
            result["distance"] = distance

        if self.bm25 is not None:
            scores = self.bm25.get_scores(
                re.findall(r"\b\w+\b", tool_input.question.casefold())
            )
            lexical_ranks = sorted(
                range(len(scores)), key=lambda index: (-scores[index], index)
            )
            rank = 0
            for index in lexical_ranks:
                if scores[index] <= 0 or rank >= RESULT_COUNT:
                    break
                document_id = self.corpus_ids[index]
                result = ranked_results.setdefault(
                    document_id,
                    {
                        "document": self.corpus_documents[index],
                        "metadata": self.corpus_metadatas[index] or {},
                        "distance": None,
                        "score": 0.0,
                    },
                )
                result["score"] += 1 / (rank + 1)
                rank += 1

        context_chunks = []
        ordered_results = sorted(
            ranked_results.values(), key=lambda result: result["score"], reverse=True
        )[:RESULT_COUNT]
        for result in ordered_results:
            metadata = result["metadata"]
            distance = result["distance"]
            distance_text = (
                f"{distance:.4f}" if distance is not None else "indisponibila"
            )
            context_chunks.append(
                f"[Sursa: {metadata.get('source', 'necunoscuta')}; "
                f"pagina: {metadata.get('page', 'necunoscuta')}; "
                f"fragment: {metadata.get('chunk_number', 'necunoscut')}; "
                f"distanta: {distance_text}]\n{result['document']}"
            )

        context = "\n\n".join(context_chunks)
        if not context:
            context = "Nu au fost gasite rezultate relevante in documentele indexate."
        return RAGSearchOutput(context=context)


class RAGAssistant:
    def __init__(
        self,
        rag_tool: RAGSearchTool,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        router_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        api_key = router_api_key or os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            raise RuntimeError(
                "Seteaza OPEN_ROUTER_API_KEY sau OPENAI_API_KEY in mediul de lucru "
                "sau in fisierul .env."
            )
        if client is not None:
            self.client = client
        elif router_api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=router_api_key,
            )
        else:
            self.client = OpenAI(api_key=api_key)
        default_model = (
            os.getenv("OPEN_ROUTER_MODEL_NAME", "openrouter/free")
            if router_api_key
            else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
        self.model = model or default_model
        self.rag_tool = rag_tool

    def answer(self, user_input: str) -> str:
        question = UserQuestion(question=user_input).question
        messages = [
            {
                "role": "system",
                "content": (
                    "Raspunde in limba intrebarii si bazeaza-te pe contextul "
                    "returnat de tool-ul rag_search. Daca documentele nu contin "
                    "raspunsul, spune clar acest lucru. Include sursa si pagina "
                    "pentru informatiile folosite."
                ),
            },
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[RAG_SEARCH_TOOL],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        if len(tool_calls) != 1 or tool_calls[0].function.name != TOOL_NAME:
            raise RuntimeError("Agentul nu a apelat corect tool-ul rag_search.")

        tool_call = tool_calls[0]
        try:
            tool_input = RAGSearchInput.model_validate_json(tool_call.function.arguments)
        except ValidationError as error:
            raise ValueError(f"Input invalid pentru tool-ul RAG: {error}") from error
        tool_output = self.rag_tool.run(tool_input)
        validated_output = RAGSearchOutput.model_validate(tool_output)

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": validated_output.context,
                },
            ]
        )
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        answer = (final_response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Agentul LLM a returnat un raspuns gol.")
        return answer


def main() -> None:
    assistant = RAGAssistant(RAGSearchTool(get_chroma_collection()))
    print("Agentul RAG este pregatit. Scrie 'iesire' pentru a inchide.")
    while True:
        user_input = input("\nIntrebarea ta: ")
        if user_input.strip().casefold() in {"iesire", "exit", "quit"}:
            break
        try:
            print(f"\n{assistant.answer(user_input)}")
        except ValidationError as error:
            print(f"Intrebare invalida: {error}")
        except (json.JSONDecodeError, ValueError) as error:
            print(f"Eroare la validarea cautarii: {error}")


if __name__ == "__main__":
    main()
