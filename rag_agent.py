import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

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

    def run(self, tool_input: RAGSearchInput) -> RAGSearchOutput:
        results = self.collection.query(
            query_texts=[tool_input.question],
            n_results=RESULT_COUNT,
            include=["documents", "metadatas", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        context_chunks = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            context_chunks.append(
                f"[Sursa: {metadata.get('source', 'necunoscuta')}; "
                f"pagina: {metadata.get('page', 'necunoscuta')}; "
                f"fragment: {metadata.get('chunk_number', 'necunoscut')}; "
                f"distanta: {distance:.4f}]\n{document}"
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
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            raise RuntimeError(
                "Seteaza OPENAI_API_KEY in mediul de lucru sau in fisierul .env."
            )
        self.client = client or OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
