"""Service layer around the Gemini File Search API."""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from google.genai import errors as genai_errors
from google.genai import types

from app import notion_graph
from app.core.client import get_genai_client
from app.core.config import Settings, get_settings

STORE_PREFIX = "fileSearchStores/"

# Appended to the prompt when the caller gives no system prompt, so answers
# render well in the frontend's Markdown view.
DEFAULT_ANSWER_INSTRUCTION = (
    "(return your answer in markdown as sections and bullet points and also return the images "
    "if there are any for the images link in markdown like "
    '![Alt text here](/path/to/image.jpg "Optional Title")\n)'
)


class FileSearchService:
    """Wrap Gemini File Search interactions in a reusable service."""

    def __init__(self, client: Any = None, settings: Settings | None = None, notion_session: Any = None):
        self.client = client if client is not None else get_genai_client()
        self.settings = settings if settings is not None else get_settings()
        self.notion_session = notion_session

    @property
    def notion_configured(self) -> bool:
        return bool(self.settings.notion_token and self.settings.notion_database_id)

    def create_store(self, display_name: str | None = None) -> dict[str, Any]:
        """Create a store, seeded with the Notion knowledge graph when Notion is configured."""
        if not self.notion_configured:
            return {"store": self._create_store(display_name), "ingest": None}
        return self._create_store_from_notion(display_name)

    def sync_graph_document(self, store_name: str, display_name: str | None = None) -> dict[str, Any]:
        """Replace a store with a fresh one holding a newly generated Notion graph."""
        notion_graph.builder_from_settings(self.settings)  # fail fast before deleting anything
        with self._graph_file() as graph_path:
            try:
                self.client.file_search_stores.delete(
                    name=self._resolve_store_resource(store_name), config={"force": True}
                )
            except genai_errors.ClientError as exc:
                if exc.code != 404:
                    raise
            store = self._create_store(display_name)
            ingest = self.upload_file(store["name"], graph_path, os.path.basename(graph_path))
        return {"store": store, "ingest": ingest}

    def list_stores(self) -> list[dict[str, Any]]:
        return [self._store_to_dict(store) for store in self.client.file_search_stores.list()]

    def upload_file(self, store_name: str, file_path: str, display_name: str | None = None) -> dict[str, Any]:
        """Upload a file into a store and wait (up to the configured timeout) for ingestion."""
        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=self._resolve_store_resource(store_name),
            config={"display_name": display_name} if display_name else None,
        )

        interval = self.settings.upload_poll_interval_seconds
        max_polls = math.ceil(self.settings.upload_timeout_seconds / interval)
        for _ in range(max_polls):
            if operation.done:
                break
            time.sleep(interval)
            operation = self.client.operations.get(operation)

        return {"name": getattr(operation, "name", None), "done": bool(getattr(operation, "done", False))}

    def query_store(
        self,
        store_name: str,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run a grounded Gemini prompt against a store, with optional prior turns."""
        contents = [
            {"role": message["role"], "parts": [{"text": message["content"]}]}
            for message in conversation_history or []
            if message["content"].strip()
        ]
        current = prompt if system_prompt else f"{prompt}\n{DEFAULT_ANSWER_INSTRUCTION}\nANSWER:\n"
        contents.append({"role": "user", "parts": [{"text": current}]})

        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[self._resolve_store_resource(store_name)],
                    )
                )
            ],
            system_instruction=system_prompt or None,
        )
        response = self.client.models.generate_content(
            model=model or self.settings.default_model,
            contents=contents,
            config=config,
        )
        return self._response_to_dict(response)

    def delete_store(self, store_name: str) -> str:
        store_resource = self._resolve_store_resource(store_name)
        self.client.file_search_stores.delete(name=store_resource, config={"force": True})
        return store_resource

    def delete_all_stores(self) -> list[str]:
        deleted: list[str] = []
        for store in self.client.file_search_stores.list():
            self.client.file_search_stores.delete(name=store.name, config={"force": True})
            deleted.append(store.name)
        return deleted

    def _create_store(self, display_name: str | None) -> dict[str, Any]:
        store = self.client.file_search_stores.create(
            config={"display_name": display_name or self.settings.default_store_display_name}
        )
        return self._store_to_dict(store)

    def _create_store_from_notion(self, display_name: str | None) -> dict[str, Any]:
        # Build the graph first so a Notion failure doesn't leave an empty store behind.
        with self._graph_file() as graph_path:
            store = self._create_store(display_name)
            ingest = self.upload_file(store["name"], graph_path, os.path.basename(graph_path))
        return {"store": store, "ingest": ingest}

    @contextmanager
    def _graph_file(self) -> Iterator[str]:
        """Write the Notion graph to a temporary JSON file and remove it afterwards."""
        fd, path = tempfile.mkstemp(prefix="knowledge_graph_", suffix=".json")
        os.close(fd)
        try:
            notion_graph.build_graph_file(path, self.settings, session=self.notion_session)
            yield path
        finally:
            os.remove(path)

    @staticmethod
    def _store_to_dict(store: Any) -> dict[str, Any]:
        create_time = getattr(store, "create_time", None)
        if create_time is not None and hasattr(create_time, "isoformat"):
            create_time = create_time.isoformat()
        return {
            "name": getattr(store, "name", ""),
            "display_name": getattr(store, "display_name", None),
            "create_time": create_time,
        }

    @staticmethod
    def _response_to_dict(response: Any) -> dict[str, Any]:
        sources: list[dict[str, str | None]] = []
        candidates = getattr(response, "candidates", None) or []
        metadata = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            retrieved = getattr(chunk, "retrieved_context", None)
            sources.append(
                {
                    "title": getattr(retrieved, "title", None),
                    "uri": getattr(retrieved, "uri", None),
                    "chunk_id": getattr(chunk, "chunk", None),
                }
            )
        return {"text": getattr(response, "text", None) or "", "sources": sources}

    def _resolve_store_resource(self, store_name: str) -> str:
        """Accept a full resource name, a store display name, or a bare store id."""
        if store_name.startswith(STORE_PREFIX):
            return store_name
        for store in self.client.file_search_stores.list():
            if getattr(store, "display_name", None) == store_name:
                return store.name
        return f"{STORE_PREFIX}{store_name}"
