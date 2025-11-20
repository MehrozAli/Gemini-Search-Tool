"""Service layer mirroring the behaviors implemented in SearchTool.py."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from google.genai import types

from app.core.client import get_genai_client
from app.core.config import get_settings


class FileSearchService:
    """Wrap Gemini File Search interactions in a reusable service."""

    def __init__(self):
        self.client = get_genai_client()
        self.settings = get_settings()

    def create_store(self, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new file search store."""

        store = self.client.file_search_stores.create(
            config={"display_name": display_name or self.settings.default_store_display_name}
        )
        return self._store_to_dict(store)

    def list_stores(self) -> List[Dict[str, Any]]:
        """Return all available file search stores."""

        stores = self.client.file_search_stores.list()
        return [self._store_to_dict(store) for store in stores]

    def upload_file(self, store_name: str, file_path: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload a file into a store and wait for ingestion to finish."""

        # Only include display_name in config if provided
        config = {}
        if display_name:
            config["display_name"] = display_name

        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_name,
            config=config if config else None,
        )

        while not operation.done:
            time.sleep(self.settings.upload_poll_interval_seconds)
            operation = self.client.operations.get(operation)

        return self._operation_to_dict(operation)

    def query_store(self, store_name: str, prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Execute a grounded Gemini prompt against a file search store."""

        # Build the content with system prompt if provided
        content = prompt
        if not system_prompt:
            # Use default instruction if no system prompt provided
            content = f"""{prompt}\n(return your answer in markdown as sections and bullet points and also return the images if there are any for the images link in markdown like ![Alt text here](/path/to/image.jpg "Optional Title")
)\nANSWER:\n"""

        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name],
                    )
                )
            ]
        )

        # Add system instruction if system_prompt is provided
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            response = self.client.models.generate_content(
                model=model or self.settings.default_model,
                contents=content,
                config=config,
            )
            return self._response_to_dict(response)
        except Exception as e:
            # Return error information in a structured way
            return {
                "text": f"Error querying store: {str(e)}",
                "sources": []
            }

    def delete_store(self, store_name: str) -> str:
        """Delete a specific file search store (force)."""
        
        self.client.file_search_stores.delete(name=store_name, config={"force": True})
        return store_name

    def delete_all_stores(self) -> List[str]:
        """Delete every file search store (force)."""

        deleted_names: List[str] = []
        for store in self.client.file_search_stores.list():
            self.client.file_search_stores.delete(name=store.name, config={"force": True})
            deleted_names.append(store.name)
        return deleted_names

        
    @staticmethod
    def _store_to_dict(store: Any) -> Dict[str, Any]:
        create_time = getattr(store, "create_time", None)
        # Convert datetime to ISO string if present
        if create_time and hasattr(create_time, "isoformat"):
            create_time = create_time.isoformat()
        return {
            "name": getattr(store, "name", ""),
            "display_name": getattr(store, "display_name", None),
            "create_time": create_time,
        }

    @staticmethod
    def _operation_to_dict(operation: Any) -> Dict[str, Any]:
        return {
            "name": getattr(operation, "name", None),
            "done": bool(getattr(operation, "done", False)),
            "metadata": getattr(operation, "metadata", None),
        }

    @staticmethod
    def _response_to_dict(response: Any) -> Dict[str, Any]:
        # Ensure text is always a string, never None
        text = getattr(response, "text", "") or ""
        sources: List[Dict[str, Optional[str]]] = []

        try:
            candidate = response.candidates[0]
            metadata = getattr(candidate, "grounding_metadata", None)
            chunks = getattr(metadata, "grounding_chunks", []) if metadata else []
            for chunk in chunks:
                retrieved = getattr(chunk, "retrieved_context", None)
                sources.append(
                    {
                        "title": getattr(retrieved, "title", None),
                        "uri": getattr(retrieved, "uri", None),
                        "chunk_id": getattr(chunk, "chunk", None),
                    }
                )
        except (AttributeError, IndexError, TypeError):
            pass

        return {"text": text, "sources": sources}


