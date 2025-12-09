"""Service layer mirroring the behaviors implemented in SearchTool.py."""
from __future__ import annotations

import os
import tempfile
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
        """
        Create a new file search store and ingest the freshly generated Notion graph JSON.
        Returns dict with store info and ingest result.
        """

        store = self.client.file_search_stores.create(
            config={"display_name": display_name or self.settings.default_store_display_name}
        )
        store_dict = self._store_to_dict(store)

        ingest_result: Optional[Dict[str, Any]] = None
        temp_path: Optional[str] = None

        try:
            # Generate the JSON file from Notion
            temp_path = self.generate_graph_json()

            # Upload to the newly created store
            ingest_result = self.upload_file(
                store_name=store_dict["name"],
                file_path=temp_path,
                display_name=os.path.basename(temp_path),
            )
        finally:
            # Always clean up the local file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        return {"store": store_dict, "ingest": ingest_result}

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

        store_resource = self._resolve_store_resource(store_name)

        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=store_resource,
            config=config if config else None,
        )

        while not operation.done:
            time.sleep(self.settings.upload_poll_interval_seconds)
            operation = self.client.operations.get(operation)

        return self._operation_to_dict(operation)

    def generate_graph_json(self, output_path: Optional[str] = None) -> str:
        """Generate the knowledge graph JSON using the GraphRag builder (extraction only)."""

        # Defer import to avoid loading heavy deps at app startup
        from GraphRag import NotionKnowledgeGraphBuilder, DATABASE_ID

        temp_path = output_path or os.path.join(
            tempfile.gettempdir(), f"knowledge_graph_{int(time.time())}.json"
        )

        builder = NotionKnowledgeGraphBuilder(DATABASE_ID)
        pages = builder.fetch_database_pages()
        builder.build_graph(pages)
        builder.save_graph(temp_path)

        return temp_path

    def sync_graph_document(
        self, store_name: str, display_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recreate the store (force delete) and ingest freshly generated Notion graph JSON.
        Returns dict with store info and ingest result.
        """

        temp_path: Optional[str] = None
        try:
            # 1) Regenerate JSON from Notion
            temp_path = self.generate_graph_json()

            # 2) Delete the existing store (force) to remove all documents
            store_resource = self._resolve_store_resource(store_name)
            try:
                self.client.file_search_stores.delete(name=store_resource, config={"force": True})
            except Exception:
                # Ignore delete failures; proceed to create
                pass

            # 3) Recreate the store with same display name (if provided)
            new_store = self.client.file_search_stores.create(
                config={"display_name": display_name or self.settings.default_store_display_name}
            )
            store_dict = self._store_to_dict(new_store)

            # 4) Upload regenerated file
            ingest_result = self.upload_file(
                store_name=store_dict["name"],
                file_path=temp_path,
                display_name=os.path.basename(temp_path),
            )
            return {"store": store_dict, "ingest": ingest_result}
        finally:
            # 5) Always remove local artifact
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

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
        
        store_resource = self._resolve_store_resource(store_name)
        self.client.file_search_stores.delete(name=store_resource, config={"force": True})
        return store_resource

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

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _normalize_store_name(store_name: str) -> str:
        """Ensure store resource name has the expected prefix."""
        if store_name.startswith("fileSearchStores/"):
            return store_name
        return f"fileSearchStores/{store_name}"

    def _resolve_store_resource(self, store_name: str) -> str:
        """
        Resolve to a valid store resource name.
        - If already a resource, return it.
        - Else try to match by display_name from list().
        - Else fall back to normalized input.
        """
        if store_name.startswith("fileSearchStores/"):
            return store_name

        # Try to find by display_name
        try:
            for store in self.client.file_search_stores.list():
                if getattr(store, "display_name", None) == store_name:
                    return getattr(store, "name", store_name)
        except Exception:
            pass

        return self._normalize_store_name(store_name)

    def _normalize_document_name(self, store_name: str, document_name: str) -> str:
        """
        Ensure document resource name is fully qualified.
        If caller passes just a doc id or filename, stitch it with the store.
        """
        if "/documents/" in document_name:
            return document_name
        doc_id = document_name.split("/")[-1]
        return f"{self._normalize_store_name(store_name)}/documents/{doc_id}"

    def _delete_document_force(self, document_name: str) -> None:
        """Delete a document with force, ignoring missing-document errors."""
        try:
            self.client.file_search_stores.documents.delete(
                name=document_name,
                config={"force": True},
            )
        except Exception:
            # Ignore delete failures (e.g., not found) to keep sync moving
            pass


