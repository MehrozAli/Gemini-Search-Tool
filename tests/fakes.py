"""In-memory stand-in for the parts of google.genai.Client the service uses."""
from __future__ import annotations

import itertools
from datetime import datetime, timezone
from types import SimpleNamespace

from google.genai import errors


class FakeOperation(SimpleNamespace):
    pass


class FakeStores:
    def __init__(self, client: FakeClient):
        self._client = client
        self.items: dict[str, SimpleNamespace] = {}
        self.deleted: list[tuple[str, dict]] = []
        self.uploads: list[dict] = []
        self._ids = itertools.count(1)

    def create(self, config: dict):
        name = f"fileSearchStores/store-{next(self._ids)}"
        store = SimpleNamespace(
            name=name,
            display_name=config.get("display_name"),
            create_time=datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc),
        )
        self.items[name] = store
        return store

    def list(self):
        return list(self.items.values())

    def delete(self, name: str, config: dict | None = None):
        if name not in self.items:
            raise errors.ClientError(404, {"error": {"code": 404, "message": f"{name} not found"}})
        self.deleted.append((name, config or {}))
        del self.items[name]

    def upload_to_file_search_store(self, file, file_search_store_name, config=None):
        with open(file, "rb") as fh:
            content = fh.read()
        self.uploads.append(
            {"file": file, "store": file_search_store_name, "config": config, "content": content}
        )
        return FakeOperation(
            name=f"{file_search_store_name}/upload/operations/op-{len(self.uploads)}",
            done=self._client.upload_done_after == 0,
            metadata=None,
        )


class FakeOperations:
    def __init__(self, client: FakeClient):
        self._client = client
        self.polls = 0

    def get(self, operation):
        self.polls += 1
        return FakeOperation(
            name=operation.name,
            done=self.polls >= self._client.upload_done_after,
            metadata=None,
        )


class FakeModels:
    def __init__(self):
        self.calls: list[dict] = []
        self.response = make_response("A grounded answer.", [("handbook.pdf", None, "chunk-1")])
        self.error: Exception | None = None

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, upload_done_after: int = 1):
        self.upload_done_after = upload_done_after
        self.file_search_stores = FakeStores(self)
        self.operations = FakeOperations(self)
        self.models = FakeModels()


def make_response(text, chunks):
    grounding_chunks = [
        SimpleNamespace(retrieved_context=SimpleNamespace(title=title, uri=uri), chunk=chunk_id)
        for title, uri, chunk_id in chunks
    ]
    metadata = SimpleNamespace(grounding_chunks=grounding_chunks)
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(grounding_metadata=metadata)])
