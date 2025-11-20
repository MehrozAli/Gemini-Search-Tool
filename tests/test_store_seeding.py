import json

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service
from app.core.config import Settings
from app.main import app
from app.services.file_search_service import FileSearchService
from tests.notion_fixtures import FakeNotionSession


@pytest.fixture
def notion_session():
    return FakeNotionSession()


@pytest.fixture
def notion_api(fake_client, notion_session):
    settings = Settings(NOTION_TOKEN="notion-test-token", NOTION_DATABASE_ID="db-demo")
    service = FileSearchService(client=fake_client, settings=settings, notion_session=notion_session)
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_create_store_without_notion_makes_an_empty_store(api, fake_client):
    response = api.post("/api/stores", json={"display_name": "Board minutes"})

    assert response.status_code == 201
    assert response.json() == {
        "store": {
            "name": "fileSearchStores/store-1",
            "display_name": "Board minutes",
            "create_time": "2026-11-09T12:00:00+00:00",
        },
        "ingest": None,
    }
    assert fake_client.file_search_stores.uploads == []


def test_create_store_uses_default_display_name(api):
    response = api.post("/api/stores", json={})

    assert response.json()["store"]["display_name"] == "my-file-search-store"


def test_create_store_with_notion_ingests_the_graph(notion_api, fake_client):
    response = notion_api.post("/api/stores", json={"display_name": "Product docs"})

    assert response.status_code == 201
    assert response.json()["ingest"]["message"] == "File uploaded and ingested"
    upload = fake_client.file_search_stores.uploads[0]
    graph = json.loads(upload["content"])
    assert {node["id"] for node in graph["nodes"]} >= {"p-reports", "kb_kb-pdf"}
    assert upload["config"]["display_name"].startswith("knowledge_graph_")
    assert upload["config"]["display_name"].endswith(".json")


def test_notion_failure_does_not_leave_an_empty_store(fake_client):
    settings = Settings(NOTION_TOKEN="bad", NOTION_DATABASE_ID="db-demo")
    service = FileSearchService(
        client=fake_client, settings=settings, notion_session=FakeNotionSession(fail_database=True)
    )
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as test_client:
            response = test_client.post("/api/stores", json={"display_name": "Product docs"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json()["detail"].startswith("Notion API error: 401")
    assert fake_client.file_search_stores.items == {}


def test_sync_replaces_the_store_with_fresh_graph(notion_api, fake_client):
    old = fake_client.file_search_stores.create({"display_name": "Product docs"})

    response = notion_api.post(f"/api/stores/{old.name}/sync", json={"display_name": "Product docs"})

    assert response.status_code == 200
    body = response.json()
    assert body["store"]["name"] != old.name
    assert body["ingest"]["message"] == "File synced and ingested"
    assert fake_client.file_search_stores.deleted == [(old.name, {"force": True})]
    assert list(fake_client.file_search_stores.items) == [body["store"]["name"]]


def test_sync_tolerates_an_already_deleted_store(notion_api, fake_client):
    response = notion_api.post("/api/stores/fileSearchStores/gone/sync", json={})

    assert response.status_code == 200
    assert len(fake_client.file_search_stores.items) == 1


def test_sync_without_notion_is_refused_before_deleting(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    response = api.post(f"/api/stores/{store.name}/sync", json={"document_name": ""})

    assert response.status_code == 503
    assert response.json() == {"detail": "NOTION_TOKEN and NOTION_DATABASE_ID must both be set"}
    assert store.name in fake_client.file_search_stores.items


def test_graph_temp_file_is_removed(notion_api, fake_client):
    import os

    notion_api.post("/api/stores", json={})

    assert not os.path.exists(fake_client.file_search_stores.uploads[0]["file"])
