from google.genai import types


def test_root_reports_ok(api):
    assert api.get("/").json() == {"status": "ok"}


def test_list_stores_returns_iso_timestamps(api, fake_client):
    fake_client.file_search_stores.create({"display_name": "Handbook"})

    response = api.get("/api/stores")

    assert response.status_code == 200
    assert response.json() == {
        "stores": [
            {
                "name": "fileSearchStores/store-1",
                "display_name": "Handbook",
                "create_time": "2026-11-09T12:00:00+00:00",
            }
        ]
    }


def test_list_stores_empty(api):
    assert api.get("/api/stores").json() == {"stores": []}


def test_upload_file_waits_for_ingest(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    response = api.post(
        f"/api/stores/{store.name}/files",
        files={"file": ("leave-policy.md", b"# Leave\n25 days", "text/markdown")},
        params={"display_name": "Leave policy"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert body["message"] == "File uploaded and ingested"
    upload = fake_client.file_search_stores.uploads[0]
    assert upload["store"] == store.name
    assert upload["content"] == b"# Leave\n25 days"
    assert upload["config"] == {"display_name": "Leave policy"}
    assert upload["file"].endswith(".md")


def test_upload_file_removes_temp_file(api, fake_client):
    import os

    store = fake_client.file_search_stores.create({"display_name": "Handbook"})
    api.post(f"/api/stores/{store.name}/files", files={"file": ("a.txt", b"x", "text/plain")})

    assert not os.path.exists(fake_client.file_search_stores.uploads[0]["file"])


def test_query_sends_history_and_prompt(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    response = api.post(
        f"/api/stores/{store.name}/query",
        json={
            "prompt": "And for part-time staff?",
            "system_prompt": "Answer briefly.",
            "conversation_history": [
                {"role": "user", "content": "How many leave days?"},
                {"role": "model", "content": "25 days."},
                {"role": "model", "content": "   "},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "A grounded answer.",
        "sources": [{"title": "handbook.pdf", "uri": None, "chunk_id": "chunk-1"}],
    }
    call = fake_client.models.calls[0]
    assert call["model"] == "gemini-test-model"
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "How many leave days?"}]},
        {"role": "model", "parts": [{"text": "25 days."}]},
        {"role": "user", "parts": [{"text": "And for part-time staff?"}]},
    ]
    config = call["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.system_instruction == "Answer briefly."
    assert config.tools[0].file_search.file_search_store_names == [store.name]


def test_query_without_system_prompt_adds_markdown_instruction(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    api.post(f"/api/stores/{store.name}/query", json={"prompt": "Summarise", "model": "other-model"})

    call = fake_client.models.calls[0]
    assert call["model"] == "other-model"
    text = call["contents"][-1]["parts"][0]["text"]
    assert text.startswith("Summarise\n")
    assert "markdown" in text
    assert call["config"].system_instruction is None


def test_query_requires_prompt(api):
    assert api.post("/api/stores/fileSearchStores/x/query", json={}).status_code == 422


def test_delete_store_by_resource_name(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    response = api.delete(f"/api/stores/{store.name}")

    assert response.status_code == 200
    assert response.json()["deleted"] == store.name
    assert fake_client.file_search_stores.deleted == [(store.name, {"force": True})]


def test_delete_store_by_display_name(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    response = api.delete("/api/stores/Handbook")

    assert response.json()["deleted"] == store.name


def test_delete_all_stores(api, fake_client):
    first = fake_client.file_search_stores.create({"display_name": "A"})
    second = fake_client.file_search_stores.create({"display_name": "B"})

    response = api.delete("/api/stores")

    assert response.json() == {"deleted": [first.name, second.name]}
    assert fake_client.file_search_stores.items == {}


def test_health_endpoint_used_by_the_frontend(api):
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_resolves_store_display_name(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    api.post("/api/stores/Handbook/query", json={"prompt": "Hi"})

    config = fake_client.models.calls[0]["config"]
    assert config.tools[0].file_search.file_search_store_names == [store.name]


def test_upload_defaults_document_name_to_original_filename(api, fake_client):
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})

    api.post(f"/api/stores/{store.name}/files", files={"file": ("rota.pdf", b"%PDF", "application/pdf")})

    assert fake_client.file_search_stores.uploads[0]["config"] == {"display_name": "rota.pdf"}


def test_upload_stops_polling_after_timeout(fake_client, settings):
    from app.services.file_search_service import FileSearchService

    fake_client.upload_done_after = 10_000
    settings.upload_poll_interval_seconds = 5
    settings.upload_timeout_seconds = 12
    store = fake_client.file_search_stores.create({"display_name": "Handbook"})
    service = FileSearchService(client=fake_client, settings=settings)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        result = service.upload_file(store.name, tmp.name)

    assert result["done"] is False
    assert fake_client.operations.polls == 3


def test_cors_allows_any_origin_by_default(api):
    response = api.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_origins_parse_from_comma_separated_setting():
    from app.core.config import Settings

    settings = Settings(CORS_ORIGINS="http://localhost:3000, https://search.example.com,")

    assert settings.cors_origin_list == ["http://localhost:3000", "https://search.example.com"]
