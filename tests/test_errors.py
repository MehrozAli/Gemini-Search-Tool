from google.genai import errors

from app.core import client as client_module
from app.core.config import Settings


def _gemini_error(code, message):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


def test_query_failure_is_an_http_error_not_an_answer(api, fake_client):
    fake_client.models.error = _gemini_error(500, "backend exploded")

    response = api.post("/api/stores/fileSearchStores/store-1/query", json={"prompt": "Hi"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Gemini API error: backend exploded"}


def test_gemini_not_found_keeps_its_status(api, fake_client):
    fake_client.models.error = _gemini_error(404, "Store not found")

    response = api.post("/api/stores/fileSearchStores/missing/query", json={"prompt": "Hi"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Gemini API error: Store not found"


def test_rate_limit_is_passed_through(api, fake_client):
    fake_client.models.error = _gemini_error(429, "Quota exceeded")

    response = api.post("/api/stores/fileSearchStores/store-1/query", json={"prompt": "Hi"})

    assert response.status_code == 429


def test_unknown_history_role_is_rejected(api):
    response = api.post(
        "/api/stores/fileSearchStores/store-1/query",
        json={"prompt": "Hi", "conversation_history": [{"role": "assistant", "content": "x"}]},
    )

    assert response.status_code == 422


def test_missing_api_key_returns_503(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(client_module, "get_settings", lambda: Settings(GOOGLE_API_KEY=""))
    client_module.get_genai_client.cache_clear()
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/api/stores")
    finally:
        client_module.get_genai_client.cache_clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "GOOGLE_API_KEY is not set"}
