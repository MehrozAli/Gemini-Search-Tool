import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service
from app.core.config import Settings
from app.main import app
from app.services.file_search_service import FileSearchService
from tests.fakes import FakeClient


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.services.file_search_service.time.sleep", lambda _seconds: None)


@pytest.fixture
def settings():
    return Settings(GOOGLE_API_KEY="test-key-not-real", DEFAULT_MODEL="gemini-test-model")


@pytest.fixture
def fake_client():
    return FakeClient()


@pytest.fixture
def service(fake_client, settings):
    return FileSearchService(client=fake_client, settings=settings)


@pytest.fixture
def api(service):
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
