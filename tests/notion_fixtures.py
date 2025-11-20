"""A fake requests.Session that answers like the Notion API for a tiny demo workspace."""
from __future__ import annotations

import requests


def title(text):
    return {"type": "title", "title": [{"plain_text": text}]}


def rich(text):
    return {"type": "rich_text", "rich_text": [{"plain_text": text}]}


def select(name):
    return {"type": "select", "select": {"name": name}}


def relation(*ids):
    return {"type": "relation", "relation": [{"id": i} for i in ids]}


def page(page_id, name, kind, parent=(), kb=(), aliases=""):
    return {
        "id": page_id,
        "url": f"https://www.notion.so/{page_id}",
        "created_time": "2026-01-01T00:00:00.000Z",
        "last_edited_time": "2026-01-02T00:00:00.000Z",
        "properties": {
            "Name": title(name),
            "Type": select(kind),
            "Parent item": relation(*parent),
            "KB Articles": relation(*kb),
            "Aliases": rich(aliases),
        },
    }


DATABASE_PAGES = [
    page("p-reports", "Reports", "Module", aliases="analytics"),
    page("p-pdf", "PDF export", "Feature", parent=["p-reports"], kb=["kb-pdf"], aliases="Analytics, export"),
    page("p-schedule", "Scheduled reports", "Feature", parent=["p-reports"]),
    page("p-cron", "Cron settings", "Setting", parent=["p-schedule"], kb=["kb-missing"], aliases="Export"),
]

KB_PAGES = {
    "kb-pdf": {
        "id": "kb-pdf",
        "url": "https://www.notion.so/kb-pdf",
        "properties": {"Title": title("Exporting a report as PDF"), "Audience": select("Admins")},
    }
}

KB_BLOCKS = {
    "kb-pdf": [
        {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "Steps"}]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "Open a report"}]}},
        {
            "type": "paragraph",
            "paragraph": {"rich_text": [{"plain_text": "See the "}, {"plain_text": "guide", "href": "https://x.test/g"}]},
        },
        {"type": "unsupported", "unsupported": {}},
    ]
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def json(self):
        return self.payload


class FakeNotionSession:
    """Serves DATABASE_PAGES two per page so pagination is exercised."""

    def __init__(self, fail_database=False):
        self.headers = {}
        self.requests = []
        self.fail_database = fail_database

    def request(self, method, url, timeout=None, json=None, params=None):
        self.requests.append((method, url, timeout, json, params))
        path = url.removeprefix("https://api.notion.com/v1/")
        if path.startswith("databases/"):
            if self.fail_database:
                return FakeResponse({}, status=401)
            start = int((json or {}).get("start_cursor") or 0)
            batch = DATABASE_PAGES[start : start + 2]
            more = start + 2 < len(DATABASE_PAGES)
            return FakeResponse({"results": batch, "has_more": more, "next_cursor": str(start + 2) if more else None})
        if path.startswith("pages/"):
            page_id = path.split("/")[1]
            if page_id not in KB_PAGES:
                return FakeResponse({}, status=404)
            return FakeResponse(KB_PAGES[page_id])
        if path.startswith("blocks/"):
            block_id = path.split("/")[1]
            return FakeResponse({"results": KB_BLOCKS.get(block_id, []), "has_more": False})
        return FakeResponse({}, status=404)
