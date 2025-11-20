"""Build a knowledge graph from a hierarchical Notion database.

Each database row becomes a node. The "Parent item" relation becomes
PARENT_OF edges, pages linked through "KB Articles" are fetched with their
body text and attached as KB_ARTICLE nodes, and rows that share an entry
in the comma-separated "Aliases" text are joined by ALIAS_OF edges.

The API uses this to produce the JSON document it ingests into a store.
It also runs on its own:

    python -m app.notion_graph --output knowledge_graph.json --gexf knowledge_graph.gexf
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import requests

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError

logger = logging.getLogger(__name__)

NOTION_API_URL = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
REQUEST_TIMEOUT_SECONDS = 30

PARENT_OF = "PARENT_OF"
HAS_KB_ARTICLE = "HAS_KB_ARTICLE"
ALIAS_OF = "ALIAS_OF"
KB_ARTICLE = "KB_ARTICLE"

TEXT_BLOCK_PREFIXES = {
    "paragraph": "",
    "heading_1": "# ",
    "heading_2": "## ",
    "heading_3": "### ",
    "bulleted_list_item": "- ",
    "numbered_list_item": "- ",
    "quote": "> ",
    "callout": "",
    "toggle": "",
}
MEDIA_BLOCKS = ("image", "video", "file", "pdf", "embed", "bookmark")


class NotionClient:
    """Minimal Notion REST client covering the three calls the builder needs."""

    def __init__(self, token: str, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(
            method, f"{NOTION_API_URL}/{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs
        )
        response.raise_for_status()
        return response.json()

    def query_database(self, database_id: str) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}
        while True:
            data = self._request("POST", f"databases/{database_id}/query", json=payload)
            pages.extend(data.get("results", []))
            if not data.get("has_more"):
                return pages
            payload = {"start_cursor": data.get("next_cursor")}

    def get_page(self, page_id: str) -> dict[str, Any]:
        return self._request("GET", f"pages/{page_id}")

    def list_block_children(self, block_id: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        params: dict[str, Any] = {"page_size": 100}
        while True:
            data = self._request("GET", f"blocks/{block_id}/children", params=params)
            blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                return blocks
            params = {"page_size": 100, "start_cursor": data.get("next_cursor")}


def plain_text(rich_text: list[dict[str, Any]], keep_links: bool = False) -> str:
    parts = []
    for item in rich_text:
        text = item.get("plain_text", "")
        href = item.get("href")
        parts.append(f"[{text}]({href})" if keep_links and href else text)
    return "".join(parts)


def extract_property_value(prop: dict[str, Any]) -> Any:
    """Reduce a Notion property object to a plain Python value."""
    prop_type = prop.get("type")
    if prop_type in ("title", "rich_text"):
        return plain_text(prop.get(prop_type, []))
    if prop_type == "select":
        selected = prop.get("select")
        return selected.get("name") if selected else None
    if prop_type == "multi_select":
        return [item["name"] for item in prop.get("multi_select", [])]
    if prop_type == "relation":
        return [item["id"] for item in prop.get("relation", [])]
    if prop_type == "rollup":
        rollup = prop.get("rollup", {})
        if rollup.get("type") == "array":
            return [extract_property_value(item) for item in rollup.get("array", [])]
        if rollup.get("type") == "number":
            return rollup.get("number")
        return None
    if prop_type == "formula":
        formula = prop.get("formula", {})
        return formula.get(formula.get("type"))
    return prop.get(prop_type)


def extract_page_title(page: dict[str, Any]) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            title = plain_text(prop.get("title", []))
            if title:
                return title
    return "Untitled"


def block_to_text(block: dict[str, Any]) -> str:
    """Render one Notion block as Markdown-ish text; unsupported blocks give ""."""
    block_type = block.get("type")
    data = block.get(block_type, {}) if block_type else {}

    if block_type in TEXT_BLOCK_PREFIXES:
        return TEXT_BLOCK_PREFIXES[block_type] + plain_text(data.get("rich_text", []), keep_links=True)
    if block_type == "code":
        return f"```{data.get('language', '')}\n{plain_text(data.get('rich_text', []))}\n```"
    if block_type == "divider":
        return "---"
    if block_type == "to_do":
        box = "[x]" if data.get("checked") else "[ ]"
        return f"{box} {plain_text(data.get('rich_text', []))}"
    if block_type in MEDIA_BLOCKS:
        source = data.get("type")
        url = data.get(source, {}).get("url", "") if source in ("external", "file") else data.get("url", "")
        caption = plain_text(data.get("caption", []))
        label = block_type.upper() + (f": {caption}" if caption else "")
        return f"[{label}]({url})"
    return ""


class NotionKnowledgeGraphBuilder:
    """Build a directed graph from one Notion database."""

    def __init__(self, database_id: str, notion: NotionClient):
        self.database_id = database_id
        self.notion = notion
        self.graph = nx.DiGraph()
        self._kb_cache: dict[str, dict[str, Any] | None] = {}

    def fetch_database_pages(self) -> list[dict[str, Any]]:
        pages = self.notion.query_database(self.database_id)
        logger.info("Fetched %d pages from Notion", len(pages))
        return pages

    def fetch_page_content(self, page_id: str) -> str:
        try:
            blocks = self.notion.list_block_children(page_id)
        except requests.RequestException as exc:
            logger.warning("Could not fetch content for page %s: %s", page_id, exc)
            return ""
        return "\n\n".join(text for text in map(block_to_text, blocks) if text)

    def fetch_kb_article(self, page_id: str) -> dict[str, Any] | None:
        if page_id not in self._kb_cache:
            self._kb_cache[page_id] = self._load_kb_article(page_id)
        return self._kb_cache[page_id]

    def _load_kb_article(self, page_id: str) -> dict[str, Any] | None:
        try:
            page = self.notion.get_page(page_id)
        except requests.RequestException as exc:
            logger.warning("Could not fetch KB article %s: %s", page_id, exc)
            return None
        return {
            "id": page["id"],
            "url": page.get("url"),
            "created_time": page.get("created_time"),
            "last_edited_time": page.get("last_edited_time"),
            "title": extract_page_title(page),
            "properties": {
                name: extract_property_value(prop) for name, prop in page.get("properties", {}).items()
            },
            "page_content": self.fetch_page_content(page_id),
        }

    def build_graph(self, pages: list[dict[str, Any]]) -> nx.DiGraph:
        for page in pages:
            page_id = page["id"]
            node = {
                "url": page.get("url"),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
            }
            for prop_name, prop in page.get("properties", {}).items():
                node[prop_name] = extract_property_value(prop)
            node["node_type"] = node.get("Type") or "Unknown"
            node["name"] = node.get("Name") or extract_page_title(page)
            self.graph.add_node(page_id, **node)

        for page in pages:
            parent_ids = extract_property_value(page.get("properties", {}).get("Parent item", {}))
            for parent_id in parent_ids or []:
                if parent_id in self.graph:
                    self.graph.add_edge(parent_id, page["id"], relationship=PARENT_OF)

        for page_id in list(self.graph.nodes):
            for kb_id in self.graph.nodes[page_id].get("KB Articles") or []:
                article = self.fetch_kb_article(kb_id)
                if article is None:
                    continue
                kb_node_id = f"kb_{kb_id}"
                attrs = {key: value for key, value in article.items() if key != "id"}
                self.graph.add_node(
                    kb_node_id, name=article["title"], node_type=KB_ARTICLE, original_id=kb_id, **attrs
                )
                self.graph.add_edge(page_id, kb_node_id, relationship=HAS_KB_ARTICLE)

        nodes_by_alias: dict[str, list[str]] = defaultdict(list)
        for node_id, data in self.graph.nodes(data=True):
            aliases = data.get("Aliases")
            if isinstance(aliases, str):
                for alias in {a.strip().lower() for a in aliases.split(",") if a.strip()}:
                    nodes_by_alias[alias].append(node_id)
        for alias, node_ids in nodes_by_alias.items():
            for i, first in enumerate(node_ids):
                for second in node_ids[i + 1 :]:
                    # A DiGraph holds one edge per pair; never overwrite a hierarchy edge.
                    if not self.graph.has_edge(first, second) and not self.graph.has_edge(second, first):
                        self.graph.add_edge(first, second, relationship=ALIAS_OF, alias=alias)

        logger.info(
            "Built graph with %d nodes and %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _related(self, node_id: str, relationships: tuple[str, ...], outgoing: bool) -> list[str]:
        edges = self.graph.out_edges(node_id, data="relationship") if outgoing else self.graph.in_edges(
            node_id, data="relationship"
        )
        return [(target if outgoing else source) for source, target, rel in edges if rel in relationships]

    def get_parent(self, node_id: str) -> str | None:
        parents = self._related(node_id, (PARENT_OF, HAS_KB_ARTICLE), outgoing=False)
        return parents[0] if parents else None

    def get_children(self, node_id: str) -> list[str]:
        return self._related(node_id, (PARENT_OF,), outgoing=True)

    def get_descendants(self, node_id: str) -> list[str]:
        found: list[str] = []
        stack = list(reversed(self.get_children(node_id)))
        while stack:
            current = stack.pop()
            if current in found or current == node_id:
                continue
            found.append(current)
            stack.extend(reversed(self.get_children(current)))
        return found

    def get_kb_articles(self, node_id: str) -> list[dict[str, Any]]:
        return [dict(self.graph.nodes[kb]) for kb in self._related(node_id, (HAS_KB_ARTICLE,), outgoing=True)]

    def get_hierarchical_path(self, node_id: str) -> list[str]:
        path: list[str] = []
        current: str | None = node_id
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            path.insert(0, self.graph.nodes[current].get("name", "Unknown"))
            current = self.get_parent(current)
        return path

    def get_node_by_name(self, name: str) -> str | None:
        wanted = name.lower()
        for node_id, data in self.graph.nodes(data=True):
            if str(data.get("name", "")).lower() == wanted:
                return node_id
        return None

    def query_node(self, name: str) -> dict[str, Any] | None:
        node_id = self.get_node_by_name(name)
        if node_id is None:
            return None
        return {
            **self.graph.nodes[node_id],
            "id": node_id,
            "hierarchical_path": " > ".join(self.get_hierarchical_path(node_id)),
            "children": [self.graph.nodes[c].get("name") for c in self.get_children(node_id)],
            "kb_articles": [
                {"title": kb.get("name"), "url": kb.get("url"), "id": kb.get("original_id")}
                for kb in self.get_kb_articles(node_id)
            ],
        }

    def summary(self) -> dict[str, Any]:
        roots = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") != KB_ARTICLE and not self._related(node_id, (PARENT_OF,), outgoing=False)
        ]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(Counter(d.get("node_type", "Unknown") for _, d in self.graph.nodes(data=True))),
            "relationships": dict(Counter(rel for _, _, rel in self.graph.edges(data="relationship"))),
            "roots": [self.graph.nodes[r].get("name") for r in roots],
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": [{**data, "id": node_id} for node_id, data in self.graph.nodes(data=True)],
            "edges": [{"source": s, "target": t, **data} for s, t, data in self.graph.edges(data=True)],
        }

    def save_graph(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2, ensure_ascii=False)

    def save_graph_gexf(self, path: str) -> None:
        """Write a GEXF copy (for Gephi) keeping only attributes GEXF can store."""
        simple = nx.DiGraph()
        for node_id, data in self.graph.nodes(data=True):
            simple.add_node(
                node_id,
                **{key: str(data.get(key) or "") for key in ("name", "node_type", "url", "Type")},
            )
        simple.add_edges_from(self.graph.edges(data=True))
        nx.write_gexf(simple, path)


def builder_from_settings(
    settings: Settings, session: requests.Session | None = None
) -> NotionKnowledgeGraphBuilder:
    if not settings.notion_token or not settings.notion_database_id:
        raise ConfigurationError("NOTION_TOKEN and NOTION_DATABASE_ID must both be set")
    return NotionKnowledgeGraphBuilder(
        settings.notion_database_id, NotionClient(settings.notion_token, session=session)
    )


def build_graph_file(path: str, settings: Settings, session: requests.Session | None = None) -> str:
    builder = builder_from_settings(settings, session=session)
    builder.build_graph(builder.fetch_database_pages())
    builder.save_graph(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="knowledge_graph.json", help="JSON file to write")
    parser.add_argument("--gexf", help="also write a GEXF file for Gephi")
    parser.add_argument("--show", metavar="NAME", help="print one node with its path and children")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        builder = builder_from_settings(get_settings())
    except ConfigurationError as exc:
        parser.exit(2, f"error: {exc}\n")
    builder.build_graph(builder.fetch_database_pages())
    builder.save_graph(args.output)
    if args.gexf:
        builder.save_graph_gexf(args.gexf)
    print(json.dumps(builder.summary(), indent=2, ensure_ascii=False))
    if args.show:
        print(json.dumps(builder.query_node(args.show), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
