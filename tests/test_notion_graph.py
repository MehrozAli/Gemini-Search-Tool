import json
import xml.etree.ElementTree as ET

import pytest

from app import notion_graph
from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.notion_graph import (
    NotionClient,
    NotionKnowledgeGraphBuilder,
    block_to_text,
    extract_property_value,
)
from tests.notion_fixtures import FakeNotionSession


@pytest.fixture
def session():
    return FakeNotionSession()


@pytest.fixture
def builder(session):
    built = NotionKnowledgeGraphBuilder("db-demo", NotionClient("notion-test-token", session=session))
    built.build_graph(built.fetch_database_pages())
    return built


def test_client_sends_auth_headers_and_timeout(session):
    NotionClient("notion-test-token", session=session).get_page("kb-pdf")

    assert session.headers["Authorization"] == "Bearer notion-test-token"
    assert session.headers["Notion-Version"] == notion_graph.NOTION_VERSION
    assert session.requests[0][2] == notion_graph.REQUEST_TIMEOUT_SECONDS


def test_database_query_follows_pagination(session):
    pages = NotionClient("t", session=session).query_database("db-demo")

    assert [p["id"] for p in pages] == ["p-reports", "p-pdf", "p-schedule", "p-cron"]
    assert [r[3] for r in session.requests] == [{}, {"start_cursor": "2"}]


@pytest.mark.parametrize(
    ("prop", "expected"),
    [
        ({"type": "number", "number": 3}, 3),
        ({"type": "select", "select": None}, None),
        ({"type": "multi_select", "multi_select": [{"name": "a"}, {"name": "b"}]}, ["a", "b"]),
        ({"type": "url", "url": "https://x.test"}, "https://x.test"),
        ({"type": "formula", "formula": {"type": "boolean", "boolean": True}}, True),
        ({"type": "rollup", "rollup": {"type": "number", "number": 7}}, 7),
        (
            {
                "type": "rollup",
                "rollup": {"type": "array", "array": [{"type": "title", "title": [{"plain_text": "x"}]}]},
            },
            ["x"],
        ),
    ],
)
def test_extract_property_value(prop, expected):
    assert extract_property_value(prop) == expected


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"type": "divider", "divider": {}}, "---"),
        ({"type": "to_do", "to_do": {"checked": True, "rich_text": [{"plain_text": "ship"}]}}, "[x] ship"),
        (
            {"type": "code", "code": {"language": "sql", "rich_text": [{"plain_text": "select 1"}]}},
            "```sql\nselect 1\n```",
        ),
        (
            {
                "type": "image",
                "image": {"type": "external", "external": {"url": "https://x.test/a.png"}, "caption": []},
            },
            "[IMAGE](https://x.test/a.png)",
        ),
        (
            {"type": "bookmark", "bookmark": {"url": "https://x.test", "caption": [{"plain_text": "Docs"}]}},
            "[BOOKMARK: Docs](https://x.test)",
        ),
        ({"type": "child_database", "child_database": {}}, ""),
    ],
)
def test_block_to_text(block, expected):
    assert block_to_text(block) == expected


def test_graph_has_hierarchy_kb_articles_and_aliases(builder):
    graph = builder.graph
    relationships = {(s, t): rel for s, t, rel in graph.edges(data="relationship")}

    assert relationships[("p-reports", "p-pdf")] == "PARENT_OF"
    assert relationships[("p-schedule", "p-cron")] == "PARENT_OF"
    assert relationships[("p-pdf", "kb_kb-pdf")] == "HAS_KB_ARTICLE"
    assert "kb_kb-missing" not in graph
    kb = graph.nodes["kb_kb-pdf"]
    assert kb["name"] == "Exporting a report as PDF"
    assert kb["page_content"] == "## Steps\n\n- Open a report\n\nSee the [guide](https://x.test/g)"


def test_shared_alias_links_nodes_without_breaking_hierarchy(builder):
    alias_edges = [(s, t) for s, t, rel in builder.graph.edges(data="relationship") if rel == "ALIAS_OF"]

    # "analytics" is shared by a parent and its child; that PARENT_OF edge must survive.
    assert builder.graph.edges["p-reports", "p-pdf"]["relationship"] == "PARENT_OF"
    assert alias_edges == [("p-pdf", "p-cron")]
    assert builder.graph.edges["p-pdf", "p-cron"]["alias"] == "export"
    assert builder.get_children("p-reports") == ["p-pdf", "p-schedule"]
    assert builder.get_children("p-pdf") == []
    assert builder.get_hierarchical_path("p-cron") == ["Reports", "Scheduled reports", "Cron settings"]
    assert builder.get_hierarchical_path("kb_kb-pdf") == ["Reports", "PDF export", "Exporting a report as PDF"]


def test_descendants_and_query_node(builder):
    assert builder.get_descendants("p-reports") == ["p-pdf", "p-schedule", "p-cron"]

    result = builder.query_node("pdf EXPORT")
    assert result["id"] == "p-pdf"
    assert result["hierarchical_path"] == "Reports > PDF export"
    assert result["kb_articles"] == [
        {"title": "Exporting a report as PDF", "url": "https://www.notion.so/kb-pdf", "id": "kb-pdf"}
    ]
    assert builder.query_node("nothing") is None


def test_summary(builder):
    summary = builder.summary()

    assert summary["nodes"] == 5
    assert summary["relationships"] == {"PARENT_OF": 3, "HAS_KB_ARTICLE": 1, "ALIAS_OF": 1}
    assert summary["node_types"]["KB_ARTICLE"] == 1
    assert summary["roots"] == ["Reports"]


def test_saved_json_edges_point_at_node_ids(builder, tmp_path):
    path = tmp_path / "graph.json"
    builder.save_graph(str(path))

    data = json.loads(path.read_text())
    node_ids = {node["id"] for node in data["nodes"]}
    assert "kb_kb-pdf" in node_ids
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in data["edges"])


def test_gexf_export(builder, tmp_path):
    path = tmp_path / "graph.gexf"
    builder.save_graph_gexf(str(path))

    assert ET.parse(path).getroot().tag.endswith("gexf")


def test_builder_requires_token_and_database():
    with pytest.raises(ConfigurationError):
        notion_graph.builder_from_settings(Settings(NOTION_TOKEN="t"))


def test_cli_writes_graph_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(notion_graph, "get_settings", lambda: Settings(NOTION_TOKEN="t", NOTION_DATABASE_ID="db-demo"))
    monkeypatch.setattr(notion_graph.requests, "Session", FakeNotionSession)

    code = notion_graph.main(
        ["--output", str(tmp_path / "g.json"), "--gexf", str(tmp_path / "g.gexf"), "--show", "Cron settings"]
    )

    assert code == 0
    assert (tmp_path / "g.json").exists()
    assert (tmp_path / "g.gexf").exists()
    assert '"hierarchical_path": "Reports > Scheduled reports > Cron settings"' in capsys.readouterr().out


def test_cli_explains_missing_configuration(monkeypatch, capsys):
    monkeypatch.setattr(notion_graph, "get_settings", lambda: Settings())

    with pytest.raises(SystemExit) as exit_info:
        notion_graph.main([])

    assert exit_info.value.code == 2
    assert "NOTION_TOKEN and NOTION_DATABASE_ID" in capsys.readouterr().err
