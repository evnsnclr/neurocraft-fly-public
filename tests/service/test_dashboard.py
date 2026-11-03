from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "apps" / "observatory-dashboard"


class InteractiveElementParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"button", "form", "input", "select", "textarea"}:
            self.elements.append((tag, dict(attrs)))


def test_dashboard_is_trace_only_and_has_no_network_api() -> None:
    html = (DASHBOARD / "index.html").read_text()
    javascript = (DASHBOARD / "app.js").read_text()
    assert "connect-src 'none'" in html
    for forbidden in ("WebSocket", "XMLHttpRequest", "fetch(", "sendBeacon", ".submit("):
        assert forbidden not in javascript

    parser = InteractiveElementParser()
    parser.feed(html)
    assert parser.elements == [
        (
            "input",
            {
                "id": "trace-file",
                "type": "file",
                "accept": ".ndjson,.json,application/json",
            },
        )
    ]


def test_dashboard_uses_required_scientific_labels() -> None:
    html = (DASHBOARD / "index.html").read_text()
    for label in (
        "Dimensionless model activity",
        "Mode",
        "Data tier",
        "Maximum claim",
        "Fault and neutral ledger",
        "Applied primitive motors",
        "Presentation is non-authoritative",
        "Trace-only · no control channel",
    ):
        assert label in html


def test_dashboard_requires_trace_footer_and_recomputes_sha256() -> None:
    javascript = (DASHBOARD / "app.js").read_text()
    assert 'crypto.subtle.digest("SHA-256"' in javascript
    assert "required trace footer is missing" in javascript
    assert "chain link mismatch" in javascript
    assert "entry hash mismatch" in javascript
    assert "entry.payload.record_count !== records.length" in javascript


def test_dashboard_has_no_external_assets_or_inline_script() -> None:
    html = (DASHBOARD / "index.html").read_text()
    assert "http://" not in html
    assert "https://" not in html
    assert "<script>" not in html
    assert '<script src="app.js" defer></script>' in html
    assert '<link rel="stylesheet" href="styles.css" />' in html
