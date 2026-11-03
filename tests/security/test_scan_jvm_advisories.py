"""Offline unit tests for the OSV advisory scan; no test contacts api.osv.dev."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "scan_jvm_advisories.py"
REAL_LOCKFILE = ROOT / "mod" / "gradle.lockfile"

LOCKFILE_SNIPPET = """# This is a Gradle generated file for dependency locking.
# Manual edits can break the build and are not advised.
org.apache.logging.log4j:log4j-core:2.8.1=loomDevelopmentDependencies
io.netty:netty-codec:4.1.118.Final=runtimeClasspath,compileClasspath
com.mojang:authlib:6.0.57=minecraftLibraries
loom:mappings:layered+hash.2198=mappings
net.fabricmc.fabric-api:fabric-api-base:0.4.42+d1308ded19=modImplementation
loom_mappings_1_21_5_layered_hash_2198_v2.net.fabricmc.fabric-api:fabric-api-base:0.4.42=modCompile
net.minecraft:minecraft-common-0cc057f080:1.21.5-loom.mappings.1_21_5.layered+hash.2198-v2=compileClasspath
net.fabricmc:fabric-loader:0.16.10=modImplementation
empty=modApi,modRuntimeOnly
"""


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan_jvm_advisories", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


class CannedFetcher:
    """Replays scripted OSV payloads and records every call the scan made."""

    def __init__(
        self,
        post_responses: list[dict[str, Any]] | None = None,
        vuln_details: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.post_responses = list(post_responses or [])
        self.vuln_details = dict(vuln_details or {})
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []

    def post_json(self, url: str, payload: Any) -> dict[str, Any]:
        self.posts.append((url, dict(payload)))
        if not self.post_responses:
            raise AssertionError(f"unexpected POST to {url}")
        return self.post_responses.pop(0)

    def get_json(self, url: str) -> dict[str, Any]:
        self.gets.append(url)
        return self.vuln_details[url.rsplit("/", 1)[-1]]


def write_lockfile(tmp_path: Path, text: str = LOCKFILE_SNIPPET) -> Path:
    path = tmp_path / "gradle.lockfile"
    path.write_text(text, encoding="utf-8")
    return path


def write_exceptions(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "exceptions.json"
    path.write_text(
        json.dumps({"schema": MODULE.EXCEPTIONS_SCHEMA, "exceptions": entries}), encoding="utf-8"
    )
    return path


def exception_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "GHSA-jfh8-c2jp-5v3q",
        "aliases": ["CVE-2021-44228"],
        "package": "org.apache.logging.log4j:log4j-core",
        "version": "2.8.1",
        "configurations": ["loomDevelopmentDependencies"],
        "justification": "Loom build-tooling-only classpath.",
        "security_md_anchor": "jvm-advisory-exceptions",
        "expires": "2026-12-31",
    }
    entry.update(overrides)
    return entry


def component(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "group": "org.apache.logging.log4j",
        "artifact": "log4j-core",
        "version": "2.8.1",
        "configurations": ("loomDevelopmentDependencies",),
    }
    fields.update(overrides)
    return MODULE.LockedComponent(**fields)


def vulnerability(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "id": "GHSA-jfh8-c2jp-5v3q",
        "aliases": ("CVE-2021-44228",),
        "summary": "Remote code injection in Log4j",
        "severity_cvss": ("CVSS:3.1/AV:N",),
        "severity_database": "CRITICAL",
        "fixed_versions": ("2.15.0",),
        "modified": "2025-10-22T19:37:02.616807Z",
    }
    fields.update(overrides)
    return MODULE.Vulnerability(**fields)


# --- lockfile parsing -------------------------------------------------------


def test_parse_lockfile_reads_components_and_skips_comments(tmp_path: Path) -> None:
    components = MODULE.parse_lockfile(write_lockfile(tmp_path))
    assert len(components) == 8
    first = components[0]
    assert first.coordinate == "org.apache.logging.log4j:log4j-core"
    assert first.key == "org.apache.logging.log4j:log4j-core:2.8.1"
    assert components[1].configurations == ("compileClasspath", "runtimeClasspath")
    assert all(item.group != "empty" for item in components)


def test_parse_lockfile_preserves_versions_containing_plus_and_dots(tmp_path: Path) -> None:
    components = MODULE.parse_lockfile(write_lockfile(tmp_path))
    versions = {item.key.rsplit(":", 1)[-1] for item in components}
    assert "layered+hash.2198" in versions
    assert "1.21.5-loom.mappings.1_21_5.layered+hash.2198-v2" in versions


@pytest.mark.parametrize(
    "line",
    [
        "org.example:artifact=runtimeClasspath",
        "org.example:artifact:1.0:extra=runtimeClasspath",
        "org.example:artifact:1.0",
        "org.example::1.0=runtimeClasspath",
        "org.example:artifact:1.0=",
    ],
)
def test_parse_lockfile_fails_fast_on_malformed_lines(tmp_path: Path, line: str) -> None:
    path = write_lockfile(tmp_path, f"{LOCKFILE_SNIPPET}{line}\n")
    with pytest.raises(MODULE.LockfileError):
        MODULE.parse_lockfile(path)


def test_parse_lockfile_rejects_a_file_with_no_components(tmp_path: Path) -> None:
    path = write_lockfile(tmp_path, "# only a comment\nempty=modApi\n")
    with pytest.raises(MODULE.LockfileError):
        MODULE.parse_lockfile(path)


# --- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "reason_code"),
    [
        ("loom_mappings_1_21_5_layered_hash_2198_v2.net.fabricmc.fabric-api", "loom-remapped"),
        ("net.fabricmc.fabric-api", "fabric-maven"),
        ("net.fabricmc", "fabric-maven"),
        ("com.mojang", "mojang-repo"),
        ("net.minecraft", "loom-generated"),
        ("loom", "loom-generated"),
    ],
)
def test_classify_marks_non_central_groups_unscannable(group: str, reason_code: str) -> None:
    verdict = MODULE.classify(component(group=group))
    assert verdict.queryable is False
    assert verdict.reason_code == reason_code
    assert verdict.reason_text


def test_classify_marks_maven_central_groups_queryable() -> None:
    verdict = MODULE.classify(component(group="io.netty"))
    assert verdict.queryable is True
    assert verdict.reason_code == "maven-central"


def test_classify_refuses_an_unknown_group_instead_of_defaulting() -> None:
    with pytest.raises(MODULE.UnknownGroupError) as error:
        MODULE.classify(component(group="com.example.newly.added"))
    assert "com.example.newly.added" in str(error.value)


def test_real_lockfile_classification_counts_are_pinned() -> None:
    components = MODULE.parse_lockfile(REAL_LOCKFILE)
    verdicts = [MODULE.classify(item) for item in components]
    queryable = sum(1 for verdict in verdicts if verdict.queryable)
    assert len(components) == 213
    assert queryable == 73
    assert len(components) - queryable == 140


# --- querying ---------------------------------------------------------------


def test_build_queries_uses_the_maven_ecosystem_and_raw_versions() -> None:
    queries = MODULE.build_queries([component(), component(group="io.netty", version="4.1.118.F")])
    assert queries[0] == {
        "package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"},
        "version": "2.8.1",
    }
    assert queries[1]["version"] == "4.1.118.F"


def test_query_osv_batch_aligns_results_by_index_and_batches() -> None:
    components = [component(), component(group="io.netty", artifact="netty-codec")]
    fetcher = CannedFetcher(
        [{"results": [{"vulns": [{"id": "GHSA-a"}]}]}, {"results": [{}]}],
    )
    advisories = MODULE.query_osv_batch(components, fetcher, batch_size=1)
    assert advisories[components[0].key] == ("GHSA-a",)
    assert advisories[components[1].key] == ()
    assert len(fetcher.posts) == 2
    assert all(url == MODULE.QUERYBATCH_URL for url, _ in fetcher.posts)


def test_query_osv_batch_rejects_a_misaligned_result_list() -> None:
    fetcher = CannedFetcher([{"results": []}])
    with pytest.raises(MODULE.OsvUnavailableError):
        MODULE.query_osv_batch([component()], fetcher, batch_size=10)


def test_query_osv_batch_follows_pagination_instead_of_truncating() -> None:
    fetcher = CannedFetcher(
        [
            {"results": [{"vulns": [{"id": "GHSA-a"}], "next_page_token": "t1"}]},
            {"vulns": [{"id": "GHSA-b"}], "next_page_token": "t2"},
            {"vulns": [{"id": "GHSA-a"}, {"id": "GHSA-c"}]},
        ]
    )
    advisories = MODULE.query_osv_batch([component()], fetcher, batch_size=10)
    assert advisories[component().key] == ("GHSA-a", "GHSA-b", "GHSA-c")
    assert [url for url, _ in fetcher.posts[1:]] == [MODULE.QUERY_URL, MODULE.QUERY_URL]
    assert fetcher.posts[1][1]["page_token"] == "t1"


def test_query_osv_batch_refuses_endless_pagination() -> None:
    fetcher = CannedFetcher(
        [{"results": [{"vulns": [], "next_page_token": "t"}]}]
        + [{"vulns": [], "next_page_token": "t"} for _ in range(MODULE.MAX_PAGES + 2)]
    )
    with pytest.raises(MODULE.OsvUnavailableError):
        MODULE.query_osv_batch([component()], fetcher, batch_size=10)


def test_fetch_vuln_detail_extracts_aliases_severity_and_matching_fixes() -> None:
    fetcher = CannedFetcher(
        vuln_details={
            "GHSA-a": {
                "id": "GHSA-a",
                "aliases": ["CVE-2021-44228", "CVE-2000-0001"],
                "summary": "Remote code injection",
                "modified": "2025-10-22T19:37:02Z",
                "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N"}],
                "database_specific": {"severity": "CRITICAL"},
                "affected": [
                    {
                        "package": {"ecosystem": "Maven", "name": "other:artifact"},
                        "ranges": [{"events": [{"introduced": "0"}, {"fixed": "9.9.9"}]}],
                    },
                    {
                        "package": {
                            "ecosystem": "Maven",
                            "name": "org.apache.logging.log4j:log4j-core",
                        },
                        "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.15.0"}]}],
                    },
                ],
            }
        }
    )
    vuln = MODULE.fetch_vuln_detail("GHSA-a", "org.apache.logging.log4j:log4j-core", fetcher)
    assert vuln.aliases == ("CVE-2000-0001", "CVE-2021-44228")
    assert vuln.severity_cvss == ("CVSS:3.1/AV:N",)
    assert vuln.severity_database == "CRITICAL"
    assert vuln.fixed_versions == ("2.15.0",)
    assert vuln.excepted is False


def test_fetch_vuln_detail_rejects_a_mismatched_advisory_id() -> None:
    fetcher = CannedFetcher(vuln_details={"GHSA-a": {"id": "GHSA-b"}})
    with pytest.raises(MODULE.OsvUnavailableError):
        MODULE.fetch_vuln_detail("GHSA-a", "org.example:artifact", fetcher)


def test_collect_findings_skips_clean_components_and_caches_details() -> None:
    components = [component(), component(group="io.netty", artifact="netty-codec")]
    fetcher = CannedFetcher(
        [{"results": [{"vulns": [{"id": "GHSA-a"}, {"id": "GHSA-a"}]}, {"vulns": []}]}],
        {"GHSA-a": {"id": "GHSA-a", "modified": "2025-01-01T00:00:00Z"}},
    )
    findings = MODULE.collect_findings(components, fetcher, batch_size=10)
    assert len(findings) == 1
    assert findings[0].component.key == components[0].key
    assert [vuln.id for vuln in findings[0].vulns] == ["GHSA-a"]
    assert fetcher.gets == [f"{MODULE.VULN_URL}GHSA-a"]


# --- exception file ---------------------------------------------------------


def test_load_exceptions_parses_and_sorts_a_valid_entry(tmp_path: Path) -> None:
    path = write_exceptions(tmp_path, [exception_entry(configurations=["b", "a"])])
    exceptions = MODULE.load_exceptions(path)
    assert len(exceptions) == 1
    assert exceptions[0].configurations == ("a", "b")
    assert exceptions[0].expires == date(2026, 12, 31)
    assert exceptions[0].identity == (
        "GHSA-jfh8-c2jp-5v3q|org.apache.logging.log4j:log4j-core|2.8.1"
    )


def test_load_exceptions_accepts_an_empty_list(tmp_path: Path) -> None:
    assert MODULE.load_exceptions(write_exceptions(tmp_path, [])) == ()


@pytest.mark.parametrize(
    "entry",
    [
        {**exception_entry(), "unexpected": "value"},
        {key: value for key, value in exception_entry().items() if key != "expires"},
        exception_entry(expires="31-12-2026"),
        exception_entry(configurations=[]),
        exception_entry(configurations="loomDevelopmentDependencies"),
        exception_entry(aliases="CVE-2021-44228"),
        exception_entry(id=""),
        exception_entry(justification=""),
    ],
)
def test_load_exceptions_fails_fast_on_schema_violations(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    with pytest.raises(MODULE.ExceptionFileError):
        MODULE.load_exceptions(write_exceptions(tmp_path, [entry]))


def test_load_exceptions_rejects_duplicate_identities(tmp_path: Path) -> None:
    path = write_exceptions(tmp_path, [exception_entry(), exception_entry()])
    with pytest.raises(MODULE.ExceptionFileError):
        MODULE.load_exceptions(path)


def test_load_exceptions_rejects_an_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps({"schema": "other.v9", "exceptions": []}), encoding="utf-8")
    with pytest.raises(MODULE.ExceptionFileError):
        MODULE.load_exceptions(path)


def test_repository_exception_file_loads_and_is_scoped_to_the_lockfile() -> None:
    exceptions = MODULE.load_exceptions(ROOT / "docs/security/jvm-advisory-exceptions.json")
    locked = {item.key: item for item in MODULE.parse_lockfile(REAL_LOCKFILE)}
    for exception in exceptions:
        entry = locked[f"{exception.package}:{exception.version}"]
        assert set(exception.configurations) <= set(entry.configurations)
        assert exception.expires >= date(2026, 1, 1)
        assert exception.security_md_anchor == "jvm-advisory-exceptions"


# --- evaluation -------------------------------------------------------------


def test_evaluate_passes_when_every_advisory_is_excepted(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(write_exceptions(tmp_path, [exception_entry()]))
    finding = MODULE.Finding(component=component(), vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], exceptions, [component()], date(2026, 8, 18))
    assert outcome.status == "pass"
    assert outcome.failures == ()
    assert outcome.findings[0].excepted is True
    assert outcome.findings[0].vulns[0].exception_id == "GHSA-jfh8-c2jp-5v3q"


def test_evaluate_matches_an_exception_written_against_a_cve_alias(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(
        write_exceptions(tmp_path, [exception_entry(id="CVE-2021-44228", aliases=[])])
    )
    finding = MODULE.Finding(component=component(), vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], exceptions, [component()], date(2026, 8, 18))
    assert outcome.status == "pass"


def test_evaluate_fails_on_an_unexcepted_advisory() -> None:
    finding = MODULE.Finding(component=component(), vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], (), [component()], date(2026, 8, 18))
    assert outcome.status == "fail"
    assert any("unexcepted advisory GHSA-jfh8-c2jp-5v3q" in item for item in outcome.failures)
    assert outcome.findings[0].vulns[0].excepted is False


def test_evaluate_fails_on_an_expired_exception(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(write_exceptions(tmp_path, [exception_entry()]))
    finding = MODULE.Finding(component=component(), vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], exceptions, [component()], date(2027, 1, 1))
    assert outcome.status == "fail"
    assert any("expired on 2026-12-31" in item for item in outcome.failures)


def test_evaluate_fails_on_a_stale_exception_that_matches_nothing(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(write_exceptions(tmp_path, [exception_entry()]))
    outcome = MODULE.evaluate([], exceptions, [component()], date(2026, 8, 18))
    assert outcome.status == "fail"
    assert any("stale exception" in item for item in outcome.failures)


def test_evaluate_fails_when_configurations_exceed_the_locked_scope(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(
        write_exceptions(
            tmp_path,
            [exception_entry(configurations=["loomDevelopmentDependencies", "runtimeClasspath"])],
        )
    )
    finding = MODULE.Finding(component=component(), vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], exceptions, [component()], date(2026, 8, 18))
    assert outcome.status == "fail"
    assert any("runtimeClasspath" in item for item in outcome.failures)


def test_evaluate_fails_when_an_exception_names_an_absent_component(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(
        write_exceptions(tmp_path, [exception_entry(version="9.9.9")])
    )
    outcome = MODULE.evaluate([], exceptions, [component()], date(2026, 8, 18))
    assert any("absent from the lockfile" in item for item in outcome.failures)


def test_evaluate_fails_when_declared_aliases_are_not_observed(tmp_path: Path) -> None:
    exceptions = MODULE.load_exceptions(
        write_exceptions(tmp_path, [exception_entry(aliases=["CVE-2021-44228", "CVE-9999-0001"])])
    )
    finding = MODULE.Finding(component=component(), vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], exceptions, [component()], date(2026, 8, 18))
    assert any("CVE-9999-0001" in item for item in outcome.failures)


# --- report determinism and receipt ----------------------------------------


def build_sample_report(tmp_path: Path) -> dict[str, Any]:
    lockfile = write_lockfile(tmp_path)
    components = MODULE.parse_lockfile(lockfile)
    unscannable = tuple(
        (item, MODULE.classify(item)) for item in components if not MODULE.classify(item).queryable
    )
    finding = MODULE.Finding(component=components[0], vulns=(vulnerability(),))
    outcome = MODULE.evaluate([finding], (), components, date(2026, 8, 18))
    return MODULE.build_report(
        lockfile_sha256=sha256(lockfile.read_bytes()).hexdigest(),
        exceptions_sha256="0" * 64,
        components=components,
        unscannable=unscannable,
        outcome=outcome,
    )


def test_build_report_is_deterministic_and_carries_no_wall_clock(tmp_path: Path) -> None:
    first = MODULE.canonical_json(build_sample_report(tmp_path))
    second = MODULE.canonical_json(build_sample_report(tmp_path))
    assert first == second
    assert "scan_date" not in first
    assert "timestamp" not in first
    report = json.loads(first)
    assert report["components"] == {"queryable": 2, "total": 8, "unscannable": 6}
    assert report["outcome"]["status"] == "fail"
    assert {entry["reason_code"] for entry in report["unscannable"]} == {
        "fabric-maven",
        "loom-generated",
        "loom-remapped",
        "mojang-repo",
    }


def test_build_receipt_binds_the_exact_report_bytes(tmp_path: Path) -> None:
    report = build_sample_report(tmp_path)
    receipt = MODULE.build_receipt(report, scan_date=date(2026, 8, 18))
    written = tmp_path / "report.json"
    assert MODULE.write_json(written, report) == receipt["report_sha256"]
    assert receipt["scan_date"] == "2026-08-18"
    assert receipt["advisories"] == {
        "components_with_advisories": 1,
        "excepted": 0,
        "total": 1,
    }
    assert receipt["outcome_status"] == "fail"


# --- http fetcher retry policy ---------------------------------------------


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.osv.dev/v1/query", code, "boom", {}, None)  # type: ignore[arg-type]


def install_urlopen(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> None:
    remaining = list(script)

    def fake_urlopen(request: Any, timeout: float | None = None) -> FakeResponse:
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(outcome)

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", fake_urlopen)


def test_http_fetcher_retries_transient_failures_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_urlopen(
        monkeypatch,
        [http_error(429), urllib.error.URLError("reset"), b'{"results": []}'],
    )
    delays: list[float] = []
    fetcher = MODULE.HttpFetcher(timeout=1.0, retries=3, sleep=delays.append)
    assert fetcher.post_json(MODULE.QUERYBATCH_URL, {"queries": []}) == {"results": []}
    assert delays == [1.0, 4.0]


def test_http_fetcher_fails_closed_when_retries_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_urlopen(monkeypatch, [http_error(503), http_error(503)])
    fetcher = MODULE.HttpFetcher(timeout=1.0, retries=2, sleep=lambda _: None)
    with pytest.raises(MODULE.OsvUnavailableError) as error:
        fetcher.get_json(f"{MODULE.VULN_URL}GHSA-a")
    assert "after 2 attempts" in str(error.value)


def test_http_fetcher_does_not_retry_a_client_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_urlopen(monkeypatch, [http_error(400)])
    fetcher = MODULE.HttpFetcher(timeout=1.0, retries=3, sleep=lambda _: None)
    with pytest.raises(MODULE.OsvUnavailableError) as error:
        fetcher.get_json(f"{MODULE.VULN_URL}GHSA-a")
    assert "HTTP 400" in str(error.value)


# --- offline CLI reevaluation ----------------------------------------------


def test_reevaluate_replays_a_saved_report_offline_and_reports_exit_codes(
    tmp_path: Path,
) -> None:
    lockfile = write_lockfile(tmp_path)
    saved = tmp_path / "saved.json"
    saved.write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "package": "org.apache.logging.log4j:log4j-core",
                        "version": "2.8.1",
                        "vulns": [
                            {
                                "id": "GHSA-jfh8-c2jp-5v3q",
                                "aliases": ["CVE-2021-44228"],
                                "summary": "Remote code injection in Log4j",
                                "severity": {"cvss": ["CVSS:3.1/AV:N"], "database": "CRITICAL"},
                                "fixed_versions": ["2.15.0"],
                                "modified": "2025-10-22T19:37:02.616807Z",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "--lockfile",
        str(lockfile),
        "--exceptions",
        str(write_exceptions(tmp_path, [])),
        "--output",
        str(tmp_path / "report.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--reevaluate",
        str(saved),
    ]
    assert MODULE.main(argv) == 1

    excepted = write_exceptions(tmp_path, [exception_entry()])
    argv[3] = str(excepted)
    assert MODULE.main(argv) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == {"failures": [], "status": "pass"}
    assert report["findings"][0]["exception_ids"] == ["GHSA-jfh8-c2jp-5v3q"]


def test_main_returns_two_when_the_lockfile_cannot_be_parsed(tmp_path: Path) -> None:
    broken = tmp_path / "gradle.lockfile"
    broken.write_text("not-a-component-line\n", encoding="utf-8")
    assert (
        MODULE.main(
            [
                "--lockfile",
                str(broken),
                "--exceptions",
                str(write_exceptions(tmp_path, [])),
                "--output",
                str(tmp_path / "report.json"),
                "--receipt",
                str(tmp_path / "receipt.json"),
                "--reevaluate",
                str(tmp_path / "missing.json"),
            ]
        )
        == 2
    )
