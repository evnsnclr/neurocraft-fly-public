#!/usr/bin/env python3
"""Fail-closed OSV.dev advisory scan of the Fabric mod's Gradle dependency lock.

Classification precedes every query: OSV indexes Maven Central, so an empty response for
a Mojang-repo, Fabric-maven, or Loom-generated coordinate means "never indexed", not
"known clean". Those components are inventoried, never cleared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Protocol

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_LOCKFILE: Final = REPOSITORY_ROOT / "mod/gradle.lockfile"
DEFAULT_EXCEPTIONS: Final = REPOSITORY_ROOT / "docs/security/jvm-advisory-exceptions.json"
DEFAULT_OUTPUT: Final = REPOSITORY_ROOT / "artifacts/audit/jvm-advisories.json"
DEFAULT_RECEIPT: Final = REPOSITORY_ROOT / "artifacts/audit/jvm-advisories.receipt.json"
OSV_ENDPOINT: Final = "https://api.osv.dev"
QUERYBATCH_URL: Final = f"{OSV_ENDPOINT}/v1/querybatch"
QUERY_URL: Final = f"{OSV_ENDPOINT}/v1/query"
VULN_URL: Final = f"{OSV_ENDPOINT}/v1/vulns/"
USER_AGENT: Final = "neurocraft-fly-jvm-advisory-scan/1 (+https://api.osv.dev)"
JSON_HEADERS: Final = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
REPORT_SCHEMA: Final = "neurocraft.jvm-advisory-report.v1"
RECEIPT_SCHEMA: Final = "neurocraft.jvm-advisory-receipt.v1"
EXCEPTIONS_SCHEMA: Final = "neurocraft.jvm-advisory-exceptions.v1"
MAVEN_ECOSYSTEM: Final = "Maven"
RETRY_BACKOFF_SECONDS: Final = (1.0, 4.0, 16.0)
MAX_PAGES: Final = 50

#: Why OSV's Maven ecosystem cannot speak about a component, keyed by reason code.
UNSCANNABLE_REASONS: Final[Mapping[str, str]] = {
    "fabric-maven": "maven.fabricmc.net only; outside the Maven Central corpus OSV indexes.",
    "loom-generated": "Loom-generated synthetic coordinate; never published to a repository.",
    "loom-remapped": "Loom-remapped copy of a Fabric API module; never published anywhere.",
    "mojang-repo": "libraries.minecraft.net only; outside the Maven Central corpus OSV indexes.",
}

#: Ordered ``group:`` prefixes -> reason code. Order matters: the Loom-remapped prefix
#: and the Fabric API group both extend net.fabricmc-shaped names.
UNSCANNABLE_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("loom_mappings_1_21_5_layered_hash_2198_v2.", "loom-remapped"),
    ("net.fabricmc.fabric-api:", "fabric-maven"),
    ("net.fabricmc:", "fabric-maven"),
    ("com.mojang:", "mojang-repo"),
    ("net.minecraft:", "loom-generated"),
    ("loom:", "loom-generated"),
)

#: Maven Central groups (queryable). A group matched by neither table is a triage event.
_CENTRAL_GROUP_NAMES = """
    com.fasterxml.jackson.core com.github.oshi com.github.stephenc.jcip
    com.google.code.gson com.google.guava com.ibm.icu com.microsoft.azure com.nimbusds
    commons-codec commons-io commons-logging io.github.llamalad7 io.netty it.unimi.dsi
    net.java.dev.jna net.minecrell net.minidev net.sf.jopt-simple org.apache.commons
    org.apache.httpcomponents org.apache.logging.log4j org.apiguardian org.jcraft
    org.jetbrains org.jline org.joml org.junit org.junit.jupiter org.junit.platform
    org.lwjgl org.lz4 org.opentest4j org.ow2.asm org.slf4j org.vineflower
"""
_EXCEPTION_FIELD_NAMES = (
    "aliases configurations expires id justification package security_md_anchor version"
)
CENTRAL_GROUPS: Final[frozenset[str]] = frozenset(_CENTRAL_GROUP_NAMES.split())
CENTRAL_REASON: Final = "Resolved from Maven Central; inside OSV's Maven ecosystem corpus."
EXCEPTION_FIELDS: Final[frozenset[str]] = frozenset(_EXCEPTION_FIELD_NAMES.split())


class LockfileError(ValueError):
    """The Gradle lockfile is missing, empty, or has an unparseable line."""


class UnknownGroupError(ValueError):
    """A locked component belongs to a group absent from the classification table."""


class ExceptionFileError(ValueError):
    """The advisory-exception file violates its schema."""


class OsvUnavailableError(RuntimeError):
    """OSV could not be reached, or answered with an unusable payload."""


@dataclass(frozen=True)
class LockedComponent:
    """One ``group:artifact:version`` line of ``mod/gradle.lockfile``."""

    group: str
    artifact: str
    version: str
    configurations: tuple[str, ...]

    @property
    def coordinate(self) -> str:
        return f"{self.group}:{self.artifact}"

    @property
    def key(self) -> str:
        return f"{self.coordinate}:{self.version}"


@dataclass(frozen=True)
class Classification:
    """Pre-query verdict on whether OSV can speak about a component at all."""

    queryable: bool
    reason_code: str
    reason_text: str


@dataclass(frozen=True)
class Vulnerability:
    """An OSV advisory record reduced to the fields the report records."""

    id: str
    aliases: tuple[str, ...]
    summary: str
    severity_cvss: tuple[str, ...]
    severity_database: str
    fixed_versions: tuple[str, ...]
    modified: str
    excepted: bool = False
    exception_id: str = ""


@dataclass(frozen=True)
class Finding:
    """Every advisory OSV returned for one locked component."""

    component: LockedComponent
    vulns: tuple[Vulnerability, ...]

    @property
    def excepted(self) -> bool:
        return all(vuln.excepted for vuln in self.vulns)


@dataclass(frozen=True)
class AdvisoryException:
    """One exact, expiring, configuration-scoped advisory exception."""

    id: str
    aliases: tuple[str, ...]
    package: str
    version: str
    configurations: tuple[str, ...]
    justification: str
    security_md_anchor: str
    expires: date

    @property
    def identity(self) -> str:
        return f"{self.id}|{self.package}|{self.version}"


@dataclass(frozen=True)
class ScanOutcome:
    """Findings annotated with exception verdicts plus every fail-closed reason."""

    findings: tuple[Finding, ...]
    failures: tuple[str, ...]

    @property
    def status(self) -> str:
        return "fail" if self.failures else "pass"


class Fetcher(Protocol):
    """Network seam; unit tests inject canned responses and never touch OSV."""

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def get_json(self, url: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HttpFetcher:
    """``urllib`` OSV client with bounded retries that fails closed on exhaustion."""

    timeout: float = 30.0
    retries: int = 3
    sleep: Callable[[float], None] = time.sleep

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST", headers=JSON_HEADERS)
        return self._fetch(request)

    def get_json(self, url: str) -> Mapping[str, Any]:
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
        return self._fetch(request)

    def _attempt(self, request: urllib.request.Request) -> Mapping[str, Any]:
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise OsvUnavailableError(f"OSV returned a non-object payload for {request.full_url}")
        return decoded

    def _fetch(self, request: urllib.request.Request) -> Mapping[str, Any]:
        last = "no attempt was made"
        for attempt in range(self.retries):
            try:
                return self._attempt(request)
            except urllib.error.HTTPError as error:
                if error.code != 429 and error.code < 500:
                    raise OsvUnavailableError(
                        f"OSV rejected {request.full_url} with HTTP {error.code}"
                    ) from error
                last = f"HTTP {error.code}"
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last = f"{type(error).__name__}: {error}"
            if attempt + 1 < self.retries:
                self.sleep(RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)])
        raise OsvUnavailableError(
            f"OSV unreachable for {request.full_url} after {self.retries} attempts: {last}"
        )


def parse_lockfile(path: Path) -> tuple[LockedComponent, ...]:
    """Parse every locked component, failing fast on any unexpected line shape."""

    components: list[LockedComponent] = []
    text = path.read_text(encoding="utf-8")
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("empty="):
            continue
        coordinate, separator, configurations = line.partition("=")
        parts = coordinate.split(":")
        if not separator or len(parts) != 3 or not all(parts) or not configurations:
            raise LockfileError(
                f"{path}:{number}: expected group:artifact:version=configs, got {line!r}"
            )
        components.append(
            LockedComponent(
                group=parts[0],
                artifact=parts[1],
                version=parts[2],
                configurations=tuple(sorted(configurations.split(","))),
            )
        )
    if not components:
        raise LockfileError(f"{path}: no locked components were parsed")
    return tuple(components)


def classify(component: LockedComponent) -> Classification:
    """Decide, before querying, whether OSV's Maven corpus covers this component."""

    group_key = f"{component.group}:"
    for prefix, code in UNSCANNABLE_PREFIXES:
        if group_key.startswith(prefix):
            return Classification(
                queryable=False, reason_code=code, reason_text=UNSCANNABLE_REASONS[code]
            )
    if component.group in CENTRAL_GROUPS:
        return Classification(
            queryable=True, reason_code="maven-central", reason_text=CENTRAL_REASON
        )
    raise UnknownGroupError(
        f"unclassified dependency group {component.group!r} ({component.key}); "
        "triage it into CENTRAL_GROUPS or UNSCANNABLE_PREFIXES before scanning"
    )


def build_queries(components: Sequence[LockedComponent]) -> tuple[dict[str, Any], ...]:
    """Build OSV query objects; OSV evaluates affected ranges server-side."""

    return tuple(
        {
            "package": {"ecosystem": MAVEN_ECOSYSTEM, "name": component.coordinate},
            "version": component.version,
        }
        for component in components
    )


def _vuln_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    vulns = payload.get("vulns")
    if vulns is None:
        return ()
    if not isinstance(vulns, list):
        raise OsvUnavailableError("OSV returned a non-list 'vulns' field")
    identifiers: list[str] = []
    for vuln in vulns:
        identifier = vuln.get("id") if isinstance(vuln, Mapping) else None
        if not isinstance(identifier, str) or not identifier:
            raise OsvUnavailableError("OSV returned an advisory without a string id")
        identifiers.append(identifier)
    return tuple(identifiers)


def _drain_pages(
    component: LockedComponent, first: Mapping[str, Any], fetcher: Fetcher
) -> tuple[str, ...]:
    """Follow ``next_page_token`` so a large advisory set is never silently truncated."""

    identifiers = set(_vuln_ids(first))
    token = first.get("next_page_token")
    for _ in range(MAX_PAGES):
        if not isinstance(token, str) or not token:
            return tuple(sorted(identifiers))
        page = fetcher.post_json(
            QUERY_URL,
            {
                "package": {"ecosystem": MAVEN_ECOSYSTEM, "name": component.coordinate},
                "version": component.version,
                "page_token": token,
            },
        )
        identifiers.update(_vuln_ids(page))
        token = page.get("next_page_token")
    raise OsvUnavailableError(f"OSV paginated past {MAX_PAGES} pages for {component.key}")


def query_osv_batch(
    components: Sequence[LockedComponent], fetcher: Fetcher, *, batch_size: int
) -> dict[str, tuple[str, ...]]:
    """Return advisory ids per component key via ``/v1/querybatch`` (index-aligned)."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    advisories: dict[str, tuple[str, ...]] = {}
    for start in range(0, len(components), batch_size):
        chunk = tuple(components[start : start + batch_size])
        response = fetcher.post_json(
            QUERYBATCH_URL, {"queries": [dict(query) for query in build_queries(chunk)]}
        )
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(chunk):
            raise OsvUnavailableError(
                f"OSV querybatch returned {type(results).__name__} of unexpected length "
                f"for a {len(chunk)}-query batch"
            )
        for component, result in zip(chunk, results, strict=True):
            if not isinstance(result, Mapping):
                raise OsvUnavailableError("OSV querybatch returned a non-object result")
            advisories[component.key] = _drain_pages(component, result, fetcher)
    return advisories


def _severity(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    entries = payload.get("severity") or ()
    vectors = {
        entry["score"]
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("score"), str)
    }
    specific = payload.get("database_specific") or {}
    database = specific.get("severity") if isinstance(specific, Mapping) else None
    return tuple(sorted(vectors)), database if isinstance(database, str) else ""


def _fixed_versions(payload: Mapping[str, Any], coordinate: str) -> tuple[str, ...]:
    fixed: set[str] = set()
    for affected in payload.get("affected") or ():
        if not isinstance(affected, Mapping):
            continue
        package = affected.get("package") or {}
        if not isinstance(package, Mapping) or package.get("name") != coordinate:
            continue
        for entry in affected.get("ranges") or ():
            if not isinstance(entry, Mapping):
                continue
            for event in entry.get("events") or ():
                value = event.get("fixed") if isinstance(event, Mapping) else None
                if isinstance(value, str):
                    fixed.add(value)
    return tuple(sorted(fixed))


def fetch_vuln_detail(vuln_id: str, coordinate: str, fetcher: Fetcher) -> Vulnerability:
    """Resolve aliases, summary, severity, and fixes via ``/v1/vulns/{id}``."""

    payload = fetcher.get_json(f"{VULN_URL}{vuln_id}")
    identifier = payload.get("id")
    if identifier != vuln_id:
        raise OsvUnavailableError(
            f"OSV returned advisory {identifier!r} when asked for {vuln_id!r}"
        )
    aliases = payload.get("aliases") or ()
    severity_cvss, severity_database = _severity(payload)
    summary = payload.get("summary")
    modified = payload.get("modified")
    return Vulnerability(
        id=vuln_id,
        aliases=tuple(sorted(alias for alias in aliases if isinstance(alias, str))),
        summary=summary if isinstance(summary, str) else "",
        severity_cvss=severity_cvss,
        severity_database=severity_database,
        fixed_versions=_fixed_versions(payload, coordinate),
        modified=modified if isinstance(modified, str) else "",
    )


def collect_findings(
    components: Sequence[LockedComponent], fetcher: Fetcher, *, batch_size: int
) -> tuple[Finding, ...]:
    """Query OSV for the queryable components and hydrate every advisory returned."""

    advisories = query_osv_batch(components, fetcher, batch_size=batch_size)
    details: dict[tuple[str, str], Vulnerability] = {}
    findings: list[Finding] = []
    for component in components:
        identifiers = advisories.get(component.key, ())
        if not identifiers:
            continue
        vulns: list[Vulnerability] = []
        for identifier in identifiers:
            cache_key = (identifier, component.coordinate)
            if cache_key not in details:
                details[cache_key] = fetch_vuln_detail(identifier, component.coordinate, fetcher)
            vulns.append(details[cache_key])
        findings.append(
            Finding(component=component, vulns=tuple(sorted(vulns, key=lambda vuln: vuln.id)))
        )
    return tuple(sorted(findings, key=lambda finding: finding.component.key))


def _require_string_list(entry: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = entry.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ExceptionFileError(f"exception field {field!r} must be a list of strings")
    return tuple(values)


def _parse_exception(entry: Mapping[str, Any]) -> AdvisoryException:
    keys = set(entry)
    if keys != EXCEPTION_FIELDS:
        missing = sorted(EXCEPTION_FIELDS - keys)
        unknown = sorted(keys - EXCEPTION_FIELDS)
        raise ExceptionFileError(f"exception key mismatch: missing={missing} unknown={unknown}")
    scalars: dict[str, str] = {}
    for field in ("id", "package", "version", "justification", "security_md_anchor", "expires"):
        value = entry[field]
        if not isinstance(value, str) or not value:
            raise ExceptionFileError(f"exception field {field!r} must be a non-empty string")
        scalars[field] = value
    configurations = _require_string_list(entry, "configurations")
    if not configurations:
        raise ExceptionFileError(f"exception {scalars['id']} must scope at least one configuration")
    try:
        expires = date.fromisoformat(scalars["expires"])
    except ValueError as error:
        raise ExceptionFileError(f"exception {scalars['id']} has a malformed 'expires'") from error
    return AdvisoryException(
        id=scalars["id"],
        aliases=tuple(sorted(_require_string_list(entry, "aliases"))),
        package=scalars["package"],
        version=scalars["version"],
        configurations=tuple(sorted(configurations)),
        justification=scalars["justification"],
        security_md_anchor=scalars["security_md_anchor"],
        expires=expires,
    )


def load_exceptions(path: Path) -> tuple[AdvisoryException, ...]:
    """Load and schema-validate the exception file, failing fast on any drift."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema", "exceptions"}:
        raise ExceptionFileError(f"{path}: expected exactly 'schema' and 'exceptions' keys")
    if document["schema"] != EXCEPTIONS_SCHEMA:
        raise ExceptionFileError(f"{path}: unsupported schema {document['schema']!r}")
    entries = document["exceptions"]
    if not isinstance(entries, list):
        raise ExceptionFileError(f"{path}: 'exceptions' must be a list")
    exceptions: list[AdvisoryException] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ExceptionFileError(f"{path}: every exception must be an object")
        exception = _parse_exception(entry)
        if exception.identity in seen:
            raise ExceptionFileError(f"{path}: duplicate exception {exception.identity}")
        seen.add(exception.identity)
        exceptions.append(exception)
    return tuple(exceptions)


def _match_exception(
    exceptions: Sequence[AdvisoryException], finding: Finding, vuln: Vulnerability
) -> AdvisoryException | None:
    identifiers = {vuln.id, *vuln.aliases}
    for exception in exceptions:
        if (
            exception.package == finding.component.coordinate
            and exception.version == finding.component.version
            and exception.id in identifiers
        ):
            return exception
    return None


def _scope_failures(
    exception: AdvisoryException, components: Mapping[str, LockedComponent]
) -> tuple[str, ...]:
    component = components.get(f"{exception.package}:{exception.version}")
    if component is None:
        return (f"exception {exception.identity} references a component absent from the lockfile",)
    extra = tuple(sorted(set(exception.configurations) - set(component.configurations)))
    if extra:
        return (
            f"exception {exception.identity} claims configurations {list(extra)} that "
            f"{component.key} is not locked into",
        )
    return ()


def _annotate(
    findings: Sequence[Finding], exceptions: Sequence[AdvisoryException]
) -> tuple[tuple[Finding, ...], tuple[str, ...], set[str]]:
    annotated: list[Finding] = []
    failures: list[str] = []
    used: set[str] = set()
    for finding in findings:
        vulns: list[Vulnerability] = []
        for vuln in finding.vulns:
            exception = _match_exception(exceptions, finding, vuln)
            if exception is None:
                failures.append(
                    f"unexcepted advisory {vuln.id} affects {finding.component.key} "
                    f"on {list(finding.component.configurations)}"
                )
                vulns.append(vuln)
                continue
            used.add(exception.identity)
            unobserved = tuple(sorted(set(exception.aliases) - set(vuln.aliases)))
            if unobserved:
                failures.append(
                    f"exception {exception.identity} declares aliases {list(unobserved)} "
                    f"that OSV does not list for {vuln.id}"
                )
            vulns.append(replace(vuln, excepted=True, exception_id=exception.id))
        annotated.append(replace(finding, vulns=tuple(vulns)))
    return tuple(annotated), tuple(failures), used


def evaluate(
    findings: Sequence[Finding],
    exceptions: Sequence[AdvisoryException],
    components: Sequence[LockedComponent],
    today: date,
) -> ScanOutcome:
    """Apply exceptions and collect every fail-closed reason the scan found."""

    annotated, failures, used = _annotate(findings, exceptions)
    by_key = {component.key: component for component in components}
    residual: list[str] = list(failures)
    for exception in exceptions:
        if today > exception.expires:
            residual.append(
                f"exception {exception.identity} expired on {exception.expires.isoformat()}"
            )
        if exception.identity not in used:
            residual.append(f"stale exception {exception.identity} matched no advisory; remove it")
        residual.extend(_scope_failures(exception, by_key))
    return ScanOutcome(findings=annotated, failures=tuple(sorted(residual)))


def _vuln_payload(vuln: Vulnerability) -> dict[str, Any]:
    return {
        "aliases": list(vuln.aliases),
        "excepted": vuln.excepted,
        "exception_id": vuln.exception_id or None,
        "fixed_versions": list(vuln.fixed_versions),
        "id": vuln.id,
        "modified": vuln.modified,
        "severity": {"cvss": list(vuln.severity_cvss), "database": vuln.severity_database},
        "summary": vuln.summary,
    }


def build_report(
    *,
    lockfile_sha256: str,
    exceptions_sha256: str,
    components: Sequence[LockedComponent],
    unscannable: Sequence[tuple[LockedComponent, Classification]],
    outcome: ScanOutcome,
) -> dict[str, Any]:
    """Assemble the deterministic report body; no wall-clock value enters it."""

    return {
        "components": {
            "queryable": len(components) - len(unscannable),
            "total": len(components),
            "unscannable": len(unscannable),
        },
        "exceptions_sha256": exceptions_sha256,
        "findings": [
            {
                "configurations": list(finding.component.configurations),
                "excepted": finding.excepted,
                "exception_ids": sorted(
                    {vuln.exception_id for vuln in finding.vulns if vuln.exception_id}
                ),
                "package": finding.component.coordinate,
                "version": finding.component.version,
                "vulns": [_vuln_payload(vuln) for vuln in finding.vulns],
            }
            for finding in outcome.findings
        ],
        "lockfile_sha256": lockfile_sha256,
        "osv_endpoint": OSV_ENDPOINT,
        "outcome": {"failures": list(outcome.failures), "status": outcome.status},
        "schema": REPORT_SCHEMA,
        "unscannable": [
            {
                "configurations": list(component.configurations),
                "package": component.coordinate,
                "reason_code": classification.reason_code,
                "reason_text": classification.reason_text,
                "version": component.version,
            }
            for component, classification in unscannable
        ],
    }


def canonical_json(document: Mapping[str, Any]) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace."""

    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def build_receipt(report: Mapping[str, Any], *, scan_date: date) -> dict[str, Any]:
    """Bind the report bytes to counts and provenance for the evidence record."""

    findings = report["findings"]
    advisories = sum(len(finding["vulns"]) for finding in findings)
    excepted = sum(1 for finding in findings for vuln in finding["vulns"] if vuln["excepted"])
    return {
        "advisories": {
            "components_with_advisories": len(findings),
            "excepted": excepted,
            "total": advisories,
        },
        "components": dict(report["components"]),
        "exceptions_sha256": report["exceptions_sha256"],
        "lockfile_sha256": report["lockfile_sha256"],
        "osv_endpoint": OSV_ENDPOINT,
        "outcome_status": report["outcome"]["status"],
        "report_sha256": sha256(canonical_json(report).encode("utf-8")).hexdigest(),
        "scan_date": scan_date.isoformat(),
        "schema": RECEIPT_SCHEMA,
    }


def write_json(path: Path, document: Mapping[str, Any]) -> str:
    """Write canonical JSON and return the SHA-256 of the exact bytes written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(document).encode("utf-8")
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _restore_findings(
    saved: Mapping[str, Any], components: Mapping[str, LockedComponent]
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for entry in saved.get("findings", ()):
        key = f"{entry['package']}:{entry['version']}"
        component = components.get(key)
        if component is None:
            raise LockfileError(f"saved report references {key}, absent from the lockfile")
        vulns = tuple(
            Vulnerability(
                id=vuln["id"],
                aliases=tuple(vuln["aliases"]),
                summary=vuln["summary"],
                severity_cvss=tuple(vuln["severity"]["cvss"]),
                severity_database=vuln["severity"]["database"],
                fixed_versions=tuple(vuln["fixed_versions"]),
                modified=vuln["modified"],
            )
            for vuln in entry["vulns"]
        )
        findings.append(Finding(component=component, vulns=vulns))
    return tuple(findings)


def run_scan(
    *,
    lockfile: Path,
    exceptions_path: Path,
    fetcher: Fetcher | None,
    batch_size: int,
    today: date,
    reevaluate: Path | None = None,
) -> tuple[dict[str, Any], ScanOutcome]:
    """Classify, query (or replay), evaluate, and return the report body."""

    components = parse_lockfile(lockfile)
    classifications = tuple((component, classify(component)) for component in components)
    unscannable = tuple(pair for pair in classifications if not pair[1].queryable)
    queryable = tuple(component for component, verdict in classifications if verdict.queryable)
    exceptions = load_exceptions(exceptions_path)
    if reevaluate is not None:
        saved = json.loads(reevaluate.read_text(encoding="utf-8"))
        raw = _restore_findings(saved, {component.key: component for component in components})
    elif fetcher is None:
        raise OsvUnavailableError("no fetcher was supplied and no saved report was replayed")
    else:
        raw = collect_findings(queryable, fetcher, batch_size=batch_size)
    outcome = evaluate(raw, exceptions, components, today)
    report = build_report(
        lockfile_sha256=sha256(lockfile.read_bytes()).hexdigest(),
        exceptions_sha256=sha256(exceptions_path.read_bytes()).hexdigest(),
        components=components,
        unscannable=unscannable,
        outcome=outcome,
    )
    return report, outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan mod/gradle.lockfile against OSV.dev, fail-closed."
    )
    parser.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--reevaluate",
        type=Path,
        default=None,
        help="re-apply exceptions to a saved report without contacting OSV",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser


def _summarize(report: Mapping[str, Any], receipt: Mapping[str, Any]) -> str:
    counts = report["components"]
    advisories = receipt["advisories"]
    return (
        f"jvm-advisory-scan {report['outcome']['status']}: "
        f"{counts['queryable']} Central-resolved components scanned against OSV; "
        f"{counts['unscannable']} components (Mojang-repo, Fabric-maven, and Loom-generated "
        "coordinates) are unscannable by OSV and are inventoried, not cleared; "
        f"{advisories['total']} advisories across "
        f"{advisories['components_with_advisories']} components "
        f"({advisories['excepted']} excepted) "
        f"report_sha256={receipt['report_sha256']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    today = datetime.now(UTC).date()
    try:
        report, outcome = run_scan(
            lockfile=args.lockfile,
            exceptions_path=args.exceptions,
            fetcher=HttpFetcher(timeout=args.timeout, retries=args.retries),
            batch_size=args.batch_size,
            today=today,
            reevaluate=args.reevaluate,
        )
        receipt = build_receipt(report, scan_date=today)
        write_json(args.output, report)
        write_json(args.receipt, receipt)
    # ValueError covers LockfileError, UnknownGroupError, ExceptionFileError, and
    # json.JSONDecodeError; OSError covers unreadable inputs and unwritable outputs.
    except (OSError, OsvUnavailableError, ValueError) as error:
        print(f"jvm-advisory-scan error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(_summarize(report, receipt))
    for failure in outcome.failures:
        print(f"  failure: {failure}", file=sys.stderr)
    return 0 if outcome.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
