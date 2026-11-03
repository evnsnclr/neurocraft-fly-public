#!/usr/bin/env python3
"""Build the JVM CycloneDX SBOM and NOTICE inventory from locked Gradle build state.

Inputs are the deterministic graph emitted by the ``emitJvmDependencyGraph`` Gradle
task, the tracked dependency lockfile, the tracked dependency-verification metadata
(the sole source of artifact SHA-256 values), and a curated license-override policy.
Every output is byte-deterministic and every policy failure is fatal.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

GRAPH_SCHEMA: Final = "neurocraft.jvm-dependency-graph.v1"
OVERRIDES_SCHEMA: Final = "neurocraft.jvm-license-overrides.v1"
CDX_SCHEMA_URL: Final = "http://cyclonedx.org/schema/bom-1.6.schema.json"
ROOT_BOM_REF: Final = "root-component"
ROOT_NAME: Final = "neurocraft-fly-mod"
VERIFICATION_NS: Final = "{https://schema.gradle.org/dependency-verification}"

# Loom publishes byte-remapped copies of Fabric API and split Minecraft JARs under
# synthetic coordinates. They are the same logical components as their upstream
# originals, so they are folded before anything is classified or reported.
LOOM_REMAP_GROUP_PREFIX: Final = "loom_mappings_"
LOOM_REMAP_UPSTREAM_GROUP: Final = "net.fabricmc.fabric-api"
LOOM_REMAP_SUFFIXES: Final = ("-common", "-client")
LOOM_MAPPINGS_GROUP: Final = "loom"
LOOM_MAPPINGS_NAME: Final = "mappings"
MINECRAFT_GROUP: Final = "net.minecraft"
MINECRAFT_SPLIT_PREFIX: Final = "minecraft-"
MINECRAFT_LOGICAL_NAME: Final = "minecraft"
MAPPINGS_LOGICAL_NAME: Final = "mojang-mappings"

# Loom's bundling bucket. It must stay empty: the distributed mod JAR embeds no
# third-party component, which is the boundary the whole inventory rests on.
BUNDLING_CONFIGURATIONS: Final = frozenset({"include", "includeInternal", "shadow", "shade"})
REQUIRED_EMPTY_CONFIGURATION: Final = "includeInternal"

# Scope is the first matching row, so the order below is the precedence order.
SCOPE_RULES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("bundled", tuple(sorted(BUNDLING_CONFIGURATIONS))),
    (
        "game-runtime",
        (
            "loaderLibraries",
            "minecraftClientLibraries",
            "minecraftClientOnlyNamedCompile",
            "minecraftClientOnlyNamedRuntime",
            "minecraftClientRuntimeLibraries",
            "minecraftCommonNamedCompile",
            "minecraftCommonNamedRuntime",
            "minecraftLibraries",
            "minecraftNatives",
            "minecraftRuntimeLibraries",
            "minecraftServerLibraries",
            "minecraftServerRuntimeLibraries",
            "minecraftTestClientRuntimeLibraries",
        ),
    ),
    (
        "mod-api",
        (
            "modCompileClasspath",
            "modCompileClasspathClientMapped",
            "modCompileClasspathGametestMapped",
            "modCompileClasspathMainMapped",
            "modCompileClasspathMapped",
            "modImplementation",
            "modRuntimeClasspathClientMapped",
            "modRuntimeClasspathGametestMapped",
            "modRuntimeClasspathMainMapped",
        ),
    ),
    ("dev-runtime", ("loomDevelopmentDependencies",)),
    (
        "build-tooling",
        (
            "annotationProcessor",
            "cfrDecompilerClasspath",
            "clientAnnotationProcessor",
            "fernflowerDecompilerClasspath",
            "gametestAnnotationProcessor",
            "mappings",
            "testAnnotationProcessor",
            "vineflowerDecompilerClasspath",
        ),
    ),
    ("build-plugin", ("buildscriptClasspath",)),
    (
        "test",
        (
            "gametestCompileClasspath",
            "gametestRuntimeClasspath",
            "testCompileClasspath",
            "testRuntimeClasspath",
        ),
    ),
    (
        "project-compile",
        (
            "clientCompileClasspath",
            "clientRuntimeClasspath",
            "compileClasspath",
            "runtimeClasspath",
        ),
    ),
)
ALL_SCOPES: Final = frozenset(name for name, _ in SCOPE_RULES)
KNOWN_CONFIGURATIONS: Final = frozenset(name for _, names in SCOPE_RULES for name in names)

# Exact POM <name> strings that map to one SPDX identifier without judgement.
# Version-ambiguous names ("The BSD License", "GNU Lesser General Public License")
# are deliberately absent so that they fail closed into the curated override file.
LICENSE_NAME_TO_SPDX: Final[Mapping[str, str]] = {
    "apache 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "eclipse public license v2.0": "EPL-2.0",
    "eclipse public license, version 2.0": "EPL-2.0",
    "lgpl-2.1-or-later": "LGPL-2.1-or-later",
    "mit license": "MIT",
    "spdx-license-identifier: mit": "MIT",
    "the apache license, version 2.0": "Apache-2.0",
    "the apache software license, version 2.0": "Apache-2.0",
    "the mit license": "MIT",
    "unicode-3.0": "Unicode-3.0",
}

# CycloneDX only accepts SPDX enum values in license.id, so LicenseRef identifiers
# are emitted as named licenses with the terms URL that governs them.
LICENSE_REFS: Final[Mapping[str, tuple[str, str]]] = {
    "LicenseRef-Minecraft-EULA": (
        "Minecraft End User License Agreement (proprietary, not redistributable)",
        "https://www.minecraft.net/en-us/eula",
    ),
    "LicenseRef-Mojang-Mappings": (
        "Mojang official obfuscation-mapping terms (not redistributable)",
        "https://www.minecraft.net/en-us/usage-guidelines",
    ),
    "LicenseRef-Mojang-Proprietary": (
        "Mojang-published library with no declared license "
        "(Minecraft EULA and Usage Guidelines apply)",
        "https://www.minecraft.net/en-us/usage-guidelines",
    ),
}

# Fail-closed allowlist: SPDX identifier -> scopes in which it may appear.
LICENSE_POLICY: Final[Mapping[str, frozenset[str]]] = {
    "0BSD": ALL_SCOPES,
    "Apache-2.0": ALL_SCOPES,
    "BSD-2-Clause": ALL_SCOPES,
    "BSD-3-Clause": ALL_SCOPES,
    "CC0-1.0": ALL_SCOPES,
    "EPL-2.0": frozenset({"build-plugin", "test"}),
    "ISC": ALL_SCOPES,
    "LGPL-2.0-or-later": frozenset({"game-runtime"}),
    "LGPL-2.1-or-later": frozenset({"game-runtime"}),
    "LGPL-3.0-or-later": frozenset({"build-plugin"}),
    "LicenseRef-Minecraft-EULA": frozenset({"game-runtime"}),
    "LicenseRef-Mojang-Mappings": frozenset({"build-tooling"}),
    "LicenseRef-Mojang-Proprietary": frozenset({"game-runtime"}),
    "MIT": ALL_SCOPES,
    "Unicode-3.0": ALL_SCOPES,
}
STRONG_COPYLEFT_PREFIXES: Final = ("GPL-", "AGPL-", "SSPL")
NON_REDISTRIBUTABLE: Final = frozenset(LICENSE_REFS)
COPYLEFT: Final = frozenset(
    identifier for identifier in LICENSE_POLICY if identifier.startswith("LGPL-")
)

# Only groups that are genuinely absent from Maven Central carry a purl
# repository_url qualifier; everything else resolves at the purl-maven default.
REPOSITORY_URLS: Final[tuple[tuple[str, str], ...]] = (
    ("com.mojang", "https://libraries.minecraft.net"),
    ("fabric-loom", "https://maven.fabricmc.net"),
    ("net.fabricmc", "https://maven.fabricmc.net"),
)
SCOPE_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "build-plugin": "Gradle plugin classpath (Loom and its build-time libraries)",
    "build-tooling": "mappings, decompilers, and annotation processors",
    "bundled": "embedded in the distributed mod JAR (must be empty)",
    "dev-runtime": "Loom development launch environment only",
    "game-runtime": "loaded by the user's own Minecraft/Fabric runtime",
    "mod-api": "compiled against and remapped by Loom (Fabric Loader and Fabric API)",
    "project-compile": "project compile/runtime classpath only",
    "test": "JUnit and GameTest classpaths only",
}


class InventoryError(RuntimeError):
    """A fail-closed input, consistency, or license-policy violation."""


@dataclass(frozen=True)
class LicenseOverride:
    """One curated license conclusion with its human-reviewed evidence."""

    coordinate: str
    spdx: str
    evidence_url: str
    reason: str


@dataclass(frozen=True)
class ArtifactHashes:
    """SHA-256 values recorded for one component in verification-metadata.xml.

    ``derived`` carries ``(coordinate, sha256)`` pairs for Loom-produced artifacts
    that were folded into this logical component. They are never promoted into the
    CycloneDX ``hashes`` field, which must describe the component's own upstream
    artifact and nothing else.
    """

    jar_sha256: str | None
    pom_sha256: str | None
    classifier_artifacts: tuple[str, ...]
    derived: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Component:
    """One logical component after Loom folding, scope, and license resolution."""

    group: str
    name: str
    version: str
    scope: str
    spdx: str
    acknowledgement: str
    configurations: tuple[str, ...]
    depends_on: tuple[str, ...]
    source_coordinates: tuple[str, ...]
    hashes: ArtifactHashes
    homepage: str | None
    organization: str | None
    evidence_url: str | None
    loom_remapped: bool
    loom_synthetic: bool
    os_conditional: bool

    @property
    def coordinate(self) -> str:
        return f"{self.group}:{self.name}:{self.version}"

    @property
    def source_url(self) -> str | None:
        return self.homepage or self.evidence_url


@dataclass(frozen=True)
class RawComponent:
    """A folded coordinate with its merged Gradle facts, before licensing."""

    configurations: frozenset[str]
    depends_on: frozenset[str]
    source_coordinates: tuple[str, ...]
    loom_remapped: bool
    loom_synthetic: bool


def read_json(path: Path) -> Mapping[str, Any]:
    """Parse a JSON object, failing closed on anything else."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InventoryError(f"{path}: unreadable JSON ({error})") from error
    if not isinstance(payload, dict):
        raise InventoryError(f"{path}: expected a JSON object")
    return payload


def load_graph(path: Path) -> Mapping[str, Any]:
    """Load the Gradle-emitted dependency graph and validate its schema tag."""

    graph = read_json(path)
    if graph.get("schema") != GRAPH_SCHEMA:
        raise InventoryError(f"{path}: expected schema {GRAPH_SCHEMA}, got {graph.get('schema')!r}")
    for key in ("project", "components", "parentPoms", "directDependencies"):
        if key not in graph:
            raise InventoryError(f"{path}: missing required key {key!r}")
    return graph


def load_overrides(path: Path) -> Mapping[str, LicenseOverride]:
    """Load curated license conclusions, requiring every documented field."""

    payload = read_json(path)
    if payload.get("schema") != OVERRIDES_SCHEMA:
        raise InventoryError(
            f"{path}: expected schema {OVERRIDES_SCHEMA}, got {payload.get('schema')!r}"
        )
    entries = payload.get("overrides")
    if not isinstance(entries, dict):
        raise InventoryError(f"{path}: 'overrides' must be a JSON object")
    overrides: dict[str, LicenseOverride] = {}
    for coordinate, entry in entries.items():
        if not isinstance(entry, dict):
            raise InventoryError(f"{path}: override {coordinate!r} must be an object")
        missing = [field for field in ("spdx", "evidence_url", "reason") if not entry.get(field)]
        if missing:
            raise InventoryError(f"{path}: override {coordinate!r} is missing {missing}")
        overrides[coordinate] = LicenseOverride(
            coordinate=coordinate,
            spdx=str(entry["spdx"]),
            evidence_url=str(entry["evidence_url"]),
            reason=str(entry["reason"]),
        )
    return overrides


def load_lockfile(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Return the locked coordinates and the configurations that resolve empty."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InventoryError(f"{path}: unreadable lockfile ({error})") from error
    coordinates: set[str] = set()
    empty: set[str] = set()
    for line in lines:
        if line.startswith("#") or "=" not in line:
            continue
        left, right = line.split("=", 1)
        if left == "empty":
            empty.update(name.strip() for name in right.split(",") if name.strip())
        else:
            coordinates.add(left)
    if not coordinates:
        raise InventoryError(f"{path}: no locked coordinates found")
    return frozenset(coordinates), frozenset(empty)


def load_verification_hashes(path: Path) -> Mapping[str, ArtifactHashes]:
    """Read every recorded artifact SHA-256 from the dependency-verification metadata."""

    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise InventoryError(f"{path}: unreadable verification metadata ({error})") from error
    hashes: dict[str, ArtifactHashes] = {}
    for component in root.iter(f"{VERIFICATION_NS}component"):
        group = component.get("group", "")
        name = component.get("name", "")
        version = component.get("version", "")
        jar_sha256: str | None = None
        pom_sha256: str | None = None
        classifiers: list[str] = []
        for artifact in component.findall(f"{VERIFICATION_NS}artifact"):
            filename = artifact.get("name", "")
            digest = artifact.find(f"{VERIFICATION_NS}sha256")
            value = None if digest is None else digest.get("value")
            if filename == f"{name}-{version}.jar":
                jar_sha256 = value
            elif filename == f"{name}-{version}.pom":
                pom_sha256 = value
            elif filename.endswith(".jar"):
                classifiers.append(filename)
        hashes[f"{group}:{name}:{version}"] = ArtifactHashes(
            jar_sha256=jar_sha256,
            pom_sha256=pom_sha256,
            classifier_artifacts=tuple(sorted(classifiers)),
        )
    if not hashes:
        raise InventoryError(f"{path}: no verification components found")
    return hashes


def fold_coordinate(coordinate: str, minecraft_version: str) -> tuple[str, bool, bool]:
    """Map a resolved coordinate to its logical one, flagging remapped/synthetic origin."""

    group, name, version = coordinate.split(":", 2)
    if group.startswith(LOOM_REMAP_GROUP_PREFIX):
        upstream_group = group.split(".", 1)[-1]
        if upstream_group != LOOM_REMAP_UPSTREAM_GROUP:
            raise InventoryError(f"unrecognised Loom remap group in {coordinate!r}")
        for suffix in LOOM_REMAP_SUFFIXES:
            if name.endswith(suffix):
                return f"{upstream_group}:{name[: -len(suffix)]}:{version}", True, False
        raise InventoryError(f"unrecognised Loom remap artifact name in {coordinate!r}")
    if group == LOOM_MAPPINGS_GROUP and name == LOOM_MAPPINGS_NAME:
        return f"{MINECRAFT_GROUP}:{MAPPINGS_LOGICAL_NAME}:{minecraft_version}", False, True
    if group == MINECRAFT_GROUP and name.startswith(MINECRAFT_SPLIT_PREFIX):
        return f"{MINECRAFT_GROUP}:{MINECRAFT_LOGICAL_NAME}:{minecraft_version}", False, True
    return coordinate, False, False


def fold_components(graph: Mapping[str, Any]) -> Mapping[str, RawComponent]:
    """Merge Loom-derived coordinates into their logical components."""

    minecraft_version = str(graph["project"]["minecraft"])
    raw = dict[str, dict[str, Any]]()
    for coordinate, entry in sorted(graph["components"].items()):
        logical, remapped, synthetic = fold_coordinate(coordinate, minecraft_version)
        bucket = raw.setdefault(
            logical,
            {
                "configurations": set(),
                "depends_on": set(),
                "sources": [],
                "remap": False,
                "synthetic": False,
            },
        )
        bucket["configurations"].update(entry["configurations"])
        bucket["depends_on"].update(entry["dependsOn"])
        bucket["sources"].append(coordinate)
        bucket["remap"] = bool(bucket["remap"]) or remapped
        bucket["synthetic"] = bool(bucket["synthetic"]) or synthetic
    folded: dict[str, RawComponent] = {}
    for logical, bucket in raw.items():
        edges = {fold_coordinate(target, minecraft_version)[0] for target in bucket["depends_on"]}
        folded[logical] = RawComponent(
            configurations=frozenset(bucket["configurations"]),
            depends_on=frozenset(edges - {logical}),
            source_coordinates=tuple(sorted(bucket["sources"])),
            loom_remapped=bool(bucket["remap"]),
            loom_synthetic=bool(bucket["synthetic"]),
        )
    missing = sorted(
        target
        for component in folded.values()
        for target in component.depends_on
        if target not in folded
    )
    if missing:
        raise InventoryError(f"folded graph references unknown components: {missing}")
    return folded


def classify_scope(coordinate: str, configurations: frozenset[str]) -> str:
    """Return the highest-priority scope implied by configuration membership."""

    unknown = sorted(configurations - KNOWN_CONFIGURATIONS)
    if unknown:
        raise InventoryError(f"{coordinate}: unclassified Gradle configurations {unknown}")
    for scope, names in SCOPE_RULES:
        if configurations.intersection(names):
            return scope
    raise InventoryError(f"{coordinate}: no configuration membership to classify")


def normalize_license_names(names: Iterable[str]) -> tuple[frozenset[str], tuple[str, ...]]:
    """Map declared POM license names to SPDX identifiers, keeping the unmapped ones."""

    resolved: set[str] = set()
    unmapped: list[str] = []
    for raw in names:
        key = " ".join(raw.split()).lower()
        identifier = LICENSE_NAME_TO_SPDX.get(key)
        if identifier is None:
            unmapped.append(raw)
        else:
            resolved.add(identifier)
    return frozenset(resolved), tuple(sorted(unmapped))


def find_pom(graph: Mapping[str, Any], coordinate: str) -> Mapping[str, Any] | None:
    """Return the harvested POM for a coordinate, from components or parent POMs."""

    component = graph["components"].get(coordinate)
    if isinstance(component, dict) and isinstance(component.get("pom"), dict):
        pom: Mapping[str, Any] = component["pom"]
        return pom
    parent = graph["parentPoms"].get(coordinate)
    return parent if isinstance(parent, dict) else None


def declared_licenses(
    graph: Mapping[str, Any], coordinate: str, *, max_depth: int = 5
) -> tuple[str, ...]:
    """Collect declared license names, following the POM parent chain when empty."""

    seen: set[str] = set()
    current: str | None = coordinate
    for _ in range(max_depth):
        if current is None or current in seen:
            return ()
        seen.add(current)
        pom = find_pom(graph, current)
        if pom is None:
            return ()
        names = [
            str(entry["name"])
            for entry in pom.get("licenses", [])
            if isinstance(entry, dict) and entry.get("name")
        ]
        if names:
            return tuple(names)
        parent = pom.get("parent")
        current = str(parent) if parent else None
    return ()


def resolve_license(
    graph: Mapping[str, Any],
    component: RawComponent,
    coordinate: str,
    overrides: Mapping[str, LicenseOverride],
) -> tuple[str, str, str | None]:
    """Return (spdx, acknowledgement, evidence_url), preferring a curated override."""

    override = overrides.get(coordinate)
    if override is not None:
        return override.spdx, "concluded", override.evidence_url
    identifiers, unmapped = normalize_license_names(
        name for source in component.source_coordinates for name in declared_licenses(graph, source)
    )
    if unmapped:
        raise InventoryError(
            f"{coordinate}: unrecognised declared license name(s) {list(unmapped)}; "
            "add a curated override entry with evidence"
        )
    if not identifiers:
        raise InventoryError(
            f"{coordinate}: no license is declared in its POM or parent chain; "
            "add a curated override entry with evidence"
        )
    if len(identifiers) > 1:
        raise InventoryError(
            f"{coordinate}: {sorted(identifiers)} are declared together; "
            "add a curated override entry selecting one, with evidence"
        )
    return next(iter(identifiers)), "declared", None


def pom_metadata(
    graph: Mapping[str, Any], component: RawComponent
) -> tuple[str | None, str | None]:
    """Return the first declared project URL and organization across folded sources."""

    homepage: str | None = None
    organization: str | None = None
    for source in component.source_coordinates:
        pom = find_pom(graph, source)
        if pom is None:
            continue
        homepage = homepage or (str(pom["url"]) if pom.get("url") else None)
        organization = organization or (
            str(pom["organization"]) if pom.get("organization") else None
        )
    return homepage, organization


def merge_hashes(
    coordinate: str,
    source_coordinates: Sequence[str],
    hashes: Mapping[str, ArtifactHashes],
) -> ArtifactHashes:
    """Attribute the component's own upstream hashes, keeping Loom copies separate."""

    primary = hashes.get(coordinate, ArtifactHashes(None, None, ()))
    derived = tuple(
        (source, hashes[source].jar_sha256 or "")
        for source in sorted(source_coordinates)
        if source != coordinate and hashes.get(source, ArtifactHashes(None, None, ())).jar_sha256
    )
    return ArtifactHashes(
        jar_sha256=primary.jar_sha256,
        pom_sha256=primary.pom_sha256,
        classifier_artifacts=primary.classifier_artifacts,
        derived=derived,
    )


def build_components(
    graph: Mapping[str, Any],
    folded: Mapping[str, RawComponent],
    hashes: Mapping[str, ArtifactHashes],
    overrides: Mapping[str, LicenseOverride],
) -> tuple[Component, ...]:
    """Resolve scope, license, hashes, and metadata for every logical component."""

    os_ignored = {str(entry) for entry in graph.get("osIgnoredDependencies", [])}
    components: list[Component] = []
    for coordinate in sorted(folded):
        raw = folded[coordinate]
        group, name, version = coordinate.split(":", 2)
        spdx, acknowledgement, evidence = resolve_license(graph, raw, coordinate, overrides)
        homepage, organization = pom_metadata(graph, raw)
        components.append(
            Component(
                group=group,
                name=name,
                version=version,
                scope=classify_scope(coordinate, raw.configurations),
                spdx=spdx,
                acknowledgement=acknowledgement,
                configurations=tuple(sorted(raw.configurations)),
                depends_on=tuple(sorted(raw.depends_on)),
                source_coordinates=raw.source_coordinates,
                hashes=merge_hashes(coordinate, raw.source_coordinates, hashes),
                homepage=homepage,
                organization=organization,
                evidence_url=evidence,
                loom_remapped=raw.loom_remapped,
                loom_synthetic=raw.loom_synthetic,
                os_conditional=f"{group}:{name}" in os_ignored,
            )
        )
    return tuple(components)


def os_conditional_pair_names(graph: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the graph-declared OS-conditional group:name pairs, sorted."""

    return tuple(
        sorted(
            {
                ":".join(str(entry).split(":", 2)[:2])
                for entry in graph.get("osIgnoredDependencies", [])
            }
        )
    )


def normalize_os_conditional_components(
    components: Sequence[Component],
    graph: Mapping[str, Any],
    hashes: Mapping[str, ArtifactHashes],
    overrides: Mapping[str, LicenseOverride],
) -> tuple[Component, ...]:
    """Emit both OS-conditional natives identically on every build platform.

    The two dependency-locking ``ignoredDependencies`` resolve on some host
    platforms and not others, so any platform-varying field (configurations,
    graph edges, POM-derived metadata) would make the tracked inventory drift
    between macOS and Linux builds. Both natives are therefore always emitted,
    built only from platform-independent sources: their verification-metadata
    hash entries and their curated override entries.
    """

    pair_names = os_conditional_pair_names(graph)
    kept = [item for item in components if f"{item.group}:{item.name}" not in pair_names]
    stripped = [
        replace(
            item,
            depends_on=tuple(
                edge
                for edge in item.depends_on
                if ":".join(edge.split(":", 2)[:2]) not in pair_names
            ),
        )
        for item in kept
    ]
    for pair in pair_names:
        matches = sorted(entry for entry in hashes if entry.startswith(f"{pair}:"))
        if len(matches) != 1:
            raise InventoryError(
                f"{pair}: expected exactly one verification-metadata version, found {matches}"
            )
        coordinate = matches[0]
        group, name, version = coordinate.split(":", 2)
        override = overrides.get(coordinate)
        if override is None:
            raise InventoryError(
                f"{coordinate}: OS-conditional natives require a curated override entry "
                "so both build platforms conclude the same license"
            )
        stripped.append(
            Component(
                group=group,
                name=name,
                version=version,
                scope="game-runtime",
                spdx=override.spdx,
                acknowledgement="concluded",
                configurations=(),
                depends_on=(),
                source_coordinates=(coordinate,),
                hashes=merge_hashes(coordinate, (coordinate,), hashes),
                homepage=None,
                organization=None,
                evidence_url=override.evidence_url,
                loom_remapped=False,
                loom_synthetic=False,
                os_conditional=True,
            )
        )
    return tuple(sorted(stripped, key=lambda item: item.coordinate))


def enforce_policy(
    components: Sequence[Component],
    locked: frozenset[str],
    empty_configurations: frozenset[str],
    overrides: Mapping[str, LicenseOverride],
    minecraft_version: str,
) -> None:
    """Fail closed on bundling, coverage, or license-allowlist violations."""

    failures: list[str] = []
    bundled = [item.coordinate for item in components if item.scope == "bundled"]
    if bundled:
        failures.append(f"the distributed mod JAR must bundle nothing, found {bundled}")
    if REQUIRED_EMPTY_CONFIGURATION not in empty_configurations:
        failures.append(
            f"lockfile does not record {REQUIRED_EMPTY_CONFIGURATION!r} as an empty "
            "configuration, so the no-bundling boundary is unproven"
        )
    coordinates = {item.coordinate for item in components}
    sources = {source for item in components for source in item.source_coordinates}
    uncovered = sorted(
        coordinate
        for coordinate in locked
        if coordinate not in sources and coordinate not in coordinates
    )
    if uncovered:
        failures.append(f"locked coordinates absent from the inventory: {uncovered}")
    stale = sorted(set(overrides) - coordinates)
    if stale:
        failures.append(f"override entries match no resolved component: {stale}")
    unverified = sorted(item.coordinate for item in components if not _has_recorded_hash(item))
    if unverified:
        failures.append(f"components with no SHA-256 in verification-metadata.xml: {unverified}")
    for item in components:
        failures.extend(_license_failures(item))
    if failures:
        raise InventoryError(
            f"JVM license policy failed for Minecraft {minecraft_version}:\n  - "
            + "\n  - ".join(failures)
        )


def _has_recorded_hash(item: Component) -> bool:
    """Whether verification-metadata.xml pins at least one artifact for a component."""

    return bool(
        item.hashes.jar_sha256
        or item.hashes.pom_sha256
        or item.hashes.classifier_artifacts
        or item.hashes.derived
    )


def _license_failures(item: Component) -> list[str]:
    """Return the allowlist violations for one component."""

    if any(item.spdx.startswith(prefix) for prefix in STRONG_COPYLEFT_PREFIXES):
        return [f"{item.coordinate}: strong copyleft license {item.spdx} is never permitted"]
    allowed = LICENSE_POLICY.get(item.spdx)
    if allowed is None:
        return [f"{item.coordinate}: license {item.spdx} is not on the allowlist"]
    if item.scope not in allowed:
        return [
            f"{item.coordinate}: license {item.spdx} is permitted only in "
            f"{sorted(allowed)} scope(s), found {item.scope!r}"
        ]
    return []


def purl_for(item: Component) -> str | None:
    """Return the Maven purl, omitted for locally derived Loom artifacts."""

    if item.loom_synthetic:
        return None
    purl = f"pkg:maven/{item.group}/{item.name}@{item.version}"
    for prefix, url in REPOSITORY_URLS:
        if item.group == prefix or item.group.startswith(f"{prefix}."):
            return f"{purl}?repository_url={quote(url, safe='')}"
    return purl


def license_entry(item: Component) -> dict[str, Any]:
    """Render the CycloneDX license object for a component."""

    named = LICENSE_REFS.get(item.spdx)
    if named is None:
        return {"license": {"acknowledgement": item.acknowledgement, "id": item.spdx}}
    title, url = named
    return {
        "license": {
            "acknowledgement": item.acknowledgement,
            "name": f"{item.spdx}: {title}",
            "url": url,
        }
    }


def component_properties(item: Component) -> list[dict[str, str]]:
    """Render the neurocraft:* evidence properties for a component."""

    values: dict[str, str] = {
        "neurocraft:configurations": ",".join(item.configurations),
        "neurocraft:scope": item.scope,
    }
    if item.hashes.pom_sha256:
        values["neurocraft:pom-sha256"] = item.hashes.pom_sha256
    if item.hashes.classifier_artifacts:
        values["neurocraft:classifier-artifacts"] = ",".join(item.hashes.classifier_artifacts)
    if item.hashes.derived:
        values["neurocraft:derived-artifact-sha256"] = ",".join(
            f"{source}={digest}" for source, digest in item.hashes.derived
        )
    if item.loom_remapped:
        values["neurocraft:loom-remapped"] = "true"
    if item.loom_synthetic:
        values["neurocraft:integrity"] = "loom-trusted-artifact"
        values["neurocraft:loom-synthetic"] = "true"
        values["neurocraft:source-coordinates"] = ",".join(item.source_coordinates)
    if item.os_conditional:
        values["neurocraft:os-conditional"] = "true"
    if item.spdx in NON_REDISTRIBUTABLE:
        values["neurocraft:redistributable"] = "false"
    if item.evidence_url:
        values["neurocraft:license-evidence"] = item.evidence_url
    return [{"name": key, "value": values[key]} for key in sorted(values)]


def external_references(item: Component) -> list[dict[str, str]]:
    """Render deduplicated CycloneDX external references for a component."""

    references: dict[tuple[str, str], None] = {}
    if item.homepage:
        references[("website", item.homepage)] = None
    if item.evidence_url:
        references[("license", item.evidence_url)] = None
    return [
        {"type": kind, "url": url} for kind, url in sorted(references, key=lambda pair: pair[::-1])
    ]


def render_component(item: Component) -> dict[str, Any]:
    """Render one CycloneDX component entry."""

    entry: dict[str, Any] = {
        "bom-ref": item.coordinate,
        "group": item.group,
        "licenses": [license_entry(item)],
        "name": item.name,
        "properties": component_properties(item),
        "type": "library",
        "version": item.version,
    }
    purl = purl_for(item)
    if purl is not None:
        entry["purl"] = purl
    if item.hashes.jar_sha256:
        entry["hashes"] = [{"alg": "SHA-256", "content": item.hashes.jar_sha256}]
    references = external_references(item)
    if references:
        entry["externalReferences"] = references
    if item.organization:
        entry["publisher"] = item.organization
    return entry


def build_sbom(
    graph: Mapping[str, Any], components: Sequence[Component], overrides_path: Path
) -> dict[str, Any]:
    """Assemble the reproducible CycloneDX 1.6 document."""

    project = graph["project"]
    minecraft_version = str(project["minecraft"])
    known = {item.coordinate for item in components}
    direct = sorted(
        {
            fold_coordinate(str(target), minecraft_version)[0]
            for target in graph["directDependencies"]
        }
        & known
    )
    dependencies: list[dict[str, Any]] = [{"dependsOn": direct, "ref": ROOT_BOM_REF}]
    for item in components:
        entry: dict[str, Any] = {"ref": item.coordinate}
        if item.depends_on:
            entry["dependsOn"] = list(item.depends_on)
        dependencies.append(entry)
    return {
        "$schema": CDX_SCHEMA_URL,
        "bomFormat": "CycloneDX",
        "components": [render_component(item) for item in components],
        "dependencies": dependencies,
        "metadata": {
            "component": {
                "bom-ref": ROOT_BOM_REF,
                "description": (
                    "Fabric mod build graph for the connectome-to-Minecraft research "
                    f"controller (Minecraft {minecraft_version}, Fabric Loader "
                    f"{project['loader']}, Fabric API {project['fabricApi']}, "
                    f"Loom {project['loom']})"
                ),
                "group": str(project["group"]),
                "name": ROOT_NAME,
                "type": "application",
                "version": str(project["version"]),
            },
            "properties": [
                {"name": "cdx:reproducible", "value": "true"},
                {"name": "neurocraft:bundled-components", "value": "0"},
                {
                    "name": "neurocraft:locked-coordinates",
                    "value": str(graph["lockedComponentCount"]),
                },
            ],
            "tools": {
                "components": [
                    {
                        "description": (
                            "Repository-owned converter from the locked Gradle graph, "
                            "dependency-verification metadata, and curated license "
                            "overrides to CycloneDX"
                        ),
                        "name": "scripts/build_jvm_inventory.py",
                        "type": "application",
                        "version": "repo",
                    }
                ]
            },
        },
        "specVersion": "1.6",
        "version": 1,
    }


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """Render a GitHub-flavoured Markdown table."""

    def row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return [row(headers), row(["---"] * len(headers)), *(row(cells) for cells in rows)]


def _counts(values: Iterable[str]) -> list[tuple[str, int]]:
    """Count occurrences, returned sorted by key for deterministic output."""

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items())


def _notice_header(graph: Mapping[str, Any], components: Sequence[Component]) -> list[str]:
    """Render the scope statement, regeneration command, and summary tables."""

    project = graph["project"]
    return [
        "# JVM third-party notices (build-time inventory)",
        "",
        "This inventory covers the **entire Gradle build graph** of the Fabric mod: the "
        "Minecraft/Fabric runtime the game itself loads, the Loom plugin classpath, "
        "mappings and decompilers, and the JUnit/GameTest classpaths.",
        "",
        "**Nothing listed here is redistributed.** The project mod JAR "
        "bundles zero third-party "
        "components: `mod/build.gradle` declares no `include`/shadow dependency and Loom's "
        "`includeInternal` bundling configuration resolves empty in `mod/gradle.lockfile`. "
        "Every component below is compiled against, remapped, run in a developer/test JVM, "
        "or fetched by the user's own Gradle and Minecraft caches under their own terms.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "cd mod && ./gradlew --no-daemon --dependency-verification strict \\",
        "    emitJvmDependencyGraph",
        "python3 scripts/build_jvm_inventory.py \\",
        "    --graph mod/build/reports/jvm-dependency-graph.json \\",
        "    --verification mod/gradle/verification-metadata.xml \\",
        "    --lockfile mod/gradle.lockfile \\",
        "    --overrides docs/compliance/jvm-license-overrides.json \\",
        "    --sbom artifacts/sbom/jvm.cdx.json \\",
        "    --notices docs/compliance/JVM_THIRD_PARTY_NOTICES.md",
        "```",
        "",
        "SHA-256 values in the machine-readable SBOM come only from the tracked "
        "`mod/gradle/verification-metadata.xml`; this tool never downloads anything. "
        "A purl carries a `repository_url` qualifier only for groups that are absent "
        "from Maven Central (`com.mojang` on libraries.minecraft.net, `net.fabricmc*` "
        "and `fabric-loom` on maven.fabricmc.net).",
        "",
        "## Build tuple",
        "",
        *_table(
            ["Pin", "Value"],
            [
                ["Minecraft", f"`{project['minecraft']}`"],
                ["Fabric Loader", f"`{project['loader']}`"],
                ["Fabric API", f"`{project['fabricApi']}`"],
                ["Fabric Loom", f"`{project['loom']}`"],
                ["Locked coordinates", str(graph["lockedComponentCount"])],
                [
                    # The raw graph size varies by host platform (the two
                    # OS-conditional natives resolve on different OSes), so the
                    # tracked inventory reports the platform-normalized count.
                    "Resolved coordinates (platform-normalized)",
                    str(
                        len(
                            [
                                key
                                for key in graph["components"]
                                if ":".join(key.split(":", 2)[:2])
                                not in os_conditional_pair_names(graph)
                            ]
                        )
                        + len(os_conditional_pair_names(graph))
                    ),
                ],
                ["Logical components", str(len(components))],
            ],
        ),
        "",
        "Loom folding accounts for the difference: byte-remapped Fabric API copies fold "
        "into their upstream modules and the two split Minecraft JARs fold into one "
        "Minecraft component.",
        "",
        "## Components by scope",
        "",
        *_table(
            ["Scope", "Components", "Meaning"],
            [
                [f"`{scope}`", str(count), SCOPE_DESCRIPTIONS[scope]]
                for scope, count in [
                    *_counts(item.scope for item in components),
                    ("bundled", 0),
                ]
            ],
        ),
        "",
        "## Components by license",
        "",
        *_table(
            ["License", "Components"],
            [
                [f"`{spdx}`", str(count)]
                for spdx, count in _counts(item.spdx for item in components)
            ],
        ),
    ]


def _notice_sections(components: Sequence[Component]) -> list[str]:
    """Render one component table per SPDX identifier."""

    lines: list[str] = ["", "## Component inventory"]
    for spdx, _ in _counts(item.spdx for item in components):
        named = LICENSE_REFS.get(spdx)
        lines += [
            "",
            f"### {spdx}" + (f" \u2014 {named[0]}" if named else ""),
            "",
            *_table(
                ["Component", "Version", "Scope", "Source", "Basis"],
                [
                    [
                        f"`{item.group}:{item.name}`",
                        f"`{item.version}`",
                        f"`{item.scope}`",
                        f"[{item.source_url}]({item.source_url})" if item.source_url else "\u2014",
                        item.acknowledgement,
                    ]
                    for item in components
                    if item.spdx == spdx
                ],
            ),
        ]
    return lines


def _notice_footer(
    components: Sequence[Component], overrides: Mapping[str, LicenseOverride]
) -> list[str]:
    """Render attribution, copyleft, non-redistributable, and evidence sections."""

    return [
        "",
        "## Attribution obligations",
        "",
        "Apache-2.0 components carry NOTICE requirements that attach on redistribution. "
        "This repository redistributes none of them, so this document is the standing "
        "attribution record for the build environment rather than a shipped NOTICE file. "
        "Anyone republishing a built artifact that embeds these components must reproduce "
        "the corresponding upstream NOTICE files; the source URLs above locate them.",
        "",
        "## Copyleft components",
        "",
        *_table(
            ["Component", "License", "Scope", "Why it is acceptable"],
            [
                [
                    f"`{item.coordinate}`",
                    f"`{item.spdx}`",
                    f"`{item.scope}`",
                    overrides[item.coordinate].reason
                    if item.coordinate in overrides
                    else "declared in its own POM",
                ]
                for item in components
                if item.spdx in COPYLEFT
            ],
        ),
        "",
        "## Non-redistributable components",
        "",
        *_table(
            ["Component", "Terms", "Scope"],
            [
                [f"`{item.coordinate}`", f"`{item.spdx}`", f"`{item.scope}`"]
                for item in components
                if item.spdx in NON_REDISTRIBUTABLE
            ],
        ),
        "",
        "These are obtained by the user's own Gradle build and Minecraft launcher into "
        "ignored caches. They are never committed, published, or embedded in a project "
        "artifact.",
        "",
        "## Curated license conclusions",
        "",
        "Each entry below could not be derived from Maven metadata alone. The full "
        "reasoning is tracked in `docs/compliance/jvm-license-overrides.json`.",
        "",
        *_table(
            ["Component", "Concluded", "Evidence"],
            [
                [
                    f"`{coordinate}`",
                    f"`{overrides[coordinate].spdx}`",
                    f"[{overrides[coordinate].evidence_url}]({overrides[coordinate].evidence_url})",
                ]
                for coordinate in sorted(overrides)
            ],
        ),
        "",
        "## Minecraft and Mojang material",
        "",
        "NeuroCraft Fly is not an official Minecraft product and is not approved by or "
        "associated with Mojang or Microsoft. Minecraft, the Mojang-published libraries, "
        "and the official obfuscation mappings are proprietary; follow the current "
        "[Minecraft Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines) "
        "and the applicable EULA.",
    ]


def render_notices(
    graph: Mapping[str, Any],
    components: Sequence[Component],
    overrides: Mapping[str, LicenseOverride],
) -> str:
    """Render the tracked NOTICE/attribution inventory."""

    lines = _notice_header(graph, components)
    lines += _notice_sections(components)
    lines += _notice_footer(components, overrides)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_text(path: Path, text: str) -> int:
    """Write UTF-8 text with LF endings, creating parents, and return the byte count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return len(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the six explicit input/output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--lockfile", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    return parser.parse_args(argv)


def build_inventory(
    graph_path: Path, verification_path: Path, lockfile_path: Path, overrides_path: Path
) -> tuple[Mapping[str, Any], tuple[Component, ...], Mapping[str, LicenseOverride]]:
    """Load every input, fold, classify, and enforce the license policy."""

    graph = load_graph(graph_path)
    overrides = load_overrides(overrides_path)
    locked, empty_configurations = load_lockfile(lockfile_path)
    hashes = load_verification_hashes(verification_path)
    folded = fold_components(graph)
    components = normalize_os_conditional_components(
        build_components(graph, folded, hashes, overrides), graph, hashes, overrides
    )
    enforce_policy(
        components, locked, empty_configurations, overrides, str(graph["project"]["minecraft"])
    )
    return graph, components, overrides


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the SBOM and notices, or report the policy failure and exit nonzero."""

    args = parse_args(argv)
    try:
        graph, components, overrides = build_inventory(
            args.graph, args.verification, args.lockfile, args.overrides
        )
        sbom = build_sbom(graph, components, args.overrides)
        sbom_bytes = write_text(args.sbom, json.dumps(sbom, indent=2, sort_keys=True) + "\n")
        notices_bytes = write_text(args.notices, render_notices(graph, components, overrides))
    except InventoryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"logical components: {len(components)}")
    for scope, count in _counts(item.scope for item in components):
        print(f"  scope {scope}: {count}")
    for spdx, count in _counts(item.spdx for item in components):
        print(f"  license {spdx}: {count}")
    print(f"{args.sbom}: {sbom_bytes} bytes")
    print(f"{args.notices}: {notices_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
