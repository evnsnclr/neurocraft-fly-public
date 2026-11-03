"""Unit tests for the JVM SBOM/NOTICE converter's folding, licensing, and policy gates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "build_jvm_inventory.py"
REAL_GRAPH = ROOT / "mod" / "build" / "reports" / "jvm-dependency-graph.json"
REAL_OVERRIDES = ROOT / "docs" / "compliance" / "jvm-license-overrides.json"

APACHE = [{"name": "The Apache Software License, Version 2.0", "url": "https://apache.org"}]
REMAP_GROUP = "loom_mappings_1_21_5_layered_hash_2198_v2.net.fabricmc.fabric-api"
MINECRAFT_SPLIT = "1.21.5-loom.mappings.1_21_5.layered+hash.2198-v2"
MAPPED = "modCompileClasspathMapped"
NETTY = "io.netty:netty-buffer:4.1.118.Final"
NETTY_PARENT = "io.netty:netty-parent:4.1.118.Final"


def load_module() -> Any:
    """Load the converter the same way tests/governance loads its script under test."""

    spec = importlib.util.spec_from_file_location("build_jvm_inventory", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Frozen dataclasses resolve their own module during class creation, so the
    # module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def pom(
    licenses: list[dict[str, str]] | None = None,
    *,
    parent: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Build a harvested-POM record in the shape the Gradle task emits."""

    return {
        "name": None,
        "description": None,
        "url": url,
        "organization": None,
        "organizationUrl": None,
        "licenses": licenses or [],
        "parent": parent,
    }


def component(
    configurations: list[str],
    *,
    depends_on: list[str] | None = None,
    component_pom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one resolved-component record."""

    return {
        "configurations": sorted(configurations),
        "dependsOn": sorted(depends_on or []),
        "pom": component_pom,
    }


def graph_payload(
    components: dict[str, Any],
    *,
    parent_poms: dict[str, Any] | None = None,
    direct: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete synthetic dependency graph."""

    return {
        "schema": MODULE.GRAPH_SCHEMA,
        "project": {
            "group": "dev.neurocraft",
            "name": "neurocraft-fly-mod",
            "version": "0.1.0",
            "minecraft": "1.21.5",
            "loader": "0.16.10",
            "fabricApi": "0.128.2+1.21.5",
            "loom": "1.10.5",
        },
        "osIgnoredDependencies": [],
        "lockedComponentCount": len(components),
        "directDependencies": sorted(direct or []),
        "pomUnavailable": [],
        "unresolvedParents": [],
        "parentPoms": parent_poms or {},
        "components": components,
    }


def verification_xml(coordinates: list[str]) -> str:
    """Build verification metadata pinning one JAR and one POM per coordinate."""

    body = []
    for index, coordinate in enumerate(sorted(coordinates)):
        group, name, version = coordinate.split(":", 2)
        digest = f"{index:064x}"
        body.append(
            f'<component group="{group}" name="{name}" version="{version}">'
            f'<artifact name="{name}-{version}.jar"><sha256 value="{digest}"/></artifact>'
            f'<artifact name="{name}-{version}.pom"><sha256 value="{digest}"/></artifact>'
            f"</component>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<verification-metadata xmlns="https://schema.gradle.org/dependency-verification">'
        f"<components>{''.join(body)}</components></verification-metadata>"
    )


def write_inputs(
    tmp_path: Path,
    graph: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
    locked: list[str] | None = None,
    empty: list[str] | None = None,
    pinned: list[str] | None = None,
) -> dict[str, Path]:
    """Materialise the four converter inputs and return their paths."""

    paths = {
        "graph": tmp_path / "graph.json",
        "verification": tmp_path / "verification-metadata.xml",
        "lockfile": tmp_path / "gradle.lockfile",
        "overrides": tmp_path / "overrides.json",
        "sbom": tmp_path / "jvm.cdx.json",
        "notices": tmp_path / "NOTICES.md",
    }
    paths["graph"].write_text(json.dumps(graph), encoding="utf-8")
    pinned_coordinates = pinned if pinned is not None else sorted(graph["components"])
    paths["verification"].write_text(verification_xml(pinned_coordinates), encoding="utf-8")
    lock_lines = ["# generated"]
    for coordinate in sorted(locked if locked is not None else graph["components"]):
        configurations = graph["components"].get(coordinate, {}).get("configurations", ["x"])
        lock_lines.append(f"{coordinate}={','.join(configurations)}")
    empty_names = ["includeInternal"] if empty is None else empty
    lock_lines.append("empty=" + ",".join(empty_names) if empty_names else "empty=")
    paths["lockfile"].write_text("\n".join(lock_lines) + "\n", encoding="utf-8")
    paths["overrides"].write_text(
        json.dumps({"schema": MODULE.OVERRIDES_SCHEMA, "overrides": overrides or {}}),
        encoding="utf-8",
    )
    return paths


def build(paths: dict[str, Path]) -> Any:
    """Run the load/fold/classify/policy pipeline over prepared inputs."""

    return MODULE.build_inventory(
        paths["graph"], paths["verification"], paths["lockfile"], paths["overrides"]
    )


def minimal_graph() -> dict[str, Any]:
    """One Apache-2.0 game-runtime component: the smallest graph that passes policy."""

    return graph_payload(
        {NETTY: component(["minecraftLibraries"], component_pom=pom(APACHE))},
        direct=[NETTY],
    )


def test_minimal_graph_passes_and_emits_a_declared_license(tmp_path: Path) -> None:
    _, components, _ = build(write_inputs(tmp_path, minimal_graph()))
    assert [item.coordinate for item in components] == [NETTY]
    assert components[0].spdx == "Apache-2.0"
    assert components[0].acknowledgement == "declared"
    assert components[0].scope == "game-runtime"
    assert components[0].hashes.jar_sha256 == f"{0:064x}"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("Apache-2.0", "Apache-2.0"),
        ("The Apache Software License, Version 2.0", "Apache-2.0"),
        ("  apache   license,   Version 2.0 ", "Apache-2.0"),
        ("The MIT License", "MIT"),
        ("SPDX-License-Identifier: MIT", "MIT"),
        ("Eclipse Public License v2.0", "EPL-2.0"),
        ("BSD 3-Clause", "BSD-3-Clause"),
    ],
)
def test_license_name_normalization_is_case_and_whitespace_insensitive(
    declared: str, expected: str
) -> None:
    identifiers, unmapped = MODULE.normalize_license_names([declared])
    assert identifiers == frozenset({expected})
    assert unmapped == ()


@pytest.mark.parametrize("declared", ["The BSD License", "GNU Lesser General Public License"])
def test_version_ambiguous_license_names_stay_unmapped(declared: str) -> None:
    identifiers, unmapped = MODULE.normalize_license_names([declared])
    assert identifiers == frozenset()
    assert unmapped == (declared,)


def test_license_is_inherited_through_the_pom_parent_chain(tmp_path: Path) -> None:
    graph = graph_payload(
        {NETTY: component(["minecraftLibraries"], component_pom=pom(parent=NETTY_PARENT))},
        parent_poms={
            NETTY_PARENT: pom(parent="org.sonatype.oss:oss-parent:7"),
            "org.sonatype.oss:oss-parent:7": pom(APACHE),
        },
    )
    _, components, _ = build(write_inputs(tmp_path, graph))
    assert components[0].spdx == "Apache-2.0"
    assert components[0].acknowledgement == "declared"


def test_parent_chain_cycle_does_not_hang_and_fails_closed(tmp_path: Path) -> None:
    graph = graph_payload(
        {"a:b:1": component(["minecraftLibraries"], component_pom=pom(parent="p:q:1"))},
        parent_poms={"p:q:1": pom(parent="a:b:1")},
    )
    with pytest.raises(MODULE.InventoryError, match="no license is declared"):
        build(write_inputs(tmp_path, graph))


def test_loom_remapped_fabric_api_folds_into_its_upstream_component(tmp_path: Path) -> None:
    upstream = "net.fabricmc.fabric-api:fabric-api-base:0.4.62"
    graph = graph_payload(
        {
            upstream: component(["modImplementation"], component_pom=pom(APACHE)),
            f"{REMAP_GROUP}:fabric-api-base-common:0.4.62": component([MAPPED]),
            f"{REMAP_GROUP}:fabric-api-base-client:0.4.62": component([MAPPED]),
        },
    )
    _, components, _ = build(write_inputs(tmp_path, graph))
    assert [item.coordinate for item in components] == [upstream]
    assert components[0].loom_remapped is True
    assert len(components[0].source_coordinates) == 3
    # The upstream JAR hash is the component's own; Loom's copies stay separate evidence.
    assert components[0].hashes.jar_sha256 == f"{2:064x}"
    assert [source for source, _ in components[0].hashes.derived] == [
        f"{REMAP_GROUP}:fabric-api-base-client:0.4.62",
        f"{REMAP_GROUP}:fabric-api-base-common:0.4.62",
    ]


def test_split_minecraft_jars_fold_into_one_non_redistributable_component(
    tmp_path: Path,
) -> None:
    graph = graph_payload(
        {
            f"net.minecraft:minecraft-common-0cc057f080:{MINECRAFT_SPLIT}": component(
                ["minecraftLibraries"]
            ),
            f"net.minecraft:minecraft-clientOnly-0cc057f080:{MINECRAFT_SPLIT}": component(
                ["minecraftClientLibraries"]
            ),
            "loom:mappings:layered+hash.2198": component(["mappings"]),
        },
    )
    overrides = {
        "net.minecraft:minecraft:1.21.5": {
            "spdx": "LicenseRef-Minecraft-EULA",
            "evidence_url": "https://www.minecraft.net/en-us/eula",
            "reason": "proprietary",
        },
        "net.minecraft:mojang-mappings:1.21.5": {
            "spdx": "LicenseRef-Mojang-Mappings",
            "evidence_url": "https://www.minecraft.net/en-us/usage-guidelines",
            "reason": "proprietary",
        },
    }
    _, components, _ = build(write_inputs(tmp_path, graph, overrides=overrides))
    assert [item.coordinate for item in components] == [
        "net.minecraft:minecraft:1.21.5",
        "net.minecraft:mojang-mappings:1.21.5",
    ]
    assert all(item.loom_synthetic for item in components)
    assert all(item.acknowledgement == "concluded" for item in components)
    # Synthetic folds never claim a single upstream JAR hash of their own.
    assert all(item.hashes.jar_sha256 is None for item in components)
    assert MODULE.purl_for(components[0]) is None


def test_scope_precedence_prefers_game_runtime_over_test_membership() -> None:
    scope = MODULE.classify_scope(
        "x:y:1", frozenset({"testRuntimeClasspath", "minecraftLibraries", "compileClasspath"})
    )
    assert scope == "game-runtime"


def test_unknown_gradle_configuration_fails_closed() -> None:
    with pytest.raises(MODULE.InventoryError, match="unclassified Gradle configurations"):
        MODULE.classify_scope("x:y:1", frozenset({"someNewLoomConfiguration"}))


def test_bundled_scope_must_be_empty(tmp_path: Path) -> None:
    graph = graph_payload(
        {"a:b:1": component(["includeInternal", "minecraftLibraries"], component_pom=pom(APACHE))}
    )
    with pytest.raises(MODULE.InventoryError, match="must bundle nothing"):
        build(write_inputs(tmp_path, graph, empty=["localRuntime"]))


def test_missing_empty_bundling_configuration_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(MODULE.InventoryError, match="no-bundling boundary is unproven"):
        build(write_inputs(tmp_path, minimal_graph(), empty=["localRuntime"]))


def test_locked_coordinate_absent_from_the_graph_fails_closed(tmp_path: Path) -> None:
    paths = write_inputs(
        tmp_path,
        minimal_graph(),
        locked=[NETTY, "io.netty:netty-codec:4.1.118.Final"],
    )
    with pytest.raises(MODULE.InventoryError, match="locked coordinates absent"):
        build(paths)


def test_component_without_recorded_verification_hashes_fails_closed(tmp_path: Path) -> None:
    # The metadata is well-formed but pins some other component, so the resolved one
    # would enter the SBOM with no integrity evidence at all.
    with pytest.raises(MODULE.InventoryError, match="no SHA-256 in verification-metadata"):
        build(write_inputs(tmp_path, minimal_graph(), pinned=["org.other:unrelated:1.0"]))


def test_component_without_any_declared_license_requires_an_override(tmp_path: Path) -> None:
    graph = graph_payload({"com.mojang:authlib:6.0.58": component(["minecraftLibraries"])})
    with pytest.raises(MODULE.InventoryError, match="no license is declared"):
        build(write_inputs(tmp_path, graph))


def test_unrecognised_license_name_requires_an_override(tmp_path: Path) -> None:
    graph = graph_payload(
        {
            "org.jcraft:jorbis:0.0.17": component(
                ["minecraftLibraries"],
                component_pom=pom([{"name": "GNU Lesser General Public License", "url": ""}]),
            )
        }
    )
    with pytest.raises(MODULE.InventoryError, match="unrecognised declared license name"):
        build(write_inputs(tmp_path, graph))


def test_multiple_declared_licenses_require_a_curated_selection(tmp_path: Path) -> None:
    graph = graph_payload(
        {
            "net.java.dev.jna:jna:5.15.0": component(
                ["minecraftLibraries"],
                component_pom=pom(
                    [
                        {"name": "LGPL-2.1-or-later", "url": ""},
                        {"name": "Apache-2.0", "url": ""},
                    ]
                ),
            )
        }
    )
    with pytest.raises(MODULE.InventoryError, match="are declared together"):
        build(write_inputs(tmp_path, graph))
    overrides = {
        "net.java.dev.jna:jna:5.15.0": {
            "spdx": "Apache-2.0",
            "evidence_url": "https://github.com/java-native-access/jna/blob/master/LICENSE",
            "reason": "dual licensed; this project elects Apache-2.0",
        }
    }
    _, components, _ = build(write_inputs(tmp_path, graph, overrides=overrides))
    assert (components[0].spdx, components[0].acknowledgement) == ("Apache-2.0", "concluded")


def test_strong_copyleft_is_never_permitted(tmp_path: Path) -> None:
    graph = graph_payload({"a:b:1": component(["minecraftLibraries"])})
    overrides = {
        "a:b:1": {"spdx": "GPL-3.0-only", "evidence_url": "https://example.invalid", "reason": "x"}
    }
    with pytest.raises(MODULE.InventoryError, match="strong copyleft"):
        build(write_inputs(tmp_path, graph, overrides=overrides))


def test_license_outside_the_allowlist_fails_closed(tmp_path: Path) -> None:
    graph = graph_payload({"a:b:1": component(["minecraftLibraries"])})
    overrides = {
        "a:b:1": {"spdx": "MPL-2.0", "evidence_url": "https://example.invalid", "reason": "x"}
    }
    with pytest.raises(MODULE.InventoryError, match="is not on the allowlist"):
        build(write_inputs(tmp_path, graph, overrides=overrides))


@pytest.mark.parametrize(
    ("spdx", "configuration"),
    [
        ("EPL-2.0", "minecraftLibraries"),
        ("LGPL-3.0-or-later", "minecraftLibraries"),
        ("LGPL-2.0-or-later", "buildscriptClasspath"),
        ("LicenseRef-Mojang-Proprietary", "testRuntimeClasspath"),
    ],
)
def test_restricted_licenses_are_rejected_outside_their_scope(
    tmp_path: Path, spdx: str, configuration: str
) -> None:
    graph = graph_payload({"a:b:1": component([configuration])})
    overrides = {"a:b:1": {"spdx": spdx, "evidence_url": "https://example.invalid", "reason": "x"}}
    with pytest.raises(MODULE.InventoryError, match="is permitted only in"):
        build(write_inputs(tmp_path, graph, overrides=overrides))


def test_override_matching_no_component_is_stale_and_fails_closed(tmp_path: Path) -> None:
    overrides = {
        "gone:away:1": {"spdx": "MIT", "evidence_url": "https://example.invalid", "reason": "x"}
    }
    with pytest.raises(MODULE.InventoryError, match="match no resolved component"):
        build(write_inputs(tmp_path, minimal_graph(), overrides=overrides))


def test_incomplete_override_entry_is_rejected(tmp_path: Path) -> None:
    overrides = {"a:b:1": {"spdx": "MIT", "reason": "x"}}
    with pytest.raises(MODULE.InventoryError, match="is missing"):
        build(write_inputs(tmp_path, minimal_graph(), overrides=overrides))


def test_graph_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    graph = minimal_graph()
    graph["schema"] = "neurocraft.jvm-dependency-graph.v0"
    with pytest.raises(MODULE.InventoryError, match="expected schema"):
        build(write_inputs(tmp_path, graph))


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("com.mojang", "?repository_url=https%3A%2F%2Flibraries.minecraft.net"),
        ("net.fabricmc.fabric-api", "?repository_url=https%3A%2F%2Fmaven.fabricmc.net"),
        ("org.lwjgl", ""),
    ],
)
def test_purl_carries_a_repository_qualifier_only_off_maven_central(
    group: str, expected: str
) -> None:
    item = MODULE.Component(
        group=group,
        name="thing",
        version="1.0",
        scope="game-runtime",
        spdx="MIT",
        acknowledgement="declared",
        configurations=(),
        depends_on=(),
        source_coordinates=(),
        hashes=MODULE.ArtifactHashes(None, None, ()),
        homepage=None,
        organization=None,
        evidence_url=None,
        loom_remapped=False,
        loom_synthetic=False,
        os_conditional=False,
    )
    assert MODULE.purl_for(item) == f"pkg:maven/{group}/thing@1.0{expected}"


def test_outputs_are_byte_identical_across_runs(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, minimal_graph())
    arguments = [
        "--graph",
        str(paths["graph"]),
        "--verification",
        str(paths["verification"]),
        "--lockfile",
        str(paths["lockfile"]),
        "--overrides",
        str(paths["overrides"]),
        "--sbom",
        str(paths["sbom"]),
        "--notices",
        str(paths["notices"]),
    ]
    assert MODULE.main(arguments) == 0
    first = (paths["sbom"].read_bytes(), paths["notices"].read_bytes())
    assert MODULE.main(arguments) == 0
    assert (paths["sbom"].read_bytes(), paths["notices"].read_bytes()) == first
    assert first[0].endswith(b"\n") and first[1].endswith(b"\n")
    document = json.loads(first[0])
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.6"
    assert "serialNumber" not in document
    assert "timestamp" not in document.get("metadata", {})
    assert document["dependencies"][0] == {
        "dependsOn": [NETTY],
        "ref": "root-component",
    }


def test_main_reports_policy_failure_with_a_nonzero_exit(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, graph_payload({"a:b:1": component(["minecraftLibraries"])}))
    assert (
        MODULE.main(
            [
                "--graph",
                str(paths["graph"]),
                "--verification",
                str(paths["verification"]),
                "--lockfile",
                str(paths["lockfile"]),
                "--overrides",
                str(paths["overrides"]),
                "--sbom",
                str(paths["sbom"]),
                "--notices",
                str(paths["notices"]),
            ]
        )
        == 1
    )
    assert not paths["sbom"].exists()


def test_tracked_overrides_file_is_well_formed_and_policy_compatible() -> None:
    overrides = MODULE.load_overrides(REAL_OVERRIDES)
    assert overrides, "the tracked override policy must not be empty"
    for coordinate, entry in overrides.items():
        assert coordinate.count(":") >= 2, coordinate
        assert entry.spdx in MODULE.LICENSE_POLICY, (coordinate, entry.spdx)
        assert entry.evidence_url.startswith("https://"), coordinate
        assert len(entry.reason) > 40, coordinate


@pytest.mark.skipif(not REAL_GRAPH.is_file(), reason="requires a Gradle emitJvmDependencyGraph run")
def test_real_emitted_graph_satisfies_the_license_policy() -> None:
    _, components, _ = MODULE.build_inventory(
        REAL_GRAPH,
        ROOT / "mod" / "gradle" / "verification-metadata.xml",
        ROOT / "mod" / "gradle.lockfile",
        REAL_OVERRIDES,
    )
    assert not [item for item in components if item.scope == "bundled"]
    assert {item.scope for item in components} <= MODULE.ALL_SCOPES
    assert {item.spdx for item in components} <= set(MODULE.LICENSE_POLICY)
