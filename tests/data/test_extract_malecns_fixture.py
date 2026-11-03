from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.feather as feather


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import extract_malecns_fixture as extractor  # noqa: E402


ANNOTATIONS_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS_FILENAME = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILENAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

ANNOTATION_SCHEMA = pa.schema(
    [
        ("bodyId", pa.int64()),
        ("type", pa.string()),
        ("instance", pa.string()),
        ("rootSide", pa.string()),
        ("somaSide", pa.string()),
        ("superclass", pa.string()),
        ("subclass", pa.string()),
        ("entryNerve", pa.string()),
    ]
)
TRANSMITTER_SCHEMA = pa.schema(
    [
        ("body", pa.int64()),
        ("predicted_nt", pa.string()),
        ("consensus_nt", pa.string()),
        ("ground_truth", pa.string()),
        ("celltype_predicted_nt", pa.string()),
    ]
)
WEIGHT_SCHEMA = pa.schema(
    [
        ("body_pre", pa.int64()),
        ("body_post", pa.int64()),
        ("weight", pa.int64()),
    ]
)


def annotation(
    body_id: int,
    *,
    neuron_type: str | None = None,
    instance: str | None = None,
    root_side: str | None = None,
    soma_side: str | None = None,
    superclass: str | None = None,
    subclass: str | None = None,
    entry_nerve: str | None = None,
) -> dict[str, Any]:
    return {
        "bodyId": body_id,
        "type": neuron_type,
        "instance": instance,
        "rootSide": root_side,
        "somaSide": soma_side,
        "superclass": superclass,
        "subclass": subclass,
        "entryNerve": entry_nerve,
    }


def base_annotations() -> list[dict[str, Any]]:
    return [
        # Side evidence intentionally comes from all three allowed sources.
        annotation(10, neuron_type="JO-E-alpha", instance="JO-E-alpha", root_side="L"),
        annotation(
            11,
            neuron_type="camp-L",
            instance="camp-L",
            root_side="L",
            subclass="campaniform sensilla",
            entry_nerve="ADMN",
        ),
        annotation(19, instance="intermediate-19"),
        annotation(20, instance="intermediate-20"),
        annotation(21, instance="intermediate-21"),
        annotation(30, neuron_type="DNg07", instance="DNg07_L"),
        annotation(31, neuron_type="AN27X008", instance="AN27X008_L"),
        annotation(40, instance="intermediate-40"),
        annotation(
            50,
            instance="motor-50",
            soma_side="L",
            superclass="vnc_motor",
            subclass="wm",
        ),
        annotation(110, neuron_type="JO-E-beta", instance="JO-E-beta", root_side="R"),
        annotation(
            111,
            neuron_type="camp-R",
            instance="camp-R",
            root_side="R",
            subclass="campaniform sensilla",
            entry_nerve="DMetaN",
        ),
        annotation(120, instance="intermediate-120"),
        annotation(130, neuron_type="DNg07", instance="DNg07_R"),
        annotation(131, neuron_type="AN27X008", instance="AN27X008_R"),
        annotation(140, instance="intermediate-140"),
        annotation(
            150,
            instance="motor-150",
            soma_side="R",
            superclass="vnc_motor",
            subclass="wm",
        ),
    ]


def base_weights() -> list[tuple[int, int, int]]:
    return [
        # L maximum-bottleneck route; its first source body ID wins the tie.
        (10, 20, 15),
        (20, 30, 14),
        (30, 40, 17),
        (40, 50, 14),
        # Equal bottleneck but lexicographically later complete tuple.
        (11, 21, 14),
        (21, 30, 14),
        # AN27X008 is complete, but has a lower complete bottleneck.
        (10, 19, 13),
        (19, 31, 13),
        (31, 50, 20),
        # R complete route.
        (110, 120, 16),
        (120, 130, 16),
        (130, 140, 16),
        (140, 150, 16),
        # Induced closure includes 7 and excludes 4.
        (20, 40, 7),
        (40, 20, 4),
    ]


def base_transmitters(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "body": row["bodyId"],
            "predicted_nt": "acetylcholine",
            "consensus_nt": None,
            "ground_truth": None,
            "celltype_predicted_nt": None,
        }
        for row in annotations
        if row["bodyId"] != 10
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class SyntheticSource:
    def __init__(
        self,
        root: Path,
        *,
        annotations: list[dict[str, Any]] | None = None,
        weights: list[tuple[int, int, int]] | None = None,
        transmitters: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = root
        self.cache = root / "cache"
        self.cache.mkdir(parents=True)
        self.annotations = annotations if annotations is not None else base_annotations()
        self.weights = weights if weights is not None else base_weights()
        self.transmitters = (
            transmitters
            if transmitters is not None
            else base_transmitters(self.annotations)
        )
        self.manifest_path = root / "male-cns-v1.0.sources.json"
        self.output = root / "fixture.json"
        self.scratch = root / "scratch"
        self.write_sources()

    def write_sources(self) -> None:
        annotation_path = self.cache / ANNOTATIONS_FILENAME
        transmitter_path = self.cache / TRANSMITTERS_FILENAME
        weight_path = self.cache / WEIGHTS_FILENAME
        feather.write_feather(
            pa.Table.from_pylist(self.annotations, schema=ANNOTATION_SCHEMA),
            annotation_path,
            compression="uncompressed",
        )
        feather.write_feather(
            pa.Table.from_pylist(self.transmitters, schema=TRANSMITTER_SCHEMA),
            transmitter_path,
            compression="uncompressed",
        )
        feather.write_feather(
            pa.Table.from_pylist(
                [
                    {"body_pre": pre, "body_post": post, "weight": weight}
                    for pre, post, weight in self.weights
                ],
                schema=WEIGHT_SCHEMA,
            ),
            weight_path,
            compression="uncompressed",
        )
        self.write_manifest()

    def write_manifest(self) -> None:
        entries = []
        for source_id, filename, generation in (
            ("annotations", ANNOTATIONS_FILENAME, "annotation-generation"),
            ("neurotransmitters", TRANSMITTERS_FILENAME, "transmitter-generation"),
            ("weights", WEIGHTS_FILENAME, "weight-generation"),
        ):
            path = self.cache / filename
            entries.append(
                {
                    "id": source_id,
                    "filename": filename,
                    "url": f"https://example.invalid/{filename}?generation={generation}",
                    "generation": generation,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        manifest = {
            "schema_version": 1,
            "dataset": {
                "id": "male-cns:v1.0",
                "release_date": "2026-06-08",
                "license": "CC BY 4.0",
                "license_spdx": "CC-BY-4.0",
                "citation": {
                    "authors": "Berg et al.",
                    "title": (
                        "Sexual dimorphism in the complete connectome of the "
                        "Drosophila male central nervous system"
                    ),
                    "doi": "10.1101/2025.10.09.680999",
                    "url": "https://doi.org/10.1101/2025.10.09.680999",
                    "version": "bioRxiv v2, 2025-10-30",
                    "status": "preprint",
                },
            },
            "default_cache_dir": "data/cache/male-cns-v1.0",
            "sources": entries,
        }
        self.manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

    def extract(self, output: Path | None = None) -> dict[str, Any]:
        return extractor.extract_fixture(
            self.manifest_path,
            self.cache,
            output or self.output,
            scratch_dir=self.scratch,
            expected_manifest_sha256=sha256(self.manifest_path),
        )


class FixtureExtractionTests(unittest.TestCase):
    def test_repository_manifest_matches_extractor_production_pin(self) -> None:
        self.assertEqual(
            sha256(
                REPOSITORY_ROOT
                / "data"
                / "manifests"
                / "male-cns-v1.0.sources.json"
            ),
            extractor.EXPECTED_MANIFEST_SHA256,
        )

    def test_exact_selection_lineage_closure_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = SyntheticSource(Path(directory))
            document = source.extract()

            self.assertEqual(document["dataset_id"], "male-cns:v1.0")
            self.assertEqual(document["fixture_id"], "mcns-v1.0-flight-power-v1")
            self.assertEqual(
                document["selected_paths"]["L"]["complete_body_id_tuple"],
                [10, 20, 30, 40, 50],
            )
            self.assertEqual(document["selected_paths"]["L"]["bottleneck_weight"], 14)
            self.assertEqual(
                document["selected_paths"]["R"]["complete_body_id_tuple"],
                [110, 120, 130, 140, 150],
            )
            self.assertEqual(document["selected_paths"]["R"]["bottleneck_weight"], 16)

            self.assertEqual(
                [
                    edge["source_row"]
                    for edge in document["selected_paths"]["L"]["upstream"]["edges"]
                ],
                [0, 1],
            )
            self.assertEqual(
                [
                    edge["source_row"]
                    for edge in document["selected_paths"]["L"]["downstream"]["edges"]
                ],
                [2, 3],
            )
            emitted = {
                (edge["body_pre"], edge["body_post"]): (edge["weight"], edge["source_row"])
                for edge in document["edges"]
            }
            self.assertEqual(emitted[(20, 40)], (7, 13))
            self.assertNotIn((40, 20), emitted)
            self.assertEqual(document["counts"]["emitted_edge_rows"], 9)
            self.assertEqual(document["counts"]["duplicate_directed_pairs"], 0)
            self.assertEqual(document["rights"]["license_spdx"], "CC-BY-4.0")
            self.assertEqual(
                document["rights"]["redistribution_decision"],
                "permitted_with_attribution_and_change_notice",
            )
            self.assertIn("selected 10 source neurons", document["rights"]["changes"])
            self.assertIn("all 9 directed source edges", document["rights"]["changes"])

            for side in ("L", "R"):
                counts = document["candidate_resolution"]["by_side"][side]
                self.assertEqual(counts["sensory"]["count"], 2)
                self.assertEqual(counts["DNg07"]["count"], 1)
                self.assertEqual(counts["AN27X008"]["count"], 1)
                self.assertEqual(counts["motor"]["count"], 1)
            self.assertEqual(
                document["candidate_resolution"]["total_unique_role_candidates"], 10
            )

            nodes = {node["body_id"]: node for node in document["nodes"]}
            self.assertEqual(nodes[10]["side"], "L")
            self.assertEqual(nodes[10]["side_evidence"], {"rootSide": "L"})
            self.assertEqual(nodes[30]["side_evidence"], {"instance_suffix": "L"})
            self.assertEqual(nodes[50]["side_evidence"], {"somaSide": "L"})
            self.assertEqual(nodes[10]["transmitter"]["status"], "missing_source_row")
            self.assertTrue(all(edge["sign_status"] == "unknown" for edge in document["edges"]))

            manifest_sha = sha256(source.manifest_path)
            self.assertEqual(document["input_receipt"]["manifest"]["sha256"], manifest_sha)
            self.assertEqual(
                document["input_receipt"]["manifest"]["required_sha256"],
                manifest_sha,
            )
            self.assertEqual(
                [entry["id"] for entry in document["input_receipt"]["sources"]],
                ["annotations", "neurotransmitters", "weights"],
            )
            unhashed = copy.deepcopy(document)
            hashes = unhashed.pop("hashes")
            self.assertEqual(
                hashlib.sha256(canonical_bytes(unhashed)).hexdigest(),
                hashes["payload_sha256"],
            )
            self.assertEqual(json.loads(source.output.read_text()), document)

            second_output = Path(directory) / "fixture-second.json"
            second = source.extract(second_output)
            self.assertEqual(source.output.read_bytes(), second_output.read_bytes())
            self.assertEqual(document, second)

    def test_direction_is_body_pre_to_body_post(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = base_weights()
            reversed_left = [
                (post, pre, weight) if pre < 100 and post < 100 else (pre, post, weight)
                for pre, post, weight in original
            ]
            source = SyntheticSource(Path(directory), weights=reversed_left)
            with self.assertRaisesRegex(
                extractor.CircuitResolutionError,
                "no complete directed sensory->named->motor route on side L",
            ):
                source.extract()

    def test_unclear_transmitter_is_preserved_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_annotations()
            transmitters = base_transmitters(rows)
            target = next(row for row in transmitters if row["body"] == 20)
            target["predicted_nt"] = "unclear"
            source = SyntheticSource(
                Path(directory), annotations=rows, transmitters=transmitters
            )
            document = source.extract()
            nodes = {node["body_id"]: node for node in document["nodes"]}
            self.assertEqual(
                nodes[20]["transmitter"]["status"], "unknown_transmitter"
            )
            self.assertEqual(
                nodes[20]["transmitter"]["record"]["predicted_nt"], "unclear"
            )
            self.assertEqual(nodes[20]["transmitter"]["sign_status"], "unknown")

    def test_four_edge_segment_fails_three_hop_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            right = [edge for edge in base_weights() if edge[0] >= 100 and edge[1] >= 100]
            too_long_left = [
                (10, 18, 20),
                (18, 19, 20),
                (19, 20, 20),
                (20, 30, 20),
                (30, 50, 20),
            ]
            source = SyntheticSource(Path(directory), weights=too_long_left + right)
            with self.assertRaisesRegex(
                extractor.CircuitResolutionError,
                "each segment <= 3 edges",
            ):
                source.extract()

    def test_no_fuzzy_type_substitution_for_missing_named_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_annotations()
            for row in rows:
                if row["type"] == "DNg07":
                    row["type"] = "DNg07_variant"
            source = SyntheticSource(Path(directory), annotations=rows)
            with self.assertRaisesRegex(
                extractor.CircuitResolutionError, "missing exact DNg07 candidate"
            ):
                source.extract()

    def test_side_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_annotations()
            next(row for row in rows if row["bodyId"] == 30)["somaSide"] = "R"
            source = SyntheticSource(Path(directory), annotations=rows)
            with self.assertRaisesRegex(
                extractor.CircuitResolutionError, "side mismatch for body 30"
            ):
                source.extract()

    def test_incomplete_required_side_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_annotations()
            target = next(row for row in rows if row["bodyId"] == 131)
            target["instance"] = "AN27X008"
            target["rootSide"] = None
            target["somaSide"] = None
            source = SyntheticSource(Path(directory), annotations=rows)
            with self.assertRaisesRegex(
                extractor.CircuitResolutionError, "incomplete side for required role body 131"
            ):
                source.extract()

    def test_ambiguous_role_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_annotations()
            target = next(row for row in rows if row["bodyId"] == 10)
            target["superclass"] = "vnc_motor"
            target["subclass"] = "wm"
            source = SyntheticSource(Path(directory), annotations=rows)
            with self.assertRaisesRegex(
                extractor.CircuitResolutionError, "ambiguous role annotation for body 10"
            ):
                source.extract()

    def test_source_hash_mismatch_fails_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = SyntheticSource(Path(directory))
            manifest = json.loads(source.manifest_path.read_text())
            weights = next(item for item in manifest["sources"] if item["id"] == "weights")
            weights["sha256"] = "0" * 64
            source.manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                extractor.SourceVerificationError, "source 'weights' SHA-256 mismatch"
            ):
                source.extract()

    def test_manifest_bytes_must_match_the_supplied_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = SyntheticSource(Path(directory))
            with self.assertRaisesRegex(
                extractor.ManifestError, "source manifest SHA-256 mismatch"
            ):
                extractor.extract_fixture(
                    source.manifest_path,
                    source.cache,
                    source.output,
                    scratch_dir=source.scratch,
                    expected_manifest_sha256="0" * 64,
                )

    def test_duplicate_pair_reports_exact_zero_based_rows_even_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_weights() + [(10, 20, 1)]
            source = SyntheticSource(Path(directory), weights=rows)
            with self.assertRaisesRegex(
                extractor.SourceDataError,
                r"duplicate directed edge \(10, 20\).*\[0, 15\]",
            ):
                source.extract()

    def test_uint64_duplicate_key_range_is_checked_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_weights() + [(2**32, 50, 1)]
            source = SyntheticSource(Path(directory), weights=rows)
            with self.assertRaisesRegex(
                extractor.SourceDataError, "body ID above uint32"
            ):
                source.extract()

    def test_path_search_does_not_select_unannotated_intermediate_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = base_weights() + [(10, 999, 100), (999, 31, 100)]
            source = SyntheticSource(Path(directory), weights=rows)
            document = source.extract()
            self.assertEqual(
                document["selected_paths"]["L"]["complete_body_id_tuple"],
                [10, 20, 30, 40, 50],
            )
            self.assertEqual(document["selected_paths"]["L"]["bottleneck_weight"], 14)

    def test_row_permutation_preserves_semantic_selection_but_updates_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = SyntheticSource(root / "first")
            first_document = first.extract()

            annotations = list(reversed(base_annotations()))
            transmitters = list(reversed(base_transmitters(base_annotations())))
            weights = list(reversed(base_weights()))
            second = SyntheticSource(
                root / "second",
                annotations=annotations,
                transmitters=transmitters,
                weights=weights,
            )
            second_document = second.extract()

            self.assertEqual(
                first_document["hashes"]["semantic_sha256"],
                second_document["hashes"]["semantic_sha256"],
            )
            self.assertNotEqual(
                first_document["hashes"]["payload_sha256"],
                second_document["hashes"]["payload_sha256"],
            )
            self.assertEqual(
                first_document["selected_paths"]["L"]["complete_body_id_tuple"],
                second_document["selected_paths"]["L"]["complete_body_id_tuple"],
            )
            first_edge = next(
                edge
                for edge in first_document["edges"]
                if (edge["body_pre"], edge["body_post"]) == (10, 20)
            )
            second_edge = next(
                edge
                for edge in second_document["edges"]
                if (edge["body_pre"], edge["body_post"]) == (10, 20)
            )
            self.assertEqual(first_edge["source_row"], 0)
            self.assertEqual(second_edge["source_row"], len(weights) - 1)


if __name__ == "__main__":
    unittest.main()
