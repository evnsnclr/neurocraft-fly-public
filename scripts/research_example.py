#!/usr/bin/env python3
"""Regenerate and verify a small source-derived, synthetic-input research example.

Uses the existing Python reference engine. No Minecraft process, full-connectome
artifact, learning, body controller, or external download participates.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_neural import DynamicsConfig, NeuralEngine, SignOverlay, Stimulation, WeightOverlay

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/fixtures/mcns-v1.0-flight-power-v1.json"
DEFAULT_DATASET = ROOT / "research/example-v1"
CONDITIONS = ("original", "shuffled_weights", "no_input")
POOLS = {
    "sensory_L": (805722,),
    "sensory_R": (77185,),
    "DNg07_L": (27143,),
    "DNg07_R": (28949,),
    "hg1_motor_L": (800241,),
    "hg1_motor_R": (800917,),
}
TARGETS = (805722, 77185)
STEPS = 100
ABS_TOL = 1e-12
REL_TOL = 1e-10
SOFTWARE_FILES = (
    "scripts/research_example.py",
    "packages/connectome-data/src/neurocraft_connectome/loader.py",
    "packages/connectome-data/src/neurocraft_connectome/model.py",
    "packages/neural-runtime/src/neurocraft_neural/engine.py",
    "packages/neural-runtime/src/neurocraft_neural/model.py",
    "packages/neural-runtime/src/neurocraft_neural/adapters.py",
)
COLUMNS = {
    "nodes": (
        "dense_index",
        "body_id",
        "type_name",
        "instance",
        "side",
        "roles",
        "superclass",
        "consensus_nt",
        "annotation_source_row",
        "transmitter_source_row",
        "canonical_sign_status",
    ),
    "edges": (
        "condition",
        "edge_index",
        "body_pre",
        "body_post",
        "source_row",
        "raw_count",
        "canonical_sign_status",
        "assigned_sign",
        "sign_provenance",
        "overlay_count",
        "overlay_source_edge_index",
        "effective_weight",
        "delay_steps",
    ),
    "inputs": (
        "condition",
        "step_before",
        "time_ms",
        "body_id",
        "requested_drive",
        "applied_drive",
    ),
    "activity": ("condition", "step", "time_ms", "body_id", "activity"),
    "readouts": ("condition", "step", "time_ms", "pool", "mean_activity"),
}
FLOAT_COLUMNS = {
    "edges": {"effective_weight"},
    "inputs": {"requested_drive", "applied_drive"},
    "activity": {"activity"},
    "readouts": {"mean_activity"},
    "nodes": set(),
}
DATASET_README = """# Synthetic pulses through a MaleCNS fixture

This is a generated, headless teaching dataset: 14 source-derived neurons and
22 directed edges from the pinned MaleCNS v1.0 flight-power fixture. It is not a
whole-brain experiment, an escape-circuit experiment, a learned policy, a
Minecraft session, or a biological measurement. Signed activity is dimensionless
model activity, not firing rate. Negative values are permitted.

From the repository root, after the documented environment setup:

```sh
uv run --locked python scripts/research_example.py verify
uv run --locked python scripts/research_example.py generate --output artifacts/my-example
uv run --locked python scripts/research_example.py verify --dataset artifacts/my-example
```

Generation refuses an existing destination. The checked-in example is preserved;
use a new directory for another run. `verify` checks the complete file inventory,
all payload hashes, code hashes, pinned source identity, schemas, conditions and
configuration, then replays the experiment. Default replay requires identical
bytes. `--tolerant` permits only finite numerical differences satisfying
abs(stored - replay) <= max(1e-12, 1e-10 * max(abs(stored), abs(replay)))
(Python math.isclose semantics) in declared CSV floating columns and floating
summary values. Identifiers, configuration, source and code
hashes remain exact. The stored payload must still match its own manifest hashes.
This accommodates possible Python/libm differences; it does not certify a platform.
Checksums detect corruption and replay checks consistency; neither is a signature
or proof that an arbitrary external producer is trustworthy.

## Experiment

Each condition begins with zero current activity, zero delayed history, no lesion,
no noise and zero baseline. The recurrence is the existing `NeuralEngine` reference
backend (10 ms step, 50 ms leak time, recurrent gain 0.9, tanh, float64). An edge of
numerical delay d reads a[n-d] when computing a[n+1]; default d=1 therefore does
not mean an instantaneous contribution from a[n]. All steps commit synchronously.

The engineered direct drive p[n] is +0.8 on source body 805722 (SNpp10, L) at
step_before 10 through 29, then +0.8 on body 77185 (JO-EV3, R) at step_before 50
through 69. Other neurons receive zero direct drive, and no sensory adapter is
used. These pulses do not implement measured touch or wind stimuli. Both target
rows are present in inputs.csv for every interval, including zero intervals.

- `original`: source endpoints and raw counts with declared modeled signs.
- `shuffled_weights`: same endpoints, sign and delay overlays, but raw-count
  weights permuted across edges by `WeightOverlay.shuffled`, seed 7. Incoming L1
  normalization is recomputed from the assigned counts. This is not rewiring and
  does not preserve per-target weight totals or normalized weight distributions.
- `no_input`: original weights with the identical requested schedule recorded,
  but every applied drive zero. This is an input-ablation control.

The coarse presynaptic consensus sign policy assigns +1 for acetylcholine,
dopamine, octopamine, serotonin and tyramine; -1 for GABA, glutamate and histamine;
0 for other/unknown labels. These are engineered assumptions, not receptor-level
measurements. Every canonical source-edge sign remains `unknown`.

Readouts are explicitly chosen singleton inspection pools: sensory_L=805722,
sensory_R=77185, DNg07_L=27143, DNg07_R=28949, hg1_motor_L=800241,
hg1_motor_R=800917. Each readout is the unscaled signed arithmetic mean of its
members. These are not the Minecraft program-selection pools and emit no action.
The pool names and source IDs are in manifest.json; no behavior is inferred.

## Files and schema

- nodes.csv: one row per source neuron, ascending body ID; roles separated by `;`.
  Blank cells mean missing source metadata, not an inferred annotation.
- edges.csv: one row per condition and source edge; target-major edge order.
  `raw_count` and `source_row` always describe the unchanged source edge.
  `overlay_source_edge_index` identifies the donor edge for `overlay_count`.
  `effective_weight` is assigned_sign * overlay_count / incoming overlay-count sum.
- inputs.csv: one row per condition, interval and target; `step_before` names n
  before p[n] is applied, and time_ms = n * 10.
- activity.csv: every neuron at committed steps 0..100, including the zero reset;
  time_ms = step * 10. Values are signed dimensionless activity.
- readouts.csv: every named pool at those same committed steps.
- summary.json: measured peak absolute activity, first peak step and signed value
  at that peak; a rectangle sum of absolute samples at steps 1..100 times 10 ms;
  and original/shuffled differences. Peaks are within this finite observation
  window; a final-sample peak does not establish settling or a global maximum.
  No success criterion or cherry-picked trial.
- manifest.json: schemas, dynamics, exact source/overlay identities, source rights,
  pool and input definitions, code hashes, payload hashes and verification policy.
  It has no timestamp, machine path, Git HEAD dependency or self-referential hash.

Source rows refer to the original source objects recorded in the fixture manifest,
not row numbers in these CSVs. Source and overlay identities are separate.
Compare the measured controls even if shuffling retains the response. This small,
selected graph cannot establish a full-network advantage or biological necessity.

## Attribution and rights

Copied neuron/edge structure and annotations retain the upstream CC BY 4.0 terms;
retain attribution and this change notice. Project-generated simulated responses,
input schedules, summaries, code and associated documentation are released under
the project MIT license. This does not relicense underlying third-party data.
The mixed edges.csv and manifest.json preserve the separate source-data terms.

MaleCNS v1.0 connectome data by FlyEM (HHMI Janelia), the University of Cambridge
Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research.
Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system", bioRxiv v2 (2025-10-30),
https://doi.org/10.1101/2025.10.09.680999, the reference current at acquisition;
published as "Sexual dimorphism in the complete Drosophila male central nervous
system connectome", Cell 189(18):5504-5526.e15 (2026),
https://doi.org/10.1016/j.cell.2026.08.015.
Source: https://male-cns.janelia.org/download/
License: https://creativecommons.org/licenses/by/4.0/legalcode.en

Changes: transcribed the already selected 14 neurons and 22 edges into CSV without
changing source values; added declared sign/count/delay overlays, synthetic input
histories, simulated activity and engineered inspection summaries. The upstream
fixture selection and bulk-source hashes are preserved in manifest.json. No bulk
source objects, Minecraft binaries or account material are included.
"""


class ResearchExampleError(ValueError):
    """An example cannot be generated or verified under its declared contract."""


def load_graph() -> RuntimeGraph:
    """Load the independently pinned 14-node fixture; never download data."""
    return load_malecns_fixture(FIXTURE)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _load_json(data: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ResearchExampleError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ResearchExampleError(f"non-finite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _csv_bytes(name: str, rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS[name], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: format(value, ".17g") if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    return buffer.getvalue().encode()


def _readouts(graph: RuntimeGraph, activity: Sequence[float]) -> dict[str, float]:
    return {
        name: math.fsum(activity[graph.index_of(body)] for body in members) / len(members)
        for name, members in POOLS.items()
    }


def simulate(
    *, shuffle_seed: int = 7, pulse_amplitude: float = 0.8, recurrent_gain: float = 0.9
) -> dict[str, Any]:
    """Run all three reset conditions; return metadata, typed tables and summary.

    Keyword changes are exploratory runs, not the versioned example-v1 reference.
    This function neither writes files nor changes the underlying graph.
    """
    if isinstance(pulse_amplitude, bool) or not math.isfinite(pulse_amplitude):
        raise ResearchExampleError("pulse_amplitude must be finite")
    graph = load_graph()
    config = DynamicsConfig(recurrent_gain=recurrent_gain)
    signs = SignOverlay.coarse_presynaptic_consensus(graph)
    shuffled = WeightOverlay.shuffled(graph, seed=shuffle_seed, provenance="research_example_v1")
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in COLUMNS}
    tables["nodes"] = [
        {
            key: (";".join(value) if key == "roles" else value)
            for key, value in asdict(node).items()
            if key in COLUMNS["nodes"]
        }
        for node in graph.nodes
    ]
    conditions: dict[str, Any] = {}
    measured: dict[str, Any] = {}
    samples: dict[str, list[dict[str, float]]] = {}
    for condition in CONDITIONS:
        weights = shuffled if condition == "shuffled_weights" else WeightOverlay.canonical(graph)
        engine = NeuralEngine(graph, sign_overlay=signs, config=config, weight_overlay=weights)
        state = engine.reset()
        conditions[condition] = {
            "weight_kind": weights.kind,
            "weight_overlay_sha256": weights.identity_sha256,
            "control_identity_sha256": engine.control_identity_sha256,
            "topology_overlay_sha256": engine.topology_overlay.identity_sha256,
            "delay_overlay_sha256": engine.delay_overlay.identity_sha256,
            "shuffle_seed": weights.seed,
            "weight_provenance": weights.provenance,
            "applied_input": "zero" if condition == "no_input" else "prescribed_direct_pulses",
        }
        for edge in graph.edges:
            index = edge.dense_index
            tables["edges"].append(
                {
                    "condition": condition,
                    "edge_index": index,
                    "body_pre": edge.body_pre,
                    "body_post": edge.body_post,
                    "source_row": edge.source_row,
                    "raw_count": edge.raw_count,
                    "canonical_sign_status": edge.canonical_sign_status,
                    "assigned_sign": signs.values[index],
                    "sign_provenance": signs.provenance[index],
                    "overlay_count": weights.values[index],
                    "overlay_source_edge_index": weights.source_edge_indices[index],
                    "effective_weight": engine.effective_weights[index],
                    "delay_steps": engine.delay_overlay.steps[index],
                }
            )
        samples[condition] = []
        for step in range(STEPS + 1):
            pool_values = _readouts(graph, state.activity)
            samples[condition].append(pool_values)
            for node, value in zip(graph.nodes, state.activity, strict=True):
                tables["activity"].append(
                    {
                        "condition": condition,
                        "step": step,
                        "time_ms": step * 10,
                        "body_id": node.body_id,
                        "activity": value,
                    }
                )
            for pool, value in pool_values.items():
                tables["readouts"].append(
                    {
                        "condition": condition,
                        "step": step,
                        "time_ms": step * 10,
                        "pool": pool,
                        "mean_activity": value,
                    }
                )
            if step == STEPS:
                break
            requested = {
                TARGETS[0]: pulse_amplitude if 10 <= step < 30 else 0.0,
                TARGETS[1]: pulse_amplitude if 50 <= step < 70 else 0.0,
            }
            applied = {
                body: 0.0 if condition == "no_input" else value for body, value in requested.items()
            }
            for body in TARGETS:
                tables["inputs"].append(
                    {
                        "condition": condition,
                        "step_before": step,
                        "time_ms": step * 10,
                        "body_id": body,
                        "requested_drive": requested[body],
                        "applied_drive": applied[body],
                    }
                )
            stimulation = Stimulation.from_body_values(graph, "research_example_v1", applied)
            state = engine.step(stimulation=stimulation).state
        measured[condition] = {}
        for pool in POOLS:
            values = [row[pool] for row in samples[condition]]
            peak_step = max(range(len(values)), key=lambda index: abs(values[index]))
            measured[condition][pool] = {
                "peak_abs_activity": abs(values[peak_step]),
                "first_peak_step": peak_step,
                "signed_activity_at_peak": values[peak_step],
                "absolute_activity_sum_ms": math.fsum(abs(value) for value in values[1:]) * 10,
                "final_activity": values[-1],
            }
    fixture = json.loads(FIXTURE.read_bytes())
    summary = {
        "scope": "synthetic_direct_stimulation_of_selected_fixture_no_behavior_measurement",
        "conditions": measured,
        "comparison": {
            pool: {
                "original_shuffled_max_abs_difference": max(
                    abs(left[pool] - right[pool])
                    for left, right in zip(
                        samples["original"], samples["shuffled_weights"], strict=True
                    )
                ),
                "shuffled_response_retained": measured["shuffled_weights"][pool][
                    "peak_abs_activity"
                ]
                > 0,
            }
            for pool in POOLS
        },
    }
    metadata = {
        "schema_version": 1,
        "example_id": "malecns-flight-power-synthetic-pulses-v1",
        "scope": summary["scope"],
        "source_graph": asdict(graph.manifest),
        "source_selection_policy": fixture["selection_policy"],
        "source_rights": fixture["rights"],
        "example_data_rights": {
            "copied_source_structure_and_annotations": "CC-BY-4.0",
            "project_generated_simulation_outputs": "MIT",
            "mixed_files_retain_separate_source_terms": ["edges.csv", "manifest.json"],
        },
        "project_code_and_documentation_license": "MIT",
        "dynamics": asdict(config),
        "dynamics_identity_sha256": config.identity_sha256,
        "sign_policy": {
            "name": "coarse_presynaptic_consensus",
            "identity_sha256": signs.identity_sha256,
            "canonical_status": "unknown",
            "provenance": "engineered_presynaptic_consensus_not_receptor_resolved",
        },
        "conditions": conditions,
        "input_schedule": {
            "kind": "synthetic_direct_stimulation_p_not_sensory_adapter",
            "steps": STEPS,
            "all_other_direct_drive": 0.0,
            "pulses": [
                {
                    "body_id": TARGETS[0],
                    "start_step_inclusive": 10,
                    "end_step_exclusive": 30,
                    "amplitude": pulse_amplitude,
                },
                {
                    "body_id": TARGETS[1],
                    "start_step_inclusive": 50,
                    "end_step_exclusive": 70,
                    "amplitude": pulse_amplitude,
                },
            ],
        },
        "initialization": {
            "activity": 0.0,
            "delayed_history": 0.0,
            "lesions": [],
            "reset_before_each_condition": True,
        },
        "readout_policy": {
            "selection": "engineered_inspection_only_no_motor_command",
            "aggregation": "signed_arithmetic_mean_no_gain_or_clipping",
            "pools": {name: list(members) for name, members in POOLS.items()},
        },
    }
    return {"tables": tables, "summary": summary, "metadata": metadata}


def _payloads() -> dict[str, bytes]:
    result = simulate()
    files = {f"{name}.csv": _csv_bytes(name, rows) for name, rows in result["tables"].items()}
    files["summary.json"] = _json_bytes(result["summary"])
    files["README.md"] = DATASET_README.encode()
    manifest = {
        **result["metadata"],
        "schemas": {
            name + ".csv": {
                "columns": list(columns),
                "numeric_tolerance_columns": sorted(FLOAT_COLUMNS[name]),
                "row_count": len(result["tables"][name]),
            }
            for name, columns in COLUMNS.items()
        },
        "software": {
            "engine": "neurocraft_neural.NeuralEngine",
            "backend": "python_single_thread_reference",
            "python_requires": ">=3.11,<3.14",
            "source_sha256": {
                name: sha256((ROOT / name).read_bytes()).hexdigest() for name in SOFTWARE_FILES
            },
        },
        "verification": {
            "default": "byte_exact_replay",
            "optional": "numerical_replay",
            "absolute_tolerance": ABS_TOL,
            "relative_tolerance": REL_TOL,
            "hash_integrity_required_in_both_modes": True,
        },
        "files": {
            name: {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in sorted(files.items())
        },
    }
    files["manifest.json"] = _json_bytes(manifest)
    return files


def generate(output: Path) -> dict[str, Any]:
    """Create a new dataset directory, refusing to overwrite any existing path."""
    files = _payloads()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ResearchExampleError("output already exists; choose a new directory") from exc
    for name, payload in files.items():
        with (output / name).open("xb") as handle:
            handle.write(payload)
    return _load_json(files["manifest.json"])  # type: ignore[no-any-return]


def _stored_files(dataset: Path) -> dict[str, bytes]:
    expected = {*[name + ".csv" for name in COLUMNS], "summary.json", "README.md", "manifest.json"}
    if dataset.is_symlink() or not dataset.is_dir():
        raise ResearchExampleError("dataset must be a regular directory")
    entries = list(dataset.iterdir())
    if {path.name for path in entries} != expected:
        raise ResearchExampleError("dataset inventory is incomplete or contains extra files")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ResearchExampleError("dataset payloads must be regular files")
    files = {path.name: path.read_bytes() for path in entries}
    try:
        manifest = _load_json(files["manifest.json"])
        recorded = manifest["files"]
        if set(recorded) != expected - {"manifest.json"}:
            raise ResearchExampleError("manifest payload inventory differs")
        for name, payload in files.items():
            if name == "manifest.json":
                continue
            identity = {"sha256": sha256(payload).hexdigest(), "bytes": len(payload)}
            if recorded[name] != identity:
                raise ResearchExampleError(f"payload hash/size mismatch: {name}")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ResearchExampleError("invalid manifest") from exc
    return files


def _compare_json(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected):
        raise ResearchExampleError(f"replay type mismatch: {label}")
    if isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, abs_tol=ABS_TOL, rel_tol=REL_TOL
        ):
            raise ResearchExampleError(f"numerical replay mismatch: {label}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ResearchExampleError(f"replay key mismatch: {label}")
        for key in expected:
            _compare_json(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ResearchExampleError(f"replay length mismatch: {label}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare_json(left, right, f"{label}[{index}]")
    elif actual != expected:
        raise ResearchExampleError(f"replay value mismatch: {label}")


def verify(dataset: Path = DEFAULT_DATASET, *, tolerant: bool = False) -> dict[str, Any]:
    """Verify stored integrity, exact source/config, and freshly executed responses."""
    stored = _stored_files(dataset)
    replay = _payloads()
    manifest = _load_json(stored["manifest.json"])
    expected_manifest = _load_json(replay["manifest.json"])
    # Payload digests must verify stored bytes, but replay floats may differ in tolerant mode.
    if _json_bytes(
        {key: value for key, value in manifest.items() if key != "files"}
    ) != _json_bytes({key: value for key, value in expected_manifest.items() if key != "files"}):
        raise ResearchExampleError("source, software, schema or configuration identity mismatch")
    for name, expected in replay.items():
        if stored[name] == expected:
            continue
        if not tolerant:
            raise ResearchExampleError(f"byte-exact replay mismatch: {name}")
        if name == "manifest.json":
            continue
        if name == "summary.json":
            _compare_json(_load_json(stored[name]), _load_json(expected), name)
        elif name.endswith(".csv"):
            table = name.removesuffix(".csv")
            left = list(csv.reader(io.StringIO(stored[name].decode())))
            right = list(csv.reader(io.StringIO(expected.decode())))
            if len(left) != len(right) or left[0] != right[0]:
                raise ResearchExampleError(f"CSV schema/length mismatch: {name}")
            for row_index, (a, b) in enumerate(zip(left[1:], right[1:], strict=True), start=1):
                if len(a) != len(b):
                    raise ResearchExampleError(f"CSV column mismatch: {name}:{row_index}")
                for column, x, y in zip(right[0], a, b, strict=True):
                    if column in FLOAT_COLUMNS[table]:
                        try:
                            _compare_json(float(x), float(y), f"{name}:{row_index}:{column}")
                        except ValueError as exc:
                            raise ResearchExampleError(str(exc)) from exc
                    elif x != y:
                        raise ResearchExampleError(f"CSV identity mismatch: {name}:{row_index}")
        else:
            raise ResearchExampleError(f"non-numerical payload mismatch: {name}")
    return {
        "verified": True,
        "mode": "numerical_replay" if tolerant else "byte_exact_replay",
        "example_id": manifest["example_id"],
        "file_count": len(stored),
        "source_graph_identity": manifest["source_graph"]["runtime_identity_sha256"],
    }


def load_dataset(dataset: Path = DEFAULT_DATASET, *, replay: bool = True) -> dict[str, Any]:
    """Read CSV rows as strings plus summary/manifest; verify before notebook use.

    With replay=False only stored payload integrity is checked. This is explicitly
    weaker than verify(): never treat that option as model reproduction.
    """
    if replay:
        verify(dataset)
    files = _stored_files(dataset)
    return {
        "tables": {
            name: list(csv.DictReader(io.StringIO(files[name + ".csv"].decode())))
            for name in COLUMNS
        },
        "summary": _load_json(files["summary.json"]),
        "manifest": _load_json(files["manifest.json"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate", help="generate into a new directory")
    create.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="verify integrity and replay")
    check.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    check.add_argument("--tolerant", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (
            generate(args.output)
            if args.command == "generate"
            else verify(args.dataset, tolerant=args.tolerant)
        )
    except (ResearchExampleError, OSError) as exc:
        parser.exit(1, f"research example: {exc}\n")
    print(
        json.dumps(
            {"generated": True, "example_id": result["example_id"]}
            if args.command == "generate"
            else result,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
