#!/usr/bin/env python3
"""Stage-4 readout census: which labeled populations a body program could read.

`PLAN.md` stage 4 asks for a recorded population selection — identifier, count,
sidedness, overlap refusals — for each behavior it intends to wire. This script
produces that record for the motor and descending candidates, through the same
digest-verified annotation join `docs/receipts/READOUT-CENSUS-01.md` used, and
writes a dated document under ``docs/research/``.

What this is
------------
An **annotation census**. Every candidate figure is a count of the pinned table's
own ``type`` / ``class`` / ``subclass`` / ``somaSide`` strings over the retained
nodes of one verified artifact, plus set arithmetic against the populations the
shipped runtime already binds. Nothing in the census section executes dynamics,
and a population appearing there carries no signal claim, no behavior claim, and
no recommendation that it be wired.

The optional ``--activity`` pass is separate and is labelled separately: it steps
the real controller through the six held conditions this script declares,
following the escape-v2 probe battery recorded in D-066, and reports the pooled
activity of two candidates. That is a first measurement of two populations under
six synthetic sensor frames, not evidence that either drives anything.

Refusals
--------
Nothing is written unless every gate passes: the four annotation-join gates from
:mod:`scripts.readout_annotations` (pinned digest, bitwise ``bodyId`` identity,
per-superclass reconciliation, unique in-range pointers), one count gate per
runtime population, a non-empty gate on every candidate selector, and the overlap
refusal — a candidate declared disjoint that captures a body the runtime already
reads refuses the whole report rather than being recorded quietly.

Determinism
-----------
Both output files are a pure function of the verified artifact, the pinned
annotations table, and the declared selectors. Under ``--activity`` they also
depend on the pinned escape-v2 readout calibration and the pinned
neurotransmitter table, both of which the controller digest-verifies at load and
both of which the document names. Re-running writes byte-identical files; no
timestamp, elapsed time, or host detail enters either document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

# ``python scripts/analyze_readout_candidates.py`` places ``scripts/`` rather
# than the repository root on sys.path; add this script's resolved repository
# root so the sibling analysis modules keep one import identity under direct CLI
# and test use, the convention analyze_readout.py already follows.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurocraft_connectome import load_full_runtime_artifact
from neurocraft_connectome.full_runtime import FullRuntimeManifest
from neurocraft_connectome.model import canonical_json_bytes
from neurocraft_controller.config import ControllerConfiguration
from neurocraft_controller.embodied_senses import (
    EXPECTED_BRISTLE_NODES,
    EXPECTED_LC4_LEFT_NODES,
    EXPECTED_LC4_RIGHT_NODES,
    EXPECTED_VISUAL_NODES,
)
from neurocraft_controller.escape_reflex import EXPECTED_TTMN_NODES
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)

from scripts.analyze_readout import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_ARTIFACT_DIR,
    TRACKED_MANIFEST_SHA256,
    artifact_identity,
    declared_annotations_sha256,
)
from scripts.reachability_graph import mapped_numpy
from scripts.readout_annotations import (
    IntegrityGate,
    RetainedAnnotations,
    annotation_gates,
    file_sha256,
    gates_report,
    key_of,
    require_all_gates,
    take_retained_annotations,
)
from scripts.readout_candidates import (
    RECORDS_OVERLAP,
    SIDE_COLUMN,
    CandidateSelector,
    candidate_cross_overlaps,
    census_candidates,
    require_declared_overlaps,
    require_nonempty,
    runtime_population_gates,
    runtime_populations,
)

SCHEMA: Final = "neurocraft.analysis.readout_candidates.v1"
REPORT_DATE: Final = "2026-09-02"
DEFAULT_OUTPUT: Final = Path("docs/research/readout-census-2026-09-02.json")

#: Member counts the shipped runtime declares for the populations it binds.
#: Imported rather than spelled, so a change in the controller is a gate failure
#: here rather than a silent disagreement between two hand-typed numbers.
RUNTIME_POPULATION_EXPECTED: Final[Mapping[str, int]] = {
    "wing_motor_readout_52": len(LEFT_WING_MOTOR_BODY_IDS) + len(RIGHT_WING_MOTOR_BODY_IDS),
    "ttmn_jump_pool": EXPECTED_TTMN_NODES,
    "lc4_left": EXPECTED_LC4_LEFT_NODES,
    "lc4_right": EXPECTED_LC4_RIGHT_NODES,
    "visual_class": EXPECTED_VISUAL_NODES,
    "mechanosensory_bristle": EXPECTED_BRISTLE_NODES,
    "faithful_sensory_entries": 2,
    "faithful_motor_entries": 2,
}

#: The declared candidates, in report order. Each is one annotation selector; the
#: notes are provenance and policy, never a function reading.
CANDIDATE_SELECTORS: Final[tuple[CandidateSelector, ...]] = (
    CandidateSelector(
        name="dna01_steering_pair",
        type_name="DNa01",
        note=(
            "descending neuron, not a motor stage. escape_reflex.py's recorded"
            " reasoning declared the motor stage as the readout, so binding a DN is a"
            " taxonomy change and needs its own decision, not just a new selector"
        ),
    ),
    CandidateSelector(
        name="dna04_steering_pair",
        type_name="DNa04",
        note="descending neuron; same taxonomy question as DNa01",
    ),
    CandidateSelector(
        name="dnp01_giant_fiber_pair",
        type_name="DNp01",
        note=(
            "POLICY-EXCLUDED. escape_reflex.py states that DNp01 itself is never read"
            " directly: the declared readout is the motor stage and the giant fibre"
            " stays interior recurrent context. Censused so the exclusion is visible,"
            " not proposed"
        ),
    ),
    CandidateSelector(
        name="dnp09_pair",
        type_name="DNp09",
        note=(
            "censused because an exploratory, unreceipted scratchpad probe reported it"
            " FLAT under held obstacles; that measurement is not reproduced here and"
            " this row is anatomy only"
        ),
    ),
    CandidateSelector(
        name="dna_prefix_aggregate",
        superclass="descending_neuron",
        type_alpha_prefix="DNa",
        note="contains DNa01 and DNa04; aggregation is not an independent readout",
    ),
    CandidateSelector(
        name="dnp_prefix_aggregate",
        superclass="descending_neuron",
        type_alpha_prefix="DNp",
        note=(
            "contains DNp01 and DNp09. The alpha-prefix rule keeps the longer DNpe"
            " run out; the strict-side gate column is the thing to read on this row"
        ),
    ),
    CandidateSelector(
        name="dna_startswith_aggregate",
        superclass="descending_neuron",
        type_startswith="DNa",
        note=(
            "the same family under the naive rule a reader reaches for first. Censused"
            " beside the alpha-prefix row so the difference between the two is a"
            " measured number in this document rather than a surprise later"
        ),
    ),
    CandidateSelector(
        name="dnp_startswith_aggregate",
        superclass="descending_neuron",
        type_startswith="DNp",
        note=(
            "the naive rule again, and the one that matters most: it sweeps in every"
            " type whose name merely begins with those letters, which is a different"
            " and larger population than the DNp prefix class"
        ),
    ),
    CandidateSelector(
        name="cb_neck_motor_pool",
        superclass="cb_motor",
        subclass="nm",
        note="brain-side neck motor neurons; a genuine motor stage",
    ),
    CandidateSelector(
        name="vnc_neck_motor_pool",
        superclass="vnc_motor",
        subclass="nm",
        note=(
            "the VNC half of the neck motor population. Censused so that a 'neck pool'"
            " is never ambiguous between the two halves or their union"
        ),
    ),
    CandidateSelector(
        name="haltere_motor_pool",
        superclass="vnc_motor",
        subclass="hm",
        note="haltere motor neurons; a genuine motor stage",
    ),
    CandidateSelector(
        name="wing_motor_subclass_all",
        superclass="vnc_motor",
        subclass="wm",
        overlap_policy=RECORDS_OVERLAP,
        note=(
            "the whole wing-motor subclass, censused as the superset the declared"
            " 52-body readout and the TTMn pair are drawn from. It necessarily overlaps"
            " both and is never a candidate readout in its own right"
        ),
    ),
    CandidateSelector(
        name="off_pool_wing_steering",
        superclass="vnc_motor",
        subclass="wm",
        excludes=("wing_motor_readout_52", "ttmn_jump_pool"),
        overlap_policy=RECORDS_OVERLAP,
        note=(
            "the wing-motor cells no shipped readout covers. Recorded with"
            " RECORDS_OVERLAP because subtracting the 52 and the TTMn pair is NOT"
            " enough: the remainder still captures a declared faithful motor entry,"
            " which the row's overlap column names"
        ),
    ),
    CandidateSelector(
        name="off_pool_wing_steering_disjoint",
        superclass="vnc_motor",
        subclass="wm",
        excludes=("wing_motor_readout_52", "ttmn_jump_pool", "faithful_motor_entries"),
        note=(
            "the same population with the faithful motor entries excluded as well; this"
            " is the variant a strict disjointness gate accepts"
        ),
    ),
    CandidateSelector(
        name="front_leg_motor_pool",
        superclass="vnc_motor",
        subclass="fl",
        note="front-leg motor neurons",
    ),
    CandidateSelector(
        name="middle_leg_motor_pool",
        superclass="vnc_motor",
        subclass="ml",
        note="middle-leg motor neurons",
    ),
    CandidateSelector(
        name="hind_leg_motor_pool",
        superclass="vnc_motor",
        subclass="hl",
        note="hind-leg motor neurons",
    ),
    CandidateSelector(
        name="antennal_motor_pool",
        superclass="cb_motor",
        subclass="am",
        note="antennal motor neurons",
    ),
    CandidateSelector(
        name="proboscis_motor_pool",
        superclass="cb_motor",
        subclass="pm",
        note="proboscis motor neurons",
    ),
    CandidateSelector(
        name="proboscis_mn9_pair",
        type_name="MN9",
        note="one named proboscis motor type; nested inside the proboscis pool",
    ),
    CandidateSelector(
        name="abdominal_motor_pool",
        superclass="vnc_motor",
        subclass="ad",
        note="abdominal motor neurons",
    ),
)

#: Candidates the optional activity pass measures, and the runtime pools it
#: measures beside them for scale. Two leads only: the pass is a first
#: measurement, not a battery.
ACTIVITY_CANDIDATES: Final = ("dna01_steering_pair", "cb_neck_motor_pool")
ACTIVITY_REFERENCE_POPULATIONS: Final = ("wing_motor_readout_52", "ttmn_jump_pool")

#: The six held conditions, declared here and following the escape-v2 probe
#: battery recorded in D-066, at its ray values: 30 ticks from a fresh reset,
#: ray_max 6.0, blocked ray 0.5, ray slots front=0 / left=2 / right=3. No
#: committed script in this repository declares this condition set, so it is
#: declared here rather than cited as an existing pinned battery.
ACTIVITY_HOLD_TICKS: Final = 30
ACTIVITY_BASELINE_TICKS: Final = 10
ACTIVITY_RAY_MAX: Final = 6.0
ACTIVITY_BLOCKED_RAY: Final = 0.5
ACTIVITY_BASELINE_CONDITION: Final = "all_open_bright"
ACTIVITY_CONDITIONS: Final[tuple[tuple[str, int | None, int, bool], ...]] = (
    # (condition, blocked ray slot or None, local light, raw horizontal collision)
    ("all_open_bright", None, 15, False),
    ("all_open_dark", None, 0, False),
    ("blocked_front", 0, 15, False),
    ("blocked_left", 2, 15, False),
    ("blocked_right", 3, 15, False),
    ("collision_on", None, 15, True),
)
DEFAULT_NEUROTRANSMITTERS: Final = Path(
    "data/cache/male-cns-v1.0/body-neurotransmitters-male-cns-v1.0.feather"
)
DEFAULT_CALIBRATION: Final = Path("data/calibration/full-readout-calibration-escape-v2.json")

#: Everything this document is not. Rendered verbatim into the markdown so the
#: bounds travel with the numbers rather than living in a commit message.
INTERPRETATION_BOUNDS: Final[tuple[tuple[str, str], ...]] = (
    (
        "annotation labels are dataset labels",
        "type, class, subclass and somaSide are strings the source annotators wrote."
        " They are not measured functions. This census assigns no behavior to any"
        " neuron and does not verify that two cells sharing a type string are the same"
        " kind of cell.",
    ),
    (
        "a census is not a selection",
        "a population appearing here exists with the stated shape. Nothing here"
        " selects it, binds it, or recommends it, and 'anatomically eligible' means"
        " only that a strict left/right, disjoint-from-the-runtime gate would accept"
        " it.",
    ),
    (
        "no signal, no behavior",
        "the census section executes no dynamics. It is not evidence that any"
        " population responds to anything, drives anything, or would be legible in"
        " the game.",
    ),
    (
        "counts are of one pinned artifact",
        "every figure is over the 166,700 retained nodes of one verified build. A"
        " different retention rule or a different release would give different"
        " numbers, and the integrity gates exist to make that impossible to miss.",
    ),
)


class ReadoutCandidateError(RuntimeError):
    """The candidate census could not be produced exactly as declared."""


# ------------------------------------------------------------------- census


def build_census(
    annotations: RetainedAnnotations,
    *,
    identity: Mapping[str, Any],
    join_gates: Sequence[IntegrityGate],
    selectors: Sequence[CandidateSelector] = CANDIDATE_SELECTORS,
    expected: Mapping[str, int] = RUNTIME_POPULATION_EXPECTED,
    wing_body_ids: Sequence[int] | None = None,
    faithful_sensory_body_ids: Sequence[int] | None = None,
    faithful_motor_body_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Produce the deterministic census body, refusing on any failed gate.

    The three body-ID arguments default to the shipped controller's own literals
    and exist so the tests can drive the whole census over a synthetic frame at
    synthetic scale, the way ``select_ttmn_pool`` is pure over its columns.
    """

    configuration = ControllerConfiguration()
    populations = runtime_populations(
        annotations,
        wing_body_ids=(
            LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS
            if wing_body_ids is None
            else wing_body_ids
        ),
        faithful_sensory_body_ids=(
            (configuration.left_sensor_body_id, configuration.right_sensor_body_id)
            if faithful_sensory_body_ids is None
            else faithful_sensory_body_ids
        ),
        faithful_motor_body_ids=(
            (configuration.left_motor_body_id, configuration.right_motor_body_id)
            if faithful_motor_body_ids is None
            else faithful_motor_body_ids
        ),
    )
    gates = [*join_gates, *runtime_population_gates(populations, expected)]
    require_all_gates(gates)
    censuses = census_candidates(annotations, selectors, populations)
    require_nonempty(censuses)
    require_declared_overlaps(censuses)
    return {
        "schema": SCHEMA,
        "report_date": REPORT_DATE,
        "artifact": dict(identity),
        "integrity_gates": gates_report(gates),
        "runtime_populations": {
            "definition": (
                "every population the shipped runtime already reads or writes, against"
                " which each candidate's membership is differenced. Selector-derived"
                " members are re-derived here from the same annotation columns the"
                " controller binds them from; literal members are read from the"
                " controller source"
            ),
            "populations": [population.as_json() for population in populations],
        },
        "candidates": {
            "definition": (
                "one row per declared candidate selector: the exact selector, its"
                " members, their somaSide split, and its intersection with every"
                " runtime population. Anatomy only"
            ),
            "strict_side_gate": (
                "'strict_side_gate_passes' asks whether a selector shaped like"
                " escape_reflex.select_ttmn_pool — which refuses any somaSide outside"
                " {L, R} — would accept this population as written"
            ),
            "rows": [census.as_json() for census in censuses],
        },
        "candidate_cross_overlaps": {
            "definition": (
                "candidate pairs that share members. Candidates are allowed to nest,"
                " but two nested candidates are not two independent readouts"
            ),
            "rows": list(candidate_cross_overlaps(censuses)),
        },
        "interpretation_bounds": {name: text for name, text in INTERPRETATION_BOUNDS},
    }


# ---------------------------------------------------------------- activity


@dataclass(frozen=True, slots=True)
class ActivityPool:
    """The dense indices the activity pass reads, whole and split by soma side."""

    members: tuple[int, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]


def activity_pools(
    census_body: Mapping[str, Any],
    annotations: RetainedAnnotations,
) -> dict[str, ActivityPool]:
    """Dense indices of the pools the activity pass reads, from the census body.

    Sides come from the same ``somaSide`` column the census counted, so the
    left/right traces below partition the whole-pool trace exactly.
    """

    index_of = {int(body): position for position, body in enumerate(annotations.body_ids.tolist())}
    sides = annotations.column(SIDE_COLUMN)

    def _pool(indices: Sequence[int]) -> ActivityPool:
        members = tuple(int(index) for index in indices)
        return ActivityPool(
            members=members,
            left=tuple(index for index in members if key_of(sides[index]) == "L"),
            right=tuple(index for index in members if key_of(sides[index]) == "R"),
        )

    pools: dict[str, ActivityPool] = {}
    for row in census_body["candidates"]["rows"]:
        if row["name"] in ACTIVITY_CANDIDATES:
            pools[row["name"]] = _pool(row["dense_indices"])
    for population in census_body["runtime_populations"]["populations"]:
        if population["name"] in ACTIVITY_REFERENCE_POPULATIONS:
            pools[population["name"]] = _pool(
                [index_of[int(body)] for body in population["body_ids"]]
            )
    missing = sorted(set(ACTIVITY_CANDIDATES + ACTIVITY_REFERENCE_POPULATIONS) - set(pools))
    if missing:
        raise ReadoutCandidateError(
            f"the activity pass names pools the census did not produce: {missing}"
        )
    return pools


def _condition_frame(slot: int | None, light: int, collision: bool) -> Any:
    from neurocraft_protocol import SensorFrame

    rays = [ACTIVITY_RAY_MAX] * 6
    if slot is not None:
        rays[slot] = ACTIVITY_BLOCKED_RAY
    distances = (rays[0], rays[1], rays[2], rays[3], rays[4], rays[5])
    return SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=collision,
        vertical_collision=False,
        in_fluid=False,
        local_light=light,
        ray_max_distance_m=ACTIVITY_RAY_MAX,
        ray_distances_m=distances,
    )


def _sign_relation(tail: float, baseline: float) -> str:
    """Whether a condition's pooled mean sits on the same side of zero as the baseline.

    ``ratio_to_baseline`` is a ratio of magnitudes, so it reads the same whether
    a pool grew in the direction it was already going or crossed zero and came
    out the other side. Those are different measurements and the second one is
    not an amplification, so the relation is recorded beside the ratio rather
    than left for a reader to reconstruct from two signed columns.
    """

    if tail == 0.0 or baseline == 0.0:
        return "zero"
    return "same" if (tail > 0.0) == (baseline > 0.0) else "reversed"


def measure_activity(
    pools: Mapping[str, ActivityPool],
    *,
    artifact_directory: Path,
    annotations_path: Path,
    body_annotations_path: Path,
    calibration_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Step the escape-v2 controller through the six held conditions, pooling activity.

    Read-only over the controller: pooled values come from
    ``engine.activity()`` after each committed tick, the same read-only path
    ``pool_readout_means`` uses. Nothing is bound, declared, projected, or fed
    back; no candidate influences any motor output.
    """

    from neurocraft_controller.full_brain_controller import FullBrainController
    from neurocraft_protocol import Observation

    controller = FullBrainController.from_artifact(
        run_id="readout-candidate-activity",
        entity_id="census-fly",
        artifact_directory=artifact_directory,
        annotations_path=annotations_path,
        calibration_path=calibration_path,
        expected_manifest_sha256=manifest_sha256,
        embodied_senses=True,
        body_annotations_path=body_annotations_path,
        escape_reflex=True,
        escape_version=2,
    )
    names = sorted(pools)

    def _mean(activity: Any, indices: Sequence[int]) -> float | None:
        if not indices:
            return None
        return math.fsum(float(activity[index]) for index in indices) / len(indices)

    series: dict[str, dict[str, list[float]]] = {}
    sided: dict[str, dict[str, list[float | None]]] = {}
    tick = 0
    try:
        for condition, slot, light, collision in ACTIVITY_CONDITIONS:
            controller.reset()
            frame = _condition_frame(slot, light, collision)
            traces: dict[str, list[float]] = {name: [] for name in names}
            side_traces: dict[str, list[float | None]] = {
                f"{name}:{side}": [] for name in names for side in ("L", "R")
            }
            for _ in range(ACTIVITY_HOLD_TICKS):
                controller.step(
                    Observation(
                        identity=controller.identity,
                        observation_seq=tick,
                        game_tick=tick,
                        sent_monotonic_ns=tick * 50_000_000,
                        sensors=frame,
                    ),
                    command_seq=tick,
                )
                tick += 1
                activity = controller.engine.activity()
                for name in names:
                    pool = pools[name]
                    whole = _mean(activity, pool.members)
                    assert whole is not None
                    traces[name].append(whole)
                    side_traces[f"{name}:L"].append(_mean(activity, pool.left))
                    side_traces[f"{name}:R"].append(_mean(activity, pool.right))
            series[condition] = traces
            sided[condition] = side_traces
        identity = {
            "adapter_hash": controller.identity.adapter_hash,
            "decoder_hash": controller.identity.decoder_hash,
            "configuration_sha256": controller.configuration.identity_sha256,
            "readout_scale": controller.readout_scale,
        }
    finally:
        controller.engine.end_run()

    def _tail_mean(values: Sequence[float]) -> float:
        return math.fsum(values[-ACTIVITY_BASELINE_TICKS:]) / ACTIVITY_BASELINE_TICKS

    def _side_tail(condition: str, name: str, side: str) -> float | None:
        values = sided[condition][f"{name}:{side}"]
        if any(value is None for value in values):
            return None
        return _tail_mean([value for value in values if value is not None])

    baselines = {name: _tail_mean(series[ACTIVITY_BASELINE_CONDITION][name]) for name in names}
    rows: list[dict[str, Any]] = []
    for condition, _slot, _light, _collision in ACTIVITY_CONDITIONS:
        for name in names:
            values = series[condition][name]
            tail = _tail_mean(values)
            baseline = baselines[name]
            peak = max(values, key=abs)
            left = _side_tail(condition, name, "L")
            right = _side_tail(condition, name, "R")
            rows.append(
                {
                    "condition": condition,
                    "pool": name,
                    "members": len(pools[name].members),
                    "pooled_mean_last_10_ticks": tail,
                    "pooled_peak_signed": peak,
                    "baseline_pooled_mean": baseline,
                    "ratio_to_baseline": (abs(tail) / abs(baseline) if baseline != 0.0 else None),
                    "sign_relative_to_baseline": _sign_relation(tail, baseline),
                    "left_pooled_mean_last_10_ticks": left,
                    "right_pooled_mean_last_10_ticks": right,
                    "left_minus_right": (None if left is None or right is None else left - right),
                }
            )
    identical: list[dict[str, Any]] = []
    for condition, _slot, _light, _collision in ACTIVITY_CONDITIONS:
        if condition == ACTIVITY_BASELINE_CONDITION:
            continue
        for name in names:
            identical.append(
                {
                    "condition": condition,
                    "pool": name,
                    "bit_identical_to_baseline_condition": (
                        series[condition][name] == series[ACTIVITY_BASELINE_CONDITION][name]
                    ),
                }
            )
    return {
        "status": "first measurement, one deterministic run, no repeats",
        "definition": (
            "unit-weight pooled mean activity of each pool after every committed tick,"
            " under the six held conditions declared in this script, following the"
            " escape-v2 probe battery recorded in D-066:"
            f" {ACTIVITY_HOLD_TICKS} ticks from a fresh reset, ray_max"
            f" {ACTIVITY_RAY_MAX}, blocked ray {ACTIVITY_BLOCKED_RAY}, ray slots"
            " front=0 / left=2 / right=3"
        ),
        "baseline": (
            f"the {ACTIVITY_BASELINE_CONDITION} condition's own last-"
            f"{ACTIVITY_BASELINE_TICKS}-tick pooled mean, per pool"
        ),
        "ratio_caveat": (
            "ratio_to_baseline is |tail| / |baseline|, a ratio of magnitudes, and it"
            " hides two different things. First, it divides by a pooled mean that can"
            " sit near zero, so where the baseline column is small the ratio is a"
            " denominator artifact and says nothing. Second, it hides a sign change:"
            " where the signed columns straddle zero the value describes a POLARITY"
            " REVERSAL, not an amplification, and the sign column marks every such"
            " row 'reversed'. Read the signed columns"
        ),
        "bounds": (
            "these are pooled activity values under six synthetic sensor frames. They"
            " are not evidence that any pool drives, selects, or modulates anything,"
            " no pool measured here is bound to any motor axis, and a single"
            " deterministic run gives no dispersion estimate"
        ),
        "controller_identity": identity,
        "additional_pinned_inputs": {
            "definition": (
                "the files this pass depends on beyond the artifact and the body"
                " annotations the census already names. The controller digest-verifies"
                " both at load; the digests below are recomputed here so the document"
                " states them rather than assuming them"
            ),
            "files": [
                {
                    "role": "escape-v2 readout calibration",
                    "path": str(calibration_path),
                    "sha256": file_sha256(calibration_path),
                },
                {
                    "role": "neurotransmitter table, the sign overlay's source",
                    "path": str(annotations_path),
                    "sha256": file_sha256(annotations_path),
                },
            ],
        },
        "conditions": [name for name, _slot, _light, _collision in ACTIVITY_CONDITIONS],
        "rows": rows,
        "bit_identity_against_baseline_condition": {
            "definition": (
                "whether a condition's whole 30-tick pooled trace is exactly equal,"
                f" value for value, to the same pool's trace under"
                f" {ACTIVITY_BASELINE_CONDITION}. A True row means the condition"
                " changed nothing at all for that pool — not that it changed little"
            ),
            "rows": identical,
        },
    }


# ------------------------------------------------------------------ markdown


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _members_cell(row: Mapping[str, Any], limit: int = 20) -> str:
    body_ids = [int(body) for body in row["body_ids"]]
    if len(body_ids) <= limit:
        return ", ".join(str(body) for body in body_ids)
    return (
        f"{len(body_ids)} bodies, {body_ids[0]}…{body_ids[-1]};"
        " full list in the JSON beside this file"
    )


def _sentence(text: str) -> str:
    """Capitalize a fragment written for a JSON field so it reads as prose."""

    return text[:1].upper() + text[1:] if text else text


def _optional(value: float | None) -> str:
    return "—" if value is None else f"{value:+.6e}"


def _side_cell(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{side} {count}" for side, count in sorted(counts.items())) or "—"


def _population(census: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for population in census["runtime_populations"]["populations"]:
        if population["name"] == name:
            entry: Mapping[str, Any] = population
            return entry
    raise ReadoutCandidateError(f"the census carries no runtime population {name!r}")


def _candidate(census: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in census["candidates"]["rows"]:
        if row["name"] == name:
            entry: Mapping[str, Any] = row
            return entry
    raise ReadoutCandidateError(f"the census carries no candidate {name!r}")


def _prefix_rule_caution(census: Mapping[str, Any]) -> str:
    """The measured cost of the naive prefix rule, rendered from the census.

    Both readings are candidates in this report precisely so the difference is a
    measured number; writing that number as prose would let it drift the first
    time a selector changed.
    """

    dnp_prefix = _candidate(census, "dnp_prefix_aggregate")
    dnp_naive = _candidate(census, "dnp_startswith_aggregate")
    dna_prefix = _candidate(census, "dna_prefix_aggregate")
    dna_naive = _candidate(census, "dna_startswith_aggregate")
    midline = ", ".join(
        f"{entry['soma_side']} (body {entry['body_id']})"
        for entry in dnp_naive["non_strict_side_members"]
    )
    return (
        "- **The two ways of writing 'the DNp family' are different populations, and"
        " only one of them passes the strict side gate.** `dnp_prefix_aggregate` uses"
        " the alpha-prefix rule `scripts/readout_census.py` already uses — the maximal"
        " leading run of letters, so `DNpe001` lands in `DNpe` and not in `DNp`."
        " `dnp_startswith_aggregate` is the naive `startswith` filter, and it sweeps"
        f" in the longer runs: {dnp_naive['members']} bodies against"
        f" {dnp_prefix['members']}, so the naive rule adds"
        f" {dnp_naive['members'] - dnp_prefix['members']} bodies the prefix class"
        f" excludes. On the DNa side it is {dna_naive['members']} against"
        f" {dna_prefix['members']}, an extra"
        f" {dna_naive['members'] - dna_prefix['members']}. The difference is not"
        " cosmetic: the naive DNp population carries midline members"
        f" ({midline or 'none'}), so a `select_ttmn_pool`-shaped selector refuses the"
        " whole population, while the prefix class"
        f" ({dnp_prefix['left']} L / {dnp_prefix['right']} R) passes. This is the same"
        " trap `docs/receipts/READOUT-CENSUS-01.md` recorded for the word"
        " `descending`, one column over. Aggregating a DN family is in any case not an"
        " independent readout."
    )


def _wing_pool_caution(census: Mapping[str, Any]) -> str:
    """The declared wing readout's own side split, rendered from the measurement.

    Written from the census rather than typed as prose: the 52-body readout is a
    literal list in `flight_pools.py`, its side split is a measured property of
    that list, and which faithful motor entries it happens to contain is exactly
    the sort of thing a hand-typed sentence gets wrong after an edit.
    """

    wing = _population(census, "wing_motor_readout_52")
    faithful = _population(census, "faithful_motor_entries")
    members = set(int(body) for body in wing["body_ids"])
    inside = sorted(body for body in map(int, faithful["body_ids"]) if body in members)
    outside = sorted(body for body in map(int, faithful["body_ids"]) if body not in members)
    return (
        "- **The declared 52-body wing readout is not side-balanced, and it splits the"
        f" faithful motor pair.** Its measured `somaSide` split is"
        f" {_side_cell(wing['soma_side_counts'])}. Of the two bodies"
        " `ControllerConfiguration` names as faithful motor entries it contains"
        f" {inside} and not {outside}. Both facts are measurements of the literal list"
        " in `flight_pools.py`, not defects proved here — but a readout that reads"
        " more of one side than the other is the first thing to rule out before a"
        " lopsided steering signal is blamed on the wiring."
    )


def _overlap_cell(row: Mapping[str, Any]) -> str:
    overlaps: Mapping[str, Sequence[int]] = row["overlaps"]
    if not overlaps:
        return "none"
    return "; ".join(f"{name} {list(bodies)}" for name, bodies in sorted(overlaps.items()))


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render the census (and any activity block) as a deterministic document."""

    census = document["census"]
    artifact = census["artifact"]
    lines: list[str] = [
        f"# Readout census — candidate body-program populations ({census['report_date']})",
        "",
        f"- **Census date:** {census['report_date']}",
        f"- **Artifact:** `{artifact['artifact_id']}` (manifest"
        f" `{artifact['manifest_file_sha256']}`), {artifact['node_count']:,} retained"
        f" nodes and {artifact['edge_count']:,} retained edges",
        "- **Annotations:** `data/cache/male-cns-v1.0/"
        "body-annotations-male-cns-v1.0-minconf-0.5.feather`, joined through"
        " `annotation_source_rows` and digest-verified against the manifest",
        "- **Evidence state:** measured (annotation census), unit-tested, locally"
        " executed. Not a receipt, not a gate promotion, not a selection."
        + (
            " A separate first activity measurement is recorded in the last section"
            " and is labelled as one; nothing in the census depends on it."
            if document.get("activity") is not None
            else ""
        ),
        f"- **Census digest:** `{document['census_sha256']}`, over the annotation"
        " census body only — the activity block, when present, is outside it",
        "",
        "## What this is, and what it is not",
        "",
        "This is an **annotation census**. Every candidate figure below is a count of"
        " the pinned table's own `type` / `class` / `subclass` / `somaSide` strings"
        " over the retained nodes of one verified artifact, plus set arithmetic"
        " against the populations the shipped runtime already binds.",
        "",
        "**No activity, no signal, no behavior.** The census executes no dynamics."
        " A population listed here exists with the stated shape and nothing more."
        " It is not evidence that any of these neurons respond to anything, drive"
        " anything, or would be visible in the game. Where the word *eligible*"
        " appears it means one thing only: a strict left/right, disjoint-from-the-"
        "runtime gate would accept the population as written.",
        "",
    ]
    if document.get("activity") is not None:
        lines += [
            "This document also carries a **first activity measurement** in its last"
            " section, which does step the real controller. It is reported separately,"
            " under its own bounds, and it is a measurement of pooled activity under"
            " six synthetic sensor frames — not evidence that any population drives,"
            " selects, or modulates anything.",
            "",
        ]
    for name, text in INTERPRETATION_BOUNDS:
        lines.append(f"- **{_sentence(name)}.** {_sentence(text)}")
    lines.append("")

    gates = census["integrity_gates"]
    lines += [
        "## Integrity gates",
        "",
        f"{len(gates['gates'])} gates evaluated, `all_passed = {gates['all_passed']}`."
        " Nothing is written unless every one of them holds, so a published document"
        " is evidence that they did.",
        "",
    ]
    lines += _table(
        ("Gate", "Result"),
        [(f"`{gate['name']}`", "PASS" if gate["passed"] else "FAIL") for gate in gates["gates"]],
    )
    lines += [
        "",
        "The first four are the annotation-join gates"
        " `docs/receipts/READOUT-CENSUS-01.md` already used. The `runtime_population_*`"
        " gates are this document's own: each pins the member count the shipped"
        " controller declares for a population it binds, so a passing report is"
        " evidence that this census read the same artifact the controller reads.",
        "",
        "## Populations the runtime already reads",
        "",
        "Every candidate below is differenced against all of these. Selector-derived"
        " memberships are re-derived here from the same annotation columns the"
        " controller binds them from; literal memberships are read out of the"
        " controller source.",
        "",
    ]
    lines += _table(
        ("Population", "Selector", "Members", "somaSide", "Origin"),
        [
            (
                f"`{population['name']}`",
                f"`{population['selector']}`",
                str(population["members"]),
                _side_cell(population["soma_side_counts"]),
                population["origin"],
            )
            for population in census["runtime_populations"]["populations"]
        ],
    )
    lines += [
        "",
        "## Candidate census",
        "",
        "`Strict L/R` is whether a selector shaped like"
        " `escape_reflex.select_ttmn_pool` — which refuses any `somaSide` outside"
        " `{L, R}` — would accept the population as written. `Eligible` is `Strict"
        " L/R` and disjointness together, and is an anatomical verdict only.",
        "",
    ]
    lines += _table(
        ("Candidate", "Members", "L", "R", "Other side", "Strict L/R", "Disjoint", "Eligible"),
        [
            (
                f"`{row['name']}`",
                str(row["members"]),
                str(row["left"]),
                str(row["right"]),
                ", ".join(
                    f"{entry['soma_side']} (body {entry['body_id']})"
                    for entry in row["non_strict_side_members"]
                )
                or "—",
                "yes" if row["strict_side_gate_passes"] else "**no**",
                "yes" if row["disjoint_from_runtime_populations"] else "**no**",
                "yes" if row["anatomically_eligible"] else "**no**",
            )
            for row in census["candidates"]["rows"]
        ],
    )
    lines += ["", "### Selectors, members, and overlaps", ""]
    lines += _table(
        ("Candidate", "Exact selector", "Body IDs", "Overlap with runtime populations"),
        [
            (
                f"`{row['name']}`",
                f"`{row['selector']}`",
                _members_cell(row),
                _overlap_cell(row),
            )
            for row in census["candidates"]["rows"]
        ],
    )
    lines += ["", "### Notes carried on each candidate", ""]
    for row in census["candidates"]["rows"]:
        if row["note"]:
            lines.append(f"- **`{row['name']}`** — {row['note']}")
    lines.append("")

    cross = census["candidate_cross_overlaps"]["rows"]
    lines += ["### Candidates that share members", ""]
    if cross:
        lines += _table(
            ("Candidates", "Shared members", "Relation"),
            [
                (
                    " + ".join(f"`{name}`" for name in row["candidates"]),
                    str(row["shared_members"]),
                    row["relation"],
                )
                for row in cross
            ],
        )
    else:
        lines.append("No declared candidate shares a member with another.")
    lines += [
        "",
        "Nesting is allowed and expected — a named type sits inside its subclass pool"
        " — but two nested candidates are not two independent readouts, and pooling"
        " both would read the same neurons twice under two names.",
        "",
        "## Negatives, exclusions, and cautions",
        "",
        "- **`dnp01_giant_fiber_pair` is policy-excluded, not unavailable.**"
        ' `escape_reflex.py:18-20` says, verbatim: "DNp01 itself is never read'
        " directly: the declared readout is the motor stage, and the giant fiber"
        ' stays interior recurrent context." The pair is censused here so the'
        " exclusion is visible; reversing it is a decision, not a selector change.",
        "- **`dnp09_pair` — flat, per an exploratory measurement that is not"
        " reproduced here.** An uncommitted, unreceipted scratchpad probe"
        " (2026-09-01) reported DNp09 slightly *decreasing* under every held"
        " condition it tried. That reading is exploratory and unconfirmed; nothing in"
        " this document measures it. It is recorded so the obvious literature"
        " candidate for a 'walk forward' readout is not re-proposed without someone"
        " re-measuring it first.",
        _prefix_rule_caution(census),
        _wing_pool_caution(census),
        "- **`off_pool_wing_steering` captures a declared faithful motor entry.**"
        " Subtracting the 52-body wing readout and the TTMn pair is not sufficient:"
        " the remainder still contains a body that `ControllerConfiguration` names as"
        " a faithful motor entry, and the overlap column above names it. The existing"
        " `escape_reflex._refuse_ttmn_overlaps` gate checks the two faithful *sensory*"
        " entries and the wing pool, so this capture would pass it silently. The"
        " `off_pool_wing_steering_disjoint` row is the variant a strict gate accepts.",
    ]
    if document.get("activity") is not None:
        lines.append(
            "- **A near-zero baseline inflates every ratio, and a sign change hides"
            " inside one.** Every ratio in the activity section below is"
            " |tail| / |baseline|. Where the baseline sits at the numerical floor the"
            " ratio is a denominator artifact, and where the signed columns straddle"
            " zero it describes a polarity reversal rather than an amplification. The"
            " table's sign column marks those rows; read the signed columns."
        )
    lines.append("")

    activity = document.get("activity")
    lines += ["## First activity measurement", ""]
    if activity is None:
        lines += [
            "Not run for this document. The census above is annotation-only.",
            "",
        ]
    else:
        identity = activity["controller_identity"]
        lines += [
            f"**{_sentence(activity['status'])}.** {_sentence(activity['definition'])}.",
            "",
            f"Baseline: {activity['baseline']}. Sided means are over that pool's own"
            " left and right members, whose counts are in the tables above and are not"
            " always equal.",
            "",
            f"**Bounds.** {_sentence(activity['bounds'])}. {_sentence(activity['ratio_caveat'])}.",
            "",
            "Controller identity:"
            f" adapter `{identity['adapter_hash']}`,"
            f" decoder `{identity['decoder_hash']}`,"
            f" configuration `{identity['configuration_sha256']}`,"
            f" readout scale `{identity['readout_scale']!r}`.",
            "",
            f"{_sentence(activity['additional_pinned_inputs']['definition'])}:",
            "",
        ]
        lines += _table(
            ("Role", "Path", "SHA-256"),
            [
                (entry["role"], f"`{entry['path']}`", f"`{entry['sha256']}`")
                for entry in activity["additional_pinned_inputs"]["files"]
            ],
        )
        lines.append("")
        lines += _table(
            (
                "Condition",
                "Pool",
                "Members",
                "Pooled mean (last 10 ticks)",
                "Signed peak",
                "Ratio to baseline",
                "Sign vs baseline",
                "L pooled",
                "R pooled",
                "L minus R",
            ),
            [
                (
                    row["condition"],
                    f"`{row['pool']}`",
                    str(row["members"]),
                    f"{row['pooled_mean_last_10_ticks']:+.6e}",
                    f"{row['pooled_peak_signed']:+.6e}",
                    "—" if row["ratio_to_baseline"] is None else f"{row['ratio_to_baseline']:.2f}x",
                    "**reversed**"
                    if row["sign_relative_to_baseline"] == "reversed"
                    else row["sign_relative_to_baseline"],
                    _optional(row["left_pooled_mean_last_10_ticks"]),
                    _optional(row["right_pooled_mean_last_10_ticks"]),
                    _optional(row["left_minus_right"]),
                )
                for row in activity["rows"]
            ],
        )
        identity_rows = activity["bit_identity_against_baseline_condition"]["rows"]
        unchanged = [row for row in identity_rows if row["bit_identical_to_baseline_condition"]]
        lines += [
            "",
            "### Conditions that changed nothing at all",
            "",
            _sentence(activity["bit_identity_against_baseline_condition"]["definition"]) + ".",
            "",
        ]
        if unchanged:
            lines += _table(
                ("Condition", "Pool"),
                [(row["condition"], f"`{row['pool']}`") for row in unchanged],
            )
            lines += [
                "",
                "Every pair listed above is **exactly equal**, tick for tick, to the"
                f" same pool under `{ACTIVITY_BASELINE_CONDITION}` — a held condition"
                " that moved nothing, not one that moved a little. This is not new:"
                " D-066 already recorded that under escape-v2 the held `collision_on`"
                " condition is bit-identical to the untouched all-open-bright"
                " condition on all five axes, the jump value, and the state digest, at"
                " all 30 ticks, and it gives the mechanism — the encoder's first-tick"
                " rule `m0 = raw0` makes the running mean a fixed point, so a body"
                " already in contact at reset perceives that contact as identically"
                " zero. What this table adds is that the same holds at the pooled"
                " activity of these four populations. A held contact bit is therefore"
                " the wrong input to wire a touch behavior to; the transient is where"
                " the signal is.",
                "",
            ]
        else:
            lines += [
                "Every condition changed every pool's trace by at least one value.",
                "",
            ]

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/analyze_readout_candidates.py \\",
        f"  --output {DEFAULT_OUTPUT}" + (" --activity" if activity is not None else ""),
        "```",
        "",
        "Both files are a pure function of the verified artifact, the pinned"
        " annotations table, and the declared selectors"
        + (
            ", plus the two further pinned inputs the activity section names above —"
            " the escape-v2 readout calibration and the neurotransmitter table the"
            " sign overlay is computed from"
            if activity is not None
            else ""
        )
        + ": no timestamp, elapsed time, or host detail enters either, so a re-run"
        " writes byte-identical bytes.",
        "",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


# -------------------------------------------------------------------- output


def write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write ``payload``, creating the parent directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def document_bytes(document: Mapping[str, Any]) -> bytes:
    """The JSON document's exact bytes, sorted and newline-terminated."""

    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def write_documents(output: Path, document: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write the JSON body and its markdown rendering, and return both paths."""

    markdown_path = output.with_suffix(".md")
    write_bytes(output, document_bytes(document))
    write_bytes(markdown_path, render_markdown(document).encode("utf-8"))
    return output, markdown_path


# ----------------------------------------------------------------------- CLI


def _print_summary(document: Mapping[str, Any]) -> None:
    census = document["census"]
    gates = census["integrity_gates"]
    print(f"artifact {census['artifact']['artifact_id']}")
    print(f"integrity gates: {len(gates['gates'])} evaluated, all_passed={gates['all_passed']}")
    header = f"{'candidate':<34} {'n':>5} {'L':>5} {'R':>5}  strict  disjoint"
    print(header)
    for row in census["candidates"]["rows"]:
        print(
            f"{row['name']:<34} {row['members']:>5} {row['left']:>5} {row['right']:>5}"
            f"  {'yes' if row['strict_side_gate_passes'] else 'NO ':>6}"
            f"  {'yes' if row['disjoint_from_runtime_populations'] else 'NO'}"
        )
    activity = document.get("activity")
    if activity is not None:
        print("condition          pool                            last10        ratio")
        for row in activity["rows"]:
            ratio = row["ratio_to_baseline"]
            print(
                f"{row['condition']:<18} {row['pool']:<30}"
                f" {row['pooled_mean_last_10_ticks']:+.4e}"
                f"  {'n/a' if ratio is None else f'{ratio:8.2f}x'}"
            )
    print(f"census_sha256 {document['census_sha256']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--manifest-sha256", default=TRACKED_MANIFEST_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--activity",
        action="store_true",
        help=(
            "additionally step the escape-v2 controller through the six held"
            " conditions and record pooled activity for the two named candidates."
            " Loads the artifact a second time; roughly 6 GB resident"
        ),
    )
    parser.add_argument("--neurotransmitters", type=Path, default=DEFAULT_NEUROTRANSMITTERS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with load_full_runtime_artifact(
        args.artifact, expected_manifest_sha256=args.manifest_sha256
    ) as artifact:
        manifest: FullRuntimeManifest = artifact.manifest
        identity = artifact_identity(manifest)
        declared_sha256 = declared_annotations_sha256(manifest)
        retained_by_superclass = dict(manifest.retained_by_superclass)
        declared_rows = manifest.counts["annotation_source_rows"]
        source_rows = np.array(mapped_numpy(artifact.annotation_source_rows, "<u8"))
        body_ids = np.array(mapped_numpy(artifact.body_ids, "<u8"))

    annotations, _taken = take_retained_annotations(
        args.annotations, source_rows=source_rows, body_ids=body_ids
    )
    census = build_census(
        annotations,
        identity=identity,
        join_gates=annotation_gates(
            annotations,
            declared_annotations_sha256=declared_sha256,
            declared_source_rows=declared_rows,
            retained_by_superclass=retained_by_superclass,
        ),
    )
    document: dict[str, Any] = {
        "census": census,
        "census_sha256": hashlib.sha256(canonical_json_bytes(census)).hexdigest(),
        "activity": None,
    }
    if args.activity:
        document["activity"] = measure_activity(
            activity_pools(census, annotations),
            artifact_directory=args.artifact,
            annotations_path=args.neurotransmitters,
            body_annotations_path=args.annotations,
            calibration_path=args.calibration,
            manifest_sha256=args.manifest_sha256,
        )
    json_path, markdown_path = write_documents(args.output, document)
    _print_summary(document)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
