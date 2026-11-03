"""Strict preregistered control matrices; results live in separate receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arena import read_bounded_file_bytes
from .identity import RunMode, canonical_sha256

MAX_EVALUATION_PROTOCOL_BYTES = 1024 * 1024
_MANDATORY_MODES = {
    RunMode.FAITHFUL,
    RunMode.SHUFFLED_WEIGHT,
    RunMode.DEGREE_PRESERVING_RANDOM,
    RunMode.LESION,
    RunMode.SCRIPTED_BASELINE,
    RunMode.PASSIVE_PHYSICS,
    RunMode.SIGN_SENSITIVITY,
    RunMode.ADAPTER_PERMUTATION,
    RunMode.DECODER_PERMUTATION,
}
_MANDATORY_CONDITION_IDS = {
    "faithful",
    "shuffled-weight",
    "degree-preserving",
    "dng07-lesion",
    "dng07-sham",
    "dng07-contralateral",
    "dng07-half-dose",
    "scripted",
    "passive-physics",
    "no-input",
    "null-graph",
    "frozen-state",
    "sign-zero",
    "sensor-permutation",
    "motor-permutation",
    "synthetic",
}


@dataclass(frozen=True, slots=True)
class ControlCondition:
    condition_id: str
    mode: RunMode
    intervention: str
    graph_changes: bool
    receives_sensors: bool


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    protocol_id: str
    status: str
    hypothesis: str
    arenas: tuple[str, ...]
    seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    held_out_seeds: tuple[int, ...]
    replicates: int
    conditions: tuple[ControlCondition, ...]
    primary_metrics: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.raw)

    @property
    def planned_trial_count(self) -> int:
        return len(self.arenas) * len(self.seeds) * self.replicates * len(self.conditions)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a non-empty string array")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicates")
    return result


def _parse_condition(value: Any) -> ControlCondition:
    if not isinstance(value, dict):
        raise ValueError("each condition must be an object")
    required = {"condition_id", "mode", "intervention", "graph_changes", "receives_sensors"}
    if set(value) != required:
        raise ValueError(f"condition fields must be exactly {sorted(required)}")
    graph_changes = value["graph_changes"]
    receives_sensors = value["receives_sensors"]
    if not isinstance(graph_changes, bool) or not isinstance(receives_sensors, bool):
        raise ValueError("condition boolean fields must be booleans")
    return ControlCondition(
        condition_id=str(value["condition_id"]),
        mode=RunMode(value["mode"]),
        intervention=str(value["intervention"]),
        graph_changes=graph_changes,
        receives_sensors=receives_sensors,
    )


def load_evaluation_protocol(path: str | Path) -> EvaluationProtocol:
    document = read_bounded_file_bytes(
        Path(path),
        maximum_bytes=MAX_EVALUATION_PROTOCOL_BYTES,
        label="evaluation protocol",
    )
    raw = json.loads(document.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evaluation protocol root must be an object")
    if raw.get("protocol_id") != "neurocraft-control-matrix-v2":
        raise ValueError("this loader accepts only control-matrix-v2")
    required = {
        "protocol_id",
        "status",
        "amendment",
        "hypothesis",
        "arenas",
        "seeds",
        "seed_split",
        "replicates",
        "conditions",
        "primary_metrics",
        "primary_endpoint",
        "fixed_budgets",
        "analysis",
        "claim_boundary",
    }
    if set(raw) != required:
        raise ValueError(f"protocol fields must be exactly {sorted(required)}")
    if raw["status"] != "preregistered_no_trials_run":
        raise ValueError("this loader accepts only a pre-run protocol, never embedded results")
    amendment = raw["amendment"]
    if not isinstance(amendment, dict) or set(amendment) != {
        "supersedes",
        "prior_sha256",
        "timing",
        "reason",
    }:
        raise ValueError("amendment fields are incomplete or unknown")
    if (
        amendment["supersedes"] != "neurocraft-control-matrix-v1"
        or amendment["prior_sha256"]
        != "1584a474b394995fc2701c996c404cae15955726beb08cc788f6de2d17631d6e"
        or amendment["timing"] != "before_any_minecraft_control_matrix_trial"
        or not isinstance(amendment["reason"], str)
        or not amendment["reason"]
    ):
        raise ValueError("control-matrix-v2 amendment lineage is invalid")
    arenas = _string_tuple(raw["arenas"], "arenas")
    primary_metrics = _string_tuple(raw["primary_metrics"], "primary_metrics")
    seeds_value = raw["seeds"]
    if (
        not isinstance(seeds_value, list)
        or not seeds_value
        or not all(isinstance(seed, int) and 0 <= seed <= 2**63 - 1 for seed in seeds_value)
    ):
        raise ValueError("seeds must be non-negative signed 64-bit integers")
    seeds = tuple(seeds_value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds contain duplicates")
    seed_split = raw["seed_split"]
    if not isinstance(seed_split, dict) or set(seed_split) != {
        "development",
        "held_out_confirmation",
        "freeze_rule",
    }:
        raise ValueError(
            "seed_split fields must be development, held_out_confirmation, freeze_rule"
        )
    development_value = seed_split["development"]
    held_out_value = seed_split["held_out_confirmation"]
    if (
        not isinstance(development_value, list)
        or not development_value
        or not all(isinstance(seed, int) for seed in development_value)
        or not isinstance(held_out_value, list)
        or not held_out_value
        or not all(isinstance(seed, int) for seed in held_out_value)
    ):
        raise ValueError("development and held-out seed splits must be non-empty integer arrays")
    development_seeds = tuple(development_value)
    held_out_seeds = tuple(held_out_value)
    if set(development_seeds).intersection(held_out_seeds):
        raise ValueError("development and held-out seed splits must be disjoint")
    if set(development_seeds).union(held_out_seeds) != set(seeds):
        raise ValueError("seed split must partition the declared seeds exactly")
    if not isinstance(seed_split["freeze_rule"], str) or not seed_split["freeze_rule"]:
        raise ValueError("seed_split.freeze_rule must be a non-empty string")
    replicates = raw["replicates"]
    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    conditions_value = raw["conditions"]
    if not isinstance(conditions_value, list) or not conditions_value:
        raise ValueError("conditions must be a non-empty array")
    conditions = tuple(_parse_condition(item) for item in conditions_value)
    condition_ids = [condition.condition_id for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition IDs contain duplicates")
    missing_condition_ids = _MANDATORY_CONDITION_IDS.difference(condition_ids)
    if missing_condition_ids:
        raise ValueError(f"mandatory control conditions missing: {sorted(missing_condition_ids)}")
    observed_modes = {condition.mode for condition in conditions}
    missing = _MANDATORY_MODES.difference(observed_modes)
    if missing:
        missing_names = sorted(mode.value for mode in missing)
        raise ValueError(f"mandatory control modes missing: {missing_names}")
    if not any(not condition.receives_sensors for condition in conditions):
        raise ValueError("protocol requires a no-input control")
    primary_endpoint = raw["primary_endpoint"]
    required_endpoint = {
        "metric",
        "comparison",
        "population",
        "direction",
        "minimum_effect",
        "decision_rule",
    }
    if not isinstance(primary_endpoint, dict) or set(primary_endpoint) != required_endpoint:
        raise ValueError(f"primary_endpoint fields must be exactly {sorted(required_endpoint)}")
    if not all(isinstance(value, str) and value for value in primary_endpoint.values()):
        raise ValueError("primary_endpoint values must be non-empty strings")
    if primary_endpoint["metric"] not in primary_metrics:
        raise ValueError("primary endpoint metric must be one of primary_metrics")
    analysis = raw["analysis"]
    required_analysis = {
        "pairing",
        "primary_comparisons",
        "interval",
        "multiplicity",
        "trial_count",
        "replicate_role",
        "replicate_aggregation",
        "precision_rule",
        "missing_trial",
        "exclusions",
        "exploratory_labels",
    }
    if not isinstance(analysis, dict) or set(analysis) != required_analysis:
        raise ValueError(f"analysis fields must be exactly {sorted(required_analysis)}")
    comparisons = analysis.get("primary_comparisons")
    if not isinstance(comparisons, list) or primary_endpoint["comparison"] not in comparisons:
        raise ValueError("primary endpoint comparison must be declared in primary_comparisons")
    planned_trial_count = len(arenas) * len(seeds) * replicates * len(conditions)
    if analysis.get("trial_count") != planned_trial_count:
        raise ValueError("analysis.trial_count must equal the declared matrix size")
    if (
        "not independent statistical units" not in analysis["replicate_role"]
        or "arena-seed-condition" not in analysis["replicate_aggregation"]
    ):
        raise ValueError("replicates must be explicitly non-independent and pre-aggregated")
    return EvaluationProtocol(
        protocol_id=str(raw["protocol_id"]),
        status=str(raw["status"]),
        hypothesis=str(raw["hypothesis"]),
        arenas=arenas,
        seeds=seeds,
        development_seeds=development_seeds,
        held_out_seeds=held_out_seeds,
        replicates=replicates,
        conditions=conditions,
        primary_metrics=primary_metrics,
        raw=raw,
    )
