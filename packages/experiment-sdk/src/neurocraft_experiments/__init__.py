"""Experiment controls and evidence identities for NeuroCraft Fly."""

from .arena import ArenaSpec, load_arena
from .baseline import PrimitiveAction, RawPhysicalObservation, scripted_baseline
from .controls import Edge, degree_preserving_rewire, lesion_mask, shuffled_weights
from .evaluation import ControlCondition, EvaluationProtocol, load_evaluation_protocol
from .identity import DataTier, RunMode, TrialIdentity, canonical_sha256
from .nulls import (
    PERMUTED_SIGN_PROVENANCE_PREFIX,
    permuted_sign_overlay,
    permuted_sign_pairs,
)
from .pilot import (
    PILOT_SCHEMA,
    RESULT_SCHEMA,
    PhysicsConfiguration,
    PilotCondition,
    PilotProtocol,
    build_pilot_result,
    load_pilot_protocol,
    policy_sensor_fields,
    result_bytes,
    run_pilot_trial,
    select_policy_action,
    summarize_trials,
    verify_reexecution,
    verify_result_envelope,
    write_result_no_clobber,
)
from .preregister import ExperimentPlan, TrialCell
from .reservoir import (
    SUPPORTED_CONDITIONS,
    ReservoirConditionError,
    build_condition_engine,
    drive_and_collect,
    reservoir_run_identity,
    ridge_readout,
)

__all__ = [
    "PERMUTED_SIGN_PROVENANCE_PREFIX",
    "PILOT_SCHEMA",
    "RESULT_SCHEMA",
    "SUPPORTED_CONDITIONS",
    "ArenaSpec",
    "ControlCondition",
    "DataTier",
    "Edge",
    "EvaluationProtocol",
    "ExperimentPlan",
    "PhysicsConfiguration",
    "PilotCondition",
    "PilotProtocol",
    "PrimitiveAction",
    "RawPhysicalObservation",
    "ReservoirConditionError",
    "RunMode",
    "TrialCell",
    "TrialIdentity",
    "build_condition_engine",
    "build_pilot_result",
    "canonical_sha256",
    "degree_preserving_rewire",
    "drive_and_collect",
    "lesion_mask",
    "load_arena",
    "load_evaluation_protocol",
    "load_pilot_protocol",
    "permuted_sign_overlay",
    "permuted_sign_pairs",
    "policy_sensor_fields",
    "reservoir_run_identity",
    "result_bytes",
    "ridge_readout",
    "run_pilot_trial",
    "scripted_baseline",
    "select_policy_action",
    "shuffled_weights",
    "summarize_trials",
    "verify_reexecution",
    "verify_result_envelope",
    "write_result_no_clobber",
]
