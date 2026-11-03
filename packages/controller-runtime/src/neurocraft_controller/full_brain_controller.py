"""Opt-in full-MaleCNS wing-motor controller profile (``full_mcns_wing_v1``).

This module drives the same sensory entries and wing-motor readout the 128-node
steering profile uses, but the recurrent context between them is the ENTIRE
verified MaleCNS runtime artifact -- 166,700 neurons / 25,582,938 signed edges
-- executed by the array-native backend
(:class:`neurocraft_neural.full_engine.ArrayNativeEngine`). Same I/O boundary,
whole-brain recurrent dynamics.

It is deliberately **not** a :class:`~neurocraft_controller.controller.ConnectomeController`
subclass: the parent hard-binds the byte-frozen reference engine and a fixture
``RuntimeGraph``, neither of which exists at full scale. Instead this class
duck-types the exact surface the service consumes
(``identity`` / ``step(observation, command_seq=...)`` / ``reset()`` /
``last_evidence`` / ``configuration``), which is the same new-module pattern
``flight_controller.py`` used to avoid touching pilot-pinned bytes.

Claim discipline (ADR 0008 mode taxonomy)
-----------------------------------------
* The edge signs come from the disclosed coarse presynaptic-consensus policy
  (``SIGN-OVERLAY-02``); they are a modeling choice over annotation data, not
  measured physiology. Any run of this profile must carry that caveat.
* The array-native backend has its **own dynamics identity**
  (``array_native_csr_float64_uniform_delay1``), distinct from the reference
  engine the fixture pilots pin.
* Building this controller is evidence of construction only. No claim is made
  here that the full graph has driven Minecraft; that run has not happened.

What is byte-shared with the pinned boundary
--------------------------------------------
* Feature encoding: the pinned :class:`~neurocraft_controller.features.PhysicalFeatureEncoder`.
* Sensor channels: ``ConnectomeController._sensor_channels`` over the pinned
  default configuration (entry bodies 805722 / 77185), re-indexed into the
  full graph by ``FullRuntimeGraph.index_of``.
* Motor pools: the declared somaSide split from
  :mod:`neurocraft_controller.flight_pools` (25 left + 27 right, unit weights),
  gain 8.0 / dead zone 0.0 declared a priori (AB-10: never re-tuned).
* Primitive projection: ``ConnectomeController._project_motor_frame`` and its
  pinned ``PROJECTION_MATERIAL``, called directly.

Declared readout boundary calibration
-------------------------------------
The a-priori gain 8.0 was declared for the *fixture-scale* boundary, where a
52-member pool mean is orders of magnitude larger than it is here: at full
scale the row-normalized weights dilute a two-entry sensory signal across each
motor neuron's thousands of mostly-quiet inputs, so the pool means -- and the
actuation derived from them -- sit near 1e-5. ``readout_scale`` is one scalar
boundary constant that closes that unit gap. It is:

* **declared, not tuned**: measured once by ``scripts/calibrate_full_readout.py``
  under a named deterministic observation stream and frozen into a committed
  calibration document, whose SHA-256 is bound into both the configuration
  identity and the decoder identity (a different calibration is a different
  decoder);
* **one number, applied uniformly**: the same scalar multiplies both pools
  after the declared gain and before the existing clip and dead zone. There is
  no baseline subtraction, no per-axis shaping, and no change to the dynamics
  or to the gain the engine executes -- a persistent readout bias stays visible
  in the calibrated output rather than being removed;
* **not a behavior claim**: rescaling the readout changes the units of what the
  brain already produced. Nothing here is evidence that behavior improved.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from neurocraft_neural.adapters import MotorFrame, SensorChannel
from neurocraft_neural.full_controls import (
    FullControlIdentity,
    FullControlKind,
    FullControlView,
    build_control_view,
)
from neurocraft_neural.full_engine import ArrayNativeEngine, ArrayStepResult
from neurocraft_neural.spmv_kernel import MAX_THREADS, load_kernel
from neurocraft_protocol import (
    ACTION_TARGET_OFFSET_TICKS,
    Action,
    MotorVector,
    Observation,
    ProtocolError,
    RunIdentity,
    RunMode,
    SensorFrame,
    parse_json_frame,
    require_sensor_contract,
)

from .config import ControllerConfiguration, canonical_sha256
from .controller import ConnectomeController, ControllerIntegrityError
from .embodied_senses import (
    EEM_MODE_TAXONOMY,
    EemEncoderDocumentEscapeV2,
    EemEncoderDocumentV2,
    EmbodiedSensesBinding,
    _verified_graph_array,
    apply_temporal_contrast,
    bind_embodied_senses,
    bind_embodied_senses_escape_v2,
    bind_embodied_senses_v2,
    encode_channel_values,
    load_embodied_population_binding,
    population_drives,
)
from .escape_reflex import (
    ESCAPE_JUMP_FORMULA,
    ESCAPE_JUMP_FORMULA_V2,
    TtmnPoolBinding,
    load_escape_ttmn_binding,
)
from .events_senses import (
    CompletedEventReadout,
    EventEncoderBinding,
    EventFeatureEncoder,
    EventInputState,
    EventPopulationBinding,
    EventPulse,
    completed_event_readout,
    load_event_populations,
)
from .features import EntityRayFeatureEncoder, PhysicalFeatureEncoder
from .flight_pools import LEFT_WING_MOTOR_BODY_IDS, RIGHT_WING_MOTOR_BODY_IDS
from .interaction_senses import (
    InteractionEncoderBinding,
    InteractionFeatureEncoder,
    InteractionInputState,
    InteractionPulse,
)
from .loom_senses import EemEncoderDocumentLoomV1, bind_embodied_senses_loom_v1
from .loom_senses_v2 import (
    EemEncoderDocumentLoomV2,
    bind_embodied_senses_loom_v2,
    encode_entity_channel_values,
)
from .steering_readout import SteeringPairBinding, load_steering_binding

FULL_BRAIN_CONFIG_SCHEMA = "neurocraft.controller.full_brain_config.v1"
FULL_BRAIN_SENSORY_BINDING_SCHEMA = "neurocraft.full_brain.sensory_binding.v1"
FULL_BRAIN_ENCODED_STIMULATION_SCHEMA = "neurocraft.full_brain.encoded_stimulation.v1"
FULL_BRAIN_DECODER_SCHEMA = "neurocraft.full_brain.decoder.v1"

#: Engineered-Encoder Mode (ADR 0008 Mode 3) schema identifiers. The EEM
#: profile carries its own configuration schema so an engineered-encoder run
#: can never be described by a wing (``full_brain_config.v1``) identity, its
#: own adapter-identity schema chaining the population binding and encoder
#: document, and its own stimulation-provenance schema.
FULL_BRAIN_EEM_CONFIG_SCHEMA = "neurocraft.controller.full_brain_eem_config.v1"
FULL_BRAIN_EEM_ADAPTER_SCHEMA = "neurocraft.boundary_adapter.identity.eem.v1"
FULL_BRAIN_EEM_ENCODED_STIMULATION_SCHEMA = "neurocraft.full_brain.encoded_stimulation.eem.v1"

#: Temporal-contrast Engineered-Encoder profile (``full_mcns_eem_v2``): its
#: own configuration schema (an eem-v2 run can never be described by a v1 EEM
#: or wing identity) and its own stimulation-provenance schema chaining the
#: raw channel values, the contrast baselines actually used, the encoded
#: contrast values, and the per-population drives.
FULL_BRAIN_EEM_V2_CONFIG_SCHEMA = "neurocraft.controller.full_brain_eem_v2_config.v1"
FULL_BRAIN_EEM_V2_ENCODED_STIMULATION_SCHEMA = "neurocraft.full_brain.encoded_stimulation.eem.v2"

#: Escape-reflex Engineered-Encoder profile (``full_mcns_escape_v1``,
#: identity recorded in experiments/preregistration/eem-escape-v1.md): the
#: eem v1 sensory boundary verbatim (absolute encoder, same populations and
#: per-capita placement) plus a declared third motor pool -- the TTMn
#: jump-motoneuron pair -- whose pooled activity adds one declared term to
#: the vertical thrust axis. Its own configuration schema (an escape run can
#: never be described by a wing or eem identity) and its own decoder schema
#: chaining the wing readout, the TTMn membership, the jump mapping formula,
#: both declared scales, and the encoder identity.
FULL_BRAIN_ESCAPE_CONFIG_SCHEMA = "neurocraft.controller.full_brain_escape_config.v1"
FULL_BRAIN_ESCAPE_DECODER_SCHEMA = "neurocraft.full_brain.decoder.escape.v1"

#: Escape-reflex v2 profile (``full_mcns_escape_v2``, identity recorded in
#: experiments/preregistration/eem-escape-v2.md): the escape lane's declared
#: follow-up to the recorded D-063/D-064 live observations. Two structural
#: changes against escape v1 -- the ``touch`` channel alone moves under the
#: frozen D-059 temporal-contrast formula (the escape-v2 encoder document),
#: and the jump mapping is rectified to the non-negative half of its declared
#: codomain (``ESCAPE_JUMP_FORMULA_V2``). Its own configuration and decoder
#: schemas (an escape-v2 run can never be described by a wing, eem, or
#: escape-v1 identity) and its own stimulation-provenance schema.
FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA = "neurocraft.controller.full_brain_escape_config.v2"
FULL_BRAIN_ESCAPE_V2_DECODER_SCHEMA = "neurocraft.full_brain.decoder.escape.v2"
FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA = (
    "neurocraft.full_brain.encoded_stimulation.eem.escape.v2"
)

#: Approach-gated looming profile (``full_mcns_loom_v1``, declared in
#: ``docs/encoders/eem-loom-v1.md``): the escape-v2 DECODER unchanged (the
#: rectified TTMn jump term plus the wing common term) over the loom-v1
#: encoder document, whose three threat channels drive their LC4 populations
#: with the rectified rise of proximity above a declared leaky mean instead of
#: with the absolute level. Its own configuration schema (a loom run can never
#: be described by a wing, eem, escape-v1, or escape-v2 configuration
#: identity) and its own per-step stimulation-provenance schema. The decoder
#: document is deliberately the escape-v2 one verbatim -- the loom lane
#: changes the sensory boundary only -- and its identity still separates from
#: escape-v2's because the decoder material chains
#: ``encoder_document_sha256``.
FULL_BRAIN_LOOM_V2_CONFIG_SCHEMA = "neurocraft.controller.full_brain_loom_config.v2"
FULL_BRAIN_EEM_LOOM_V2_ENCODED_STIMULATION_SCHEMA = (
    "neurocraft.full_brain.eem_loom_v2.stimulation.v1"
)
FULL_BRAIN_LOOM_V1_CONFIG_SCHEMA = "neurocraft.controller.full_brain_loom_config.v1"
FULL_BRAIN_EEM_LOOM_V1_ENCODED_STIMULATION_SCHEMA = (
    "neurocraft.full_brain.encoded_stimulation.eem.loom.v1"
)

#: The declared name of the escape profile's third motor pool.
ESCAPE_JUMP_POOL_NAME = "jump_motor"

#: Schema of the committed readout calibration document written by
#: ``scripts/calibrate_full_readout.py`` and consumed by
#: :meth:`FullBrainController.from_artifact`.
FULL_READOUT_CALIBRATION_SCHEMA = "neurocraft.full_readout_calibration.v1"

#: Schema of the escape profile's committed calibration document
#: (``scripts/calibrate_full_readout.py --profile escape``): the v1 layout
#: extended with named scales {wing_scale, jump_scale} measured in ONE
#: standard-stream run, plus a ``jump_readout`` section binding the TTMn
#: pool identity. A distinct schema so the v1 loader refuses an escape
#: document and the escape loader refuses every v1 document, in both
#: directions.
FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA = "neurocraft.full_readout_calibration.escape.v1"

#: Schema of the escape-v2 profile's committed calibration document
#: (``scripts/calibrate_full_readout.py --profile escape-v2``): the escape-v1
#: layout byte-for-byte under its own schema and stream id, cross-bound to
#: the escape-v2 encoder identity and the TTMn pool identity. A distinct
#: schema so every other boundary's loader refuses an escape-v2 document and
#: the escape-v2 loader refuses every other document, in both directions.
FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA = "neurocraft.full_readout_calibration.escape.v2"

#: Schema of the loom-v1 profile's committed calibration document
#: (``scripts/calibrate_full_readout.py --profile loom-v1``): the escape
#: layout byte-for-byte under its own schema and stream id, cross-bound to the
#: loom-v1 encoder identity and the TTMn pool identity. A distinct schema so
#: every other boundary's loader refuses a loom-v1 document and the loom-v1
#: loader refuses every other document, in both directions.
FULL_READOUT_CALIBRATION_LOOM_V2_SCHEMA = "neurocraft.full_readout_calibration.loom.v2"
FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA = "neurocraft.full_readout_calibration.loom.v1"

#: Upper bound on the declared readout scale. The measured full-scale pool
#: magnitudes are ~1e-5, so a legitimate calibration lands in the thousands;
#: the bound exists to reject nonsense (and overflow) rather than to shape
#: anything.
MAX_READOUT_SCALE = 1.0e6

#: Sign-overlay provenance prefix chained into the runtime view identity. The
#: suffix is the executed overlay's ``identity_sha256``, which itself binds the
#: int8 sign bytes and the artifact manifest (see ``full_signs.py``).
SIGN_OVERLAY_POLICY_PREFIX = "coarse_presynaptic_consensus_v1"

#: External pin for the full MaleCNS runtime artifact's ``manifest.json``.
#: Portable-builder candidate, regenerated from verified pinned sources on
#: 2026-10-24. Structural arrays and lineage are byte-identical to the historical
#: ``docs/receipts/RUNTIME-MCNS-01.json`` artifact; the builder implementation
#: and manifest identities supersede it. Historical receipts retain their scope.
#: ``load_full_runtime_artifact`` refuses any directory whose manifest bytes do
#: not hash to exactly this value, so a colocated manifest can never authorize
#: itself.
FULL_MCNS_ARTIFACT_MANIFEST_SHA256 = (
    "0485f7fe5aeede0a23e71d8191825d5b8129abc46a38c10563c94c604a840546"
)

_SHA256_HEX_LENGTH = 64


class FullBrainProfileError(RuntimeError):
    """The full-brain profile could not be constructed exactly as declared."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: str, label: str) -> str:
    if not _is_sha256(value):
        raise FullBrainProfileError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class FullBrainConfiguration:
    """Declared assumptions of the full-brain wing profile, under its own schema.

    ``mode`` is faithful (the unchanged default), shuffled_weight, or no_input.
    Each control requires its matching content-addressed view binding. A lesion
    remains unsupported: it would re-verify and re-digest
    166,700-node state around every ablation change
    (``ArrayNativeEngine.set_ablated`` runs two unconditional full-view
    verifications), which exceeds the per-tick compute budget. That limit is
    disclosed here and enforced below rather than silently degraded.

    ``kernel_threads`` is an execution parameter, not an identity input: the C
    kernel is bit-identical across thread counts (its identity material pins
    ``thread_count_affects_results: false``), so the thread count is excluded
    from :meth:`to_identity_material` exactly as it is excluded from the
    backend's dynamics identity. ``None`` defers to the kernel loader's default
    (the ``NEUROCRAFT_SPMV_THREADS`` environment variable, else a CPU-derived
    cap of 8).

    ``readout_scale`` and ``readout_calibration_sha256`` are a matched pair and
    are validated in both directions: the default ``1.0`` is the *uncalibrated*
    readout and must carry no digest, while any other scale must name the
    64-hex committed calibration document it was read from. Both enter
    :meth:`to_identity_material`, so a calibrated controller can never be
    confused with an uncalibrated one.
    """

    schema: str = FULL_BRAIN_CONFIG_SCHEMA
    mode: str = "faithful"
    sign_policy: str = "coarse_presynaptic_consensus"
    neural_steps_per_game_tick: int = 5
    motor_pool_gain: float = 8.0
    motor_dead_zone: float = 0.0
    kernel_threads: int | None = None
    readout_scale: float = 1.0
    readout_calibration_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_CONFIG_SCHEMA:
            raise ValueError("unsupported full-brain controller configuration schema")
        self._validate_shared()

    def _validate_shared(self) -> None:
        """Everything but the schema gate, shared with the EEM subclass verbatim."""

        if self.mode == "lesion":
            raise ValueError(
                "the full-brain profile v1 supports faithful mode only: lesion "
                "interventions re-verify and re-digest full-graph state around every "
                "ablation change, which exceeds the per-tick compute budget at "
                "166,700 nodes; this is a disclosed v1 limitation, not a hidden fallback"
            )
        if self.mode not in ("faithful", "shuffled_weight", "no_input"):
            raise ValueError(
                "full-brain controller mode must be faithful, shuffled_weight, or no_input"
            )
        if self.sign_policy != "coarse_presynaptic_consensus":
            raise ValueError(
                "the full-brain profile v1 executes only the disclosed "
                "coarse_presynaptic_consensus sign overlay"
            )
        if self.neural_steps_per_game_tick != 5:
            raise ValueError("the v1 clock contract requires exactly five neural steps per tick")
        if not math.isfinite(self.motor_pool_gain) or not 0.0 <= self.motor_pool_gain <= 8.0:
            raise ValueError("motor_pool_gain must be finite and in [0, 8]")
        if not math.isfinite(self.motor_dead_zone) or not 0.0 <= self.motor_dead_zone < 1.0:
            raise ValueError("motor_dead_zone must be finite and in [0, 1)")
        if self.kernel_threads is not None and (
            isinstance(self.kernel_threads, bool)
            or not isinstance(self.kernel_threads, int)
            or not 1 <= self.kernel_threads <= MAX_THREADS
        ):
            raise ValueError(f"kernel_threads must be None or an integer in [1, {MAX_THREADS}]")
        if (
            not math.isfinite(self.readout_scale)
            or not 0.0 < self.readout_scale <= MAX_READOUT_SCALE
        ):
            raise ValueError(f"readout_scale must be finite and in (0, {MAX_READOUT_SCALE}]")
        if self.readout_scale == 1.0:
            if self.readout_calibration_sha256 != "":
                raise ValueError(
                    "the uncalibrated readout (readout_scale 1.0) must carry no "
                    "readout_calibration_sha256"
                )
        elif not _is_sha256(self.readout_calibration_sha256):
            raise ValueError(
                "a calibrated readout_scale requires the lowercase SHA-256 hex digest "
                "of the committed calibration document"
            )

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "motor_dead_zone": float(self.motor_dead_zone).hex(),
            "motor_pool_gain": float(self.motor_pool_gain).hex(),
            "neural_steps_per_game_tick": self.neural_steps_per_game_tick,
            "readout_calibration_sha256": self.readout_calibration_sha256,
            "readout_scale": float(self.readout_scale).hex(),
            "schema": self.schema,
            "sign_policy": self.sign_policy,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


@dataclass(frozen=True, slots=True)
class FullBrainEemConfiguration(FullBrainConfiguration):
    """Declared assumptions of the Engineered-Encoder profile (``full_mcns_eem_v1``).

    ADR 0008 Mode 3 (Engineered-Encoder Mode), never Faithful Organism Mode:
    ``mode_taxonomy`` is pinned to ``"engineered_encoder"`` and enters the
    identity material, and ``encoder_identity_sha256`` must name the exact
    encoder document driving the labeled populations, so an EEM run always
    carries a configuration identity distinct from every wing identity. All
    shared parameters keep the wing profile's validation verbatim. ``mode``
    remains the *controls*-taxonomy field (faithful = no control overlay)
    exactly as it does for the trained-readout profile.
    """

    schema: str = FULL_BRAIN_EEM_CONFIG_SCHEMA
    mode_taxonomy: str = EEM_MODE_TAXONOMY
    encoder_identity_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_EEM_CONFIG_SCHEMA:
            raise ValueError("unsupported engineered-encoder configuration schema")
        if self.mode_taxonomy != EEM_MODE_TAXONOMY:
            raise ValueError(
                "the engineered-encoder configuration must declare mode_taxonomy "
                f"{EEM_MODE_TAXONOMY!r}"
            )
        if not _is_sha256(self.encoder_identity_sha256):
            raise ValueError(
                "the engineered-encoder configuration must name its encoder document "
                "identity as a lowercase SHA-256 hex digest"
            )
        self._validate_shared()

    def to_identity_material(self) -> dict[str, Any]:
        # Explicit base call rather than zero-arg super(): dataclass slots
        # recreation and closure cells interact badly across versions, and the
        # identity material must never depend on that subtlety.
        return {
            **FullBrainConfiguration.to_identity_material(self),
            "encoder_identity_sha256": self.encoder_identity_sha256,
            "mode_taxonomy": self.mode_taxonomy,
        }


@dataclass(frozen=True, slots=True)
class FullBrainEemV2Configuration(FullBrainEemConfiguration):
    """Declared assumptions of the temporal-contrast EEM profile (``full_mcns_eem_v2``).

    Identical parameter surface and validation to the v1 EEM configuration,
    under its own schema so the two can never share a configuration identity;
    ``encoder_identity_sha256`` must name a v2 (temporal-contrast) encoder
    document, and the controller refuses a v1/v2 config-document mix in both
    directions. The contrast constants themselves (alpha, formulas, the touch
    exception) live in the hashed encoder document, so they are chained into
    this identity through ``encoder_identity_sha256`` rather than repeated
    here. Still ADR 0008 Mode 3, never Faithful Organism Mode.
    """

    schema: str = FULL_BRAIN_EEM_V2_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_EEM_V2_CONFIG_SCHEMA:
            raise ValueError(
                "unsupported temporal-contrast engineered-encoder configuration schema"
            )
        if self.mode_taxonomy != EEM_MODE_TAXONOMY:
            raise ValueError(
                "the engineered-encoder configuration must declare mode_taxonomy "
                f"{EEM_MODE_TAXONOMY!r}"
            )
        if not _is_sha256(self.encoder_identity_sha256):
            raise ValueError(
                "the engineered-encoder configuration must name its encoder document "
                "identity as a lowercase SHA-256 hex digest"
            )
        self._validate_shared()


@dataclass(frozen=True, slots=True)
class FullBrainEscapeConfiguration(FullBrainEemConfiguration):
    """Declared assumptions of the escape-reflex profile (``full_mcns_escape_v1``).

    The eem v1 configuration surface -- ``encoder_identity_sha256`` must name
    the v1 (absolute) encoder document, ``mode_taxonomy`` stays
    ``engineered_encoder`` -- under its own schema, plus one field:
    ``jump_scale``, the declared boundary scale of the TTMn jump readout.
    ``readout_scale`` remains the wing pools' scale (re-measured under this
    profile so ONE calibration document binds the whole boundary), and the
    two scales pair with ``readout_calibration_sha256`` in both directions:
    the uncalibrated profile carries scale 1.0 on both and no digest, while a
    calibrated profile must carry both measured scales from the single named
    escape calibration document. Still ADR 0008 Mode 3, never Faithful
    Organism Mode.
    """

    schema: str = FULL_BRAIN_ESCAPE_CONFIG_SCHEMA
    jump_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_ESCAPE_CONFIG_SCHEMA:
            raise ValueError("unsupported escape-reflex configuration schema")
        self._validate_escape_boundary()

    def _validate_escape_boundary(self) -> None:
        """Everything but the schema gate, shared by every escape-lane generation.

        Extracted verbatim from the escape-v1 body that escape-v2 and loom-v1
        each duplicated; every message and every ordering is unchanged, so the
        three configuration identities and their refusals are byte-identical
        to what each class validated before.
        """

        if self.mode_taxonomy != EEM_MODE_TAXONOMY:
            raise ValueError(
                "the engineered-encoder configuration must declare mode_taxonomy "
                f"{EEM_MODE_TAXONOMY!r}"
            )
        if not _is_sha256(self.encoder_identity_sha256):
            raise ValueError(
                "the engineered-encoder configuration must name its encoder document "
                "identity as a lowercase SHA-256 hex digest"
            )
        if not math.isfinite(self.jump_scale) or not 0.0 < self.jump_scale <= MAX_READOUT_SCALE:
            raise ValueError(f"jump_scale must be finite and in (0, {MAX_READOUT_SCALE}]")
        if self.jump_scale == 1.0:
            if self.readout_calibration_sha256 != "":
                raise ValueError(
                    "a calibrated escape configuration must carry the measured "
                    "jump_scale from its calibration document (jump_scale 1.0 is "
                    "the uncalibrated readout)"
                )
        elif self.readout_calibration_sha256 == "":
            raise ValueError(
                "a declared jump_scale requires the SHA-256 of the committed escape "
                "calibration document it was read from"
            )
        self._validate_shared()

    def to_identity_material(self) -> dict[str, Any]:
        return {
            **FullBrainEemConfiguration.to_identity_material(self),
            "jump_scale": float(self.jump_scale).hex(),
        }


@dataclass(frozen=True, slots=True)
class FullBrainEscapeV2Configuration(FullBrainEscapeConfiguration):
    """Declared assumptions of the escape-reflex v2 profile (``full_mcns_escape_v2``).

    The escape-v1 parameter surface and validation verbatim, under its own
    schema so the two can never share a configuration identity;
    ``encoder_identity_sha256`` must name the escape-v2 (transient-touch)
    encoder document, and the controller refuses every other
    configuration/encoder pairing in both directions. The rectified jump
    formula and the touch-contrast constants live in the hashed decoder and
    encoder documents, so they are chained into this identity through
    ``encoder_identity_sha256`` and the decoder identity rather than repeated
    here. Still ADR 0008 Mode 3, never Faithful Organism Mode.
    """

    schema: str = FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_ESCAPE_V2_CONFIG_SCHEMA:
            raise ValueError("unsupported escape-reflex v2 configuration schema")
        self._validate_escape_boundary()


@dataclass(frozen=True, slots=True)
class FullBrainLoomV1Configuration(FullBrainEscapeConfiguration):
    """Declared assumptions of the approach-gated loom profile (``full_mcns_loom_v1``).

    The escape parameter surface and validation verbatim, under its own schema
    so no two of the four escape-lane generations can share a configuration
    identity; ``encoder_identity_sha256`` must name the loom-v1
    (approach-gated) encoder document, and the controller refuses every other
    configuration/encoder pairing in both directions. The rectification set,
    alpha, and the declared formulas live in the hashed encoder document, and
    the rectified jump formula lives in the (unchanged) escape-v2 decoder
    document, so both are chained into this identity through
    ``encoder_identity_sha256`` and the decoder identity rather than repeated
    here. Still ADR 0008 Mode 3 (Engineered-Encoder Mode), never Faithful
    Organism Mode.
    """

    schema: str = FULL_BRAIN_LOOM_V1_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_LOOM_V1_CONFIG_SCHEMA:
            raise ValueError("unsupported approach-gated loom configuration schema")
        self._validate_escape_boundary()


@dataclass(frozen=True, slots=True)
class FullBrainLoomV2Configuration(FullBrainLoomV1Configuration):
    """Entity-gated loom encoder over the existing rectified jump arithmetic."""

    schema: str = FULL_BRAIN_LOOM_V2_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FULL_BRAIN_LOOM_V2_CONFIG_SCHEMA:
            raise ValueError("unsupported entity-gated loom configuration schema")
        self._validate_escape_boundary()


@dataclass(frozen=True, slots=True)
class FullBrainEventsConfiguration(FullBrainLoomV2Configuration):
    """Candidate composite; inherited encoder/calibration fields name components only."""

    schema: str = "neurocraft.full_brain.events.configuration.v1"
    event_encoder_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != "neurocraft.full_brain.events.configuration.v1":
            raise ValueError("unsupported candidate event configuration schema")
        self._validate_escape_boundary()
        if not _is_sha256(self.event_encoder_sha256):
            raise ValueError("candidate event configuration requires its composite encoder hash")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            **FullBrainLoomV2Configuration.to_identity_material(self),
            "event_encoder_sha256": self.event_encoder_sha256,
            "calibration_scope": "reused loom-v2 component; composite not behavior-validated",
        }


@dataclass(frozen=True, slots=True)
class FullBrainInteractionsConfiguration(FullBrainEventsConfiguration):
    """V4 actor transport and separate bristle attack input; no new body policy implied."""

    schema: str = "neurocraft.full_brain.interactions.configuration.v2"

    def __post_init__(self) -> None:
        if self.schema != "neurocraft.full_brain.interactions.configuration.v2":
            raise ValueError("unsupported interaction configuration schema")
        self._validate_escape_boundary()
        if not _is_sha256(self.event_encoder_sha256):
            raise ValueError("interaction configuration requires its composite encoder hash")


#: The configuration types that declare the RECTIFIED (escape-v2) jump decoder
#: binding. The loom lane reuses that decoder document unchanged, so the
#: v1/v2 decoder-generation pairing gate is expressed over this tuple rather
#: than over ``FullBrainEscapeV2Configuration`` alone.
RECTIFIED_JUMP_CONFIGURATIONS: tuple[type[FullBrainEscapeConfiguration], ...] = (
    FullBrainEscapeV2Configuration,
    FullBrainLoomV1Configuration,
)


@dataclass(frozen=True, slots=True)
class BoundSensorTarget:
    """One sensory entry body resolved to its full-graph dense index."""

    body_id: int
    dense_index: int
    weight: float


@dataclass(frozen=True, slots=True)
class BoundSensorChannel:
    """One pinned sensor channel with its targets bound to full-graph indices."""

    name: str
    unit: str
    gain: float
    offset: float
    clip_min: float
    clip_max: float
    saturation: str
    targets: tuple[BoundSensorTarget, ...]


@dataclass(frozen=True, slots=True)
class FullSensoryBinding:
    """The canonical sensory-binding document, hashed into ``adapter_hash``."""

    graph_manifest_sha256: str
    node_count: int
    channels: tuple[BoundSensorChannel, ...]

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "clip_max": float(channel.clip_max).hex(),
                    "clip_min": float(channel.clip_min).hex(),
                    "gain": float(channel.gain).hex(),
                    "name": channel.name,
                    "offset": float(channel.offset).hex(),
                    "saturation": channel.saturation,
                    "targets": [
                        [target.body_id, target.dense_index, float(target.weight).hex()]
                        for target in channel.targets
                    ],
                    "unit": channel.unit,
                }
                for channel in self.channels
            ],
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "node_count": self.node_count,
            "schema": FULL_BRAIN_SENSORY_BINDING_SCHEMA,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


@dataclass(frozen=True, slots=True)
class BoundPoolMember:
    """One wing-motor pool member resolved to its full-graph dense index."""

    body_id: int
    dense_index: int
    weight: float


@dataclass(frozen=True, slots=True)
class BoundMotorPool:
    """One soma-side wing-motor pool over full-graph indices."""

    name: str
    gain: float
    dead_zone: float
    members: tuple[BoundPoolMember, ...]

    @property
    def denominator(self) -> float:
        """Summed absolute member weight, exactly as ``MotorDecoder`` divides."""

        return math.fsum(abs(member.weight) for member in self.members)


@dataclass(frozen=True, slots=True)
class FullMotorBinding:
    """The canonical readout document, hashed into ``decoder_hash``.

    The decoder identity chains the executed sign-overlay identity: the same
    pools over the same graph with a different sign overlay are a *different
    decoder identity*, because the readout's meaning depends on the disclosed
    sign policy that shaped the activity it reads. For the same reason the
    declared ``readout_scale`` and the digest of the calibration document it
    was measured from are part of the identity: a rescaled readout is a
    different readout.
    """

    graph_manifest_sha256: str
    sign_overlay_identity_sha256: str
    pools: tuple[BoundMotorPool, ...]
    projection_sha256: str
    readout_scale: float = 1.0
    readout_calibration_sha256: str = ""

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "pools": [
                {
                    "dead_zone": float(pool.dead_zone).hex(),
                    "gain": float(pool.gain).hex(),
                    "members": [
                        [member.body_id, member.dense_index, float(member.weight).hex()]
                        for member in pool.members
                    ],
                    "name": pool.name,
                }
                for pool in self.pools
            ],
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "primitive_projection_sha256": self.projection_sha256,
            "readout_calibration_sha256": self.readout_calibration_sha256,
            "readout_scale": float(self.readout_scale).hex(),
            "schema": FULL_BRAIN_DECODER_SCHEMA,
            "sign_overlay_identity_sha256": self.sign_overlay_identity_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


def projection_identity_sha256() -> str:
    """Identity of the pinned two-pool primitive projection, derived not copied."""

    return canonical_sha256(ConnectomeController.PROJECTION_MATERIAL)


def bind_sensor_channels(
    *,
    channels: Sequence[SensorChannel],
    index_of: Callable[[int], int],
    node_count: int,
    graph_manifest_sha256: str,
) -> FullSensoryBinding:
    """Resolve the pinned sensor channels against a full-graph index function.

    Only existence is checked here: the full artifact carries no per-node
    role metadata, so the sensory/motor role vetting is inherited from the
    fixture-scope constants these body IDs are reused from verbatim.
    """

    _require_sha256(graph_manifest_sha256, "graph_manifest_sha256")
    bound_channels: list[BoundSensorChannel] = []
    names: set[str] = set()
    for channel in channels:
        if channel.name in names:
            raise FullBrainProfileError(f"duplicate sensor channel name {channel.name!r}")
        names.add(channel.name)
        targets: list[BoundSensorTarget] = []
        seen_indices: set[int] = set()
        for target in channel.targets:
            try:
                dense_index = index_of(target.body_id)
            except KeyError as exc:
                raise FullBrainProfileError(
                    f"sensor channel {channel.name!r} entry body {target.body_id} is not "
                    "present in the full runtime artifact"
                ) from exc
            if not 0 <= dense_index < node_count:
                raise FullBrainProfileError(
                    f"sensor channel {channel.name!r} entry index {dense_index} is out of range"
                )
            if dense_index in seen_indices:
                raise FullBrainProfileError(
                    f"sensor channel {channel.name!r} repeats entry body {target.body_id}"
                )
            seen_indices.add(dense_index)
            if not math.isfinite(target.weight) or target.weight == 0.0:
                raise FullBrainProfileError(
                    f"sensor channel {channel.name!r} weight must be finite and nonzero"
                )
            targets.append(
                BoundSensorTarget(
                    body_id=target.body_id, dense_index=dense_index, weight=target.weight
                )
            )
        bound_channels.append(
            BoundSensorChannel(
                name=channel.name,
                unit=channel.unit,
                gain=channel.gain,
                offset=channel.offset,
                clip_min=channel.clip_min,
                clip_max=channel.clip_max,
                saturation=channel.saturation,
                targets=tuple(targets),
            )
        )
    return FullSensoryBinding(
        graph_manifest_sha256=graph_manifest_sha256,
        node_count=node_count,
        channels=tuple(bound_channels),
    )


def _bind_pool(
    name: str,
    body_ids: Sequence[int],
    *,
    index_of: Callable[[int], int],
    gain: float,
    dead_zone: float,
) -> BoundMotorPool:
    members: list[BoundPoolMember] = []
    seen: set[int] = set()
    for body_id in body_ids:
        try:
            dense_index = index_of(body_id)
        except KeyError as exc:
            raise FullBrainProfileError(
                f"wing-motor pool {name!r} member body {body_id} is not present in the "
                "full runtime artifact"
            ) from exc
        if dense_index in seen:
            raise FullBrainProfileError(f"wing-motor pool {name!r} repeats body {body_id}")
        seen.add(dense_index)
        members.append(BoundPoolMember(body_id=body_id, dense_index=dense_index, weight=1.0))
    if not members:
        raise FullBrainProfileError(f"wing-motor pool {name!r} is empty")
    return BoundMotorPool(name=name, gain=gain, dead_zone=dead_zone, members=tuple(members))


def bind_wing_motor_pools(
    *,
    index_of: Callable[[int], int],
    gain: float,
    dead_zone: float,
    graph_manifest_sha256: str,
    sign_overlay_identity_sha256: str,
    readout_scale: float = 1.0,
    readout_calibration_sha256: str = "",
) -> FullMotorBinding:
    """Bind the declared 25+27 somaSide wing-motor split to full-graph indices.

    Membership is exactly ``flight_pools.LEFT_WING_MOTOR_BODY_IDS`` /
    ``RIGHT_WING_MOTOR_BODY_IDS`` -- the same declared readout the steering
    profile uses -- with unit weight per member, so each pool value is the mean
    activity of its side times the a-priori gain and the declared
    ``readout_scale`` (1.0 = uncalibrated).
    """

    _require_sha256(graph_manifest_sha256, "graph_manifest_sha256")
    _require_sha256(sign_overlay_identity_sha256, "sign_overlay_identity_sha256")
    return FullMotorBinding(
        graph_manifest_sha256=graph_manifest_sha256,
        sign_overlay_identity_sha256=sign_overlay_identity_sha256,
        pools=(
            _bind_pool(
                "left_motor",
                LEFT_WING_MOTOR_BODY_IDS,
                index_of=index_of,
                gain=gain,
                dead_zone=dead_zone,
            ),
            _bind_pool(
                "right_motor",
                RIGHT_WING_MOTOR_BODY_IDS,
                index_of=index_of,
                gain=gain,
                dead_zone=dead_zone,
            ),
        ),
        projection_sha256=projection_identity_sha256(),
        readout_scale=readout_scale,
        readout_calibration_sha256=readout_calibration_sha256,
    )


@dataclass(frozen=True, slots=True)
class EscapeMotorBinding:
    """The canonical escape decoder document, hashed into ``decoder_hash``.

    Chains the untouched wing readout document (whose material already binds
    the sign overlay, the pinned projection, the wing scale, and the
    calibration digest), the TTMn pool membership, the declared jump mapping
    formula, the declared jump scale, and the encoder identity the boundary
    was calibrated under. A different membership, formula, scale, or encoder
    is a different decoder.
    """

    wing: FullMotorBinding
    jump_pool: BoundMotorPool
    jump_scale: float
    ttmn_pool_identity_sha256: str
    encoder_identity_sha256: str
    jump_formula: str = ESCAPE_JUMP_FORMULA

    def __post_init__(self) -> None:
        if self.jump_formula != ESCAPE_JUMP_FORMULA:
            raise FullBrainProfileError(
                "the escape decoder executes only the declared jump mapping formula; "
                "a different formula is a different (undeclared) decoder"
            )
        if self.jump_pool.name != ESCAPE_JUMP_POOL_NAME:
            raise FullBrainProfileError(
                f"the escape jump pool must be named {ESCAPE_JUMP_POOL_NAME!r}"
            )
        if len(self.jump_pool.members) != 2:
            raise FullBrainProfileError(
                "the escape jump pool must carry exactly the two TTMn members"
            )
        if any(member.weight != 1.0 for member in self.jump_pool.members):
            raise FullBrainProfileError("the escape jump pool declares unit weights only")
        if not math.isfinite(self.jump_scale) or not 0.0 < self.jump_scale <= MAX_READOUT_SCALE:
            raise FullBrainProfileError(
                f"jump_scale must be finite and in (0, {MAX_READOUT_SCALE}]"
            )
        _require_sha256(self.ttmn_pool_identity_sha256, "ttmn_pool_identity_sha256")
        _require_sha256(self.encoder_identity_sha256, "encoder_identity_sha256")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "encoder_document_sha256": self.encoder_identity_sha256,
            "jump_formula": self.jump_formula,
            "jump_pool": {
                "dead_zone": float(self.jump_pool.dead_zone).hex(),
                "gain": float(self.jump_pool.gain).hex(),
                "members": [
                    [member.body_id, member.dense_index, float(member.weight).hex()]
                    for member in self.jump_pool.members
                ],
                "name": self.jump_pool.name,
            },
            "jump_scale": float(self.jump_scale).hex(),
            "readout_calibration_sha256": self.wing.readout_calibration_sha256,
            "schema": FULL_BRAIN_ESCAPE_DECODER_SCHEMA,
            "ttmn_pool_binding_sha256": self.ttmn_pool_identity_sha256,
            "wing": self.wing.to_identity_material(),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())


@dataclass(frozen=True, slots=True)
class EscapeMotorBindingV2(EscapeMotorBinding):
    """The canonical escape-v2 decoder document, hashed into ``decoder_hash``.

    The escape-v1 chaining verbatim -- the untouched wing readout document,
    the TTMn pool membership, the declared jump scale, the calibration
    document digest, and the encoder identity -- under its own decoder schema
    (``neurocraft.full_brain.decoder.escape.v2``), with the jump mapping
    replaced by the declared rectified formula ``ESCAPE_JUMP_FORMULA_V2``:
    negative pooled TTMn activity maps to zero, and the clip range is the
    non-negative half of the declared codomain. A different membership,
    formula, scale, encoder, or calibration is a different decoder.
    """

    jump_formula: str = ESCAPE_JUMP_FORMULA_V2

    def __post_init__(self) -> None:
        if self.jump_formula != ESCAPE_JUMP_FORMULA_V2:
            raise FullBrainProfileError(
                "the escape-v2 decoder executes only the declared rectified jump "
                "mapping formula; a different formula is a different (undeclared) "
                "decoder"
            )
        if self.jump_pool.name != ESCAPE_JUMP_POOL_NAME:
            raise FullBrainProfileError(
                f"the escape jump pool must be named {ESCAPE_JUMP_POOL_NAME!r}"
            )
        if len(self.jump_pool.members) != 2:
            raise FullBrainProfileError(
                "the escape jump pool must carry exactly the two TTMn members"
            )
        if any(member.weight != 1.0 for member in self.jump_pool.members):
            raise FullBrainProfileError("the escape jump pool declares unit weights only")
        if not math.isfinite(self.jump_scale) or not 0.0 < self.jump_scale <= MAX_READOUT_SCALE:
            raise FullBrainProfileError(
                f"jump_scale must be finite and in (0, {MAX_READOUT_SCALE}]"
            )
        _require_sha256(self.ttmn_pool_identity_sha256, "ttmn_pool_identity_sha256")
        _require_sha256(self.encoder_identity_sha256, "encoder_identity_sha256")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            **EscapeMotorBinding.to_identity_material(self),
            "schema": FULL_BRAIN_ESCAPE_V2_DECODER_SCHEMA,
        }


def bind_escape_jump_pool(
    ttmn: TtmnPoolBinding, *, gain: float, dead_zone: float
) -> BoundMotorPool:
    """The declared TTMn jump pool: two unit-weight members, mean readout."""

    return BoundMotorPool(
        name=ESCAPE_JUMP_POOL_NAME,
        gain=gain,
        dead_zone=dead_zone,
        members=tuple(
            BoundPoolMember(body_id=member.body_id, dense_index=member.dense_index, weight=1.0)
            for member in ttmn.members
        ),
    )


@dataclass(frozen=True, slots=True)
class FullReadoutCalibration:
    """One committed readout-calibration document, parsed from its own bytes.

    Every field is derived from the single ``bytes`` object passed to
    :func:`load_readout_calibration` -- including ``document_sha256`` -- so
    there is no read-then-reread window in which the file could change between
    the digest and the value that is executed.
    """

    document_sha256: str
    readout_scale: float
    stream_id: str
    artifact_manifest_sha256: str
    sign_overlay_identity_sha256: str
    #: Empty for a wing calibration (no ``encoder`` section in the document);
    #: the encoder document identity for an EEM calibration. The pairing is
    #: enforced in both directions by :meth:`assert_binds`, so the wing
    #: calibration can never be loaded for the EEM boundary and vice versa.
    encoder_identity_sha256: str = ""

    def assert_binds(
        self,
        *,
        artifact_manifest_sha256: str,
        sign_overlay_identity_sha256: str,
        encoder_identity_sha256: str = "",
    ) -> None:
        """Fail closed unless the document was measured on exactly this boundary."""

        if self.artifact_manifest_sha256 != artifact_manifest_sha256:
            raise FullBrainProfileError(
                "readout calibration was measured against a different artifact manifest "
                f"({self.artifact_manifest_sha256} != {artifact_manifest_sha256})"
            )
        if self.sign_overlay_identity_sha256 != sign_overlay_identity_sha256:
            raise FullBrainProfileError(
                "readout calibration was measured under a different sign overlay "
                f"({self.sign_overlay_identity_sha256} != {sign_overlay_identity_sha256})"
            )
        if self.encoder_identity_sha256 != encoder_identity_sha256:
            if self.encoder_identity_sha256 == "":
                raise FullBrainProfileError(
                    "readout calibration was measured without an engineered-encoder "
                    "binding (a wing calibration); the EEM boundary refuses it"
                )
            if encoder_identity_sha256 == "":
                raise FullBrainProfileError(
                    "readout calibration was measured under an engineered-encoder "
                    "binding; the wing boundary refuses it"
                )
            raise FullBrainProfileError(
                "readout calibration was measured under a different encoder document "
                f"({self.encoder_identity_sha256} != {encoder_identity_sha256})"
            )


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = document.get(name)
    if not isinstance(section, dict):
        raise FullBrainProfileError(f"readout calibration is missing the {name!r} section")
    return section


def load_readout_calibration(raw: bytes) -> FullReadoutCalibration:
    """Parse and validate calibration bytes; never trusts a field it can derive.

    The scale is read from its float hex, which is exact; the document's
    decimal copy is a human convenience and is required to agree with the hex
    rather than being used.
    """

    if not isinstance(raw, bytes):
        raise FullBrainProfileError("readout calibration must be supplied as bytes")
    try:
        document = parse_json_frame(raw)
    except ProtocolError as exc:
        raise FullBrainProfileError(f"readout calibration is not a valid document: {exc}") from exc
    if document.get("schema") != FULL_READOUT_CALIBRATION_SCHEMA:
        raise FullBrainProfileError("unsupported readout calibration schema")
    stream = _section(document, "stream")
    artifact = _section(document, "artifact")
    overlay = _section(document, "sign_overlay")
    calibration = _section(document, "calibration")
    stream_id = stream.get("stream_id")
    if not isinstance(stream_id, str) or not stream_id:
        raise FullBrainProfileError("readout calibration declares no stream id")
    manifest_sha256 = artifact.get("manifest_file_sha256")
    overlay_sha256 = overlay.get("identity_sha256")
    if not isinstance(manifest_sha256, str) or not _is_sha256(manifest_sha256):
        raise FullBrainProfileError(
            "readout calibration must name the artifact manifest digest it was measured against"
        )
    if not isinstance(overlay_sha256, str) or not _is_sha256(overlay_sha256):
        raise FullBrainProfileError(
            "readout calibration must name the sign overlay identity it was measured under"
        )
    scale_hex = calibration.get("readout_scale_float_hex")
    if not isinstance(scale_hex, str):
        raise FullBrainProfileError("readout calibration carries no readout_scale_float_hex")
    try:
        scale = float.fromhex(scale_hex)
    except ValueError as exc:
        raise FullBrainProfileError("readout calibration scale hex is malformed") from exc
    if not math.isfinite(scale) or not 0.0 < scale <= MAX_READOUT_SCALE:
        raise FullBrainProfileError(
            f"readout calibration scale must be finite and in (0, {MAX_READOUT_SCALE}]"
        )
    if scale == 1.0:
        # The pairing rule makes a declared scale of exactly 1.0
        # unrepresentable (1.0 must carry no digest), so refuse it here with
        # this error type instead of letting replace() raise a bare ValueError.
        raise FullBrainProfileError(
            "readout calibration produced the identity scale 1.0; there is nothing to declare"
        )
    decimal = calibration.get("readout_scale")
    if not isinstance(decimal, float) or float(decimal) != scale:
        raise FullBrainProfileError(
            "readout calibration decimal scale disagrees with its float hex"
        )
    # The ``encoder`` section is absent from wing (v1) documents and required
    # to carry a well-formed identity when present; its value feeds the
    # symmetric cross-binding refusal in ``assert_binds``.
    encoder_identity = ""
    encoder = document.get("encoder")
    if encoder is not None:
        if not isinstance(encoder, dict):
            raise FullBrainProfileError("readout calibration encoder section must be an object")
        encoder_identity_value = encoder.get("identity_sha256")
        if not isinstance(encoder_identity_value, str) or not _is_sha256(encoder_identity_value):
            raise FullBrainProfileError(
                "readout calibration encoder section must name the encoder document "
                "identity as a lowercase SHA-256 hex digest"
            )
        encoder_identity = encoder_identity_value
    return FullReadoutCalibration(
        document_sha256=sha256(raw).hexdigest(),
        readout_scale=scale,
        stream_id=stream_id,
        artifact_manifest_sha256=manifest_sha256,
        sign_overlay_identity_sha256=overlay_sha256,
        encoder_identity_sha256=encoder_identity,
    )


@dataclass(frozen=True, slots=True)
class EscapeReadoutCalibration:
    """One committed escape calibration document, parsed from its own bytes.

    Carries BOTH declared scales -- ``wing_scale`` for the wing pools
    (re-measured under the escape profile) and ``jump_scale`` for the TTMn
    jump readout -- measured in one standard-stream run, and names the exact
    encoder document AND TTMn pool membership the boundary was measured
    under. As with :class:`FullReadoutCalibration`, every field including
    ``document_sha256`` derives from the single ``bytes`` object.
    """

    document_sha256: str
    wing_scale: float
    jump_scale: float
    stream_id: str
    artifact_manifest_sha256: str
    sign_overlay_identity_sha256: str
    encoder_identity_sha256: str
    ttmn_pool_identity_sha256: str

    def assert_binds(
        self,
        *,
        artifact_manifest_sha256: str,
        sign_overlay_identity_sha256: str,
        encoder_identity_sha256: str,
        ttmn_pool_identity_sha256: str,
    ) -> None:
        """Fail closed unless the document was measured on exactly this boundary."""

        if self.artifact_manifest_sha256 != artifact_manifest_sha256:
            raise FullBrainProfileError(
                "escape calibration was measured against a different artifact manifest "
                f"({self.artifact_manifest_sha256} != {artifact_manifest_sha256})"
            )
        if self.sign_overlay_identity_sha256 != sign_overlay_identity_sha256:
            raise FullBrainProfileError(
                "escape calibration was measured under a different sign overlay "
                f"({self.sign_overlay_identity_sha256} != {sign_overlay_identity_sha256})"
            )
        if self.encoder_identity_sha256 != encoder_identity_sha256:
            raise FullBrainProfileError(
                "escape calibration was measured under a different encoder document "
                f"({self.encoder_identity_sha256} != {encoder_identity_sha256})"
            )
        if self.ttmn_pool_identity_sha256 != ttmn_pool_identity_sha256:
            raise FullBrainProfileError(
                "escape calibration was measured under a different TTMn pool binding "
                f"({self.ttmn_pool_identity_sha256} != {ttmn_pool_identity_sha256})"
            )


def _named_scale(calibration: Mapping[str, Any], name: str, *, label: str = "escape") -> float:
    """One named scale from its exact float hex, decimal copy required to agree."""

    scale_hex = calibration.get(f"{name}_float_hex")
    if not isinstance(scale_hex, str):
        raise FullBrainProfileError(f"{label} calibration carries no {name}_float_hex")
    try:
        scale = float.fromhex(scale_hex)
    except ValueError as exc:
        raise FullBrainProfileError(f"{label} calibration {name} hex is malformed") from exc
    if not math.isfinite(scale) or not 0.0 < scale <= MAX_READOUT_SCALE:
        raise FullBrainProfileError(
            f"{label} calibration {name} must be finite and in (0, {MAX_READOUT_SCALE}]"
        )
    if scale == 1.0:
        raise FullBrainProfileError(
            f"{label} calibration produced the identity {name} 1.0; there is nothing to declare"
        )
    decimal = calibration.get(name)
    if not isinstance(decimal, float) or float(decimal) != scale:
        raise FullBrainProfileError(
            f"{label} calibration decimal {name} disagrees with its float hex"
        )
    return scale


#: How each calibration schema is named in a refusal message, with its
#: article. One row per boundary a two-scale loader can be handed; the loader
#: below refuses every row but its own, so the refusals stay mutual in every
#: direction, and an unlisted schema falls through to "unsupported".
_JUMP_READOUT_CALIBRATION_NAMES: dict[str, str] = {
    FULL_READOUT_CALIBRATION_SCHEMA: "a wing/eem",
    FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA: "an escape-v1",
    FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA: "an escape-v2",
    FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA: "a loom-v1",
    FULL_READOUT_CALIBRATION_LOOM_V2_SCHEMA: "a loom-v2",
}


def _load_jump_readout_calibration(
    raw: bytes,
    *,
    expected_schema: str,
    label: str,
    profile: str,
    peer_refuser: str | None = None,
) -> EscapeReadoutCalibration:
    """Parse one two-scale calibration; refuse every other boundary by name.

    Shared by the escape-v1, escape-v2, and loom-v1 loaders, which differ only
    in which schema they accept and how they name themselves. ``label`` names
    the boundary in every parse/document message and in the wing/eem refusal;
    ``profile`` is the ``--profile`` value that measures it. ``peer_refuser``
    exists only to preserve one frozen irregularity byte-for-byte: the
    escape-v1 loader has always called itself "the escape boundary" when
    refusing a wing/eem document but "the escape-v1 boundary" when refusing
    another two-scale document, and that committed wording is what the
    existing tests match.
    """

    document = _parse_escape_calibration_frame(raw, label=label)
    schema = document.get("schema")
    if schema != expected_schema:
        named = _JUMP_READOUT_CALIBRATION_NAMES.get(schema) if isinstance(schema, str) else None
        if named is None:
            raise FullBrainProfileError(f"unsupported {label} readout calibration schema")
        refuser = (
            label
            if peer_refuser is None or schema == FULL_READOUT_CALIBRATION_SCHEMA
            else peer_refuser
        )
        raise FullBrainProfileError(
            f"this document is {named} calibration ({schema}); the {refuser} "
            f"boundary refuses it -- measure with scripts/calibrate_full_readout.py "
            f"--profile {profile}"
        )
    return _escape_calibration_from_document(raw, document, label=label)


def load_escape_readout_calibration(raw: bytes) -> EscapeReadoutCalibration:
    """Parse and validate escape calibration bytes; refuses every other document.

    The wing, eem-v1, and eem-v2 calibrations all carry the v1 schema and are
    refused here by name, and the escape-v2 and loom-v1 schemas are refused by
    name; each of those loaders symmetrically refuses the escape schema, so no
    profile can ever execute another profile's scales.
    """

    return _load_jump_readout_calibration(
        raw,
        expected_schema=FULL_READOUT_CALIBRATION_ESCAPE_SCHEMA,
        label="escape",
        profile="escape",
        peer_refuser="escape-v1",
    )


def load_escape_v2_readout_calibration(raw: bytes) -> EscapeReadoutCalibration:
    """Parse and validate escape-v2 calibration bytes; refuses every other document.

    The mirror of :func:`load_escape_readout_calibration` for the escape-v2
    boundary: the v1 (wing/eem) schema, the escape-v1 schema, and the loom-v1
    schema are refused by name, so the mutual refusal between the escape-v2
    calibration and every other calibration boundary holds in both directions.
    """

    return _load_jump_readout_calibration(
        raw,
        expected_schema=FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA,
        label="escape-v2",
        profile="escape-v2",
    )


def load_loom_v1_readout_calibration(raw: bytes) -> EscapeReadoutCalibration:
    """Parse and validate loom-v1 calibration bytes; refuses every other document.

    The mirror of :func:`load_escape_v2_readout_calibration` for the loom-v1
    boundary: the wing/eem schema and both escape schemas are refused by name,
    and each of those loaders refuses the loom-v1 schema by name in turn, so
    the mutual refusal holds in every direction. The parsed shape is the
    escape layout ({wing_scale, jump_scale} plus the encoder and TTMn pool
    identities) because the loom lane changes the sensory boundary only.
    """

    return _load_jump_readout_calibration(
        raw,
        expected_schema=FULL_READOUT_CALIBRATION_LOOM_V1_SCHEMA,
        label="loom-v1",
        profile="loom-v1",
    )


def load_loom_v2_readout_calibration(raw: bytes) -> EscapeReadoutCalibration:
    """Load the new calibration, refusing every historical boundary."""

    return _load_jump_readout_calibration(
        raw,
        expected_schema=FULL_READOUT_CALIBRATION_LOOM_V2_SCHEMA,
        label="loom-v2",
        profile="loom-v2",
    )


def _parse_escape_calibration_frame(raw: bytes, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, bytes):
        raise FullBrainProfileError(f"{label} readout calibration must be supplied as bytes")
    try:
        return parse_json_frame(raw)
    except ProtocolError as exc:
        raise FullBrainProfileError(
            f"{label} readout calibration is not a valid document: {exc}"
        ) from exc


def _escape_calibration_from_document(
    raw: bytes, document: Mapping[str, Any], *, label: str
) -> EscapeReadoutCalibration:
    """The shared escape-layout parse, after the caller's schema gate passed."""

    stream = _section(document, "stream")
    artifact = _section(document, "artifact")
    overlay = _section(document, "sign_overlay")
    calibration = _section(document, "calibration")
    encoder = _section(document, "encoder")
    jump_readout = _section(document, "jump_readout")
    stream_id = stream.get("stream_id")
    if not isinstance(stream_id, str) or not stream_id:
        raise FullBrainProfileError(f"{label} readout calibration declares no stream id")
    manifest_sha256 = artifact.get("manifest_file_sha256")
    overlay_sha256 = overlay.get("identity_sha256")
    encoder_sha256 = encoder.get("identity_sha256")
    ttmn_sha256 = jump_readout.get("ttmn_pool_binding_sha256")
    for value, field_label in (
        (manifest_sha256, "artifact manifest digest"),
        (overlay_sha256, "sign overlay identity"),
        (encoder_sha256, "encoder document identity"),
        (ttmn_sha256, "TTMn pool binding identity"),
    ):
        if not isinstance(value, str) or not _is_sha256(value):
            raise FullBrainProfileError(
                f"{label} readout calibration must name the {field_label} it was measured under"
            )
    assert isinstance(manifest_sha256, str)
    assert isinstance(overlay_sha256, str)
    assert isinstance(encoder_sha256, str)
    assert isinstance(ttmn_sha256, str)
    return EscapeReadoutCalibration(
        document_sha256=sha256(raw).hexdigest(),
        wing_scale=_named_scale(calibration, "wing_scale", label=label),
        jump_scale=_named_scale(calibration, "jump_scale", label=label),
        stream_id=stream_id,
        artifact_manifest_sha256=manifest_sha256,
        sign_overlay_identity_sha256=overlay_sha256,
        encoder_identity_sha256=encoder_sha256,
        ttmn_pool_identity_sha256=ttmn_sha256,
    )


@dataclass(frozen=True, slots=True)
class FullBrainStepEvidence:
    """Light per-step evidence; never materializes 166,700 Python floats.

    ``state_after_activity`` is a **declared subset** of the committed activity
    vector: the entry-node activities followed by every wing-motor pool member,
    in the fixed order given by ``activity_subset_body_ids`` (entry bodies in
    channel-declaration order without repeats, then the left pool, then the
    right pool -- 54 values for the real artifact). The full-vector identity is
    carried by ``state_after_sha256``; the subset exists only so presentation
    surfaces (for example ``PresentationTelemetryWriter``) can show boundary
    activity without copying the whole brain state per tick.

    The EEM profile keeps this subset unchanged (a disclosed choice): copying
    the ~8.4k embodied population members per tick is not cheap, membership is
    pinned by the hashed population binding, and the per-population drives are
    recorded in ``encoded_stimulation`` under ``eem:``/``eem_drive:`` keys.
    """

    observation: Observation
    observation_sha256: str
    feature_values: Mapping[str, float]
    encoded_stimulation: Mapping[str, float]
    encoded_stimulation_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    state_after_activity: tuple[float, ...]
    activity_subset_body_ids: tuple[int, ...]
    neural_step_start: int
    neural_step_end: int
    motor_frame: MotorFrame
    command_seq: int
    motors: MotorVector
    action_sha256: str
    #: The escape profile's decoded jump value (post-clip, post-dead-zone),
    #: the exact term added to the vertical projection this step. ``None``
    #: for every other profile.
    jump_value: float | None = None
    #: Only controls add these fields. Under no_input the original encoder
    #: output remains here, while encoded_stimulation reports executed zeros.
    encoder_stimulation_before_control: Mapping[str, float] | None = None
    control_identity_sha256: str | None = None
    executed_stimulation_sha256: str | None = None


class FullBrainController:
    """Whole-brain recurrent context behind the pinned wing-motor boundary.

    Duck-types the service's ``Controller``/``ResettableController`` protocols.
    The constructor accepts prepared components so tests can inject small
    synthetic views; :meth:`from_artifact` performs the real verified loading.
    """

    def __init__(
        self,
        *,
        run_id: str,
        entity_id: str,
        session_epoch: int = 0,
        configuration: FullBrainConfiguration | None = None,
        engine: ArrayNativeEngine,
        sensory_binding: FullSensoryBinding,
        motor_binding: FullMotorBinding,
        embodied: EmbodiedSensesBinding | None = None,
        escape: EscapeMotorBinding | None = None,
        steering: SteeringPairBinding | None = None,
        control: FullControlView | None = None,
        event_encoder: EventEncoderBinding | None = None,
    ) -> None:
        config = configuration or FullBrainConfiguration()
        if not isinstance(config, FullBrainConfiguration):
            raise FullBrainProfileError(
                "the full-brain controller requires a FullBrainConfiguration"
            )
        self._control = self._validate_control(control, config, engine, sensory_binding)
        if isinstance(config, FullBrainEventsConfiguration) != (event_encoder is not None):
            raise FullBrainProfileError("event configuration and composite encoder must be paired")
        if isinstance(config, FullBrainInteractionsConfiguration) != isinstance(
            event_encoder, InteractionEncoderBinding
        ):
            raise FullBrainProfileError("interaction configuration and V4 encoder must be paired")
        if event_encoder is not None:
            assert isinstance(config, FullBrainEventsConfiguration)
            if config.event_encoder_sha256 != event_encoder.identity_sha256:
                raise FullBrainProfileError("candidate event encoder identity mismatch")
            if (
                event_encoder.populations.graph_hash != sensory_binding.graph_manifest_sha256
                or event_encoder.populations.node_count != engine.view.node_count
                or embodied is None
                or escape is None
                or event_encoder.populations.annotations_hash
                != embodied.populations.annotations_sha256
                or event_encoder.background_encoder_sha256 != embodied.encoder.identity_sha256
                or event_encoder.component_decoder_sha256 != escape.identity_sha256
                or event_encoder.component_calibration_sha256 != config.readout_calibration_sha256
            ):
                raise FullBrainProfileError("event boundary names different verified components")
            if isinstance(event_encoder, InteractionEncoderBinding) and (
                event_encoder.embodied_populations.identity_sha256
                != embodied.populations.identity_sha256
            ):
                raise FullBrainProfileError("attack proxy differs from actual embodied populations")
        self._event_inputs = (
            InteractionInputState(event_encoder)
            if isinstance(event_encoder, InteractionEncoderBinding)
            else None
            if event_encoder is None
            else EventInputState(event_encoder)
        )
        self._last_event_readout: CompletedEventReadout | None = None

        if embodied is None and isinstance(config, FullBrainEemConfiguration):
            raise FullBrainProfileError(
                "an engineered-encoder configuration requires the embodied senses binding"
            )
        if embodied is not None and not isinstance(config, FullBrainEemConfiguration):
            raise FullBrainProfileError(
                "the embodied senses binding requires a FullBrainEemConfiguration"
            )
        if embodied is not None and isinstance(config, FullBrainLoomV2Configuration) != isinstance(
            embodied.encoder, EemEncoderDocumentLoomV2
        ):
            raise FullBrainProfileError("loom-v2 configuration and encoder must be paired")
        if embodied is not None and isinstance(config, FullBrainLoomV1Configuration) != isinstance(
            embodied.encoder, EemEncoderDocumentLoomV1
        ):
            # Both directions, and checked FIRST because it is the most
            # specific gate: the loom encoder document rides the same
            # rectified decoder as escape-v2, so without this gate an
            # escape-v2 configuration over a loom encoder would be refused
            # only by the (less precise) escape-v2 message below, and a loom
            # configuration over any escape-v2 document not at all.
            raise FullBrainProfileError(
                "the loom-v1 (approach-gated) configuration and the loom-v1 "
                "encoder document must be paired; every other pairing is refused"
            )
        if embodied is not None and isinstance(
            config, FullBrainEscapeV2Configuration
        ) != isinstance(embodied.encoder, EemEncoderDocumentEscapeV2):
            # Both directions: an escape-v2 configuration over any other
            # encoder document (the v1 absolute encoder, or the eem-v2
            # all-channel-contrast document) refuses, and every other
            # configuration over the escape-v2 encoder document refuses.
            raise FullBrainProfileError(
                "the escape-v2 (transient-touch) configuration and the escape-v2 "
                "encoder document must be paired; every other pairing is refused"
            )
        if (
            embodied is not None
            and not isinstance(
                embodied.encoder, EemEncoderDocumentEscapeV2 | EemEncoderDocumentLoomV1
            )
            and isinstance(config, FullBrainEemV2Configuration)
            != isinstance(embodied.encoder, EemEncoderDocumentV2)
        ):
            raise FullBrainProfileError(
                "the temporal-contrast (v2) engineered-encoder configuration and the "
                "v2 encoder document must be paired; a v1/v2 mix is refused"
            )
        if escape is None and isinstance(config, FullBrainEscapeConfiguration):
            raise FullBrainProfileError(
                "an escape-reflex configuration requires the escape motor binding"
            )
        if escape is not None and not isinstance(config, FullBrainEscapeConfiguration):
            raise FullBrainProfileError(
                "the escape motor binding requires a FullBrainEscapeConfiguration"
            )
        if escape is not None and isinstance(config, RECTIFIED_JUMP_CONFIGURATIONS) != isinstance(
            escape, EscapeMotorBindingV2
        ):
            raise FullBrainProfileError(
                "the escape-v2 and loom-v1 configurations both declare the "
                "escape-v2 (rectified) decoder binding and must be paired with "
                "it; the escape-v1 configuration must be paired with the "
                "unrectified binding. A v1/v2 escape mix is refused"
            )
        if sensory_binding.graph_manifest_sha256 != motor_binding.graph_manifest_sha256:
            raise FullBrainProfileError(
                "sensory and motor bindings were derived from different artifacts"
            )
        node_count = engine.view.node_count
        if sensory_binding.node_count != node_count:
            raise FullBrainProfileError(
                "sensory binding node count does not match the bound engine view"
            )
        pool_names = tuple(pool.name for pool in motor_binding.pools)
        if set(pool_names) != {"left_motor", "right_motor"} or len(pool_names) != 2:
            raise FullBrainProfileError(
                "the pinned primitive projection requires exactly the left_motor and "
                "right_motor pools"
            )
        if (
            motor_binding.readout_scale != config.readout_scale
            or motor_binding.readout_calibration_sha256 != config.readout_calibration_sha256
        ):
            raise FullBrainProfileError(
                "motor binding readout calibration does not match the declared configuration"
            )
        for pool in motor_binding.pools:
            if pool.gain != config.motor_pool_gain or pool.dead_zone != config.motor_dead_zone:
                raise FullBrainProfileError(
                    f"pool {pool.name!r} gain/dead-zone does not match the declared configuration"
                )
            for member in pool.members:
                if not 0 <= member.dense_index < node_count:
                    raise FullBrainProfileError(
                        f"pool {pool.name!r} member index {member.dense_index} is out of range"
                    )
        for channel in sensory_binding.channels:
            for target in channel.targets:
                if not 0 <= target.dense_index < node_count:
                    raise FullBrainProfileError(
                        f"sensor channel {channel.name!r} index {target.dense_index} "
                        "is out of range"
                    )

        self.configuration = config
        self.engine = engine
        self.features = (
            InteractionFeatureEncoder()
            if isinstance(event_encoder, InteractionEncoderBinding)
            else EventFeatureEncoder()
            if event_encoder is not None
            else EntityRayFeatureEncoder()
            if isinstance(config, FullBrainLoomV2Configuration)
            else PhysicalFeatureEncoder()
        )
        self._sensory_binding = sensory_binding
        self._motor_binding = motor_binding
        self._graph_manifest_sha256 = sensory_binding.graph_manifest_sha256
        self._node_count = node_count

        channel_names = {channel.name for channel in sensory_binding.channels}
        if channel_names != set(self.features.channel_names):
            raise FullBrainProfileError(
                "sensory binding channels do not match the pinned feature schema"
            )

        self._embodied = embodied
        self._population_indices: tuple[tuple[str, np.ndarray[Any, np.dtype[np.int64]]], ...] = ()
        if embodied is None:
            adapter_hash = canonical_sha256(
                {
                    "feature_schema_sha256": self.features.identity_sha256,
                    "neural_adapter_sha256": sensory_binding.identity_sha256,
                    "schema": "neurocraft.boundary_adapter.identity.v1",
                }
            )
        else:
            # Narrowing for the type checker; enforced with a real refusal above.
            assert isinstance(config, FullBrainEemConfiguration)
            adapter_hash = self._validate_embodied(embodied, config, sensory_binding)
        if event_encoder is not None:
            adapter_hash = canonical_sha256(
                {
                    "schema": "neurocraft.events.adapter.v1",
                    "background_adapter_sha256": adapter_hash,
                    "event_encoder_sha256": event_encoder.identity_sha256,
                }
            )
        self._escape = escape
        if escape is None:
            decoder_hash = motor_binding.identity_sha256
        else:
            # Narrowing for the type checker; enforced with real refusals above.
            assert isinstance(config, FullBrainEscapeConfiguration)
            decoder_hash = self._validate_escape(escape, config, motor_binding)
        # Field placement differs from the reference controller: there the
        # sign-overlay identity is folded into dynamics_hash; here it rides in
        # decoder_hash (via the motor binding) while dynamics_hash is the
        # backend's own kernel-bound identity. Not a collision surface: the
        # configuration hard-pins sign_policy to a single value, and the view
        # provenance additionally bakes the overlay into every per-step
        # state_sha256 — a stronger, per-step binding than construction-time.
        self.identity = RunIdentity(
            run_id=run_id,
            session_epoch=session_epoch,
            entity_id=entity_id,
            graph_hash=self._graph_manifest_sha256,
            adapter_hash=adapter_hash,
            dynamics_hash=(
                engine.dynamics_identity_sha256
                if control is None
                else canonical_sha256(
                    {
                        "schema": "neurocraft.dynamics.identity.full_control.v1",
                        "backend_dynamics_sha256": engine.dynamics_identity_sha256,
                        "control_identity_sha256": control.identity.identity_sha256,
                    }
                )
            ),
            decoder_hash=decoder_hash,
            observation_schema_hash=self.features.identity_sha256,
        )

        # The steering readout is bound AFTER the identity above, and deliberately
        # enters none of its eight fields: it adds no term to any motor axis, drives
        # nothing, and is read only by the selection policy, whose own `policy_hash`
        # chains this binding's identity. Folding it into `decoder_hash` would rotate
        # every calibration cross-binding for a readout the decoder never sees.
        # `test_steering_readout_does_not_rotate_the_run_identity` pins that.
        self._steering = steering
        if steering is not None:
            if steering.graph_manifest_sha256 != self._graph_manifest_sha256:
                raise FullBrainProfileError(
                    "the steering readout binding was derived from a different artifact"
                )
            if steering.node_count != node_count:
                raise FullBrainProfileError(
                    "steering readout binding node count does not match the bound engine view"
                )
            for steering_member in steering.members:
                if not 0 <= steering_member.dense_index < node_count:
                    raise FullBrainProfileError(
                        "steering readout member index "
                        f"{steering_member.dense_index} is out of range"
                    )

        # Declared evidence subset: entry indices in channel-declaration order
        # without repeats, then the left pool, then the right pool -- and, for
        # the escape profile only, the two TTMn jump pool members after them.
        subset: dict[int, int] = {}
        for channel in sensory_binding.channels:
            for target in channel.targets:
                subset.setdefault(target.dense_index, target.body_id)
        for pool in motor_binding.pools:
            for member in pool.members:
                subset.setdefault(member.dense_index, member.body_id)
        if escape is not None:
            for member in escape.jump_pool.members:
                subset.setdefault(member.dense_index, member.body_id)
        self._evidence_indices = np.asarray(tuple(subset), dtype=np.int64)
        self._evidence_body_ids = tuple(subset.values())

        self._last_observation_seq = -1
        self._last_game_tick = -1
        self._last_command_seq = -1
        self._neural_step_offset = 0
        self._last_evidence: FullBrainStepEvidence | None = None
        # Temporal-contrast (eem v2) running means: the only cross-tick
        # encoder state. None = declared first tick (baseline := raw_0, all
        # contrast drives 0); replaced wholesale each step by the pure
        # apply_temporal_contrast result and cleared by reset(). Always None
        # for the wing and v1 EEM profiles.
        self._eem_contrast_means: dict[str, float] | None = None
        self._eem_temporal_contrast = embodied is not None and isinstance(
            embodied.encoder, EemEncoderDocumentV2
        )
        # step() runs on a worker thread (the service de-blinds its event
        # loop) while warm_compute_path() may fire from the loop; the numpy
        # fallback kernel caches a mutable scratch buffer, so the two must
        # mutually exclude. warm skips rather than blocks when contended.
        self._compute_lock = threading.Lock()

    @staticmethod
    def _validate_control(
        control: FullControlView | None,
        config: FullBrainConfiguration,
        engine: ArrayNativeEngine,
        sensory_binding: FullSensoryBinding,
    ) -> FullControlView | None:
        if control is None:
            if config.mode != "faithful" or ":control:" in engine.view.provenance.origin:
                raise FullBrainProfileError(
                    "a control mode/view requires its declared control binding"
                )
            return None
        if not isinstance(control, FullControlView) or config.mode != control.kind.value:
            raise FullBrainProfileError("configuration mode and control binding must agree")
        if engine.view is not control.view:
            raise FullBrainProfileError("the engine must execute the bound control view")
        identity = control.identity
        if (
            identity.artifact_manifest_sha256 != sensory_binding.graph_manifest_sha256
            or identity.edge_count != engine.view.edge_count
        ):
            raise FullBrainProfileError("control binding names a different source artifact")
        if control.kind is FullControlKind.NO_INPUT:
            if (
                engine.view.view_sha256 != identity.faithful_view_sha256
                or ":control:" in engine.view.provenance.origin
            ):
                raise FullBrainProfileError("no_input must execute its unchanged faithful view")
        elif engine.view.provenance.weight_overlay_sha256 != identity.identity_sha256:
            raise FullBrainProfileError(
                "shuffled view provenance does not bind the control identity"
            )
        return control

    def _validate_embodied(
        self,
        embodied: EmbodiedSensesBinding,
        config: FullBrainEemConfiguration,
        sensory_binding: FullSensoryBinding,
    ) -> str:
        """Cross-check the EEM binding and derive the chained adapter identity.

        The adapter identity chains the pinned feature schema, the faithful
        sensory binding (retained unchanged), the population-membership
        document, and the encoder document: a different membership or a
        different declared formula is a different adapter.
        """

        if config.encoder_identity_sha256 != embodied.encoder.identity_sha256:
            raise FullBrainProfileError(
                "declared encoder identity does not match the bound encoder document"
            )
        populations = embodied.populations
        if populations.graph_manifest_sha256 != sensory_binding.graph_manifest_sha256:
            raise FullBrainProfileError(
                "embodied populations were derived from a different artifact"
            )
        if populations.node_count != self._node_count:
            raise FullBrainProfileError(
                "embodied population node count does not match the bound engine view"
            )
        faithful_indices = {
            target.dense_index for channel in sensory_binding.channels for target in channel.targets
        }
        indices: list[tuple[str, np.ndarray[Any, np.dtype[np.int64]]]] = []
        for population in populations.populations:
            for dense_index in population.dense_indices:
                if not 0 <= dense_index < self._node_count:
                    raise FullBrainProfileError(
                        f"population {population.name!r} index {dense_index} is out of range"
                    )
                if dense_index in faithful_indices:
                    raise FullBrainProfileError(
                        f"population {population.name!r} overlaps a faithful sensory entry"
                    )
            indices.append((population.name, np.asarray(population.dense_indices, dtype=np.int64)))
        self._population_indices = tuple(indices)
        return canonical_sha256(
            {
                "encoder_document_sha256": embodied.encoder.identity_sha256,
                "faithful_adapter_sha256": sensory_binding.identity_sha256,
                "feature_schema_sha256": self.features.identity_sha256,
                "population_binding_sha256": populations.identity_sha256,
                "schema": FULL_BRAIN_EEM_ADAPTER_SCHEMA,
            }
        )

    def _validate_escape(
        self,
        escape: EscapeMotorBinding,
        config: FullBrainEscapeConfiguration,
        motor_binding: FullMotorBinding,
    ) -> str:
        """Cross-check the escape binding and return the escape decoder identity.

        Runs after :meth:`_validate_embodied` (the escape configuration is an
        engineered-encoder configuration, so the embodied binding is already
        required and validated), so the population indices are available for
        the disjointness refusals.
        """

        if escape.wing != motor_binding:
            raise FullBrainProfileError(
                "the escape binding must chain exactly the bound wing motor binding"
            )
        if escape.jump_scale != config.jump_scale:
            raise FullBrainProfileError(
                "escape binding jump_scale does not match the declared configuration"
            )
        if escape.encoder_identity_sha256 != config.encoder_identity_sha256:
            raise FullBrainProfileError(
                "escape binding encoder identity does not match the declared configuration"
            )
        pool = escape.jump_pool
        if pool.gain != config.motor_pool_gain or pool.dead_zone != config.motor_dead_zone:
            raise FullBrainProfileError(
                "escape jump pool gain/dead-zone does not match the declared configuration"
            )
        wing_indices = {
            member.dense_index for wing_pool in motor_binding.pools for member in wing_pool.members
        }
        faithful_indices = {
            target.dense_index
            for channel in self._sensory_binding.channels
            for target in channel.targets
        }
        population_indices: set[int] = set()
        for _, indices in self._population_indices:
            population_indices.update(int(index) for index in indices)
        for member in pool.members:
            if not 0 <= member.dense_index < self._node_count:
                raise FullBrainProfileError(
                    f"escape jump pool index {member.dense_index} is out of range"
                )
            if member.dense_index in wing_indices:
                raise FullBrainProfileError(
                    f"escape jump pool body {member.body_id} overlaps the wing readout"
                )
            if member.dense_index in faithful_indices:
                raise FullBrainProfileError(
                    f"escape jump pool body {member.body_id} overlaps a faithful sensory entry"
                )
            if member.dense_index in population_indices:
                raise FullBrainProfileError(
                    f"escape jump pool body {member.body_id} overlaps a driven sensory population"
                )
        return escape.identity_sha256

    # ------------------------------------------------------------------ surface

    @property
    def run_mode(self) -> RunMode:
        """Mode label under the ADR 0008 taxonomy caveats.

        ``FAITHFUL`` here means: no control overlay, no scripted policy, the
        pinned boundary, and the disclosed coarse-consensus sign overlay over
        the verified artifact -- executed by the array-native backend under its
        own dynamics identity. It is not a claim of physiological fidelity.
        """

        return RunMode(self.configuration.mode)

    @property
    def control_kind(self) -> str:
        return self.configuration.mode

    @property
    def control_identity(self) -> FullControlIdentity | None:
        return None if self._control is None else self._control.identity

    @property
    def last_evidence(self) -> FullBrainStepEvidence | None:
        return self._last_evidence

    @property
    def sign_overlay_identity_sha256(self) -> str:
        """Identity of the executed sign overlay, read from the motor binding."""

        return self._motor_binding.sign_overlay_identity_sha256

    @property
    def readout_scale(self) -> float:
        """The declared readout scale currently multiplying both pool values."""

        return self._motor_binding.readout_scale

    @property
    def embodied_senses(self) -> EmbodiedSensesBinding | None:
        """The bound EEM population/encoder documents, or ``None`` for the wing profile."""

        return self._embodied

    @property
    def escape_binding(self) -> EscapeMotorBinding | None:
        """The bound escape decoder document, or ``None`` for every other profile."""

        return self._escape

    @property
    def steering_binding(self) -> SteeringPairBinding | None:
        """The bound DNa01 steering pair, or ``None`` when it was not requested.

        Present only when ``from_artifact(..., steering_readout=True)`` selected it. It
        is read by the selection policy and by nothing else: no motor axis, no decoder
        term, and no ``RunIdentity`` field depends on it.
        """

        return self._steering

    def steering_readout_means(self) -> tuple[float, float]:
        """Read-only hook: the PRE-gain, PRE-scale mean activity of each DNa01 member.

        Returns ``(left, right)`` in that order. Each value is the raw committed activity
        of the one L or one R member -- the pair carries exactly one of each, so the
        "mean" is the member's own value. Mirrors :meth:`jump_readout_mean` and
        :meth:`pool_readout_means`: it reads the engine's already-committed activity
        vector and mutates nothing.

        These are activities, not decisions and not a turn command. The selection policy
        turns them into a declared normalized difference under a declared sign
        convention; that mapping is hand-chosen engineering and lives in
        ``program_policy.py``, not here.
        """

        steering = self._steering
        if steering is None:
            raise FullBrainProfileError(
                "steering_readout_means requires the DNa01 steering binding "
                "(from_artifact(..., steering_readout=True))"
            )
        activity = self.engine.activity()
        return (
            float(activity[steering.left.dense_index]),
            float(activity[steering.right.dense_index]),
        )

    def jump_readout_mean(self) -> float:
        """Read-only hook: the PRE-gain, PRE-scale mean TTMn pool activity.

        The jump-pool counterpart of :meth:`pool_readout_means` -- exactly the
        ``fsum(weight * activity) / denominator`` term :meth:`_decode_jump`
        computes before the declared gain, the declared jump scale, the clip
        and the dead zone; the quantity
        ``scripts/calibrate_full_readout.py --profile escape`` measures.
        """

        escape = self._escape
        if escape is None:
            raise FullBrainProfileError("jump_readout_mean is defined for the escape profile only")
        activity = self.engine.activity()
        pool = escape.jump_pool
        return (
            math.fsum(
                member.weight * float(activity[member.dense_index]) for member in pool.members
            )
            / pool.denominator
        )

    def pool_readout_means(self) -> tuple[tuple[str, float], ...]:
        """Read-only hook: the PRE-gain, PRE-scale mean activity of each pool.

        This is exactly the ``fsum(weight * activity) / denominator`` term
        :meth:`_decode` computes before the declared gain, the declared scale,
        the clip and the dead zone -- the quantity
        ``scripts/calibrate_full_readout.py`` measures. It reads the engine's
        already-committed activity vector and mutates nothing; it exists so the
        calibration measurement never has to reach into the decoder's internals
        or re-derive the pool membership by hand.
        """

        activity = self.engine.activity()
        return tuple(
            (
                pool.name,
                math.fsum(
                    member.weight * float(activity[member.dense_index]) for member in pool.members
                )
                / pool.denominator,
            )
            for pool in self._motor_binding.pools
        )

    @property
    def event_encoder(self) -> EventEncoderBinding | None:
        return None if self._event_inputs is None else self._event_inputs.binding

    @property
    def session_has_stepped(self) -> bool:
        """Whether this session completed a command, preserved across neural RESET."""
        return self._last_command_seq >= 0

    @property
    def last_event_readout(self) -> CompletedEventReadout | None:
        """Immutable scalar snapshot captured in the same completed-step worker."""
        return self._last_event_readout

    def reset(self) -> None:
        """Reset neural state without rewinding the live session clocks.

        The eem-v2 running means are cleared with the neural state: the next
        step is a declared first tick again (baseline := raw_0, contrast
        drives 0), so a post-reset trajectory depends only on post-reset
        observations.
        """

        self._neural_step_offset += self.engine.step_index
        self.engine.reset()
        self._last_evidence = None
        self._eem_contrast_means = None
        self._last_event_readout = None
        if self._event_inputs is not None:
            self._event_inputs.reset()

    def reset_session(self, *, session_epoch: int) -> None:
        """Start a new neural session, keeping the bound lane and clearing its clocks.

        Unlike reset(), this permits observation/command sequences to restart
        at zero. The service must also retire the old connection and queued
        work under its execution fence before calling this method.
        """

        with self._compute_lock:
            if (
                isinstance(session_epoch, bool)
                or not isinstance(session_epoch, int)
                or session_epoch <= self.identity.session_epoch
            ):
                raise ControllerIntegrityError("new session_epoch must strictly increase")
            identity = replace(self.identity, session_epoch=session_epoch)
            self.reset()
            self._last_observation_seq = -1
            self._last_game_tick = -1
            self._last_command_seq = -1
            self._neural_step_offset = 0
            self.identity = identity

    def warm_compute_path(self, *, passes: int = 3) -> float:
        """Warm the kernel/page/ufunc path without touching neural state.

        The mod grants the FIRST live action a hard 2-tick (100 ms) deadline
        after activation, and a cold first step (thread-pool spin-up, first
        ``np.tanh`` dispatch, evicted weight pages) can spike past it and
        latch a permanent ``DEADLINE_MISSED``. This runs the same sparse
        matvec + tanh arithmetic over the real view buffers into locally
        allocated scratch, so the first real step behaves like steady state.
        Purity: the engine's buffers are only read, ``engine.step`` is never
        called, and the step index is asserted unchanged. Returns seconds.
        """

        if passes < 1:
            raise FullBrainProfileError("warm_compute_path requires at least one pass")
        if not self._compute_lock.acquire(blocking=False):
            return 0.0  # a step is computing; the path is warm by definition
        try:
            return self._warm_locked(passes)
        finally:
            self._compute_lock.release()

    def _warm_locked(self, passes: int) -> float:
        step_index_before = self.engine.step_index
        view = self.engine.view
        scratch_in = np.zeros(self._node_count, dtype=np.float64)
        scratch_out = np.empty(self._node_count, dtype=np.float64)
        started = time.perf_counter()
        for _ in range(passes):
            self.engine.kernel.matvec(
                indptr=view.indptr,
                indices=view.indices,
                values=view.weights,
                x=scratch_in,
                out=scratch_out,
                empty_rows=view.empty_rows,
            )
            np.tanh(scratch_out, out=scratch_out)
        elapsed = time.perf_counter() - started
        if self.engine.step_index != step_index_before:
            raise FullBrainProfileError("warm_compute_path must never advance neural state")
        return elapsed

    # --------------------------------------------------------------------- step

    def step(self, observation: Observation, *, command_seq: int) -> Action:
        with self._compute_lock:
            return self._step_locked(observation, command_seq=command_seq)

    def _step_locked(self, observation: Observation, *, command_seq: int) -> Action:
        self._validate_observation(observation, command_seq)
        event_pulse: EventPulse | None = None
        if self._event_inputs is not None:
            from neurocraft_protocol import SensorFrameV3

            if not isinstance(observation.sensors, SensorFrameV3):
                raise ControllerIntegrityError("event profile requires SensorFrameV3")
            if "events" in observation.clamped_fields:
                raise ControllerIntegrityError("event inputs may not be declared clamped")
            if isinstance(self._event_inputs, InteractionInputState) and (
                "ray_actor_kinds" in observation.clamped_fields
            ):
                raise ControllerIntegrityError("actor metadata may not be declared clamped")
            event_pulse = self._event_inputs.prepare(
                observation.sensors.events, observation.game_tick
            )
        state_before = self.engine.current_state()
        features = self.features.encode(observation.sensors)
        vector, assignments = self._encode_dense(features.values)
        embodied_evidence: dict[str, float] = {}
        if self._embodied is None:
            stimulation_sha256 = self._stimulation_identity(assignments)
        elif self._eem_temporal_contrast:
            raw_values, means_used, channel_values, drives = self._apply_embodied_drives_v2(
                vector, observation.sensors
            )
            stimulation_sha256 = self._stimulation_identity_embodied_v2(
                assignments, raw_values, means_used, channel_values, drives
            )
            embodied_evidence = {
                **{f"eem:{name}": value for name, value in channel_values},
                **{f"eem_drive:{name}": value for name, value in drives},
                **{f"eem_mean:{name}": value for name, value in means_used},
                **{f"eem_raw:{name}": value for name, value in raw_values},
            }
        else:
            channel_values, drives = self._apply_embodied_drives(vector, observation.sensors)
            stimulation_sha256 = self._stimulation_identity_embodied(
                assignments, channel_values, drives
            )
            embodied_evidence = {
                **{f"eem:{name}": value for name, value in channel_values},
                **{f"eem_drive:{name}": value for name, value in drives},
            }
        if self._event_inputs is not None:
            assert event_pulse is not None
            self._event_inputs.apply(vector, event_pulse)
            stimulation_sha256 = canonical_sha256(
                {
                    "schema": "neurocraft.events.stimulation.v1",
                    "background_stimulation_sha256": stimulation_sha256,
                    "event_encoder_sha256": self._event_inputs.binding.identity_sha256,
                    "pulse": event_pulse.to_dict(),
                    "events": observation.sensors.to_dict()["events"],
                }
            )
            embodied_evidence.update(
                {"event:food": float(event_pulse.food), "event:groom": float(event_pulse.groom)}
            )
            if isinstance(event_pulse, InteractionPulse):
                embodied_evidence["event:attack"] = float(event_pulse.attack)
        if self._control is not None:
            if self._control.zeroes_sensory_input:
                # Gate AFTER every faithful and engineered assignment; zeroing
                # raw sensors or only EEM would leave other inputs connected.
                vector.fill(0.0)
            stimulation_sha256 = canonical_sha256(
                {
                    "schema": "neurocraft.full_brain.controlled_stimulation.v1",
                    "control_identity_sha256": self._control.identity.identity_sha256,
                    "encoder_stimulation_sha256": stimulation_sha256,
                    "input_gate": "zero" if self._control.zeroes_sensory_input else "identity",
                }
            )
        result: ArrayStepResult | None = None
        for _ in range(self.configuration.neural_steps_per_game_tick):
            result = self.engine.step(sensory=vector, sensory_provenance_sha256=stimulation_sha256)
        assert result is not None
        activity = self.engine.activity()
        if self._event_inputs is not None:
            assert event_pulse is not None
            self._last_event_readout = completed_event_readout(
                activity, self._event_inputs.binding.populations, event_pulse
            )
        motor_frame = self._decode(activity, result.state.state_sha256)
        motors = ConnectomeController._project_motor_frame(motor_frame)
        jump_value: float | None = None
        if self._escape is not None:
            # The declared escape extension: one additive term on the vertical
            # axis (ESCAPE_JUMP_FORMULA); every other axis is passed through
            # bit-for-bit from the pinned projection.
            jump_value = self._decode_jump(activity)
            motors = MotorVector(
                forward_thrust=motors.forward_thrust,
                lateral_thrust=motors.lateral_thrust,
                vertical_thrust=min(1.0, max(-1.0, motors.vertical_thrust + jump_value)),
                yaw_torque=motors.yaw_torque,
                pitch_torque=motors.pitch_torque,
            )
        action = Action(
            identity=self.identity,
            command_seq=command_seq,
            observation_seq=observation.observation_seq,
            observation_tick=observation.game_tick,
            target_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
            valid_through_tick=observation.game_tick + ACTION_TARGET_OFFSET_TICKS,
            neural_step_start=self._neural_step_offset + state_before.step_index,
            neural_step_end=self._neural_step_offset + result.state.step_index,
            state_digest=result.state.state_sha256,
            motors=motors,
            neutral=False,
        )
        action_sha256 = canonical_sha256(action.to_dict())
        # Wing profile: exactly the faithful per-entry assignments. EEM adds
        # per-channel values and per-population drives; per-member entries are
        # deliberately absent because every member of a population receives
        # the identical per-capita value and the membership itself is pinned
        # by the hashed population binding.
        encoded_values = MappingProxyType(
            {
                **{f"{name}:{body_id}": value for name, body_id, _, value in assignments},
                **embodied_evidence,
            }
        )
        before_control: Mapping[str, float] | None = None
        if self._control is not None and self._control.zeroes_sensory_input:
            before_control = encoded_values
            # Raw sensors and contrast means remain in the explicitly named
            # pre-control evidence. Only delivered channels/assignments are
            # reported as zero; no fabricated zero raw observation is logged.
            encoded_values = MappingProxyType(
                {
                    name: 0.0
                    for name in before_control
                    if not name.startswith(("eem_raw:", "eem_mean:"))
                }
            )
        self._last_evidence = FullBrainStepEvidence(
            observation=observation,
            observation_sha256=canonical_sha256(observation.to_dict()),
            feature_values=MappingProxyType(dict(features.values)),
            encoded_stimulation=encoded_values,
            encoded_stimulation_sha256=stimulation_sha256,
            state_before_sha256=state_before.state_sha256,
            state_after_sha256=result.state.state_sha256,
            state_after_activity=tuple(float(value) for value in activity[self._evidence_indices]),
            activity_subset_body_ids=self._evidence_body_ids,
            neural_step_start=self._neural_step_offset + state_before.step_index,
            neural_step_end=self._neural_step_offset + result.state.step_index,
            motor_frame=motor_frame,
            command_seq=command_seq,
            motors=motors,
            action_sha256=action_sha256,
            jump_value=jump_value,
            encoder_stimulation_before_control=before_control,
            control_identity_sha256=(
                None if self._control is None else self._control.identity.identity_sha256
            ),
            executed_stimulation_sha256=(
                None if self._control is None else result.record.neural.encoded_stimulation_sha256
            ),
        )
        self._last_observation_seq = observation.observation_seq
        self._last_game_tick = observation.game_tick
        self._last_command_seq = command_seq
        return action

    def _validate_observation(self, observation: Observation, command_seq: int) -> None:
        """Mirror ``ConnectomeController._validate_observation`` check for check.

        The parent's final check (the bound graph identity re-read) has no
        mutable counterpart here: the bindings are frozen dataclasses and the
        engine re-pins its own view every step, so the identity cannot drift
        between construction and use.
        """

        if not isinstance(observation, Observation):
            raise ControllerIntegrityError("controller accepts only protocol Observation")
        if observation.identity != self.identity:
            raise ControllerIntegrityError("observation identity does not match controller")
        require_sensor_contract(observation.sensors, self.identity.observation_schema_hash)
        if observation.missing_fields:
            raise ControllerIntegrityError("faithful input has declared missing sensor fields")
        if observation.observation_seq <= self._last_observation_seq:
            raise ControllerIntegrityError("observation sequence is stale or duplicated")
        if observation.game_tick <= self._last_game_tick:
            raise ControllerIntegrityError("observation tick is stale or duplicated")
        if isinstance(command_seq, bool) or not isinstance(command_seq, int):
            raise ControllerIntegrityError("command sequence must be an integer")
        if command_seq <= self._last_command_seq:
            raise ControllerIntegrityError("command sequence is stale or duplicated")

    def _encode_dense(
        self, values: Mapping[str, float]
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], tuple[tuple[str, int, int, float], ...]]:
        """Dense sensory vector with ``SensoryAdapter.encode``'s exact arithmetic.

        The channel values are placed at the entry bodies' full-graph indices
        and are **not** pre-multiplied by ``sensory_gain``: the engine applies
        the gain inside its step, mirroring the reference engine's ordering.
        """

        vector = np.zeros(self._node_count, dtype=np.float64)
        assignments: list[tuple[str, int, int, float]] = []
        for channel in self._sensory_binding.channels:
            raw = float(values[channel.name])
            if not math.isfinite(raw):
                raise ControllerIntegrityError(f"observation[{channel.name}] is non-finite")
            value = channel.gain * raw + channel.offset
            value = min(channel.clip_max, max(channel.clip_min, value))
            if channel.saturation == "tanh":
                value = math.tanh(value)
            for target in channel.targets:
                contribution = value * target.weight
                vector[target.dense_index] += contribution
                assignments.append((channel.name, target.body_id, target.dense_index, contribution))
        if not bool(np.isfinite(vector).all()):
            raise ControllerIntegrityError("encoded full-graph sensory vector is non-finite")
        return vector, tuple(assignments)

    def _stimulation_identity(self, assignments: tuple[tuple[str, int, int, float], ...]) -> str:
        """Deterministic provenance digest over the canonical assignments.

        Mirrors ``EncodedStimulation.identity_sha256``'s material shape
        (adapter identity + graph identity + hashed assignments) without
        importing the fixture-bound machinery. The executed dense vector's own
        digest is derived independently by the engine from the array it
        executes, so this provenance value can never stand in for it.
        """

        return canonical_sha256(
            {
                "adapter_sha256": self._sensory_binding.identity_sha256,
                "assignments": [
                    [name, body_id, dense_index, float(value).hex()]
                    for name, body_id, dense_index, value in assignments
                ],
                "graph_identity_sha256": self._graph_manifest_sha256,
                "schema": FULL_BRAIN_ENCODED_STIMULATION_SCHEMA,
            }
        )

    def _apply_embodied_drives(
        self, vector: np.ndarray[Any, np.dtype[np.float64]], sensors: SensorFrame
    ) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
        """Place the per-capita EEM drives on top of the faithful entries.

        Every member of a population receives the identical drive value in
        [0, 1]; populations are pairwise disjoint and disjoint from the
        faithful entries (refused at binding time), so the vectorized ``+=``
        below never sums two EEM drives onto one index. Values are, as with
        the faithful channels, not pre-multiplied by the engine's sensory
        gain -- the engine applies its gain uniformly inside the step.
        """

        embodied = self._embodied
        assert embodied is not None
        channel_values = encode_channel_values(sensors)
        drives = population_drives(dict(channel_values), embodied.encoder.channels)
        drive_by_name = dict(drives)
        for name, indices in self._population_indices:
            vector[indices] += drive_by_name[name]
        if not bool(np.isfinite(vector).all()):
            raise ControllerIntegrityError("embodied sensory vector is non-finite")
        return channel_values, drives

    def _stimulation_identity_embodied(
        self,
        assignments: tuple[tuple[str, int, int, float], ...],
        channel_values: tuple[tuple[str, float], ...],
        drives: tuple[tuple[str, float], ...],
    ) -> str:
        """Provenance digest for a faithful-plus-EEM dense vector.

        The per-population drives plus the hashed population binding fully
        determine the EEM placement (per-capita, no per-member weights), so no
        per-member list is hashed; the engine still digests the executed dense
        vector independently, exactly as on the wing path.
        """

        embodied = self._embodied
        assert embodied is not None
        return canonical_sha256(
            {
                "adapter_sha256": self._sensory_binding.identity_sha256,
                "assignments": [
                    [name, body_id, dense_index, float(value).hex()]
                    for name, body_id, dense_index, value in assignments
                ],
                "channel_values": [[name, float(value).hex()] for name, value in channel_values],
                "encoder_document_sha256": embodied.encoder.identity_sha256,
                "graph_identity_sha256": self._graph_manifest_sha256,
                "population_binding_sha256": embodied.populations.identity_sha256,
                "population_drives": [[name, float(value).hex()] for name, value in drives],
                "schema": FULL_BRAIN_EEM_ENCODED_STIMULATION_SCHEMA,
            }
        )

    def _apply_embodied_drives_v2(
        self, vector: np.ndarray[Any, np.dtype[np.float64]], sensors: SensorFrame
    ) -> tuple[
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
    ]:
        """Temporal-contrast (eem v2) drives over the byte-identical v1 raw values.

        The raw values are exactly :func:`encode_channel_values` (the v1
        formulas); the executed drive is the declared contrast encoding over
        them, read from the bound v2 encoder document (alpha and the
        contrast/absolute partition are document fields, so the executable
        path cannot drift from the hashed declaration). The running means are
        the only cross-tick state: a pure function of the observation history,
        replaced wholesale each game tick, cleared by :meth:`reset`. Placement
        is per-capita on disjoint populations, exactly as v1.

        Returns ``(raw_values, means_used, channel_values, drives)``.
        """

        embodied = self._embodied
        assert embodied is not None
        encoder = embodied.encoder
        assert isinstance(encoder, EemEncoderDocumentV2)
        raw_values = (
            encode_entity_channel_values(sensors)
            if isinstance(encoder, EemEncoderDocumentLoomV2)
            else encode_channel_values(sensors)
        )
        # Which channels are rectified is a field of the bound encoder
        # document, exactly as alpha and the contrast partition are, so the
        # executable path cannot drift from the hashed declaration. Every
        # non-loom document declares none, and its arithmetic is unchanged.
        rectified_channels: tuple[str, ...] = (
            encoder.rectified_channels if isinstance(encoder, EemEncoderDocumentLoomV1) else ()
        )
        channel_values, means_used, new_means = apply_temporal_contrast(
            raw_values,
            self._eem_contrast_means,
            alpha=encoder.alpha,
            contrast_channels=encoder.contrast_channels,
            rectified_channels=rectified_channels,
        )
        self._eem_contrast_means = new_means
        drives = population_drives(dict(channel_values), encoder.channels)
        drive_by_name = dict(drives)
        for name, indices in self._population_indices:
            vector[indices] += drive_by_name[name]
        if not bool(np.isfinite(vector).all()):
            raise ControllerIntegrityError("embodied sensory vector is non-finite")
        return raw_values, means_used, channel_values, drives

    def _stimulation_identity_embodied_v2(
        self,
        assignments: tuple[tuple[str, int, int, float], ...],
        raw_values: tuple[tuple[str, float], ...],
        means_used: tuple[tuple[str, float], ...],
        channel_values: tuple[tuple[str, float], ...],
        drives: tuple[tuple[str, float], ...],
    ) -> str:
        """Provenance digest for a faithful-plus-contrast-EEM dense vector.

        Chains, beyond the v1 material, the raw (pre-contrast) channel values
        and the exact baselines each contrast channel was scored against, so
        an auditor can recompute the encoded values and the leaky-mean update
        from the digested material alone; the engine still digests the
        executed dense vector independently.
        """

        embodied = self._embodied
        assert embodied is not None
        # The escape-v2 and loom-v1 encoders ride the same contrast machinery,
        # each under its own per-step stimulation provenance schema; the eem-v2
        # schema value is byte-unchanged for every other contrast document.
        # The material below records the raw values and the exact baselines,
        # so which channels were rectified is recoverable from the chained
        # ``encoder_document_sha256`` and reproducible from the material.
        if isinstance(embodied.encoder, EemEncoderDocumentLoomV2):
            schema = FULL_BRAIN_EEM_LOOM_V2_ENCODED_STIMULATION_SCHEMA
        elif isinstance(embodied.encoder, EemEncoderDocumentLoomV1):
            schema = FULL_BRAIN_EEM_LOOM_V1_ENCODED_STIMULATION_SCHEMA
        elif isinstance(embodied.encoder, EemEncoderDocumentEscapeV2):
            schema = FULL_BRAIN_EEM_ESCAPE_V2_ENCODED_STIMULATION_SCHEMA
        else:
            schema = FULL_BRAIN_EEM_V2_ENCODED_STIMULATION_SCHEMA
        return canonical_sha256(
            {
                "adapter_sha256": self._sensory_binding.identity_sha256,
                "assignments": [
                    [name, body_id, dense_index, float(value).hex()]
                    for name, body_id, dense_index, value in assignments
                ],
                "channel_values": [[name, float(value).hex()] for name, value in channel_values],
                "contrast_means_used": [[name, float(value).hex()] for name, value in means_used],
                "encoder_document_sha256": embodied.encoder.identity_sha256,
                "graph_identity_sha256": self._graph_manifest_sha256,
                "population_binding_sha256": embodied.populations.identity_sha256,
                "population_drives": [[name, float(value).hex()] for name, value in drives],
                "raw_channel_values": [[name, float(value).hex()] for name, value in raw_values],
                "schema": schema,
            }
        )

    def _decode(
        self, activity: np.ndarray[Any, np.dtype[np.float64]], state_sha256: str
    ) -> MotorFrame:
        """Pool readout with ``MotorDecoder.decode``'s exact arithmetic.

        The only addition is the trailing multiplication by the declared
        ``readout_scale``. At the default 1.0 the product is bit-identical to
        the uncalibrated value, so the uncalibrated profile's arithmetic is
        unchanged; at a calibrated scale the same single factor applies to both
        pools, before the pinned clip and dead zone.
        """

        values: list[tuple[str, float]] = []
        readout_scale = self._motor_binding.readout_scale
        for pool in self._motor_binding.pools:
            raw = (
                pool.gain
                * math.fsum(
                    member.weight * float(activity[member.dense_index]) for member in pool.members
                )
                / pool.denominator
                * readout_scale
            )
            if not math.isfinite(raw):
                raise ControllerIntegrityError("motor readout produced a non-finite raw value")
            value = min(1.0, max(-1.0, raw))
            if abs(value) < pool.dead_zone:
                value = 0.0
            values.append((pool.name, value))
        value_tuple = tuple(values)
        identity = canonical_sha256(
            {
                "decoder_sha256": self.identity.decoder_hash,
                "graph_identity_sha256": self._graph_manifest_sha256,
                "neural_state_sha256": state_sha256,
                "values": [[name, float(value).hex()] for name, value in value_tuple],
            }
        )
        return MotorFrame(
            graph_identity_sha256=self._graph_manifest_sha256,
            decoder_sha256=self.identity.decoder_hash,
            neural_state_sha256=state_sha256,
            values=value_tuple,
            identity_sha256=identity,
        )

    def _decode_jump(self, activity: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        """The declared TTMn jump readout.

        Mirrors :meth:`_decode`'s per-pool arithmetic exactly -- mean member
        activity times the declared gain times the declared ``jump_scale``,
        then the declared clip and dead zone -- over the jump pool. The clip
        range is the one the bound decoder document declares: the full signed
        codomain for the escape-v1 binding (``ESCAPE_JUMP_FORMULA``), the
        rectified non-negative half for the escape-v2 binding
        (``ESCAPE_JUMP_FORMULA_V2``: negative pooled TTMn activity maps to
        zero, no anti-jump). The wing frame is untouched; the returned value
        is the one additive vertical term.
        """

        escape = self._escape
        assert escape is not None
        pool = escape.jump_pool
        raw = (
            pool.gain
            * math.fsum(
                member.weight * float(activity[member.dense_index]) for member in pool.members
            )
            / pool.denominator
            * escape.jump_scale
        )
        if not math.isfinite(raw):
            raise ControllerIntegrityError("jump readout produced a non-finite raw value")
        if isinstance(escape, EscapeMotorBindingV2):
            value = min(1.0, max(0.0, raw))
            if value < pool.dead_zone:
                value = 0.0
            return value
        value = min(1.0, max(-1.0, raw))
        if abs(value) < pool.dead_zone:
            value = 0.0
        return value

    # ------------------------------------------------------------- construction

    @classmethod
    def from_artifact(
        cls,
        *,
        run_id: str,
        entity_id: str,
        session_epoch: int = 0,
        configuration: FullBrainConfiguration | None = None,
        artifact_directory: Path,
        annotations_path: Path,
        calibration_path: Path | None = None,
        expected_manifest_sha256: str = FULL_MCNS_ARTIFACT_MANIFEST_SHA256,
        embodied_senses: bool = False,
        embodied_encoder_version: int = 1,
        body_annotations_path: Path | None = None,
        escape_reflex: bool = False,
        escape_version: int = 1,
        loom_encoder: bool = False,
        loom_version: int = 1,
        steering_readout: bool = False,
        control_seed: int = 7,
        control_permutation_directory: Path | None = None,
        event_inputs: bool = False,
        event_inputs_version: int = 1,
    ) -> FullBrainController:
        """Load, verify, sign, and bind the real full artifact end to end.

        The chain is: verified memory-mapped artifact (manifest pinned to
        ``expected_manifest_sha256``) -> coarse-consensus sign overlay computed
        from the digest-verified neurotransmitters table -> runtime view whose
        provenance string carries the overlay identity -> array-native engine.
        The heavyweight imports happen here rather than at module import so
        that importing this module never drags in ``pyarrow``.

        ``calibration_path`` is optional and defaults to ``None`` = the
        uncalibrated readout (scale 1.0). When supplied, the document's bytes
        are read once; its digest, its scale, and the boundary it names are all
        derived from that one read, and construction fails closed unless the
        artifact manifest and sign overlay it was measured on are exactly the
        ones loaded here.

        ``embodied_senses=True`` builds an Engineered-Encoder profile
        (ADR 0008 Mode 3): the digest-verified body-annotations table at
        ``body_annotations_path`` supplies the labeled populations, the
        declared encoder document is bound over them, and ``configuration``
        (which must then be a plain, uncalibrated ``FullBrainConfiguration``
        base -- the engineered-encoder configuration is derived here so its
        encoder identity matches the loaded binding) is converted to the
        matching EEM configuration. ``embodied_encoder_version`` selects the
        declared encoder document: ``1`` (default) is the v1 absolute-level
        encoder (``full_mcns_eem_v1``), ``2`` the temporal-contrast encoder
        (``full_mcns_eem_v2``). The calibration cross-binding then
        additionally requires the document's encoder identity to match, in
        both directions -- which also keeps the wing, eem-v1, and eem-v2
        calibrations mutually refusing.

        ``escape_reflex=True`` builds the escape profile
        (``full_mcns_escape_v1``): the eem v1 sensory boundary verbatim
        (``embodied_senses=True`` with the v1 encoder is required) plus the
        digest-verified TTMn jump pool and the declared jump mapping onto
        vertical thrust. Its ``calibration_path`` must then name an ESCAPE
        calibration document ({wing_scale, jump_scale}, own schema); the v1
        and escape calibration formats refuse each other in both directions,
        and the escape document is additionally cross-bound to the exact TTMn
        pool identity loaded here.

        ``escape_version`` selects the escape decoder generation and applies
        to the escape profile only: ``1`` (default) is the frozen escape-v1
        identity above; ``2`` builds ``full_mcns_escape_v2`` (identity
        recorded in experiments/preregistration/eem-escape-v2.md) -- the
        escape-v2 (transient-touch) encoder document over the same
        populations, the rectified jump mapping
        (``ESCAPE_JUMP_FORMULA_V2``), and an escape-v2 calibration document
        under its own schema, mutually refused with every other calibration
        boundary in both directions.

        ``loom_encoder=True`` builds ``full_mcns_loom_v1`` (declared in
        ``docs/encoders/eem-loom-v1.md``) and requires ``escape_reflex=True``
        with ``escape_version=2``: the escape-v2 DECODER unchanged over the
        loom-v1 (approach-gated) encoder document, whose three threat channels
        drive their LC4 populations with the rectified rise of proximity above
        a declared leaky mean instead of with the absolute level. Its
        ``calibration_path`` must then name a LOOM-V1 calibration document
        under its own schema, mutually refused with every other calibration
        boundary in both directions.

        ``steering_readout=True`` additionally selects the declared DNa01
        descending pair (``steering_readout.py``) and exposes it through
        :meth:`steering_readout_means`. It requires ``embodied_senses=True``
        and ``escape_reflex=True``, because the selector refuses overlap with
        the embodied populations and with the TTMn jump pool and therefore
        needs both bindings to check against. It adds NO term to any motor
        axis and enters NO ``RunIdentity`` field: the controller's identity
        with and without it is byte-identical. Reading a descending neuron
        departs from ``escape_reflex.py``'s motor-stage declaration on
        purpose; that is a recorded taxonomy decision, not a selector detail.

        Control modes reuse the same encoder, readout bindings, and calibration
        bytes. ``control_seed`` names the existing full-scale control transform;
        the resulting permutation can be saved in ``control_permutation_directory``.
        Only controls wrap the backend dynamics identity. The original path
        neither constructs a control nor changes its declared identity material.
        """

        config = configuration or FullBrainConfiguration()
        if not isinstance(event_inputs, bool):
            raise FullBrainProfileError("event_inputs must be a boolean")
        if type(event_inputs_version) is not int or event_inputs_version not in (1, 2):
            raise FullBrainProfileError("event_inputs_version must be 1 or 2")
        if not event_inputs and event_inputs_version != 1:
            raise FullBrainProfileError("event_inputs_version requires event_inputs")
        if event_inputs and not (loom_encoder and loom_version == 2 and steering_readout):
            raise FullBrainProfileError(
                "candidate events require loom-v2 components and the steering readout"
            )
        if config.mode != "faithful" and (
            isinstance(control_seed, bool)
            or not isinstance(control_seed, int)
            or not 0 <= control_seed < (1 << 64)
        ):
            raise FullBrainProfileError("control_seed must be a uint64 integer")
        if embodied_encoder_version not in (1, 2):
            raise FullBrainProfileError("embodied_encoder_version must be 1 or 2")
        if embodied_encoder_version != 1 and not embodied_senses:
            raise FullBrainProfileError(
                "embodied_encoder_version applies to the embodied-senses profile only"
            )
        if escape_version not in (1, 2):
            raise FullBrainProfileError("escape_version must be 1 or 2")
        if escape_version != 1 and not escape_reflex:
            raise FullBrainProfileError("escape_version applies to the escape-reflex profile only")
        if escape_reflex and not embodied_senses:
            raise FullBrainProfileError(
                "the escape-reflex profile requires embodied_senses=True: its sensory "
                "boundary is the eem v1 encoder over the labeled populations"
            )
        if escape_reflex and embodied_encoder_version != 1:
            raise FullBrainProfileError(
                "the escape-reflex profile is declared over the eem v1 (absolute) "
                "encoder only as far as embodied_encoder_version is concerned; a "
                "different sensory boundary is selected by escape_version / "
                "loom_encoder instead -- escape_version=2 for the escape-v2 "
                "(transient-touch) encoder, or loom_encoder=True with "
                "escape_version=2 for the loom-v1 (approach-gated) encoder"
            )
        if steering_readout and not (embodied_senses and escape_reflex):
            raise FullBrainProfileError(
                "the DNa01 steering readout requires embodied_senses=True and "
                "escape_reflex=True: its selector refuses overlap with the embodied "
                "populations and with the TTMn jump pool, so both bindings must exist"
            )
        if isinstance(loom_version, bool) or loom_version not in (1, 2):
            raise FullBrainProfileError("loom_version must be 1 or 2")
        if loom_version != 1 and not loom_encoder:
            raise FullBrainProfileError("loom_version=2 requires loom_encoder=True")
        if loom_encoder and not (escape_reflex and escape_version == 2):
            raise FullBrainProfileError(
                "the loom-v1 (approach-gated) profile requires escape_reflex=True "
                "with escape_version=2: it is the escape-v2 rectified jump decoder "
                "over the loom-v1 encoder document"
            )
        if escape_reflex:
            eem_profile_label = (
                f"full_mcns_loom_v{loom_version}"
                if loom_encoder
                else ("full_mcns_escape_v2" if escape_version == 2 else "full_mcns_escape_v1")
            )
        else:
            eem_profile_label = (
                "full_mcns_eem_v2" if embodied_encoder_version == 2 else "full_mcns_eem_v1"
            )
        if embodied_senses:
            if body_annotations_path is None:
                raise FullBrainProfileError(
                    "the embodied-senses profile requires body_annotations_path"
                )
            if type(config) is not FullBrainConfiguration:
                raise FullBrainProfileError(
                    "pass a plain FullBrainConfiguration base for embodied_senses; the "
                    "engineered-encoder configuration is derived here so its encoder "
                    "identity matches the loaded binding"
                )
            if not Path(body_annotations_path).is_file():
                raise FullBrainProfileError(
                    f"{eem_profile_label} is a local-only profile; the body-annotations "
                    f"table is not present at {body_annotations_path} (fetch it with "
                    "scripts/acquire_malecns.py --include-metadata)"
                )
        elif body_annotations_path is not None:
            raise FullBrainProfileError(
                "body_annotations_path applies to the embodied-senses profile only"
            )
        if calibration_path is None and config.readout_scale != 1.0:
            # A calibrated configuration may only enter this path through
            # calibration_path, where assert_binds verifies the document
            # against the boundary actually loaded; accepting a caller-built
            # pair here would let an arbitrary scale be attested under an
            # arbitrary digest without any document existing.
            raise FullBrainProfileError(
                "a calibrated configuration requires calibration_path so the "
                "declared document is verified against the loaded boundary"
            )
        calibration: FullReadoutCalibration | None = None
        escape_calibration: EscapeReadoutCalibration | None = None
        if calibration_path is not None:
            # Shape-validate the calibration before the multi-second verified
            # load, so a bad document fails in milliseconds; the boundary
            # cross-check below still runs against the artifact actually loaded.
            if config.readout_scale != 1.0 or config.readout_calibration_sha256 != "":
                raise FullBrainProfileError(
                    "a calibration_path was supplied alongside a configuration that "
                    "already declares a readout calibration; supply exactly one"
                )
            try:
                raw_calibration = Path(calibration_path).read_bytes()
            except OSError as exc:
                raise FullBrainProfileError(
                    f"could not read the readout calibration at {calibration_path}: {exc.strerror}"
                ) from exc
            if loom_encoder:
                escape_calibration = (
                    load_loom_v2_readout_calibration(raw_calibration)
                    if loom_version == 2
                    else load_loom_v1_readout_calibration(raw_calibration)
                )
            elif escape_reflex:
                escape_calibration = (
                    load_escape_v2_readout_calibration(raw_calibration)
                    if escape_version == 2
                    else load_escape_readout_calibration(raw_calibration)
                )
            else:
                calibration = load_readout_calibration(raw_calibration)
        if not (Path(artifact_directory) / "manifest.json").is_file():
            raise FullBrainProfileError(
                "full_mcns_wing_v1 is a local-only profile; the full artifact is not "
                f"present at {artifact_directory} (build it with "
                "scripts/build_malecns_runtime.py)"
            )
        if not Path(annotations_path).is_file():
            raise FullBrainProfileError(
                "full_mcns_wing_v1 is a local-only profile; the neurotransmitter "
                f"annotations table is not present at {annotations_path} (fetch it with "
                "scripts/acquire_malecns.py --include-metadata)"
            )
        from neurocraft_connectome.full_runtime import load_full_runtime_artifact
        from neurocraft_neural.full_signs import compute_full_sign_overlay
        from neurocraft_neural.full_view import view_from_full_runtime_graph

        graph = load_full_runtime_artifact(
            artifact_directory, expected_manifest_sha256=expected_manifest_sha256
        )
        try:
            overlay = compute_full_sign_overlay(graph, Path(annotations_path))
            view = view_from_full_runtime_graph(
                graph,
                signs=overlay.edge_signs,
                sign_provenance=f"{SIGN_OVERLAY_POLICY_PREFIX}:{overlay.identity_sha256}",
            )
            manifest_sha256 = graph.manifest.manifest_file_sha256
            control: FullControlView | None = None
            if config.mode != "faithful":
                control = build_control_view(
                    view,
                    kind=FullControlKind(config.mode),
                    seed=control_seed,
                    artifact_manifest_sha256=manifest_sha256,
                    raw_counts=(
                        _verified_graph_array(graph, "edge_raw_counts", "<u8").astype(np.int64)
                        if config.mode == "shuffled_weight"
                        else None
                    ),
                    edge_signs=overlay.edge_signs,
                    persist_directory=control_permutation_directory,
                )
                view = control.view
            embodied: EmbodiedSensesBinding | None = None
            if embodied_senses:
                assert body_annotations_path is not None
                if loom_encoder:
                    bind = (
                        bind_embodied_senses_loom_v2
                        if loom_version == 2
                        else bind_embodied_senses_loom_v1
                    )
                elif escape_reflex and escape_version == 2:
                    bind = bind_embodied_senses_escape_v2
                elif embodied_encoder_version == 2:
                    bind = bind_embodied_senses_v2
                else:
                    bind = bind_embodied_senses
                embodied = bind(
                    load_embodied_population_binding(graph, Path(body_annotations_path))
                )
            event_populations: EventPopulationBinding | None = None
            if event_inputs:
                assert body_annotations_path is not None
                event_populations = load_event_populations(graph, Path(body_annotations_path))
            escape_ttmn: TtmnPoolBinding | None = None
            if escape_reflex:
                assert embodied is not None
                assert body_annotations_path is not None
                escape_ttmn = load_escape_ttmn_binding(
                    graph, Path(body_annotations_path), embodied.populations
                )
            steering: SteeringPairBinding | None = None
            if steering_readout:
                # Selected inside the graph's lifetime (it is closed in the
                # `finally` below) and gated on both other bindings, which the
                # selector's disjointness refusals check against.
                assert embodied is not None
                assert escape_ttmn is not None
                assert body_annotations_path is not None
                steering = load_steering_binding(
                    graph, Path(body_annotations_path), embodied.populations, escape_ttmn
                )
            if calibration is not None:
                calibration.assert_binds(
                    artifact_manifest_sha256=manifest_sha256,
                    sign_overlay_identity_sha256=overlay.identity_sha256,
                    encoder_identity_sha256=(
                        "" if embodied is None else embodied.encoder.identity_sha256
                    ),
                )
                config = replace(
                    config,
                    readout_scale=calibration.readout_scale,
                    readout_calibration_sha256=calibration.document_sha256,
                )
            escape_jump_scale = 1.0
            if escape_calibration is not None:
                assert embodied is not None and escape_ttmn is not None
                escape_calibration.assert_binds(
                    artifact_manifest_sha256=manifest_sha256,
                    sign_overlay_identity_sha256=overlay.identity_sha256,
                    encoder_identity_sha256=embodied.encoder.identity_sha256,
                    ttmn_pool_identity_sha256=escape_ttmn.identity_sha256,
                )
                config = replace(
                    config,
                    readout_scale=escape_calibration.wing_scale,
                    readout_calibration_sha256=escape_calibration.document_sha256,
                )
                escape_jump_scale = escape_calibration.jump_scale
            if embodied is not None and escape_reflex:
                escape_configuration_type: type[FullBrainEscapeConfiguration]
                if loom_encoder:
                    escape_configuration_type = (
                        FullBrainLoomV2Configuration
                        if loom_version == 2
                        else FullBrainLoomV1Configuration
                    )
                elif escape_version == 2:
                    escape_configuration_type = FullBrainEscapeV2Configuration
                else:
                    escape_configuration_type = FullBrainEscapeConfiguration
                config = escape_configuration_type(
                    mode=config.mode,
                    sign_policy=config.sign_policy,
                    neural_steps_per_game_tick=config.neural_steps_per_game_tick,
                    motor_pool_gain=config.motor_pool_gain,
                    motor_dead_zone=config.motor_dead_zone,
                    kernel_threads=config.kernel_threads,
                    readout_scale=config.readout_scale,
                    readout_calibration_sha256=config.readout_calibration_sha256,
                    encoder_identity_sha256=embodied.encoder.identity_sha256,
                    jump_scale=escape_jump_scale,
                )
            elif embodied is not None:
                eem_configuration_type = (
                    FullBrainEemV2Configuration
                    if embodied_encoder_version == 2
                    else FullBrainEemConfiguration
                )
                config = eem_configuration_type(
                    mode=config.mode,
                    sign_policy=config.sign_policy,
                    neural_steps_per_game_tick=config.neural_steps_per_game_tick,
                    motor_pool_gain=config.motor_pool_gain,
                    motor_dead_zone=config.motor_dead_zone,
                    kernel_threads=config.kernel_threads,
                    readout_scale=config.readout_scale,
                    readout_calibration_sha256=config.readout_calibration_sha256,
                    encoder_identity_sha256=embodied.encoder.identity_sha256,
                )
            sensory_binding = bind_sensor_channels(
                # The pinned channel builder over the pinned default boundary:
                # entry bodies 805722 (left) and 77185 (right), the same
                # entries every fixture controller uses.
                channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
                index_of=graph.index_of,
                node_count=graph.manifest.node_count,
                graph_manifest_sha256=manifest_sha256,
            )
            motor_binding = bind_wing_motor_pools(
                index_of=graph.index_of,
                gain=config.motor_pool_gain,
                dead_zone=config.motor_dead_zone,
                graph_manifest_sha256=manifest_sha256,
                sign_overlay_identity_sha256=overlay.identity_sha256,
                readout_scale=config.readout_scale,
                readout_calibration_sha256=config.readout_calibration_sha256,
            )
            escape_binding: EscapeMotorBinding | None = None
            if escape_reflex:
                assert escape_ttmn is not None
                assert embodied is not None
                escape_binding_type = (
                    EscapeMotorBindingV2 if escape_version == 2 else EscapeMotorBinding
                )
                escape_binding = escape_binding_type(
                    wing=motor_binding,
                    jump_pool=bind_escape_jump_pool(
                        escape_ttmn,
                        gain=config.motor_pool_gain,
                        dead_zone=config.motor_dead_zone,
                    ),
                    jump_scale=escape_jump_scale,
                    ttmn_pool_identity_sha256=escape_ttmn.identity_sha256,
                    encoder_identity_sha256=embodied.encoder.identity_sha256,
                )
        finally:
            graph.close()
        event_binding: EventEncoderBinding | None = None
        if event_inputs:
            assert (
                event_populations is not None
                and embodied is not None
                and escape_binding is not None
            )
            event_arguments = (
                event_populations,
                embodied.encoder.identity_sha256,
                escape_binding.identity_sha256,
                config.readout_calibration_sha256,
            )
            event_binding = (
                InteractionEncoderBinding(*event_arguments, embodied.populations)
                if event_inputs_version == 2
                else EventEncoderBinding(*event_arguments)
            )
            event_configuration = (
                FullBrainInteractionsConfiguration
                if event_inputs_version == 2
                else FullBrainEventsConfiguration
            )
            config = event_configuration(
                **{
                    field.name: getattr(config, field.name)
                    for field in fields(config)
                    if field.name != "schema"
                },
                event_encoder_sha256=event_binding.identity_sha256,
            )
        engine = ArrayNativeEngine(view, kernel=load_kernel(thread_count=config.kernel_threads))
        return cls(
            run_id=run_id,
            entity_id=entity_id,
            session_epoch=session_epoch,
            configuration=config,
            engine=engine,
            sensory_binding=sensory_binding,
            motor_binding=motor_binding,
            embodied=embodied,
            escape=escape_binding,
            steering=steering,
            control=control,
            event_encoder=event_binding,
        )
