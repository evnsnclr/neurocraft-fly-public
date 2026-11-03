"""Frozen provenance identities for a body that the pinned controller can drive.

Nothing here touches ``RunIdentity`` or ``TrialIdentity``. Both are wire- and
receipt-bound: ``RunIdentity`` declares an exact ``FIELDS`` set enforced by the
protocol codecs and by the pinned cross-language handshake transcript, and
``TrialIdentity.identity_sha256`` hashes its own ``asdict``, so adding a field to
either would silently rotate the identity of every already-recorded artifact.
``EmbodimentIdentity`` is therefore a *sibling*: it is recorded in the trace
manifest's existing ``arena_hash`` slot and never serialized over the protocol.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import Any

from neurocraft_protocol import DeterminismLevel, canonical_json_bytes

EMBODIMENT_IDENTITY_SCHEMA = "neurocraft.embodiment.identity.v1"
BOUNDARY_IDENTITY_SCHEMA = "neurocraft.embodiment.boundary.identity.v1"

#: The v1 clock contract, frozen by ``ControllerConfiguration`` (five neural steps
#: per tick) and ``DynamicsConfig.neural_step_ms`` (10 ms). Every embodiment runs at
#: this rate; changing it is an ADR-level act, not a constructor argument.
#:
#: These live here, beside the identity that has to enforce them, rather than in
#: ``seam`` where they used to sit: ``seam`` imports ``identity``, so an identity
#: that wanted to check itself against the seam's rate could not see it, and
#: :meth:`EmbodimentIdentity._assert_clock_closes` closed the substep product
#: against the body's *own* ``control_hz`` -- a self-declaration checking itself.
#: ``seam`` re-exports both names, so ``from .seam import CONTROL_HZ`` still works.
CONTROL_HZ = 20.0
CONTROL_PERIOD_NS = 50_000_000

#: Which currency an embodiment's motor boundary consumes. Declared, never inferred.
CONSUMES_PRIMITIVE_MOTOR_VECTOR = "primitive_motor_vector"
CONSUMES_NEURAL_STATE = "neural_state"
MOTOR_CURRENCIES = frozenset({CONSUMES_PRIMITIVE_MOTOR_VECTOR, CONSUMES_NEURAL_STATE})

BOUNDARY_ROLES = frozenset({"sensory", "motor"})

#: Where one boundary channel's scalar comes from. Declared, never inferred.
#:
#: ``dynamic`` varies tick to tick with the simulation (sensory) or with the neural
#: state (motor). ``config_constant`` is pinned by a value that already lives inside
#: ``configuration_sha256``, so changing it rotates the embodiment identity today.
#: ``literal_constant`` is pinned by a source literal, which *nothing* else in the
#: identity covers -- the ``derivation`` string beside it is the only record of what
#: the value is and why. That asymmetry is exactly what makes the three-way split
#: worth hashing rather than a plain constant/varies flag.
CHANNEL_SOURCE_DYNAMIC = "dynamic"
CHANNEL_SOURCE_CONFIG_CONSTANT = "config_constant"
CHANNEL_SOURCE_LITERAL_CONSTANT = "literal_constant"
CHANNEL_SOURCES = frozenset(
    {
        CHANNEL_SOURCE_DYNAMIC,
        CHANNEL_SOURCE_CONFIG_CONSTANT,
        CHANNEL_SOURCE_LITERAL_CONSTANT,
    }
)

_HASH_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class EmbodimentIdentityError(ValueError):
    """An embodiment declared provenance that cannot be hashed honestly."""


def canonical_sha256(value: Any) -> str:
    """Hash exactly as the controller runtime does, over canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def float_hex(value: float) -> str:
    """Lossless float encoding; identity material must never round-trip decimals."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EmbodimentIdentityError("identity float must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise EmbodimentIdentityError("identity float must be finite")
    return number.hex()


def _assert_hash(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _HASH_LENGTH or not set(value) <= _HEX_DIGITS:
        raise EmbodimentIdentityError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _assert_name(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise EmbodimentIdentityError(f"{name} must be a non-empty string of at most 128 chars")
    return value


def _assert_count(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EmbodimentIdentityError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ChannelDeclaration:
    """One scalar a boundary carries, and where that scalar comes from.

    ``derivation`` is a short declared string naming how the value is produced --
    ``thorax_height_above_ground``, ``false_neuromechfly_has_no_fluid_model``. For a
    ``literal_constant`` channel it is the *only* thing in the whole identity that
    records what the value is, so it is hashed with the same weight as the name.
    """

    name: str
    source: str
    derivation: str

    def __post_init__(self) -> None:
        _assert_name(self.name, "channel.name")
        if self.source not in CHANNEL_SOURCES:
            raise EmbodimentIdentityError(
                "channel source must be one of " + ", ".join(sorted(CHANNEL_SOURCES))
            )
        _assert_name(self.derivation, "channel.derivation")

    def to_identity_material(self) -> dict[str, Any]:
        return {"derivation": self.derivation, "name": self.name, "source": self.source}


@dataclass(frozen=True, slots=True)
class BoundaryIdentity:
    """One hashed sensory or motor boundary attached to an embodiment.

    ``free_parameters`` is the count of scalars a designer could have chosen. It is
    recorded next to ``channel_count`` because high dimension is acceptable in this
    project and high parametric capacity is not: a reviewer must be able to read the
    capacity off the identity without reading the code.

    The channel convention, stated once
    -----------------------------------
    ``channels`` carries one :class:`ChannelDeclaration` per scalar the boundary
    emits, in wire order. A vector is never counted as one channel.
    ``channel_count`` is the declared width of that frame and must equal
    ``len(channels)``.

    Why there are two declared numbers
    ----------------------------------
    The count equality alone was decoration. Every shipped call site passed
    ``len(names)`` as ``channel_count``, so the check could not fail for a real body
    however wrong the body was -- the same defect it was added to fix, one level up.
    It is kept because it *is* the wire contract with the controller's sensory
    adapter and it does fire when the two sides are derived differently, but the
    number that carries the information is ``declared_dynamic_channels``: how many of
    these scalars actually vary. Nothing else in the repository can derive it, it is
    asserted against the records' own tally, and it fires on the one edit that
    matters -- silently retagging a constant as dynamic, or pinning a real channel to
    a constant without saying so.

    ``channel_names``, ``dynamic_channel_count`` and ``constant_channel_count`` are
    read off ``channels`` and are never stored, so they cannot rot. Only ``channels``
    and ``channel_count`` are hashed: the tallies are implied by the records, and
    hashing redundant derived material would recreate the decorative-number smell in
    mirror image. ``declared_dynamic_channels`` is a construction-time assertion
    input, exactly as ``flygym_boundary._assert_declared``'s argument is, and is
    likewise absent from the material.
    """

    role: str
    schema: str
    identity_sha256: str
    channel_count: int
    channels: tuple[ChannelDeclaration, ...]
    declared_dynamic_channels: int
    free_parameters: int

    def __post_init__(self) -> None:
        if self.role not in BOUNDARY_ROLES:
            raise EmbodimentIdentityError("boundary role must be 'sensory' or 'motor'")
        _assert_name(self.schema, "boundary.schema")
        _assert_hash(self.identity_sha256, "boundary.identity_sha256")
        _assert_count(self.channel_count, "boundary.channel_count", minimum=1)
        _assert_count(self.declared_dynamic_channels, "boundary.declared_dynamic_channels")
        _assert_count(self.free_parameters, "boundary.free_parameters", minimum=1)
        self._assert_channels()

    def _assert_channels(self) -> None:
        if not isinstance(self.channels, tuple) or not all(
            isinstance(entry, ChannelDeclaration) for entry in self.channels
        ):
            raise EmbodimentIdentityError(
                "boundary.channels must be a tuple of ChannelDeclaration records"
            )
        names = self.channel_names
        if len(set(names)) != len(names):
            raise EmbodimentIdentityError("boundary channel names must be unique")
        if len(self.channels) != self.channel_count:
            raise EmbodimentIdentityError(
                "boundary channel-count mismatch: computed "
                f"{len(self.channels)}, declared {self.channel_count}"
            )
        if self.dynamic_channel_count != self.declared_dynamic_channels:
            raise EmbodimentIdentityError(
                "boundary dynamic-channel mismatch: the records tally "
                f"{self.dynamic_channel_count} dynamic channels, the boundary "
                f"declares {self.declared_dynamic_channels}"
            )

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.channels)

    @property
    def dynamic_channel_count(self) -> int:
        return sum(1 for entry in self.channels if entry.source == CHANNEL_SOURCE_DYNAMIC)

    @property
    def constant_channel_count(self) -> int:
        return len(self.channels) - self.dynamic_channel_count

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "channel_count": self.channel_count,
            "channels": [entry.to_identity_material() for entry in self.channels],
            "free_parameters": self.free_parameters,
            "identity_sha256": self.identity_sha256,
            "role": self.role,
            "schema": self.schema,
            "wrapper_schema": BOUNDARY_IDENTITY_SCHEMA,
        }


@dataclass(frozen=True, slots=True)
class EmbodimentIdentity:
    """Everything that makes a body a different body.

    A trace carrying this hash can never be ambiguous about which body produced it:
    the body's name and version, the exact configuration bytes, both boundary
    identities, the seed, the control/physics clock, the declared determinism tier,
    and any environment provenance the body must record (library versions, model
    digests) all fold into one SHA-256.
    """

    embodiment_id: str
    embodiment_version: str
    configuration_sha256: str
    sensory_boundary: BoundaryIdentity
    motor_boundary: BoundaryIdentity
    motor_currency: str
    seed: int
    control_hz: float
    physics_steps_per_control_tick: int
    physics_timestep_s: float
    determinism_level: DeterminismLevel
    provenance: tuple[tuple[str, str], ...] = ()
    schema: str = EMBODIMENT_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EMBODIMENT_IDENTITY_SCHEMA:
            raise EmbodimentIdentityError("unsupported embodiment identity schema")
        _assert_name(self.embodiment_id, "embodiment_id")
        _assert_name(self.embodiment_version, "embodiment_version")
        _assert_hash(self.configuration_sha256, "configuration_sha256")
        if not isinstance(self.sensory_boundary, BoundaryIdentity):
            raise EmbodimentIdentityError("sensory_boundary must be a BoundaryIdentity")
        if not isinstance(self.motor_boundary, BoundaryIdentity):
            raise EmbodimentIdentityError("motor_boundary must be a BoundaryIdentity")
        if self.sensory_boundary.role != "sensory" or self.motor_boundary.role != "motor":
            raise EmbodimentIdentityError("embodiment boundaries are declared in the wrong roles")
        if self.motor_currency not in MOTOR_CURRENCIES:
            raise EmbodimentIdentityError(
                "motor_currency must be 'primitive_motor_vector' or 'neural_state'"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EmbodimentIdentityError("seed must be an integer")
        if not math.isfinite(self.control_hz) or not 0.0 < self.control_hz <= 1000.0:
            raise EmbodimentIdentityError("control_hz must be finite and within (0, 1000]")
        _assert_count(
            self.physics_steps_per_control_tick,
            "physics_steps_per_control_tick",
            minimum=1,
        )
        if not math.isfinite(self.physics_timestep_s) or self.physics_timestep_s <= 0.0:
            raise EmbodimentIdentityError("physics_timestep_s must be finite and positive")
        if not isinstance(self.determinism_level, DeterminismLevel):
            raise EmbodimentIdentityError("determinism_level must be a DeterminismLevel")
        self._assert_clock_closes()
        self._assert_provenance()

    def _assert_clock_closes(self) -> None:
        """The physics substeps must exactly fill one control tick at the seam rate.

        Two separate facts, and the second one used to be missing.

        First, a body whose substeps do not sum to the control period is silently
        running the controller at a different rate than it claims, which would make
        any cross-embodiment comparison meaningless.

        Second -- and this is the hole -- that closure used to be evaluated against
        ``self.control_hz``, a number the body declares about itself and that this
        class accepted anywhere in ``(0, 1000]``. Both checks then agreed with each
        other and with nothing else. Meanwhile :func:`seam.observation_timing` stamps
        every observation at :data:`CONTROL_PERIOD_NS` unconditionally, and the
        controller's five 10 ms neural steps are fixed. So a body declaring 100 Hz
        with 5 x 10 ms substeps closed its own clock, recorded ``game_tick_hz=100``
        in the manifest, and was then driven at 20 Hz with observations 50 ms apart:
        a factor-of-five lie that passed end to end. ``control_hz`` is not a free
        parameter of a body; it is the v1 contract, so it is pinned here.
        """

        if self.control_hz != CONTROL_HZ:
            raise EmbodimentIdentityError(
                f"control_hz must be the frozen v1 seam rate {CONTROL_HZ!r} Hz, not "
                f"{self.control_hz!r}: observation_timing stamps every observation "
                f"{CONTROL_PERIOD_NS} ns apart for every body, so a body declaring a "
                "different rate would be recorded under a clock it is not run at"
            )
        control_period_s = 1.0 / self.control_hz
        simulated_s = self.physics_steps_per_control_tick * self.physics_timestep_s
        if not math.isclose(simulated_s, control_period_s, rel_tol=1e-9, abs_tol=1e-15):
            raise EmbodimentIdentityError(
                "physics_steps_per_control_tick * physics_timestep_s must equal 1 / control_hz "
                f"(got {simulated_s!r} vs {control_period_s!r})"
            )

    def _assert_provenance(self) -> None:
        if not isinstance(self.provenance, tuple):
            raise EmbodimentIdentityError("provenance must be a tuple of key/value pairs")
        keys: list[str] = []
        for entry in self.provenance:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise EmbodimentIdentityError("provenance entries must be (key, value) pairs")
            key, value = entry
            _assert_name(key, "provenance key")
            if not isinstance(value, str) or len(value) > 256:
                raise EmbodimentIdentityError(
                    f"provenance value for {key!r} must be a string of at most 256 chars"
                )
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise EmbodimentIdentityError("provenance keys must be unique")
        if list(keys) != sorted(keys):
            raise EmbodimentIdentityError("provenance keys must be sorted for canonical hashing")

    def to_identity_material(self) -> dict[str, Any]:
        return {
            "configuration_sha256": self.configuration_sha256,
            "control_hz": float_hex(self.control_hz),
            "determinism_level": self.determinism_level.value,
            "embodiment_id": self.embodiment_id,
            "embodiment_version": self.embodiment_version,
            "motor_boundary": self.motor_boundary.to_identity_material(),
            "motor_currency": self.motor_currency,
            "physics_steps_per_control_tick": self.physics_steps_per_control_tick,
            "physics_timestep_s": float_hex(self.physics_timestep_s),
            "provenance": [[key, value] for key, value in self.provenance],
            "schema": self.schema,
            "seed": self.seed,
            "sensory_boundary": self.sensory_boundary.to_identity_material(),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_identity_material())

    def provenance_mapping(self) -> Mapping[str, str]:
        return dict(self.provenance)


def sorted_provenance(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Normalize a provenance mapping into the sorted tuple the identity requires."""

    return tuple(sorted((str(key), str(value)) for key, value in values.items()))


# -- simulator-presence evidence ------------------------------------------------
#
# A body's ``provenance()`` names the third-party libraries it says produced its
# physics. Nothing checked whether any of them was actually installed, so a body
# built on a hand-written stand-in for a simulator -- ``FlyGymEmbodiment`` accepts a
# ``modules=`` handle that bypasses ``require_flygym`` entirely -- recorded
# ``flygym_version: "2.1.0-stub"`` inside a hash preimage and produced a trace that
# said nothing anywhere about the simulator being absent.
#
# The check below is deliberately *not* a string sniff for "stub". A hostile stub
# would report ``2.1.0``. It resolves the declared distribution through
# ``importlib.metadata`` in the process that is doing the running: a distribution
# that is not installed cannot have executed a single step, whatever it called
# itself. That direction is sound. The other direction is not, and is worded that
# way everywhere it is reported: a distribution being installed is not evidence that
# the run used it, because ``modules=`` can still hand the body a stand-in.

#: Provenance keys ending in this suffix name a Python distribution and its version.
PROVENANCE_VERSION_SUFFIX = "_version"

#: The body declares no third-party simulator library at all (the reference body:
#: its physics is its own declared arithmetic, so there is no library to be absent).
SIMULATOR_EVIDENCE_NONE_DECLARED = "no_declared_simulator_libraries"
#: Every declared library resolves, in this process, to exactly the declared version.
SIMULATOR_EVIDENCE_INSTALLED = "declared_simulator_libraries_installed"
#: At least one declared library is missing or resolves to a different version, so it
#: cannot have executed anything during the run.
SIMULATOR_EVIDENCE_ABSENT = "declared_simulator_libraries_absent"


@dataclass(frozen=True, slots=True)
class LibraryEvidence:
    """One declared simulator library, and what the environment actually has."""

    distribution: str
    declared_version: str
    installed_version: str | None

    @property
    def present(self) -> bool:
        return self.installed_version is not None and self.installed_version == (
            self.declared_version
        )

    def to_receipt_entry(self) -> dict[str, Any]:
        return {
            "declared_version": self.declared_version,
            "distribution": self.distribution,
            "installed_version": self.installed_version,
            "present": self.present,
        }


def declared_runtime_libraries(provenance: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Every ``<distribution>_version`` fact in a provenance mapping, sorted.

    The convention is the one both shipped bodies already follow -- ``flygym_version``,
    ``mujoco_version`` -- read off the keys rather than declared in a second table
    that could disagree with them.
    """

    libraries = [
        (key[: -len(PROVENANCE_VERSION_SUFFIX)], value)
        for key, value in provenance.items()
        if key.endswith(PROVENANCE_VERSION_SUFFIX) and key != PROVENANCE_VERSION_SUFFIX
    ]
    return tuple(sorted(libraries))


def resolve_library_evidence(provenance: Mapping[str, str]) -> tuple[LibraryEvidence, ...]:
    """Resolve each declared library against this interpreter's installed set."""

    evidence: list[LibraryEvidence] = []
    for distribution, declared in declared_runtime_libraries(provenance):
        try:
            installed: str | None = distribution_version(distribution)
        except PackageNotFoundError:
            installed = None
        evidence.append(
            LibraryEvidence(
                distribution=distribution,
                declared_version=declared,
                installed_version=installed,
            )
        )
    return tuple(evidence)


def simulator_evidence(evidence: tuple[LibraryEvidence, ...]) -> str:
    """Classify a run's simulator presence. Three states, never inferred from names."""

    if not evidence:
        return SIMULATOR_EVIDENCE_NONE_DECLARED
    if all(entry.present for entry in evidence):
        return SIMULATOR_EVIDENCE_INSTALLED
    return SIMULATOR_EVIDENCE_ABSENT
