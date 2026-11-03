"""Typed configuration, overlays, state, and digest contracts."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import sha256_json

ZERO_SHA256 = "0" * 64
UINT64_MAX = (1 << 64) - 1


class NeuralConfigurationError(ValueError):
    """A neural configuration violates the declared low-capacity contract."""


class NeuralIntegrityError(RuntimeError):
    """State, overlay, snapshot, or trace identity does not verify."""


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeuralConfigurationError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise NeuralConfigurationError(f"{label} must be finite")
    return result


def _float_hex(value: float) -> str:
    return float(value).hex()


def _float_bytes(values: Iterable[float]) -> bytes:
    encoded = bytearray()
    for value in values:
        if not math.isfinite(value):
            raise NeuralIntegrityError("non-finite value cannot be hashed as neural state")
        encoded.extend(struct.pack(">d", value))
    return bytes(encoded)


def _float_tuple_digest(values: Sequence[float]) -> str:
    return sha256(_float_bytes(values)).hexdigest()


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    """Versioned engineering assumptions for the ADR-0002 recurrence."""

    neural_step_ms: float = 10.0
    tau_ms: float = 50.0
    recurrent_gain: float = 0.9
    sensory_gain: float = 1.0
    baseline: float = 0.0
    default_delay_steps: int = 1
    activation: str = "tanh"
    normalization: str = "incoming_raw_count_l1"
    precision: str = "float64"
    backend: str = "python_single_thread_reference"
    noise: str = "none"

    def __post_init__(self) -> None:
        for name in (
            "neural_step_ms",
            "tau_ms",
            "recurrent_gain",
            "sensory_gain",
            "baseline",
        ):
            value = _finite_float(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.neural_step_ms <= 0 or self.tau_ms <= 0:
            raise NeuralConfigurationError("neural_step_ms and tau_ms must be positive")
        if self.recurrent_gain < 0:
            raise NeuralConfigurationError("recurrent_gain must be non-negative")
        if (
            isinstance(self.default_delay_steps, bool)
            or not isinstance(self.default_delay_steps, int)
            or not 1 <= self.default_delay_steps <= 64
        ):
            raise NeuralConfigurationError("default_delay_steps must be in [1, 64]")
        fixed = {
            "activation": "tanh",
            "normalization": "incoming_raw_count_l1",
            "precision": "float64",
            "backend": "python_single_thread_reference",
            "noise": "none",
        }
        for field, expected in fixed.items():
            if getattr(self, field) != expected:
                raise NeuralConfigurationError(f"reference backend requires {field}={expected!r}")

    @property
    def leak(self) -> float:
        return math.exp(-self.neural_step_ms / self.tau_ms)

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "activation": self.activation,
                "backend": self.backend,
                "baseline": _float_hex(self.baseline),
                "default_delay_steps": self.default_delay_steps,
                "neural_step_ms": _float_hex(self.neural_step_ms),
                "noise": self.noise,
                "normalization": self.normalization,
                "precision": self.precision,
                "recurrent_gain": _float_hex(self.recurrent_gain),
                "sensory_gain": _float_hex(self.sensory_gain),
                "tau_ms": _float_hex(self.tau_ms),
            }
        )


def _control_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise NeuralConfigurationError("control seed must be a uint64 integer")
    return value


def _nonempty_provenance(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NeuralConfigurationError(f"{label} needs non-empty named provenance")
    return value


@dataclass(frozen=True, slots=True)
class WeightOverlay:
    """A provenance-bound edge-weight control that never mutates source counts."""

    graph_support_sha256: str
    graph_raw_counts_sha256: str
    kind: str
    values: tuple[int, ...]
    source_edge_indices: tuple[int, ...]
    seed: int | None
    provenance: str
    identity_sha256: str

    @classmethod
    def canonical(cls, graph: RuntimeGraph) -> WeightOverlay:
        graph.assert_integrity()
        return cls(
            graph_support_sha256=graph.edge_support_sha256,
            graph_raw_counts_sha256=graph.raw_counts_sha256,
            kind="canonical_source_raw_counts",
            values=tuple(edge.raw_count for edge in graph.edges),
            source_edge_indices=tuple(range(len(graph.edges))),
            seed=None,
            provenance="canonical_source_raw_counts",
            identity_sha256=ZERO_SHA256,
        )

    @classmethod
    def shuffled(
        cls,
        graph: RuntimeGraph,
        *,
        seed: int,
        provenance: str,
    ) -> WeightOverlay:
        graph.assert_integrity()
        resolved_seed = _control_seed(seed)
        resolved_provenance = _nonempty_provenance(provenance, "shuffled-weight control")
        seed_bytes = struct.pack(">Q", resolved_seed)
        permutation = tuple(
            sorted(
                range(len(graph.edges)),
                key=lambda index: (
                    sha256(seed_bytes + struct.pack(">Q", index)).digest(),
                    index,
                ),
            )
        )
        values = tuple(graph.edges[index].raw_count for index in permutation)
        material = {
            "graph_raw_counts_sha256": graph.raw_counts_sha256,
            "graph_support_sha256": graph.edge_support_sha256,
            "kind": "shuffled_weight",
            "provenance": resolved_provenance,
            "seed": resolved_seed,
            "source_edge_indices": list(permutation),
            "values": list(values),
        }
        return cls(
            graph_support_sha256=graph.edge_support_sha256,
            graph_raw_counts_sha256=graph.raw_counts_sha256,
            kind="shuffled_weight",
            values=values,
            source_edge_indices=permutation,
            seed=resolved_seed,
            provenance=resolved_provenance,
            identity_sha256=sha256_json(material),
        )

    def assert_compatible(self, graph: RuntimeGraph) -> None:
        if (
            self.graph_support_sha256 != graph.edge_support_sha256
            or self.graph_raw_counts_sha256 != graph.raw_counts_sha256
            or len(self.values) != len(graph.edges)
            or len(self.source_edge_indices) != len(graph.edges)
        ):
            raise NeuralIntegrityError("weight overlay is not bound to this source graph")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= UINT64_MAX
            for value in self.values
        ):
            raise NeuralIntegrityError("weight overlay contains an invalid uint64 count")
        canonical = tuple(edge.raw_count for edge in graph.edges)
        if self.kind == "canonical_source_raw_counts":
            if (
                self.values != canonical
                or self.source_edge_indices != tuple(range(len(graph.edges)))
                or self.seed is not None
                or self.provenance != "canonical_source_raw_counts"
                or self.identity_sha256 != ZERO_SHA256
            ):
                raise NeuralIntegrityError("canonical weight overlay identity mismatch")
            return
        if self.kind != "shuffled_weight":
            raise NeuralIntegrityError("unknown weight overlay kind")
        if self.seed is None:
            raise NeuralIntegrityError("shuffled weight overlay is missing its seed")
        try:
            rebuilt = WeightOverlay.shuffled(graph, seed=self.seed, provenance=self.provenance)
        except NeuralConfigurationError as exc:
            raise NeuralIntegrityError(str(exc)) from exc
        if rebuilt != self:
            raise NeuralIntegrityError("weight overlay identity mismatch")


def _degree_counts(values: Sequence[int], node_count: int) -> tuple[int, ...]:
    counts = [0] * node_count
    for value in values:
        counts[value] += 1
    return tuple(counts)


def _topology_material(
    graph: RuntimeGraph,
    *,
    pre_indices: tuple[int, ...],
    post_indices: tuple[int, ...],
    seed: int,
    requested_swaps: int,
    attempted_swaps: int,
    accepted_swaps: int,
    provenance: str,
) -> dict[str, object]:
    return {
        "accepted_swaps": accepted_swaps,
        "attempted_swaps": attempted_swaps,
        "graph_raw_counts_sha256": graph.raw_counts_sha256,
        "graph_support_sha256": graph.edge_support_sha256,
        "kind": "degree_preserving_random",
        "post_indices": list(post_indices),
        "pre_indices": list(pre_indices),
        "provenance": provenance,
        "requested_swaps": requested_swaps,
        "seed": seed,
        "self_edge_policy": "preserve exact source count",
    }


@dataclass(frozen=True, slots=True)
class TopologyOverlay:
    """A directed-degree-preserving endpoint control over immutable source edges."""

    graph_support_sha256: str
    graph_raw_counts_sha256: str
    kind: str
    pre_indices: tuple[int, ...]
    post_indices: tuple[int, ...]
    seed: int | None
    requested_swaps: int
    attempted_swaps: int
    accepted_swaps: int
    provenance: str
    identity_sha256: str

    @classmethod
    def canonical(cls, graph: RuntimeGraph) -> TopologyOverlay:
        graph.assert_integrity()
        return cls(
            graph_support_sha256=graph.edge_support_sha256,
            graph_raw_counts_sha256=graph.raw_counts_sha256,
            kind="canonical_source_topology",
            pre_indices=tuple(edge.pre_index for edge in graph.edges),
            post_indices=tuple(edge.post_index for edge in graph.edges),
            seed=None,
            requested_swaps=0,
            attempted_swaps=0,
            accepted_swaps=0,
            provenance="canonical_source_topology",
            identity_sha256=ZERO_SHA256,
        )

    @classmethod
    def degree_preserving_random(
        cls,
        graph: RuntimeGraph,
        *,
        seed: int,
        requested_swaps: int | None = None,
        provenance: str,
    ) -> TopologyOverlay:
        graph.assert_integrity()
        edge_count = len(graph.edges)
        if edge_count < 2:
            raise NeuralConfigurationError(
                "degree-preserving control requires at least two source edges"
            )
        resolved_seed = _control_seed(seed)
        resolved_provenance = _nonempty_provenance(provenance, "degree-preserving topology control")
        target = edge_count if requested_swaps is None else requested_swaps
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise NeuralConfigurationError("requested swaps must be a positive integer")
        pre = [edge.pre_index for edge in graph.edges]
        post = [edge.post_index for edge in graph.edges]
        pairs = set(zip(pre, post, strict=True))
        if len(pairs) != edge_count:
            raise NeuralIntegrityError("source graph contains duplicate directed pairs")
        original_self_edges = sum(left == right for left, right in pairs)
        seed_bytes = struct.pack(">Q", resolved_seed)
        accepted = 0
        attempted = 0
        max_attempts = target * 100 + 1_000
        while accepted < target and attempted < max_attempts:
            digest = sha256(seed_bytes + struct.pack(">Q", attempted)).digest()
            first = int.from_bytes(digest[:8], "big") % edge_count
            second = int.from_bytes(digest[8:16], "big") % edge_count
            attempted += 1
            if first == second:
                continue
            old_first = (pre[first], post[first])
            old_second = (pre[second], post[second])
            new_first = (old_first[0], old_second[1])
            new_second = (old_second[0], old_first[1])
            if new_first == new_second or {new_first, new_second} == {
                old_first,
                old_second,
            }:
                continue
            if sum(left == right for left, right in (new_first, new_second)) != sum(
                left == right for left, right in (old_first, old_second)
            ):
                continue
            pairs.remove(old_first)
            pairs.remove(old_second)
            if new_first in pairs or new_second in pairs:
                pairs.add(old_first)
                pairs.add(old_second)
                continue
            pre[first], post[first] = new_first
            pre[second], post[second] = new_second
            pairs.add(new_first)
            pairs.add(new_second)
            accepted += 1
        if accepted != target:
            raise NeuralConfigurationError(
                f"degree-preserving control accepted {accepted}/{target} swaps"
            )
        pre_values = tuple(pre)
        post_values = tuple(post)
        if pre_values == tuple(edge.pre_index for edge in graph.edges) and post_values == tuple(
            edge.post_index for edge in graph.edges
        ):
            raise NeuralConfigurationError("degree-preserving control returned source topology")
        if (
            sum(left == right for left, right in zip(pre_values, post_values, strict=True))
            != original_self_edges
        ):
            raise NeuralIntegrityError("degree-preserving control changed self-edge count")
        material = _topology_material(
            graph,
            pre_indices=pre_values,
            post_indices=post_values,
            seed=resolved_seed,
            requested_swaps=target,
            attempted_swaps=attempted,
            accepted_swaps=accepted,
            provenance=resolved_provenance,
        )
        return cls(
            graph_support_sha256=graph.edge_support_sha256,
            graph_raw_counts_sha256=graph.raw_counts_sha256,
            kind="degree_preserving_random",
            pre_indices=pre_values,
            post_indices=post_values,
            seed=resolved_seed,
            requested_swaps=target,
            attempted_swaps=attempted,
            accepted_swaps=accepted,
            provenance=resolved_provenance,
            identity_sha256=sha256_json(material),
        )

    def assert_compatible(self, graph: RuntimeGraph) -> None:
        edge_count = len(graph.edges)
        if (
            self.graph_support_sha256 != graph.edge_support_sha256
            or self.graph_raw_counts_sha256 != graph.raw_counts_sha256
            or len(self.pre_indices) != edge_count
            or len(self.post_indices) != edge_count
        ):
            raise NeuralIntegrityError("topology overlay is not bound to this source graph")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(graph.nodes)
            for index in (*self.pre_indices, *self.post_indices)
        ):
            raise NeuralIntegrityError("topology overlay contains an invalid dense index")
        pairs = tuple(zip(self.pre_indices, self.post_indices, strict=True))
        if len(set(pairs)) != edge_count:
            raise NeuralIntegrityError("topology overlay contains duplicate directed pairs")
        canonical_pre = tuple(edge.pre_index for edge in graph.edges)
        canonical_post = tuple(edge.post_index for edge in graph.edges)
        if _degree_counts(self.pre_indices, len(graph.nodes)) != _degree_counts(
            canonical_pre, len(graph.nodes)
        ) or _degree_counts(self.post_indices, len(graph.nodes)) != _degree_counts(
            canonical_post, len(graph.nodes)
        ):
            raise NeuralIntegrityError("topology overlay does not preserve directed degrees")
        if sum(left == right for left, right in pairs) != sum(
            left == right for left, right in zip(canonical_pre, canonical_post, strict=True)
        ):
            raise NeuralIntegrityError("topology overlay does not preserve self-edge count")
        if self.kind == "canonical_source_topology":
            if (
                self.pre_indices != canonical_pre
                or self.post_indices != canonical_post
                or self.seed is not None
                or any((self.requested_swaps, self.attempted_swaps, self.accepted_swaps))
                or self.provenance != "canonical_source_topology"
                or self.identity_sha256 != ZERO_SHA256
            ):
                raise NeuralIntegrityError("canonical topology overlay identity mismatch")
            return
        if self.kind != "degree_preserving_random" or self.seed is None:
            raise NeuralIntegrityError("unknown or incomplete topology overlay kind")
        try:
            _control_seed(self.seed)
            _nonempty_provenance(self.provenance, "degree-preserving topology control")
        except NeuralConfigurationError as exc:
            raise NeuralIntegrityError(str(exc)) from exc
        if (
            self.requested_swaps <= 0
            or self.accepted_swaps != self.requested_swaps
            or self.attempted_swaps < self.accepted_swaps
            or (self.pre_indices == canonical_pre and self.post_indices == canonical_post)
        ):
            raise NeuralIntegrityError("topology overlay swap receipt is invalid")
        material = _topology_material(
            graph,
            pre_indices=self.pre_indices,
            post_indices=self.post_indices,
            seed=self.seed,
            requested_swaps=self.requested_swaps,
            attempted_swaps=self.attempted_swaps,
            accepted_swaps=self.accepted_swaps,
            provenance=self.provenance,
        )
        if sha256_json(material) != self.identity_sha256:
            raise NeuralIntegrityError("topology overlay identity mismatch")


@dataclass(frozen=True, slots=True)
class SignOverlay:
    """Assigned signs separated from immutable source counts and unknown status."""

    graph_support_sha256: str
    graph_raw_counts_sha256: str
    values: tuple[int, ...]
    provenance: tuple[str, ...]
    identity_sha256: str

    @classmethod
    def explicit(
        cls,
        graph: RuntimeGraph,
        values: Sequence[int],
        provenance: Sequence[str],
    ) -> SignOverlay:
        graph.assert_integrity()
        signs = tuple(values)
        origins = tuple(provenance)
        if len(signs) != len(graph.edges) or len(origins) != len(graph.edges):
            raise NeuralConfigurationError("sign overlay must contain one value per edge")
        if any(isinstance(value, bool) or value not in {-1, 0, 1} for value in signs):
            raise NeuralConfigurationError("edge signs must be exactly -1, 0, or +1")
        if any(not isinstance(item, str) or not item for item in origins):
            raise NeuralConfigurationError("every assigned sign needs named provenance")
        material = {
            "canonical_sign_status": [edge.canonical_sign_status for edge in graph.edges],
            "graph_raw_counts_sha256": graph.raw_counts_sha256,
            "graph_support_sha256": graph.edge_support_sha256,
            "provenance": list(origins),
            "values": list(signs),
        }
        return cls(
            graph_support_sha256=graph.edge_support_sha256,
            graph_raw_counts_sha256=graph.raw_counts_sha256,
            values=signs,
            provenance=origins,
            identity_sha256=sha256_json(material),
        )

    @classmethod
    def unknown_zero(cls, graph: RuntimeGraph) -> SignOverlay:
        return cls.explicit(
            graph,
            [0] * len(graph.edges),
            ["canonical_unknown_preserved_as_zero"] * len(graph.edges),
        )

    @classmethod
    def coarse_presynaptic_consensus(cls, graph: RuntimeGraph) -> SignOverlay:
        """Apply the disclosed coarse policy without changing canonical status."""

        positive = {
            "acetylcholine",
            "dopamine",
            "octopamine",
            "serotonin",
            "tyramine",
        }
        negative = {"gaba", "glutamate", "histamine"}
        values: list[int] = []
        provenance: list[str] = []
        for edge in graph.edges:
            transmitter = graph.nodes[edge.pre_index].consensus_nt
            normalized = transmitter.casefold() if transmitter else "unknown"
            if normalized in positive:
                sign = 1
            elif normalized in negative:
                sign = -1
            else:
                sign = 0
            values.append(sign)
            provenance.append(f"inferred_from_presynaptic_consensus_nt:{normalized}")
        return cls.explicit(graph, values, provenance)

    def assert_compatible(self, graph: RuntimeGraph) -> None:
        if (
            self.graph_support_sha256 != graph.edge_support_sha256
            or self.graph_raw_counts_sha256 != graph.raw_counts_sha256
            or len(self.values) != len(graph.edges)
        ):
            raise NeuralIntegrityError("sign overlay is not bound to this source graph")
        rebuilt = SignOverlay.explicit(graph, self.values, self.provenance)
        if rebuilt.identity_sha256 != self.identity_sha256:
            raise NeuralIntegrityError("sign overlay identity mismatch")


@dataclass(frozen=True, slots=True)
class DelayOverlay:
    """Integer numerical delays separated from source topology."""

    graph_support_sha256: str
    graph_raw_counts_sha256: str
    steps: tuple[int, ...]
    provenance: str
    identity_sha256: str

    @classmethod
    def explicit(
        cls,
        graph: RuntimeGraph,
        steps: Sequence[int],
        *,
        provenance: str,
    ) -> DelayOverlay:
        graph.assert_integrity()
        values = tuple(steps)
        if len(values) != len(graph.edges):
            raise NeuralConfigurationError("delay overlay must contain one delay per edge")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64
            for value in values
        ):
            raise NeuralConfigurationError("edge delays must be integer steps in [1, 64]")
        if not isinstance(provenance, str) or not provenance:
            raise NeuralConfigurationError("delay overlay needs named provenance")
        material = {
            "graph_raw_counts_sha256": graph.raw_counts_sha256,
            "graph_support_sha256": graph.edge_support_sha256,
            "provenance": provenance,
            "steps": list(values),
        }
        return cls(
            graph_support_sha256=graph.edge_support_sha256,
            graph_raw_counts_sha256=graph.raw_counts_sha256,
            steps=values,
            provenance=provenance,
            identity_sha256=sha256_json(material),
        )

    @classmethod
    def uniform(
        cls, graph: RuntimeGraph, steps: int = 1, *, provenance: str = "assigned_uniform"
    ) -> DelayOverlay:
        return cls.explicit(graph, [steps] * len(graph.edges), provenance=provenance)

    def assert_compatible(self, graph: RuntimeGraph) -> None:
        if (
            self.graph_support_sha256 != graph.edge_support_sha256
            or self.graph_raw_counts_sha256 != graph.raw_counts_sha256
            or len(self.steps) != len(graph.edges)
        ):
            raise NeuralIntegrityError("delay overlay is not bound to this source graph")
        rebuilt = DelayOverlay.explicit(graph, self.steps, provenance=self.provenance)
        if rebuilt.identity_sha256 != self.identity_sha256:
            raise NeuralIntegrityError("delay overlay identity mismatch")


@dataclass(frozen=True, slots=True)
class Stimulation:
    """A named sparse direct intervention p[n], never a motor command."""

    intervention_id: str
    graph_identity_sha256: str
    assignments: tuple[tuple[int, float], ...]
    identity_sha256: str

    @classmethod
    def from_body_values(
        cls,
        graph: RuntimeGraph,
        intervention_id: str,
        values: Mapping[int, float],
    ) -> Stimulation:
        if not isinstance(intervention_id, str) or not intervention_id:
            raise NeuralConfigurationError("stimulation needs a non-empty intervention ID")
        assignments: list[tuple[int, float]] = []
        for body_id, raw_value in values.items():
            try:
                index = graph.index_of(body_id)
            except KeyError as exc:
                raise NeuralConfigurationError(str(exc)) from exc
            value = _finite_float(raw_value, f"stimulation[{body_id}]")
            assignments.append((index, value))
        assignments.sort()
        if len({index for index, _ in assignments}) != len(assignments):
            raise NeuralConfigurationError("duplicate stimulation target")
        material = {
            "assignments": [[index, _float_hex(value)] for index, value in assignments],
            "graph_identity_sha256": graph.manifest.runtime_identity_sha256,
            "intervention_id": intervention_id,
        }
        return cls(
            intervention_id=intervention_id,
            graph_identity_sha256=graph.manifest.runtime_identity_sha256,
            assignments=tuple(assignments),
            identity_sha256=sha256_json(material),
        )

    def assert_compatible(self, graph: RuntimeGraph) -> None:
        if self.graph_identity_sha256 != graph.manifest.runtime_identity_sha256:
            raise NeuralIntegrityError("stimulation is not bound to this graph")
        material = {
            "assignments": [[index, _float_hex(value)] for index, value in self.assignments],
            "graph_identity_sha256": self.graph_identity_sha256,
            "intervention_id": self.intervention_id,
        }
        if sha256_json(material) != self.identity_sha256:
            raise NeuralIntegrityError("stimulation identity mismatch")
        for index, value in self.assignments:
            if not 0 <= index < len(graph.nodes) or not math.isfinite(value):
                raise NeuralIntegrityError("stimulation assignment is invalid")


@dataclass(frozen=True, slots=True)
class NeuralState:
    """Immutable dimensionless model activity at one committed neural index."""

    step_index: int
    graph_identity_sha256: str
    dynamics_identity_sha256: str
    sign_overlay_sha256: str
    delay_overlay_sha256: str
    weight_overlay_sha256: str
    topology_overlay_sha256: str
    activity: tuple[float, ...]
    activity_sha256: str
    state_sha256: str

    def assert_integrity(self) -> None:
        if self.step_index < 0:
            raise NeuralIntegrityError("state step index is negative")
        if any(not -1.0 <= value <= 1.0 for value in self.activity):
            raise NeuralIntegrityError("state activity lies outside [-1, 1]")
        observed_activity = _float_tuple_digest(self.activity)
        if observed_activity != self.activity_sha256:
            raise NeuralIntegrityError("state activity digest mismatch")
        observed_state = _state_digest(
            step_index=self.step_index,
            graph_identity_sha256=self.graph_identity_sha256,
            dynamics_identity_sha256=self.dynamics_identity_sha256,
            sign_overlay_sha256=self.sign_overlay_sha256,
            delay_overlay_sha256=self.delay_overlay_sha256,
            weight_overlay_sha256=self.weight_overlay_sha256,
            topology_overlay_sha256=self.topology_overlay_sha256,
            activity_sha256=self.activity_sha256,
        )
        if observed_state != self.state_sha256:
            raise NeuralIntegrityError("state identity mismatch")


def make_neural_state(
    *,
    step_index: int,
    graph_identity_sha256: str,
    dynamics_identity_sha256: str,
    sign_overlay_sha256: str,
    delay_overlay_sha256: str,
    weight_overlay_sha256: str,
    topology_overlay_sha256: str,
    activity: Sequence[float],
) -> NeuralState:
    values = tuple(float(value) for value in activity)
    activity_sha256 = _float_tuple_digest(values)
    state_sha256 = _state_digest(
        step_index=step_index,
        graph_identity_sha256=graph_identity_sha256,
        dynamics_identity_sha256=dynamics_identity_sha256,
        sign_overlay_sha256=sign_overlay_sha256,
        delay_overlay_sha256=delay_overlay_sha256,
        weight_overlay_sha256=weight_overlay_sha256,
        topology_overlay_sha256=topology_overlay_sha256,
        activity_sha256=activity_sha256,
    )
    return NeuralState(
        step_index=step_index,
        graph_identity_sha256=graph_identity_sha256,
        dynamics_identity_sha256=dynamics_identity_sha256,
        sign_overlay_sha256=sign_overlay_sha256,
        delay_overlay_sha256=delay_overlay_sha256,
        weight_overlay_sha256=weight_overlay_sha256,
        topology_overlay_sha256=topology_overlay_sha256,
        activity=values,
        activity_sha256=activity_sha256,
        state_sha256=state_sha256,
    )


def _state_digest(
    *,
    step_index: int,
    graph_identity_sha256: str,
    dynamics_identity_sha256: str,
    sign_overlay_sha256: str,
    delay_overlay_sha256: str,
    weight_overlay_sha256: str,
    topology_overlay_sha256: str,
    activity_sha256: str,
) -> str:
    material = {
        "activity_sha256": activity_sha256,
        "delay_overlay_sha256": delay_overlay_sha256,
        "dynamics_identity_sha256": dynamics_identity_sha256,
        "graph_identity_sha256": graph_identity_sha256,
        "sign_overlay_sha256": sign_overlay_sha256,
        "step_index": step_index,
    }
    if weight_overlay_sha256 != ZERO_SHA256:
        material["weight_overlay_sha256"] = weight_overlay_sha256
    if topology_overlay_sha256 != ZERO_SHA256:
        material["topology_overlay_sha256"] = topology_overlay_sha256
    return sha256_json(material)


@dataclass(frozen=True, slots=True)
class NeuralStepRecord:
    """Hash-linked neural interval record for the larger causal trace."""

    step_before: int
    step_after: int
    state_before_sha256: str
    state_after_sha256: str
    delay_before_sha256: str
    delay_after_sha256: str
    encoded_stimulation_sha256: str
    direct_stimulation_sha256: str
    ablation_mask_sha256: str
    execution_sha256: str
    previous_trace_sha256: str
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class StepResult:
    state: NeuralState
    record: NeuralStepRecord


@dataclass(frozen=True, slots=True)
class NeuralSnapshot:
    """Complete mutable reference-backend state for exact restore."""

    graph_identity_sha256: str
    dynamics_identity_sha256: str
    sign_overlay_sha256: str
    delay_overlay_sha256: str
    weight_overlay_sha256: str
    topology_overlay_sha256: str
    sensory_adapter_sha256: str | None
    step_index: int
    activity: tuple[float, ...]
    delay_history: tuple[tuple[float, ...], ...]
    ablation_mask: tuple[bool, ...]
    trace_sha256: str
    snapshot_sha256: str


def neural_snapshot_digest(snapshot: NeuralSnapshot) -> str:
    material = {
        "ablation_mask": list(snapshot.ablation_mask),
        "activity_sha256": _float_tuple_digest(snapshot.activity),
        "delay_history_sha256": [_float_tuple_digest(row) for row in snapshot.delay_history],
        "delay_overlay_sha256": snapshot.delay_overlay_sha256,
        "dynamics_identity_sha256": snapshot.dynamics_identity_sha256,
        "graph_identity_sha256": snapshot.graph_identity_sha256,
        "sensory_adapter_sha256": snapshot.sensory_adapter_sha256,
        "sign_overlay_sha256": snapshot.sign_overlay_sha256,
        "step_index": snapshot.step_index,
        "trace_sha256": snapshot.trace_sha256,
    }
    if snapshot.weight_overlay_sha256 != ZERO_SHA256:
        material["weight_overlay_sha256"] = snapshot.weight_overlay_sha256
    if snapshot.topology_overlay_sha256 != ZERO_SHA256:
        material["topology_overlay_sha256"] = snapshot.topology_overlay_sha256
    return sha256_json(material)


def delay_history_digest(history: Sequence[Sequence[float]]) -> str:
    return sha256_json([_float_tuple_digest(row) for row in history])


def ablation_mask_digest(mask: Sequence[bool]) -> str:
    return sha256(bytes(1 if item else 0 for item in mask)).hexdigest()
