"""Single-thread float64 sparse reference engine for ADR 0002."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256

from neurocraft_connectome import RuntimeGraph
from neurocraft_connectome.model import sha256_json

from neurocraft_neural.adapters import EncodedStimulation
from neurocraft_neural.model import (
    ZERO_SHA256,
    DelayOverlay,
    DynamicsConfig,
    NeuralConfigurationError,
    NeuralIntegrityError,
    NeuralSnapshot,
    NeuralState,
    NeuralStepRecord,
    SignOverlay,
    StepResult,
    Stimulation,
    TopologyOverlay,
    WeightOverlay,
    _finite_float,
    ablation_mask_digest,
    delay_history_digest,
    make_neural_state,
    neural_snapshot_digest,
)


class NeuralEngine:
    """Literal synchronous implementation of the selected recurrence.

    At committed index ``n``, an edge with delay ``d`` reads ``a[n-d]``.
    The current state is never updated in place; all targets commit together.
    """

    def __init__(
        self,
        graph: RuntimeGraph,
        *,
        sign_overlay: SignOverlay,
        config: DynamicsConfig | None = None,
        delay_overlay: DelayOverlay | None = None,
        weight_overlay: WeightOverlay | None = None,
        topology_overlay: TopologyOverlay | None = None,
        sensory_adapter_sha256: str | None = None,
    ) -> None:
        graph.assert_integrity()
        sign_overlay.assert_compatible(graph)
        resolved_config = config or DynamicsConfig()
        resolved_delay = delay_overlay or DelayOverlay.uniform(
            graph,
            resolved_config.default_delay_steps,
            provenance="dynamics_config_uniform_default",
        )
        resolved_delay.assert_compatible(graph)
        resolved_weight = weight_overlay or WeightOverlay.canonical(graph)
        resolved_weight.assert_compatible(graph)
        resolved_topology = topology_overlay or TopologyOverlay.canonical(graph)
        resolved_topology.assert_compatible(graph)
        if sensory_adapter_sha256 is not None and (
            not isinstance(sensory_adapter_sha256, str) or len(sensory_adapter_sha256) != 64
        ):
            raise NeuralConfigurationError("sensory adapter identity must be SHA-256 hex")

        self.graph = graph
        self.sign_overlay = sign_overlay
        self.delay_overlay = resolved_delay
        self.weight_overlay = resolved_weight
        self.topology_overlay = resolved_topology
        self.config = resolved_config
        self.sensory_adapter_sha256 = sensory_adapter_sha256
        self._node_count = len(graph.nodes)
        self._max_delay = max(resolved_delay.steps, default=resolved_config.default_delay_steps)

        incoming_denominator = [0] * self._node_count
        incoming_edges: list[list[int]] = [[] for _ in range(self._node_count)]
        for edge_index, (post_index, raw_count) in enumerate(
            zip(
                resolved_topology.post_indices,
                resolved_weight.values,
                strict=True,
            )
        ):
            incoming_denominator[post_index] += raw_count
            incoming_edges[post_index].append(edge_index)
        self._incoming_denominator = tuple(incoming_denominator)
        self._incoming_edges = tuple(tuple(row) for row in incoming_edges)
        self._effective_weights = tuple(
            sign_overlay.values[edge_index]
            * resolved_weight.values[edge_index]
            / max(1, incoming_denominator[resolved_topology.post_indices[edge_index]])
            for edge_index in range(len(graph.edges))
        )
        self._activity = [0.0] * self._node_count
        self._history = [[0.0] * self._node_count for _ in range(self._max_delay)]
        self._ablated = [False] * self._node_count
        self._step_index = 0
        self._trace_sha256 = ZERO_SHA256

    @property
    def effective_weights(self) -> tuple[float, ...]:
        return self._effective_weights

    @property
    def incoming_denominators(self) -> tuple[int, ...]:
        return self._incoming_denominator

    @property
    def control_identity_sha256(self) -> str:
        return sha256_json(
            {
                "topology_overlay_sha256": self.topology_overlay.identity_sha256,
                "weight_overlay_sha256": self.weight_overlay.identity_sha256,
            }
        )

    @property
    def trace_sha256(self) -> str:
        return self._trace_sha256

    @property
    def step_index(self) -> int:
        return self._step_index

    def current_state(self) -> NeuralState:
        state = make_neural_state(
            step_index=self._step_index,
            graph_identity_sha256=self.graph.manifest.runtime_identity_sha256,
            dynamics_identity_sha256=self.config.identity_sha256,
            sign_overlay_sha256=self.sign_overlay.identity_sha256,
            delay_overlay_sha256=self.delay_overlay.identity_sha256,
            weight_overlay_sha256=self.weight_overlay.identity_sha256,
            topology_overlay_sha256=self.topology_overlay.identity_sha256,
            activity=self._activity,
        )
        state.assert_integrity()
        return state

    def reset(
        self,
        activity: Mapping[int, float] | None = None,
        *,
        prehistory: Sequence[Mapping[int, float]] | None = None,
    ) -> NeuralState:
        """Reset to index zero; unspecified current/history values are zero."""

        current = self._compile_body_values(activity or {}, "reset activity")
        if any(abs(value) > 1.0 for value in current):
            raise NeuralConfigurationError("initial activity must lie in [-1, 1]")
        if prehistory is None:
            history = [[0.0] * self._node_count for _ in range(self._max_delay)]
        else:
            if len(prehistory) != self._max_delay:
                raise NeuralConfigurationError(
                    f"prehistory must provide exactly {self._max_delay} age rows"
                )
            history = [
                self._compile_body_values(row, f"prehistory age {age + 1}")
                for age, row in enumerate(prehistory)
            ]
            if any(abs(value) > 1.0 for row in history for value in row):
                raise NeuralConfigurationError("prehistory activity must lie in [-1, 1]")
        self._activity = current
        self._history = history
        self._ablated = [False] * self._node_count
        self._step_index = 0
        self._trace_sha256 = ZERO_SHA256
        return self.current_state()

    def _compile_body_values(self, values: Mapping[int, float], label: str) -> list[float]:
        if not isinstance(values, Mapping):
            raise NeuralConfigurationError(f"{label} must be a body-ID mapping")
        result = [0.0] * self._node_count
        for body_id, raw in values.items():
            try:
                index = self.graph.index_of(body_id)
            except KeyError as exc:
                raise NeuralConfigurationError(str(exc)) from exc
            result[index] = _finite_float(raw, f"{label}[{body_id}]")
        return result

    def set_ablated(self, body_ids: Sequence[int], *, ablated: bool = True) -> None:
        """Apply a persistent lesion and immediately clear its state/history."""

        if not isinstance(ablated, bool):
            raise NeuralConfigurationError("ablated flag must be boolean")
        indices: list[int] = []
        for body_id in body_ids:
            try:
                indices.append(self.graph.index_of(body_id))
            except KeyError as exc:
                raise NeuralConfigurationError(str(exc)) from exc
        for index in indices:
            self._ablated[index] = ablated
            if ablated:
                self._activity[index] = 0.0
                for age in self._history:
                    age[index] = 0.0

    def step(
        self,
        encoded: EncodedStimulation | None = None,
        *,
        stimulation: Stimulation | None = None,
    ) -> StepResult:
        """Advance one fixed neural interval atomically."""

        self.graph.assert_integrity()
        self.sign_overlay.assert_compatible(self.graph)
        self.delay_overlay.assert_compatible(self.graph)
        self.weight_overlay.assert_compatible(self.graph)
        self.topology_overlay.assert_compatible(self.graph)
        before = self.current_state()
        delay_before_sha256 = delay_history_digest(self._history)
        sensory = self._validate_encoded(encoded)
        direct = self._validate_stimulation(stimulation)
        sensory_sha256 = encoded.identity_sha256 if encoded else ZERO_SHA256
        direct_sha256 = stimulation.identity_sha256 if stimulation else ZERO_SHA256

        recurrent = [0.0] * self._node_count
        for post_index, edge_indices in enumerate(self._incoming_edges):
            contributions: list[float] = []
            for edge_index in edge_indices:
                pre_index = self.topology_overlay.pre_indices[edge_index]
                if self._ablated[pre_index]:
                    continue
                delay = self.delay_overlay.steps[edge_index]
                source_value = self._history[delay - 1][pre_index]
                contributions.append(self._effective_weights[edge_index] * source_value)
            recurrent[post_index] = math.fsum(contributions)

        leak = self.config.leak
        drive_scale = 1.0 - leak
        next_activity: list[float] = []
        for index in range(self._node_count):
            if self._ablated[index]:
                next_activity.append(0.0)
                continue
            z = (
                self.config.recurrent_gain * recurrent[index]
                + self.config.sensory_gain * sensory[index]
                + direct[index]
                + self.config.baseline
            )
            if not math.isfinite(z):
                raise NeuralIntegrityError(
                    f"non-finite drive proposed for body {self.graph.nodes[index].body_id}"
                )
            value = leak * self._activity[index] + drive_scale * math.tanh(z)
            if not math.isfinite(value):
                raise NeuralIntegrityError(
                    f"non-finite state proposed for body {self.graph.nodes[index].body_id}"
                )
            if value < -1.0 or value > 1.0:
                raise NeuralIntegrityError("bounded recurrence produced out-of-range state")
            next_activity.append(value)

        next_history = [self._activity.copy(), *[row.copy() for row in self._history[:-1]]]
        next_step = self._step_index + 1
        after = make_neural_state(
            step_index=next_step,
            graph_identity_sha256=self.graph.manifest.runtime_identity_sha256,
            dynamics_identity_sha256=self.config.identity_sha256,
            sign_overlay_sha256=self.sign_overlay.identity_sha256,
            delay_overlay_sha256=self.delay_overlay.identity_sha256,
            weight_overlay_sha256=self.weight_overlay.identity_sha256,
            topology_overlay_sha256=self.topology_overlay.identity_sha256,
            activity=next_activity,
        )
        delay_after_sha256 = delay_history_digest(next_history)
        mask_sha256 = ablation_mask_digest(self._ablated)
        execution_sha256 = sha256_json(
            {
                "ablation_mask_sha256": mask_sha256,
                "delay_after_sha256": delay_after_sha256,
                "delay_before_sha256": delay_before_sha256,
                "direct_stimulation_sha256": direct_sha256,
                "encoded_stimulation_sha256": sensory_sha256,
                "state_after_sha256": after.state_sha256,
                "state_before_sha256": before.state_sha256,
                "step_after": next_step,
                "step_before": self._step_index,
            }
        )
        trace_sha256 = sha256(
            bytes.fromhex(self._trace_sha256) + bytes.fromhex(execution_sha256)
        ).hexdigest()
        record = NeuralStepRecord(
            step_before=self._step_index,
            step_after=next_step,
            state_before_sha256=before.state_sha256,
            state_after_sha256=after.state_sha256,
            delay_before_sha256=delay_before_sha256,
            delay_after_sha256=delay_after_sha256,
            encoded_stimulation_sha256=sensory_sha256,
            direct_stimulation_sha256=direct_sha256,
            ablation_mask_sha256=mask_sha256,
            execution_sha256=execution_sha256,
            previous_trace_sha256=self._trace_sha256,
            trace_sha256=trace_sha256,
        )
        self._activity = next_activity
        self._history = next_history
        self._step_index = next_step
        self._trace_sha256 = trace_sha256
        return StepResult(state=after, record=record)

    def _validate_encoded(self, encoded: EncodedStimulation | None) -> tuple[float, ...]:
        if encoded is None:
            return (0.0,) * self._node_count
        if not isinstance(encoded, EncodedStimulation):
            raise NeuralConfigurationError(
                "neural engine accepts EncodedStimulation, never raw observations"
            )
        encoded.assert_integrity()
        if self.sensory_adapter_sha256 is None:
            raise NeuralIntegrityError("engine has no bound sensory adapter")
        if encoded.adapter_sha256 != self.sensory_adapter_sha256:
            raise NeuralIntegrityError("encoded stimulation used an unbound adapter")
        if encoded.graph_identity_sha256 != self.graph.manifest.runtime_identity_sha256:
            raise NeuralIntegrityError("encoded stimulation used a different graph")
        if len(encoded.vector) != self._node_count:
            raise NeuralIntegrityError("encoded stimulation vector length mismatch")
        return encoded.vector

    def _validate_stimulation(self, stimulation: Stimulation | None) -> tuple[float, ...]:
        result = [0.0] * self._node_count
        if stimulation is None:
            return tuple(result)
        if not isinstance(stimulation, Stimulation):
            raise NeuralConfigurationError("direct intervention must be Stimulation")
        stimulation.assert_compatible(self.graph)
        for index, value in stimulation.assignments:
            result[index] += value
        return tuple(result)

    def snapshot(self) -> NeuralSnapshot:
        provisional = NeuralSnapshot(
            graph_identity_sha256=self.graph.manifest.runtime_identity_sha256,
            dynamics_identity_sha256=self.config.identity_sha256,
            sign_overlay_sha256=self.sign_overlay.identity_sha256,
            delay_overlay_sha256=self.delay_overlay.identity_sha256,
            weight_overlay_sha256=self.weight_overlay.identity_sha256,
            topology_overlay_sha256=self.topology_overlay.identity_sha256,
            sensory_adapter_sha256=self.sensory_adapter_sha256,
            step_index=self._step_index,
            activity=tuple(self._activity),
            delay_history=tuple(tuple(row) for row in self._history),
            ablation_mask=tuple(self._ablated),
            trace_sha256=self._trace_sha256,
            snapshot_sha256="",
        )
        return replace(provisional, snapshot_sha256=neural_snapshot_digest(provisional))

    def restore(self, snapshot: NeuralSnapshot) -> NeuralState:
        if not isinstance(snapshot, NeuralSnapshot):
            raise NeuralConfigurationError("restore requires NeuralSnapshot")
        identities = {
            "graph": (
                snapshot.graph_identity_sha256,
                self.graph.manifest.runtime_identity_sha256,
            ),
            "dynamics": (snapshot.dynamics_identity_sha256, self.config.identity_sha256),
            "sign": (snapshot.sign_overlay_sha256, self.sign_overlay.identity_sha256),
            "delay": (snapshot.delay_overlay_sha256, self.delay_overlay.identity_sha256),
            "weight": (
                snapshot.weight_overlay_sha256,
                self.weight_overlay.identity_sha256,
            ),
            "topology": (
                snapshot.topology_overlay_sha256,
                self.topology_overlay.identity_sha256,
            ),
            "adapter": (snapshot.sensory_adapter_sha256, self.sensory_adapter_sha256),
        }
        mismatches = [name for name, pair in identities.items() if pair[0] != pair[1]]
        if mismatches:
            raise NeuralIntegrityError("snapshot identity mismatch: " + ", ".join(mismatches))
        if neural_snapshot_digest(snapshot) != snapshot.snapshot_sha256:
            raise NeuralIntegrityError("snapshot content digest mismatch")
        if snapshot.step_index < 0:
            raise NeuralIntegrityError("snapshot step index is negative")
        if len(snapshot.activity) != self._node_count:
            raise NeuralIntegrityError("snapshot activity length mismatch")
        if len(snapshot.delay_history) != self._max_delay or any(
            len(row) != self._node_count for row in snapshot.delay_history
        ):
            raise NeuralIntegrityError("snapshot delay history shape mismatch")
        if len(snapshot.ablation_mask) != self._node_count:
            raise NeuralIntegrityError("snapshot ablation mask length mismatch")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in snapshot.activity):
            raise NeuralIntegrityError("snapshot current activity is invalid")
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for row in snapshot.delay_history
            for value in row
        ):
            raise NeuralIntegrityError("snapshot delay history is invalid")
        if len(snapshot.trace_sha256) != 64:
            raise NeuralIntegrityError("snapshot trace identity is not SHA-256")
        try:
            bytes.fromhex(snapshot.trace_sha256)
        except ValueError as exc:
            raise NeuralIntegrityError("snapshot trace identity is not hexadecimal") from exc
        self._activity = list(snapshot.activity)
        self._history = [list(row) for row in snapshot.delay_history]
        self._ablated = list(snapshot.ablation_mask)
        self._step_index = snapshot.step_index
        self._trace_sha256 = snapshot.trace_sha256
        return self.current_state()
