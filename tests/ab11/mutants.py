"""AB-11 falsification artifacts (mutants), never Faithful Organism Mode components.

Each class below deliberately breaks the causal chain that PLAN.md AB-11 requires the
acceptance suite to defend: sensory adapter -> connectome-derived recurrent dynamics ->
motor pools -> bounded primitive actuation. They exist only so that
``tests/ab11/test_ab11_mutation_power.py`` can show that a named test fails, or a named
production detector refuses, whenever that chain is made decorative or is bypassed.

Claim discipline:

* Every mutant is ``Mutant``-prefixed; the remaining public names are builders for
  mutant artifacts.
* Nothing here is a control identity (shuffled / rewired / no-input), a fallback
  policy, or a presentable run mode. Mutant output is never evidence of behavior.
* The package lives under ``tests/`` and is therefore outside every code-identity
  bundle (``scripts/fixture_harness.py::replay_implementation_sha256`` and
  ``scripts/prepare_gametest_demo.py::_code_bundle``), so committed receipts and
  golden traces keep their meaning.

Injection idioms, both taken from production seams that already exist:

* ``NeuralEngine`` subclasses are rebound onto a freshly created
  ``ConnectomeController`` with identical constructor arguments, so every published
  identity hash (graph, adapter, dynamics, decoder, configuration) is unchanged and
  only the committed state differs.
* ``MutantTelemetryOnlyController`` and ``MutantDirectSteeringController`` satisfy the
  structural ``neurocraft_service.controller.Controller`` protocol, the same seam the
  transport test doubles use, and wrap a real controller so the neural evidence stays
  genuinely computed while the issued motors come from elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_neural import (
    EncodedStimulation,
    NeuralEngine,
    NeuralState,
    StepResult,
    Stimulation,
)
from neurocraft_neural.model import make_neural_state
from neurocraft_protocol import (
    Action,
    CausalRecord,
    MotorVector,
    Observation,
    RunIdentity,
    SensorFrame,
    TraceWriter,
    verify_trace,
)

from scripts.replay_minecraft_trace import iter_records

MUTANT_ENTITY_ID = "ab11-mutant-fly"
SCRIPTED_FORWARD_THRUST = 0.5
SCRIPTED_VERTICAL_THRUST = 0.25


def _committed_state(engine: NeuralEngine, activity: Sequence[float]) -> NeuralState:
    """Re-issue the engine's own state identity over a substituted activity vector."""

    return make_neural_state(
        step_index=engine.step_index,
        graph_identity_sha256=engine.graph.manifest.runtime_identity_sha256,
        dynamics_identity_sha256=engine.config.identity_sha256,
        sign_overlay_sha256=engine.sign_overlay.identity_sha256,
        delay_overlay_sha256=engine.delay_overlay.identity_sha256,
        weight_overlay_sha256=engine.weight_overlay.identity_sha256,
        topology_overlay_sha256=engine.topology_overlay.identity_sha256,
        activity=activity,
    )


def _substituted_result(
    engine: NeuralEngine,
    faithful: StepResult,
    activity: Sequence[float],
) -> StepResult:
    """Return the faithful step's bookkeeping with the substituted state committed."""

    state = _committed_state(engine, activity)
    return StepResult(
        state=state,
        record=replace(faithful.record, state_after_sha256=state.state_sha256),
    )


class MutantZeroPropagationEngine(NeuralEngine):
    """AB-11 mutant: the committed neural state is forced to all zeros every step.

    Falsification target: "real physical input reaches the neural state". The faithful
    arithmetic still runs, so timing, step windows, and hash-chain bookkeeping look
    alive, but nothing that was computed is ever committed.
    """

    def step(
        self,
        encoded: EncodedStimulation | None = None,
        *,
        stimulation: Stimulation | None = None,
    ) -> StepResult:
        faithful = super().step(encoded, stimulation=stimulation)
        zeros = [0.0] * len(self.graph.nodes)
        self._activity = list(zeros)
        self._history = [list(zeros) for _ in self._history]
        return _substituted_result(self, faithful, zeros)


class MutantIdentityPropagationEngine(NeuralEngine):
    """AB-11 mutant: the committed state is the clipped encoded sensory vector.

    Falsification target: "the output depends on the signed recurrence". Recurrence,
    leak, edge signs, and delays are all discarded, so a faithful controller and its
    ``unknown_zero`` sign-ablated counterfactual commit *identical* activity vectors
    and emit *identical* motors — neutral ones, because no motor node is sensory
    encoded. Their committed state digests still differ, since the sign overlay hash is
    part of state identity: the overlay changes the digest without changing anything
    causal, which is exactly what makes the sign-ablation predicate falsifiable here.
    """

    def step(
        self,
        encoded: EncodedStimulation | None = None,
        *,
        stimulation: Stimulation | None = None,
    ) -> StepResult:
        faithful = super().step(encoded, stimulation=stimulation)
        node_count = len(self.graph.nodes)
        source = encoded.vector if encoded is not None else (0.0,) * node_count
        passthrough = [min(1.0, max(-1.0, value)) for value in source]
        self._activity = list(passthrough)
        return _substituted_result(self, faithful, passthrough)


class MutantFrozenPropagationEngine(NeuralEngine):
    """AB-11 mutant: the state runs live, then freezes; only the step counter advances.

    Falsification target: "one physical sensor change alters state and motors". The
    first ``LIVE_NEURAL_STEPS`` steps are committed faithfully, so the state that
    freezes is a real, non-zero connectome state that the sensory input produced. That
    is what makes this mutant its own falsification device rather than a relabelling of
    ``MutantZeroPropagationEngine``: freezing from the all-zero initial condition would
    be observationally identical to zero propagation. After the freeze the
    five-neural-steps-per-tick window and the neural step indices stay exactly right,
    and only the state content is dead.
    """

    # Two Minecraft ticks of faithful drive at five neural steps per tick. The named
    # acceptance test drives two constant prefix ticks and then perturbs one ray, so
    # the freeze lands exactly on the tick whose propagation the predicate examines.
    LIVE_NEURAL_STEPS = 10

    def step(
        self,
        encoded: EncodedStimulation | None = None,
        *,
        stimulation: Stimulation | None = None,
    ) -> StepResult:
        if self.step_index < self.LIVE_NEURAL_STEPS:
            return super().step(encoded, stimulation=stimulation)
        frozen_activity = list(self._activity)
        frozen_history = [list(row) for row in self._history]
        faithful = super().step(encoded, stimulation=stimulation)
        self._activity = list(frozen_activity)
        self._history = frozen_history
        return _substituted_result(self, faithful, frozen_activity)


def make_mutant_engine_controller(
    graph: RuntimeGraph,
    *,
    engine_type: type[NeuralEngine],
    run_id: str,
    entity_id: str = MUTANT_ENTITY_ID,
    configuration: ControllerConfiguration | None = None,
) -> ConnectomeController:
    """Build a faithful controller, then rebind ``.engine`` with identical arguments.

    The rebind reproduces the engine construction in ``ConnectomeController.__init__``
    exactly, so the run identity, dynamics identity, and configuration hash the
    controller already published stay valid. Every caller gets a freshly constructed
    object; no production instance is ever mutated.
    """

    controller = ConnectomeController.create(
        graph,
        run_id=run_id,
        entity_id=entity_id,
        configuration=configuration,
    )
    controller.engine = engine_type(
        graph,
        sign_overlay=controller.sign_overlay,
        config=controller.dynamics,
        delay_overlay=controller.delay_overlay,
        sensory_adapter_sha256=controller.adapter.identity_sha256,
    )
    return controller


class MutantTelemetryOnlyController:
    """AB-11 mutant: genuine neural telemetry, scripted motors (a decorative model display).

    Falsification target: "the issued action is the decoder output of the state the
    evidence reports". The wrapped controller really runs, so ``last_evidence``, the
    state digests, and any HUD-visible activity are authentic; only the motor vector
    handed to Minecraft comes from a constant script. This is the shape a plausible
    cheat takes, and it is what the issued-action hash binding in
    ``MinecraftReceiptTraceRecorder.accept_issued_action`` exists to refuse.
    """

    def __init__(self, inner: ConnectomeController) -> None:
        self.inner = inner

    @property
    def identity(self) -> RunIdentity:
        return self.inner.identity

    def step(self, observation: Observation, *, command_seq: int) -> Action:
        action = self.inner.step(observation, command_seq=command_seq)
        scripted = replace(
            action.motors,
            forward_thrust=SCRIPTED_FORWARD_THRUST,
            vertical_thrust=SCRIPTED_VERTICAL_THRUST,
        )
        return replace(action, motors=scripted)

    def reset(self) -> None:
        self.inner.reset()


class MutantDirectSteeringController:
    """AB-11 mutant: motors come from a direct observation-to-action policy.

    Falsification target: the AB-02 boundary that no policy logic may live outside the
    connectome recurrence. Unlike the telemetry-only mutant, these motors vary with the
    sensor rays, so any check that only asks "does the fly move sensibly?" still
    passes. The load-bearing detectors are the issued-action hash binding and an
    independent re-decode of the evidence state.
    """

    def __init__(self, inner: ConnectomeController) -> None:
        self.inner = inner

    @property
    def identity(self) -> RunIdentity:
        return self.inner.identity

    def step(self, observation: Observation, *, command_seq: int) -> Action:
        action = self.inner.step(observation, command_seq=command_seq)
        return replace(action, motors=self._steer(observation.sensors, action.motors))

    def reset(self) -> None:
        self.inner.reset()

    @staticmethod
    def _steer(sensors: SensorFrame, motors: MotorVector) -> MotorVector:
        limit = sensors.ray_max_distance_m
        forward, _back, left, right = sensors.ray_distances_m[:4]
        clearance = min(1.0, max(0.0, forward / limit))
        turn = min(1.0, max(-1.0, (right - left) / limit))
        return replace(
            motors,
            forward_thrust=clearance,
            vertical_thrust=0.5 * clearance,
            yaw_torque=turn,
        )


def substituted_graph_trace(source: Path, destination: Path, *, graph_hash: str) -> None:
    """Rewrite a golden trace so a substituted runtime graph hash is hash-authentic.

    Manifest artifacts and every record identity (record, observation, issued action,
    actuation receipt) are rewritten together and re-chained through ``TraceWriter``,
    so the result satisfies ``verify_trace`` and the writer's own manifest/record
    artifact binding. Only an independent re-execution can still tell that the trace
    does not belong to the pinned connectome.
    """

    verification = verify_trace(source)
    manifest = verification.manifest
    substituted = replace(
        manifest,
        artifacts=replace(manifest.artifacts, runtime_graph_hash=graph_hash),
    )
    with TraceWriter(destination, substituted) as writer:
        for record in iter_records(source):
            writer.append(_with_graph_hash(record, graph_hash))


def substituted_canonical_graph_trace(
    source: Path,
    destination: Path,
    *,
    canonical_graph_hash: str,
) -> None:
    """Rewrite a golden trace's canonical graph provenance, keeping it hash-authentic.

    ``canonical_graph_hash`` is not cross-checked against record identities, so this
    substitution survives ``TraceWriter`` and ``verify_trace`` untouched and reaches
    the replay environment binding in ``scripts/replay_minecraft_trace.py``.
    """

    verification = verify_trace(source)
    manifest = verification.manifest
    substituted = replace(
        manifest,
        artifacts=replace(manifest.artifacts, canonical_graph_hash=canonical_graph_hash),
    )
    with TraceWriter(destination, substituted) as writer:
        for record in iter_records(source):
            writer.append(record)


def _with_graph_hash(record: CausalRecord, graph_hash: str) -> CausalRecord:
    """Restate one record under a substituted graph hash in a single atomic rewrite.

    Record, observation, issued action, and actuation receipt identities must move
    together: ``CausalRecord`` refuses any partially rewritten intermediate.
    """

    identity = replace(record.identity, graph_hash=graph_hash)
    receipt = record.actuation_receipt
    return replace(
        record,
        identity=identity,
        observation=replace(record.observation, identity=identity),
        issued_action=replace(record.issued_action, identity=identity),
        actuation_receipt=None if receipt is None else replace(receipt, identity=identity),
    )
