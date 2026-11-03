from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import (
    BoundaryIdentity,
    ChannelDeclaration,
    EmbodimentError,
    EmbodimentIdentityError,
    EmbodimentTransition,
    FlyGymBodyConfig,
    ReferenceArena,
    ReferenceEmbodiment,
    build_embodiment_manifest,
    embodiment_implementation_sha256,
    replay_embodiment_trace,
    run_embodiment,
)
from neurocraft_embodiment.cli import main
from neurocraft_embodiment.loop import (
    STATUS_NEURAL_AND_BODY_EXACT,
    STATUS_NEURAL_ONLY,
    embodiment_environment_label,
    replay_embodiment_trace_neural_only,
)
from neurocraft_embodiment.seam import CONTROL_PERIOD_NS, observation_timing
from neurocraft_protocol import (
    ArtifactIdentities,
    ClaimScope,
    DataTier,
    Observation,
    RunMode,
    verify_trace,
)

from .conftest import (
    FIXTURE,
    PACKAGE_SRC,
    TICKS,
    kinematic_flygym_body,
    make_controller,
    record_reference_trace,
    replay_reference_trace,
)
from .test_identity import make_identity

FLYGYM_INSTALLED = importlib.util.find_spec("flygym") is not None


def test_reference_run_produces_a_trace_the_existing_verifier_accepts(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """The headline reuse claim: no new trace schema, no new verifier."""

    output = tmp_path / "reference-embodiment.ndjson"
    receipt = record_reference_trace(graph, output)
    assert receipt["record_count"] == TICKS
    assert receipt["schema"] == "neurocraft.embodiment.receipt.v2"
    assert receipt["embodiment_id"] == "reference-arena-pointmass"
    assert receipt["motor_currency"] == "primitive_motor_vector"
    assert receipt["body_reset_before_first_observation"] is True
    assert receipt["simulator_evidence"] == "no_declared_simulator_libraries"
    assert receipt["declared_libraries"] == []
    assert receipt["provenance"]["integrator"] == "semi_implicit_euler_fixed_substep"

    verified = verify_trace(output)
    assert verified.record_count == TICKS
    assert verified.first_tick == 0
    assert verified.last_tick == TICKS - 1
    assert verified.file_sha256 == receipt["file_sha256"]
    manifest = verified.manifest
    assert manifest.mode is RunMode.FAITHFUL
    assert manifest.claim_scope is ClaimScope.G2
    assert manifest.data_tier is DataTier.MALECNS_FIXTURE
    assert manifest.game_tick_hz == 20.0
    # All four Minecraft-runtime strings keep the ``not_used`` marker that
    # ``scripts/replay_minecraft_trace.py`` refuses a trace on; ``mod`` carries the
    # body's runtime environment after it.
    for label in (manifest.java, manifest.minecraft, manifest.fabric, manifest.mod):
        assert label.startswith("not_used_in_")
    assert "simulator_evidence=no_declared_simulator_libraries" in manifest.mod


def test_manifest_arena_hash_is_the_embodiment_identity(graph: RuntimeGraph) -> None:
    embodiment = ReferenceEmbodiment(ReferenceArena())
    manifest = build_embodiment_manifest(make_controller(graph), graph, embodiment)
    assert manifest.artifacts.arena_hash == embodiment.identity.identity_sha256
    assert manifest.seed == embodiment.identity.seed


def test_a_different_body_yields_a_different_arena_hash(graph: RuntimeGraph) -> None:
    """A trace can never be ambiguous about which body produced it."""

    controller = make_controller(graph)
    first = build_embodiment_manifest(
        controller, graph, ReferenceEmbodiment(ReferenceArena(seed=0))
    )
    second = build_embodiment_manifest(
        controller, graph, ReferenceEmbodiment(ReferenceArena(seed=3))
    )
    assert first.artifacts.arena_hash != second.artifacts.arena_hash
    # Everything upstream of the body is byte-identical: same graph, same adapter,
    # same dynamics, same decoder. Only the world changed.
    assert first.artifacts.runtime_graph_hash == second.artifacts.runtime_graph_hash
    assert first.artifacts.adapter_hash == second.artifacts.adapter_hash
    assert first.artifacts.dynamics_hash == second.artifacts.dynamics_hash
    assert first.artifacts.decoder_hash == second.artifacts.decoder_hash


def test_one_pinned_controller_more_than_one_body_is_a_measured_table(
    graph: RuntimeGraph,
) -> None:
    """LOW-1: ``__init__``'s first sentence, pinned instead of asserted in prose.

    Across the reference body and the FlyGym body, twelve of the thirteen artifact
    identities in the manifest are identical and exactly one -- ``arena_hash`` --
    differs. Nothing asserted any of it: the nearest test compared two
    ``ReferenceArena`` *seeds* through one body class, which cannot see whether a
    second body class shares the controller at all.

    The FlyGym body here is stub-backed. That is irrelevant to this claim, which is
    about manifest identities and not about physics, and it is why the assertion is
    reachable without flygym installed.
    """

    controller = make_controller(graph)
    reference = build_embodiment_manifest(controller, graph, ReferenceEmbodiment(ReferenceArena()))
    flygym = build_embodiment_manifest(controller, graph, kinematic_flygym_body(graph))
    names = [field.name for field in dataclasses.fields(ArtifactIdentities)]
    differing = [
        name
        for name in names
        if getattr(reference.artifacts, name) != getattr(flygym.artifacts, name)
    ]
    assert differing == ["arena_hash"]
    assert len(names) - len(differing) == 12


def test_embodiment_code_identity_differs_from_the_fixture_harness_identity() -> None:
    from scripts.fixture_harness import replay_implementation_sha256

    assert embodiment_implementation_sha256() != replay_implementation_sha256()
    assert len(embodiment_implementation_sha256()) == 64


def test_recorded_trace_replays_exactly_in_both_brain_and_body(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    output = tmp_path / "replayable.ndjson"
    record_reference_trace(graph, output)
    result = replay_reference_trace(output)
    assert result["status"] == STATUS_NEURAL_AND_BODY_EXACT == "neural_and_body_exact"
    assert result["schema"] == "neurocraft.embodiment.replay.v2"
    assert result["record_count"] == TICKS
    assert result["body_ticks_re_executed"] == TICKS
    assert result["world_transitions_cross_linked"] == TICKS - 1
    assert result["terminal_world_transitions_not_cross_linked"] == 1


def test_replay_rejects_a_trace_recorded_by_a_different_body(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    output = tmp_path / "other-body.ndjson"
    record_reference_trace(graph, output, ticks=6)
    with pytest.raises(EmbodimentError, match="does not match the current replay environment"):
        replay_reference_trace(output, seed=9)


def test_run_rejects_a_non_positive_tick_count(graph: RuntimeGraph, tmp_path: Path) -> None:
    with pytest.raises(EmbodimentError, match="ticks must be positive"):
        run_embodiment(
            controller=make_controller(graph),
            graph=graph,
            embodiment=ReferenceEmbodiment(ReferenceArena()),
            output=tmp_path / "unused.ndjson",
            ticks=0,
        )


def test_cli_round_trips_run_then_replay(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = tmp_path / "cli.ndjson"
    assert main(["run", "--output", str(trace), "--ticks", "8"]) == 0
    run_receipt = json.loads(capsys.readouterr().out)
    assert run_receipt["record_count"] == 8

    assert main(["replay", "--trace", str(trace)]) == 0
    replay_receipt = json.loads(capsys.readouterr().out)
    assert replay_receipt["status"] == STATUS_NEURAL_AND_BODY_EXACT
    assert replay_receipt["file_sha256"] == run_receipt["file_sha256"]


# -- HIGH-1: replay must re-execute the body ------------------------------------


class _DriftingBody(ReferenceEmbodiment):
    """Overrides only ``apply``: 5.0 m of undeclared +x motion every control tick.

    The drift is committed to the body's own state, so the recorded trace is
    internally consistent: every ``world_after`` equals the next record's
    ``world_before``, and every observation is the one this body really produced.
    """

    def apply(self, command: Any) -> Any:  # type: ignore[override]
        transition = super().apply(command)
        drifted = replace(self._state, x_m=self._state.x_m + 5.0)
        self._state = drifted
        return replace(
            transition,
            world_after=replace(
                transition.world_after,
                position_m=(drifted.x_m, drifted.y_m, drifted.z_m),
            ),
        )


class _RayTamperedBody(ReferenceEmbodiment):
    """Overrides only ``observe``: saturates the three encoder-visible rays.

    ``PhysicalFeatureEncoder`` reads ``ray_distances_m[0]``, ``[2]``, ``[3]`` and
    ``local_light`` and nothing else, so these three scalars are most of what the
    graph is ever told about the world.
    """

    def observe(self, *, identity: Any, sequence: int) -> Observation:
        observation = super().observe(identity=identity, sequence=sequence)
        sensors = observation.sensors
        rays = list(sensors.ray_distances_m)
        for index in (0, 2, 3):
            rays[index] = sensors.ray_max_distance_m
        return replace(observation, sensors=replace(sensors, ray_distances_m=tuple(rays)))


class _InertBody(ReferenceEmbodiment):
    """A body that records fine and then refuses to be run again."""

    def observe(self, *, identity: Any, sequence: int) -> Observation:
        raise EmbodimentError("this body cannot re-execute")

    def apply(self, command: Any) -> Any:  # type: ignore[override]
        raise EmbodimentError("this body cannot re-execute")


def test_replay_re_executes_the_body_and_catches_undeclared_drift(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """HIGH-1: ``exact_match`` attested to the connectome and nothing else.

    ``_DriftingBody`` has a **byte-identical** ``EmbodimentIdentity`` -- asserted
    below, because that is the load-bearing fact: an identity hashes declarations
    (arena constants, channel sourcing, seed, clock, provenance) and behaviour is
    not a declaration, so no check built from the identity can ever see this. The
    only thing that can is running the body again and comparing what it produces.

    Against the shipped replay this trace was pronounced ``"status":
    "exact_match"`` with eleven cross-links and no complaint.
    """

    pristine = ReferenceEmbodiment(ReferenceArena())
    drifting = _DriftingBody(ReferenceArena())
    assert drifting.identity.identity_sha256 == pristine.identity.identity_sha256

    output = tmp_path / "drift.ndjson"
    run_embodiment(
        controller=make_controller(graph),
        graph=graph,
        embodiment=drifting,
        output=output,
        ticks=6,
    )
    with pytest.raises(EmbodimentError, match="world_after mismatch at record 0") as excinfo:
        replay_reference_trace(output)
    assert "position_m" in str(excinfo.value)

    # And the neural-only lane still passes it, which is precisely why that lane is
    # named for what it does not do rather than for a bare superlative.
    weaker = replay_embodiment_trace_neural_only(
        trace=output,
        graph_factory=lambda: load_malecns_fixture(FIXTURE),
        embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena()),
    )
    assert weaker["status"] == STATUS_NEURAL_ONLY
    assert weaker["body_ticks_re_executed"] == 0


def test_replay_catches_a_sensory_only_tamper_at_the_observation(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """The comparison unit must be the whole ``Observation``, not ``world_before``.

    ``WorldState`` carries position, velocity, yaw, pitch, three flags and
    ``in_fluid``. ``ray_distances_m``, ``ray_max_distance_m`` and ``local_light``
    live only on the ``Observation`` -- and they are the entire sensory boundary the
    pinned encoder reads. This test pins the consequence: against a pristine run the
    tampered run's **observations** first differ at record 0 while its
    ``world_before`` agrees until record 1, reached only through closed-loop
    amplification. Comparing ``world_before`` would therefore report the wrong record
    and would rely on everything downstream staying bit-exact forever.
    """

    tampered_trace = tmp_path / "rays.ndjson"
    pristine_trace = tmp_path / "pristine.ndjson"
    run_embodiment(
        controller=make_controller(graph),
        graph=graph,
        embodiment=_RayTamperedBody(ReferenceArena()),
        output=tampered_trace,
        ticks=6,
    )
    record_reference_trace(graph, pristine_trace, ticks=6)

    from neurocraft_embodiment.loop import iter_records

    tampered = tuple(iter_records(tampered_trace))
    pristine = tuple(iter_records(pristine_trace))
    first_observation_difference = next(
        index
        for index, pair in enumerate(zip(tampered, pristine, strict=True))
        if pair[0].observation != pair[1].observation
    )
    first_world_before_difference = next(
        index
        for index, pair in enumerate(zip(tampered, pristine, strict=True))
        if pair[0].world_before != pair[1].world_before
    )
    assert first_observation_difference == 0
    assert first_world_before_difference == 1

    with pytest.raises(EmbodimentError, match="observation mismatch at record 0") as excinfo:
        replay_reference_trace(tampered_trace)
    assert "sensors.ray_distances_m" in str(excinfo.value)


def test_replay_fails_when_the_body_refuses_to_re_execute(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """A body that cannot be run again fails closed; there is no weaker status."""

    output = tmp_path / "inert.ndjson"
    record_reference_trace(graph, output, ticks=4)
    with pytest.raises(EmbodimentError, match="cannot re-execute"):
        replay_embodiment_trace(
            trace=output,
            graph_factory=lambda: load_malecns_fixture(FIXTURE),
            embodiment_factory=lambda: _InertBody(ReferenceArena()),
        )


def test_neural_only_replay_is_named_for_what_it_does_not_do(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    output = tmp_path / "neural-only.ndjson"
    record_reference_trace(graph, output, ticks=6)
    result = replay_embodiment_trace_neural_only(
        trace=output,
        graph_factory=lambda: load_malecns_fixture(FIXTURE),
        embodiment_factory=lambda: ReferenceEmbodiment(ReferenceArena()),
    )
    assert result["status"] == "neural_exact_body_not_re_executed"
    assert result["body_ticks_re_executed"] == 0
    assert "was NOT" in result["evidence_boundary"]


def test_no_replay_entry_point_can_disable_body_re_execution() -> None:
    """No ``verify_body=False``: the weak lane has its own name, not a flag.

    A flag would be one keystroke from re-opening the hole and would become the
    thing an author reaches for when a body is inconvenient.
    """

    signature = inspect.signature(replay_embodiment_trace)
    # ``configuration`` reconstructs a non-default recorded controller and is still
    # checked against the trace manifest's hashes, so it widens what can be
    # replayed, not what passes. A body-verification switch remains forbidden.
    assert set(signature.parameters) == {
        "trace",
        "graph_factory",
        "embodiment_factory",
        "configuration",
    }
    assert signature.parameters == inspect.signature(replay_embodiment_trace_neural_only).parameters


def test_the_retired_exact_match_status_is_gone_from_the_package() -> None:
    """Retired outright rather than aliased: an alias lets a receipt keep saying it."""

    offenders = [
        f"{path.relative_to(PACKAGE_SRC).as_posix()}:{node.lineno}"
        for path in sorted(PACKAGE_SRC.rglob("*.py"))
        if "__pycache__" not in path.parts
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and node.value == "exact_match"
    ]
    # String *constants*, not raw text: the module docstring must stay free to name
    # the retired status when it explains why it was retired.
    assert offenders == []


# -- MEDIUM-3: the run resets the body ------------------------------------------


def test_run_resets_the_body_before_the_first_observation(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """MEDIUM-3, and the precondition that makes body-side replay sound.

    Replay's factory yields a body at its declared initial state. If a run may be
    recorded from an arbitrary state then a fresh, honest body legitimately
    disagrees at record 0. A body pre-stepped 20 ticks at full vertical thrust --
    which *sinks*, since ``thrust_accel_m_s2`` 6.0 < ``gravity_m_s2`` 9.81 --
    recorded a first ``world_before`` of y = 0.30351889908803587 m against a declared
    ``spawn_height_m`` of 1.5 m, and replayed exact.
    """

    from neurocraft_embodiment import EmbodimentCommand
    from neurocraft_protocol import MotorVector

    controller = make_controller(graph)
    body = ReferenceEmbodiment(ReferenceArena())
    controller.step(body.observe(identity=controller.identity, sequence=0), command_seq=0)
    state = controller.engine.current_state()
    for tick in range(20):
        body.apply(
            EmbodimentCommand(
                command_seq=tick,
                motors=MotorVector(vertical_thrust=1.0),
                neural_state=state,
            )
        )
    assert body.state.y_m != body.arena.spawn_height_m

    output = tmp_path / "prestepped.ndjson"
    run_embodiment(
        controller=make_controller(graph),
        graph=graph,
        embodiment=body,
        output=output,
        ticks=6,
    )
    from neurocraft_embodiment.loop import iter_records

    first = next(iter_records(output))
    assert first.world_before.position_m[1] == body.arena.spawn_height_m
    assert replay_reference_trace(output)["status"] == STATUS_NEURAL_AND_BODY_EXACT


# -- HIGH-2: a stub-backed run says so in its own receipt ------------------------


@pytest.mark.skipif(FLYGYM_INSTALLED, reason="flygym is installed in this environment")
def test_a_stub_backed_run_is_self_identifying_from_the_trace_bytes(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """HIGH-2: the receipt enumerated three exclusions and omitted the one that applied.

    ``FlyGymModules`` is a public export and ``FlyGymEmbodiment`` accepts
    ``modules=``, which bypasses ``require_flygym`` entirely. A stub-backed body
    therefore produced a full verifier-accepted trace whose bytes contained no
    occurrence of ``stub``, ``mujoco`` or the declared version -- the only mention of
    ``flygym`` anywhere was the embodiment id inside the ``not_used_in_``
    placeholders. Root cause: ``run_embodiment`` had zero call sites for
    ``provenance()``, so ``flygym_version: "2.1.0-stub"`` lived only inside the
    identity's hash preimage.

    The classification is resolved through ``importlib.metadata``, not by sniffing
    the string ``stub``: a distribution that is not installed cannot have executed a
    step whatever it calls itself. The converse is weaker and is worded as such --
    ``modules=`` can still hand an installed-flygym environment a stand-in.
    """

    output = tmp_path / "stub-flygym.ndjson"
    receipt = run_embodiment(
        controller=ConnectomeController.create(graph, run_id="embodiment-run", entity_id="fly"),
        graph=graph,
        embodiment=kinematic_flygym_body(graph),
        output=output,
        ticks=4,
    )
    assert receipt["simulator_evidence"] == "declared_simulator_libraries_absent"
    assert receipt["declared_libraries"] == [
        {
            "declared_version": "2.1.0-stub",
            "distribution": "flygym",
            "installed_version": None,
            "present": False,
        },
        {
            "declared_version": "3.9.0-stub",
            "distribution": "mujoco",
            "installed_version": None,
            "present": False,
        },
    ]
    assert "not one step of that simulator executed" in receipt["evidence_boundary"]

    manifest = verify_trace(output).manifest
    assert manifest.mod.startswith("not_used_in_flygym-neuromechfly;")
    assert "simulator_evidence=declared_simulator_libraries_absent" in manifest.mod
    assert "absent=flygym@2.1.0-stub,mujoco@3.9.0-stub" in manifest.mod
    assert "2.1.0-stub" in output.read_text(encoding="utf-8")


class _LyingProvenanceBody(ReferenceEmbodiment):
    """Its live provenance is not the provenance its identity hashed."""

    def provenance(self) -> dict[str, str]:
        return {**dict(super().provenance()), "flygym_version": "2.1.0"}


def test_run_rejects_a_body_whose_live_provenance_left_its_identity(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """``provenance()`` is only worth recording if the hash actually covers it."""

    with pytest.raises(EmbodimentError, match="does not match the provenance folded into"):
        run_embodiment(
            controller=make_controller(graph),
            graph=graph,
            embodiment=_LyingProvenanceBody(ReferenceArena()),
            output=tmp_path / "lying.ndjson",
            ticks=2,
        )


def test_the_environment_label_drops_detail_rather_than_truncating_it() -> None:
    """Half a distribution name would read as a real one, so it is dropped whole.

    The classification is the part that carries the claim and it always fits: the
    worst case is 12 + 128 (``_assert_name``'s ``embodiment_id`` bound) + 20 + 38 =
    198 of the manifest's 256 characters.
    """

    from neurocraft_embodiment.identity import LibraryEvidence

    identity = make_identity(embodiment_id="b" * 128)
    oversized = LibraryEvidence(
        distribution="x" * 120,
        declared_version="y" * 200,
        installed_version=None,
    )
    label = embodiment_environment_label(identity, (oversized,))
    assert len(label) <= 256
    assert label.endswith("simulator_evidence=declared_simulator_libraries_absent")
    assert "absent=" not in label


# -- MEDIUM-2: the control rate is the seam's, not the body's --------------------


def test_identity_refuses_a_control_rate_the_seam_does_not_run() -> None:
    """MEDIUM-2: 100 Hz with 5 x 10 ms substeps closed its own clock and passed.

    ``_assert_clock_closes`` compared the substep product against the body's own
    ``control_hz``, which ``__post_init__`` accepted anywhere in ``(0, 1000]``. Both
    checks then agreed with each other and with nothing else, while
    ``observation_timing`` stamped every observation 50 ms apart for every body and
    the manifest recorded ``game_tick_hz = 100``: a factor-of-five lie, verified end
    to end. Both shipped bodies pass ``CONTROL_HZ``, so this was latent, not live.
    """

    assert observation_timing(1)[1] - observation_timing(0)[1] == CONTROL_PERIOD_NS == 50_000_000
    with pytest.raises(EmbodimentIdentityError, match="frozen v1 seam rate"):
        make_identity(
            control_hz=100.0,
            physics_steps_per_control_tick=5,
            physics_timestep_s=0.01,
        )


# -- LOW-3: identity material covers every declared field -----------------------

#: ``(deliberately unhashed fields, derived keys that are not fields)``.
_MATERIAL_EXCEPTIONS: dict[type, tuple[frozenset[str], frozenset[str]]] = {
    # ``declared_dynamic_channels`` is a construction-time assertion input, exactly
    # as ``flygym_boundary._assert_declared``'s argument is; identity.py:150-157
    # gives the reason. Not a defect, and not to be "fixed".
    BoundaryIdentity: (frozenset({"declared_dynamic_channels"}), frozenset({"wrapper_schema"})),
    ReferenceArena: (frozenset(), frozenset({"integrator", "prng", "units"})),
}


@pytest.mark.parametrize(
    "instance",
    [
        make_identity(),
        make_identity().sensory_boundary,
        ChannelDeclaration("alpha", "dynamic", "measured"),
        ReferenceArena(),
        FlyGymBodyConfig(),
    ],
    ids=[
        "EmbodimentIdentity",
        "BoundaryIdentity",
        "ChannelDeclaration",
        "ReferenceArena",
        "FlyGymBodyConfig",
    ],
)
def test_every_identity_dataclass_hashes_every_field_it_declares(instance: Any) -> None:
    """LOW-3: completeness was a curated list of ten hand-varied field names.

    Structurally everything is covered today, but nothing compared
    ``dataclasses.fields(cls)`` against ``cls.to_identity_material()``, so a field
    added tomorrow would be unhashed with the whole suite green.
    """

    unhashed_by_design, derived = _MATERIAL_EXCEPTIONS.get(
        type(instance), (frozenset(), frozenset())
    )
    fields = {field.name for field in dataclasses.fields(instance)}
    material = set(instance.to_identity_material())
    assert fields - material == unhashed_by_design
    assert material - fields == derived


# -- the clock closure, isolated -------------------------------------------------


class _MisreportingBody(ReferenceEmbodiment):
    """A body whose transitions claim a clock its identity does not declare."""

    def __init__(self, *, steps: int | None = None, seconds: float | None = None) -> None:
        super().__init__(ReferenceArena())
        self._steps = steps
        self._seconds = seconds

    def apply(self, command: object) -> object:  # type: ignore[override]
        transition = super().apply(command)  # type: ignore[arg-type]
        return replace(
            transition,
            physics_steps=self._steps or transition.physics_steps,
            simulated_time_s=self._seconds or transition.simulated_time_s,
        )


def test_run_rejects_a_body_that_steps_a_different_number_of_times(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """LOW-10: ``physics_steps`` was validated for self-consistency and never
    cross-checked against ``identity.physics_steps_per_control_tick``.

    An embodiment could therefore step a different number of times per control tick
    than the identity -- and therefore the recorded manifest -- declares.
    """

    with pytest.raises(EmbodimentError, match="advanced 24 physics steps"):
        run_embodiment(
            controller=make_controller(graph),
            graph=graph,
            embodiment=_MisreportingBody(steps=24),
            output=tmp_path / "wrong-steps.ndjson",
            ticks=2,
        )


def test_run_rejects_a_body_that_advances_a_different_amount_of_time(
    graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    with pytest.raises(EmbodimentError, match="simulated"):
        run_embodiment(
            controller=make_controller(graph),
            graph=graph,
            embodiment=_MisreportingBody(seconds=0.04),
            output=tmp_path / "wrong-seconds.ndjson",
            ticks=2,
        )


def test_the_transition_clock_fields_are_not_related_to_each_other() -> None:
    """LOW-1: the loop's comment credited a check ``EmbodimentTransition`` never had.

    ``seam.EmbodimentTransition`` validates ``physics_steps >= 1`` and
    ``simulated_time_s > 0`` and nothing at all relates the two, so a transition
    claiming the reference body's full 25 substeps in one nanosecond is well formed
    -- 50 ms of declared physics in 1 ns. ``assert_declared_clock`` is the only thing
    closing the clock, and this is the evidence.
    """

    from neurocraft_embodiment.loop import assert_declared_clock
    from neurocraft_protocol import WorldState

    absurd = EmbodimentTransition(
        world_after=WorldState(
            position_m=(0.0, 1.0, 0.0),
            velocity_m_s=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            pitch_rad=0.0,
            on_ground=False,
            horizontal_collision=False,
            vertical_collision=False,
            in_fluid=False,
        ),
        passive_forces=(0.0, -9.81, 0.0),
        external_forces=(0.0, 0.0, 0.0),
        physics_steps=25,
        simulated_time_s=1.0e-9,
    )
    assert absurd.physics_steps == 25
    assert absurd.simulated_time_s == 1.0e-9

    with pytest.raises(EmbodimentError, match="simulated"):
        assert_declared_clock(
            absurd,
            ReferenceEmbodiment(ReferenceArena()).identity,
            tick=0,
        )
