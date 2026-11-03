"""Drive the pinned controller through any embodiment, record and replay the trace.

Nothing in the trace stack is modified or re-implemented here. This module calls
``causal_record_from_applied_action``, ``TraceWriter`` and ``verify_trace`` exactly
as ``scripts/fixture_harness.py`` does; the only new thing is that the body behind
the loop is a seam implementation rather than a hard-coded world, and that the
body's hashed :class:`EmbodimentIdentity` occupies the manifest's ``arena_hash``.

Naming rule for replay outcomes, stated once because the last string broke it
-----------------------------------------------------------------------------
A replay status names **what was re-executed and what was not**, and never carries
a bare superlative -- ``exact``, ``match``, ``verified``, ``passed`` -- without a
scope word. The retired ``"exact_match"`` failed both halves: a superlative with no
subject, which is exactly how it came to stand for a replay that re-ran the
connectome and never once touched the body. A ``ReferenceEmbodiment`` subclass
adding 5.0 m of undeclared motion per tick has a byte-identical
``EmbodimentIdentity`` -- behaviour is not a declaration -- and was pronounced
``exact_match`` against the pristine body. This axis is orthogonal to the
repository's evidence vocabulary (planned / unit-tested / benchmarked / ...), which
describes work items; the two are never mixed in one string.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import platform
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import (
    ConnectomeController,
    ControllerConfiguration,
    causal_record_from_applied_action,
    world_state_from_observation,
)
from neurocraft_protocol import (
    ArtifactIdentities,
    CausalRecord,
    ClaimScope,
    DataTier,
    DeterminismLevel,
    FloatPrecision,
    Observation,
    TraceManifest,
    TraceWriter,
    ValidationResult,
    parse_json_frame,
    verify_trace,
)

from .identity import (
    SIMULATOR_EVIDENCE_INSTALLED,
    SIMULATOR_EVIDENCE_NONE_DECLARED,
    EmbodimentIdentity,
    LibraryEvidence,
    resolve_library_evidence,
    simulator_evidence,
)
from .seam import Embodiment, EmbodimentCommand, EmbodimentError, EmbodimentTransition

_CODE_ROOTS = (
    ("packages", "connectome-data", "src"),
    ("packages", "controller-runtime", "src"),
    ("packages", "embodiment-runtime", "src"),
    ("packages", "neural-runtime", "src"),
    ("packages", "protocol", "src"),
)

#: Every replay re-executes the connectome. Only one of the two entry points also
#: re-executes the body, and the string says which.
STATUS_NEURAL_AND_BODY_EXACT = "neural_and_body_exact"
STATUS_NEURAL_ONLY = "neural_exact_body_not_re_executed"

#: The manifest has no field for a body's runtime environment, and the protocol
#: package is identity-bound, so no field may be added to it. ``mod`` carries it,
#: prefixed with the same ``not_used_in_<id>`` marker the other three environment
#: strings use -- ``scripts/replay_minecraft_trace.py`` refuses any trace whose
#: java/minecraft/fabric/mod tuple contains ``not_used``, and that refusal must keep
#: firing on all four. Everything after the marker is the embodiment lane's.
_ENVIRONMENT_LABEL_LIMIT = 256

_RUN_EVIDENCE_BASE = (
    "Deterministic in-process embodiment run. This is not a Minecraft run, "
    "not a behavioral-validation result, and not a biological claim."
)


def repository_root() -> Path:
    """Locate the checkout this package was loaded from, fail-closed.

    The embodiment lane records a code identity that must include this package's own
    sources, so it needs the checkout rather than the installed wheel. If the layout
    is not recognisable the caller gets an explicit error instead of a silently
    wrong hash.
    """

    root = Path(__file__).resolve().parents[4]
    if not (root / "uv.lock").is_file() or not (root / "packages").is_dir():
        raise EmbodimentError(
            "the embodiment loop must run from a source checkout; "
            f"{root} does not look like the repository root"
        )
    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embodiment_implementation_sha256() -> str:
    """Bind every Python source that can affect embodiment execution or replay.

    Same construction as ``scripts/fixture_harness.replay_implementation_sha256``,
    with this package's ``src`` added. It is deliberately a *different* value from
    the fixture harness's: a different set of files can change the result, so it
    must be a different code identity.
    """

    root = repository_root()
    paths = {
        path
        for parts in _CODE_ROOTS
        for path in (root.joinpath(*parts)).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        raw = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


# -- environment provenance -----------------------------------------------------


def recorded_provenance(embodiment: Embodiment) -> Mapping[str, str]:
    """Read the body's live provenance and refuse it if the hash does not cover it.

    ``EmbodimentIdentity`` hashes a provenance tuple; ``Embodiment.provenance()``
    returns a mapping. Nothing related the two, and nothing called the method at all,
    so the hashed provenance was an unverified declaration about an environment
    nobody had asked the body about at run time. Cross-checking here is what makes
    the manifest's ``arena_hash`` a faithful commitment to the environment strings
    the receipt goes on to report.
    """

    identity = embodiment.identity
    live = dict(embodiment.provenance())
    hashed = dict(identity.provenance_mapping())
    if live != hashed:
        raise EmbodimentError(
            "the body's live provenance does not match the provenance folded into "
            f"its identity hash: live={sorted(live.items())!r}, "
            f"hashed={sorted(hashed.items())!r}"
        )
    return live


def embodiment_environment_label(
    identity: EmbodimentIdentity,
    evidence: tuple[LibraryEvidence, ...],
) -> str:
    """The manifest ``mod`` string: what ran this body, readable from trace bytes.

    Before this existed a stub-backed FlyGym body produced a full verifier-accepted
    trace containing no occurrence of ``stub``, ``mujoco`` or the declared version --
    the only mention of ``flygym`` anywhere in the bytes was the embodiment id inside
    the ``not_used_in_`` placeholders.
    """

    plain = (
        f"not_used_in_{identity.embodiment_id};simulator_evidence={simulator_evidence(evidence)}"
    )
    # ``plain`` cannot overflow: ``embodiment_id`` is bounded at 128 characters by
    # ``_assert_name`` and the longest classification is 38, so the worst case is
    # 12 + 128 + 20 + 38 = 198 of the manifest's 256. Only the optional ``absent=``
    # detail can, and it is dropped whole rather than truncated -- half a
    # distribution name reads as a real one, and the classification is the part that
    # carries the claim.
    absent = [
        f"{entry.distribution}@{entry.declared_version}" for entry in evidence if not entry.present
    ]
    if not absent:
        return plain
    detailed = f"{plain};absent=" + ",".join(absent)
    return detailed if len(detailed) <= _ENVIRONMENT_LABEL_LIMIT else plain


def run_evidence_boundary(evidence: tuple[LibraryEvidence, ...]) -> str:
    """State the run's boundary, including which simulator did *not* run.

    The negative direction is the sound one: a distribution that is not installed in
    this interpreter cannot have executed a step, whatever version string the body
    declared. The positive direction is weaker and is worded as what it is -- a body
    can still be handed a stand-in through ``FlyGymEmbodiment(modules=...)``, so
    "installed" is never reported as "used".
    """

    classification = simulator_evidence(evidence)
    if classification == SIMULATOR_EVIDENCE_NONE_DECLARED:
        return (
            f"{_RUN_EVIDENCE_BASE} The body declares no third-party simulator "
            "library, so there is no simulator result to claim here."
        )
    declared = ", ".join(f"{entry.distribution} {entry.declared_version}" for entry in evidence)
    if classification == SIMULATOR_EVIDENCE_INSTALLED:
        return (
            f"{_RUN_EVIDENCE_BASE} The body declares {declared}, and those "
            "distributions resolve to those versions in this interpreter. Installed "
            "is not used: a body can still be constructed against a substitute "
            "module handle, so this is not evidence that the simulator drove the run."
        )
    missing = ", ".join(
        f"{entry.distribution} {entry.declared_version} "
        f"(installed: {entry.installed_version or 'absent'})"
        for entry in evidence
        if not entry.present
    )
    return (
        f"{_RUN_EVIDENCE_BASE} The body declares {declared}, and {missing} does not "
        "resolve in this interpreter, so not one step of that simulator executed "
        "during this run. Nothing recorded here is a result of that simulator."
    )


def assert_declared_clock(
    transition: EmbodimentTransition,
    identity: EmbodimentIdentity,
    *,
    tick: int,
) -> None:
    """Close one transition's clock against the identity that will be recorded.

    ``EmbodimentTransition`` checks each clock field on its own -- at least one
    physics step, some simulated time -- and nothing at all relates the two to each
    other or to any declared rate: a transition reporting 500 steps and 1 ns is well
    formed. One copy, two callers: the run loop and the body-side replay.
    """

    if transition.physics_steps != identity.physics_steps_per_control_tick:
        raise EmbodimentError(
            f"tick {tick} advanced {transition.physics_steps} physics steps "
            f"but the embodiment identity declares "
            f"{identity.physics_steps_per_control_tick}"
        )
    declared_s = identity.physics_steps_per_control_tick * identity.physics_timestep_s
    if not math.isclose(transition.simulated_time_s, declared_s, rel_tol=1e-9, abs_tol=1e-15):
        raise EmbodimentError(
            f"tick {tick} advanced {transition.simulated_time_s!r} simulated "
            f"seconds but the embodiment identity declares {declared_s!r}"
        )


def build_embodiment_manifest(
    controller: ConnectomeController,
    graph: RuntimeGraph,
    embodiment: Embodiment,
    *,
    claim_scope: ClaimScope = ClaimScope.G2,
    data_tier: DataTier = DataTier.MALECNS_FIXTURE,
) -> TraceManifest:
    """Manifest for an embodiment run; ``arena_hash`` carries the body identity."""

    root = repository_root()
    lock = root / "uv.lock"
    identity = embodiment.identity
    absent = f"not_used_in_{identity.embodiment_id}"
    evidence = resolve_library_evidence(recorded_provenance(embodiment))
    artifacts = ArtifactIdentities(
        source_hash=graph.manifest.source_manifest_sha256,
        canonical_graph_hash=graph.manifest.canonical_graph_sha256,
        runtime_graph_hash=graph.manifest.runtime_identity_sha256,
        fixture_hash=graph.manifest.fixture_file_sha256,
        code_hash=embodiment_implementation_sha256(),
        dependency_lock_hash=sha256_file(lock) if lock.is_file() else "0" * 64,
        adapter_hash=controller.identity.adapter_hash,
        dynamics_hash=controller.identity.dynamics_hash,
        sign_hash=controller.sign_overlay.identity_sha256,
        decoder_hash=controller.identity.decoder_hash,
        arena_hash=identity.identity_sha256,
        configuration_hash=controller.configuration.identity_sha256,
        observation_schema_hash=controller.identity.observation_schema_hash,
    )
    return TraceManifest(
        run_id=controller.identity.run_id,
        initial_session_epoch=controller.identity.session_epoch,
        mode=controller.run_mode,
        data_tier=data_tier,
        claim_scope=claim_scope,
        artifacts=artifacts,
        game_tick_hz=identity.control_hz,
        neural_step_ms=controller.dynamics.neural_step_ms,
        precision=FloatPrecision.FLOAT64,
        backend=controller.dynamics.backend,
        thread_count=1,
        seed=identity.seed,
        determinism_level=identity.determinism_level,
        machine=platform.machine() or "unknown",
        os=platform.system() or "unknown",
        python=platform.python_version(),
        java=absent,
        minecraft=absent,
        fabric=absent,
        mod=embodiment_environment_label(identity, evidence),
        privacy_policy="local_trace_no_secrets_v1",
        expected_chunks=1,
        checkpoint_interval_ticks=100,
    )


def run_embodiment(
    *,
    controller: ConnectomeController,
    graph: RuntimeGraph,
    embodiment: Embodiment,
    output: Path,
    ticks: int,
    claim_scope: ClaimScope = ClaimScope.G2,
    data_tier: DataTier = DataTier.MALECNS_FIXTURE,
) -> dict[str, Any]:
    """Run ``ticks`` control ticks and write a trace the existing verifier accepts.

    The body is ``reset()`` before the first observation. That is not a courtesy: it
    is the precondition that makes body-side replay sound. Replay's factory yields a
    body at its declared initial state, so a run recorded from an arbitrary state
    would make an honest body disagree at record 0. A body pre-stepped 20 ticks at
    full vertical thrust used to record a first ``world_before`` of y = 0.3035 m
    against a declared ``spawn_height_m`` of 1.5 m, and replay called it exact.
    """

    if ticks < 1:
        raise EmbodimentError("ticks must be positive")
    provenance = recorded_provenance(embodiment)
    evidence = resolve_library_evidence(provenance)
    manifest = build_embodiment_manifest(
        controller,
        graph,
        embodiment,
        claim_scope=claim_scope,
        data_tier=data_tier,
    )
    identity = embodiment.identity
    embodiment.reset()
    with TraceWriter(output, manifest) as writer:
        for tick in range(ticks):
            observation = embodiment.observe(identity=controller.identity, sequence=tick)
            start = time.perf_counter_ns()
            action = controller.step(observation, command_seq=tick)
            compute_ns = time.perf_counter_ns() - start
            neural_state = controller.engine.current_state()
            if neural_state.state_sha256 != action.state_digest:
                raise EmbodimentError(
                    "engine state does not match the issued action digest; "
                    "the body would be driven by an unverified state"
                )
            transition = embodiment.apply(
                EmbodimentCommand(
                    command_seq=tick,
                    motors=action.motors,
                    neural_state=neural_state,
                )
            )
            assert_declared_clock(transition, identity, tick=tick)
            writer.append(
                causal_record_from_applied_action(
                    controller,
                    record_seq=tick,
                    observation=observation,
                    world_after=transition.world_after,
                    world_after_observation_seq=observation.observation_seq + 1,
                    world_after_game_tick=observation.game_tick + 1,
                    passive_forces=transition.passive_forces,
                    external_forces=transition.external_forces,
                    compute_ns=compute_ns,
                )
            )
    verified = verify_trace(output)
    return {
        "body_reset_before_first_observation": True,
        "claim_scope": claim_scope.value,
        "declared_libraries": [entry.to_receipt_entry() for entry in evidence],
        "embodiment_id": identity.embodiment_id,
        "embodiment_identity_sha256": identity.identity_sha256,
        "embodiment_version": identity.embodiment_version,
        "evidence_boundary": run_evidence_boundary(evidence),
        "file_sha256": verified.file_sha256,
        "final_hash": verified.final_hash,
        "first_tick": verified.first_tick,
        "last_tick": verified.last_tick,
        "motor_currency": identity.motor_currency,
        "path": str(output),
        "provenance": dict(provenance),
        "record_count": verified.record_count,
        "schema": "neurocraft.embodiment.receipt.v2",
        "simulator_evidence": simulator_evidence(evidence),
    }


def iter_records(path: Path) -> Iterator[CausalRecord]:
    with path.open("rb") as stream:
        for raw in stream:
            envelope = parse_json_frame(raw.rstrip(b"\n"))
            if envelope.get("kind") == "record":
                yield CausalRecord.from_dict(envelope["payload"])


# -- replay ---------------------------------------------------------------------


def _first_difference(recorded: Any, replayed: Any, *, prefix: str) -> str | None:
    """Name the first field on which two protocol dataclasses disagree.

    Depth-first over ``dataclasses.fields`` so the message says
    ``observation.sensors.ray_distances_m`` rather than dumping two whole frames.
    """

    for field in dataclasses.fields(recorded):
        left = getattr(recorded, field.name)
        right = getattr(replayed, field.name)
        if left == right:
            continue
        path = f"{prefix}.{field.name}"
        if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
            nested = _first_difference(left, right, prefix=path)
            if nested is not None:
                return nested
        return f"{path}: recorded {left!r}, re-executed {right!r}"
    return None


def _assert_body_reproduced_record(
    *,
    record: CausalRecord,
    observation: Observation,
    transition: EmbodimentTransition,
    identity: EmbodimentIdentity,
) -> None:
    """Assert the re-driven body reproduced this record's whole physical boundary.

    The comparison unit is the entire :class:`Observation`, not ``world_before``.
    ``WorldState`` carries position, velocity, yaw, pitch, three flags and
    ``in_fluid``; ``ray_distances_m``, ``ray_max_distance_m`` and ``local_light``
    exist only on the ``Observation`` -- and those are 100% of what the pinned
    encoder feeds the graph. A body tampered *only* in ``observe`` to saturate the
    three encoder-visible rays keeps a byte-identical identity hash and first
    diverges in ``world_before`` a whole record later, and in ``world_after`` only at
    1e-11 m. That works today solely because everything is bit-exact; it is not a
    guarantee, and it dies the moment anyone introduces a tolerance.
    """

    if observation != record.observation:
        difference = _first_difference(record.observation, observation, prefix="observation")
        raise EmbodimentError(
            f"body re-execution observation mismatch at record {record.record_seq}: "
            f"{difference or 'observations differ'}"
        )
    assert_declared_clock(transition, identity, tick=record.record_seq)
    if transition.world_after != record.world_after:
        difference = _first_difference(
            record.world_after, transition.world_after, prefix="world_after"
        )
        raise EmbodimentError(
            f"body re-execution world_after mismatch at record {record.record_seq}: "
            f"{difference or 'world states differ'}"
        )
    for name in ("passive_forces", "external_forces"):
        recorded_forces = getattr(record, name)
        replayed_forces = getattr(transition, name)
        if replayed_forces != recorded_forces:
            raise EmbodimentError(
                f"body re-execution {name} mismatch at record {record.record_seq}: "
                f"recorded {recorded_forces!r}, re-executed {replayed_forces!r}"
            )


def _replay(
    *,
    trace: Path,
    graph_factory: Callable[[], RuntimeGraph],
    embodiment_factory: Callable[[], Embodiment],
    configuration: ControllerConfiguration | None = None,
    re_execute_body: bool,
) -> dict[str, Any]:
    """The one replay engine. Both entry points are thin wrappers so they cannot drift.

    ``configuration`` reconstructs a controller recorded under a non-default
    ``ControllerConfiguration``; the trace manifest's configuration, adapter and
    decoder hashes are still compared afterwards, so a wrong configuration fails
    closed exactly as before. ``None`` keeps the historical default.
    """

    verification = verify_trace(trace)
    manifest = verification.manifest
    records = tuple(iter_records(trace))
    if len(records) != verification.record_count or not records:
        raise EmbodimentError("trace record count is empty or inconsistent")
    first = records[0]
    graph = graph_factory()
    embodiment = embodiment_factory()
    controller = ConnectomeController.create(
        graph,
        run_id=first.identity.run_id,
        entity_id=first.identity.entity_id,
        session_epoch=first.identity.session_epoch,
        configuration=configuration,
    )
    if controller.identity != first.identity:
        raise EmbodimentError("current controller identity does not match the trace")
    root = repository_root()
    lock = root / "uv.lock"
    artifacts = manifest.artifacts
    if (
        artifacts.runtime_graph_hash != graph.manifest.runtime_identity_sha256
        or artifacts.canonical_graph_hash != graph.manifest.canonical_graph_sha256
        or artifacts.source_hash != graph.manifest.source_manifest_sha256
        or artifacts.code_hash != embodiment_implementation_sha256()
        or artifacts.dependency_lock_hash != (sha256_file(lock) if lock.is_file() else "0" * 64)
        or artifacts.adapter_hash != controller.identity.adapter_hash
        or artifacts.dynamics_hash != controller.identity.dynamics_hash
        or artifacts.sign_hash != controller.sign_overlay.identity_sha256
        or artifacts.decoder_hash != controller.identity.decoder_hash
        or artifacts.arena_hash != embodiment.identity.identity_sha256
        or artifacts.configuration_hash != controller.configuration.identity_sha256
        or artifacts.observation_schema_hash != controller.identity.observation_schema_hash
    ):
        raise EmbodimentError("trace manifest does not match the current replay environment")
    identity = embodiment.identity
    if re_execute_body:
        if identity.determinism_level is not DeterminismLevel.SAME_BUILD_EXACT:
            raise EmbodimentError(
                f"body-side replay requires a {DeterminismLevel.SAME_BUILD_EXACT.value} "
                f"body; {identity.embodiment_id} declares "
                f"{identity.determinism_level.value}, and v1 defines no preregistered "
                "replay tolerance to compare against"
            )
        embodiment.reset()

    indexed = {
        (record.observation.observation_seq, record.observation.game_tick): (index, record)
        for index, record in enumerate(records)
    }
    if len(indexed) != len(records):
        raise EmbodimentError("trace contains duplicate observation lineage")
    final_observation = records[-1].observation
    cross_linked = 0
    terminal = 0
    for record_index, record in enumerate(records):
        if record.validation is not ValidationResult.APPLIED or record.command_seq is None:
            raise EmbodimentError("embodiment replay requires applied commands")
        if world_state_from_observation(record.observation) != record.world_before:
            raise EmbodimentError(f"world-before mismatch at record {record.record_seq}")
        lineage = (record.world_after_observation_seq, record.world_after_game_tick)
        target = indexed.get(lineage)
        if target is not None:
            target_index, target_record = target
            if target_index <= record_index or record.world_after != target_record.world_before:
                raise EmbodimentError(f"world-after lineage mismatch at record {record.record_seq}")
            cross_linked += 1
        elif (
            record.world_after_observation_seq > final_observation.observation_seq
            and record.world_after_game_tick > final_observation.game_tick
        ):
            terminal += 1
        else:
            raise EmbodimentError(
                f"world-after lineage is missing within trace at record {record.record_seq}"
            )
        observed = (
            embodiment.observe(
                identity=controller.identity,
                sequence=record.observation.observation_seq,
            )
            if re_execute_body
            else None
        )
        if controller.engine.current_state().activity != record.state_before_activity:
            raise EmbodimentError(f"full state-before mismatch at record {record.record_seq}")
        action = controller.step(record.observation, command_seq=record.command_seq)
        evidence = controller.last_evidence
        if evidence is None:
            raise EmbodimentError(f"controller evidence missing at record {record.record_seq}")
        if (
            action.state_digest != record.state_after_digest
            or action.motors != record.decoder_output
            or action.motors != record.applied_action
            or action.neural_step_start != record.neural_step_start
            or action.neural_step_end != record.neural_step_end
            or dict(evidence.encoded_stimulation) != dict(record.encoded_stimulation)
            or evidence.state_after_activity != record.state_after_activity
        ):
            raise EmbodimentError(f"causal replay mismatch at record {record.record_seq}")
        if observed is not None:
            transition = embodiment.apply(
                EmbodimentCommand(
                    command_seq=record.command_seq,
                    motors=action.motors,
                    neural_state=controller.engine.current_state(),
                )
            )
            _assert_body_reproduced_record(
                record=record,
                observation=observed,
                transition=transition,
                identity=identity,
            )
    libraries = resolve_library_evidence(recorded_provenance(embodiment))
    return {
        "body_ticks_re_executed": len(records) if re_execute_body else 0,
        "declared_libraries": [entry.to_receipt_entry() for entry in libraries],
        "embodiment_id": identity.embodiment_id,
        "embodiment_identity_sha256": identity.identity_sha256,
        "evidence_boundary": _replay_evidence_boundary(re_execute_body=re_execute_body),
        "file_sha256": verification.file_sha256,
        "final_state_sha256": records[-1].state_after_digest,
        "record_count": len(records),
        "replay_implementation_sha256": embodiment_implementation_sha256(),
        "schema": "neurocraft.embodiment.replay.v2",
        "simulator_evidence": simulator_evidence(libraries),
        "status": STATUS_NEURAL_AND_BODY_EXACT if re_execute_body else STATUS_NEURAL_ONLY,
        "terminal_world_transitions_not_cross_linked": terminal,
        "world_transitions_cross_linked": cross_linked,
    }


def _replay_evidence_boundary(*, re_execute_body: bool) -> str:
    common = (
        "Same-build in-process replay of an embodiment trace: hash chain, manifest, "
        "world-state lineage cross-links and neural re-step, with terminal lineage "
        "counted separately. Not a Minecraft run, not a behavioral-validation "
        "result, not a biological claim."
    )
    if re_execute_body:
        return (
            f"{common} The body was reset and re-driven with the re-derived command "
            "sequence, and reproduced every observation, world_after and declared "
            "force triple exactly."
        )
    return (
        f"{common} The body was constructed and its identity compared; it was NOT "
        "run, so nothing here constrains what the body would have done."
    )


def replay_embodiment_trace(
    *,
    trace: Path,
    graph_factory: Callable[[], RuntimeGraph],
    embodiment_factory: Callable[[], Embodiment],
    configuration: ControllerConfiguration | None = None,
) -> dict[str, Any]:
    """Re-execute a recorded trace in both brain and body, and assert both exact.

    There is deliberately no ``verify_body=False`` parameter. A flag would be one
    keystroke from re-opening the hole and would become the thing an author reaches
    for when a body is inconvenient; :func:`replay_embodiment_trace_neural_only`
    carries the weaker guarantee under a name that says so.

    Re-executing the body costs nothing in reachability: ``embodiment_factory()`` is
    already called and its identity already compared, so a FlyGym trace already
    cannot be replayed without flygym installed.
    """

    return _replay(
        trace=trace,
        graph_factory=graph_factory,
        embodiment_factory=embodiment_factory,
        configuration=configuration,
        re_execute_body=True,
    )


def replay_embodiment_trace_neural_only(
    *,
    trace: Path,
    graph_factory: Callable[[], RuntimeGraph],
    embodiment_factory: Callable[[], Embodiment],
    configuration: ControllerConfiguration | None = None,
) -> dict[str, Any]:
    """The old guarantee, honestly named: the brain is re-executed, the body is not."""

    return _replay(
        trace=trace,
        graph_factory=graph_factory,
        embodiment_factory=embodiment_factory,
        configuration=configuration,
        re_execute_body=False,
    )
