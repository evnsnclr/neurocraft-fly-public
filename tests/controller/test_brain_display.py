"""Small synthetic bindings and real controller steps; no full artifact or live apps."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller import brain_display as display
from neurocraft_controller.brain_display import (
    AtomicBrainDisplayPublisher,
    BrainDisplayBinding,
    BrainDisplayError,
    BrainDisplayFrame,
    BrainDisplayProducer,
    PopulationMean,
    load_brain_display_binding,
    select_brain_display_binding,
)
from neurocraft_controller.embodied_senses import (
    ANNOTATION_SOURCE_ID,
    EmbodiedPopulation,
    EmbodiedPopulationBinding,
    EmbodiedSensesError,
    faithful_entry_body_ids,
)
from neurocraft_controller.escape_reflex import TtmnPoolBinding, select_ttmn_pool
from neurocraft_controller.flight_pools import LEFT_WING_MOTOR_BODY_IDS, RIGHT_WING_MOTOR_BODY_IDS
from neurocraft_controller.full_brain_controller import (
    EscapeMotorBindingV2,
    FullBrainController,
    FullBrainLoomV1Configuration,
    bind_escape_jump_pool,
    bind_sensor_channels,
    bind_wing_motor_pools,
)
from neurocraft_controller.loom_senses import bind_embodied_senses_loom_v1
from neurocraft_controller.program_policy import ProgramSelectingController
from neurocraft_controller.steering_readout import select_dna01_pair
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_protocol import Action, ProgramAction, RunIdentity, canonical_json_bytes

from tests.controller.test_embodied_senses import _observation, _sensors, _synthetic_view

GRAPH_HASH = "ab" * 32
ANNOTATIONS_HASH = "cd" * 32


class TinySpace:
    """Exact display counts, two small auxiliary sensory pools, all declared motor bodies.

    Auxiliary visual/bristle counts are deliberately synthetic. Only the display's
    LC4/DNp01/TTMn counts and selection invariants are under test here.
    """

    def __init__(self) -> None:
        rows: list[tuple[int, str | None, str | None]] = []
        rows += [(100_000 + i, "LC4", "L") for i in range(71)]
        rows += [(101_000 + i, "LC4", "R") for i in range(55)]
        rows += [(102_000, "DNp01", "L"), (102_001, "DNp01", "R")]
        rows += [(800_146, "TTMn", "R"), (804_642, "TTMn", "L")]
        rows += [(103_000, "DNa01", "L"), (103_001, "DNa01", "R")]
        rows += [(104_000, None, None), (104_001, None, None)]
        rows += [(body, None, None) for body in faithful_entry_body_ids()]
        rows += [
            (body, None, None) for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS
        ]
        rows.sort()
        self.bodies = [body for body, _, _ in rows]
        self.types = [label for _, label, _ in rows]
        self.sides = [side for _, _, side in rows]
        self.index = {body: i for i, body in enumerate(self.bodies)}
        self.annotations_hash = ANNOTATIONS_HASH

    def populations(self) -> EmbodiedPopulationBinding:
        groups = []
        for name, bodies in (
            ("lc4_left", tuple(100_000 + i for i in range(71))),
            ("lc4_right", tuple(101_000 + i for i in range(55))),
            ("visual", (104_000,)),
            ("bristle", (104_001,)),
        ):
            groups.append(
                EmbodiedPopulation(
                    name,
                    "synthetic display fixture",
                    bodies,
                    tuple(self.index[body] for body in bodies),
                )
            )
        return EmbodiedPopulationBinding(
            GRAPH_HASH, self.annotations_hash, len(self.bodies), tuple(groups)
        )

    def arguments(self) -> dict[str, Any]:
        return dict(
            type_column=self.types,
            soma_side_column=self.sides,
            body_ids=self.bodies,
            index_of=self.index.__getitem__,
            node_count=len(self.bodies),
            graph_manifest_sha256=GRAPH_HASH,
            annotations_sha256=self.annotations_hash,
            populations=self.populations(),
        )

    def ttmn(self) -> TtmnPoolBinding:
        return select_ttmn_pool(**self.arguments())

    def select(self, **overrides: Any) -> BrainDisplayBinding:
        args = self.arguments()
        args["graph_hash"] = args.pop("graph_manifest_sha256")
        args["annotations_hash"] = args.pop("annotations_sha256")
        args["ttmn_pool"] = self.ttmn()
        args.update(overrides)
        return select_brain_display_binding(**args)

    def controller(self) -> FullBrainController:
        senses = bind_embodied_senses_loom_v1(self.populations())
        ttmn = self.ttmn()
        steering = select_dna01_pair(**self.arguments(), ttmn_pool=ttmn)
        sensory = bind_sensor_channels(
            channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
            index_of=self.index.__getitem__,
            node_count=len(self.bodies),
            graph_manifest_sha256=GRAPH_HASH,
        )
        motor = bind_wing_motor_pools(
            index_of=self.index.__getitem__,
            gain=8.0,
            dead_zone=0.0,
            graph_manifest_sha256=GRAPH_HASH,
            sign_overlay_identity_sha256="ef" * 32,
        )
        escape = EscapeMotorBindingV2(
            wing=motor,
            jump_pool=bind_escape_jump_pool(ttmn, gain=8.0, dead_zone=0.0),
            jump_scale=1.0,
            ttmn_pool_identity_sha256=ttmn.identity_sha256,
            encoder_identity_sha256=senses.encoder.identity_sha256,
        )
        return FullBrainController(
            run_id="brain-display-test",
            entity_id="synthetic-fly",
            configuration=FullBrainLoomV1Configuration(
                encoder_identity_sha256=senses.encoder.identity_sha256
            ),
            engine=ArrayNativeEngine(_synthetic_view(len(self.bodies))),
            sensory_binding=sensory,
            motor_binding=motor,
            embodied=senses,
            escape=escape,
            steering=steering,
        )


def _frame(sequence: int = 0) -> BrainDisplayFrame:
    identity = RunIdentity(
        "display-test", 0, "test-fly", GRAPH_HASH, "11" * 32, "22" * 32, "33" * 32, "44" * 32
    )
    return BrainDisplayFrame(
        identity,
        sequence,
        sequence,
        sequence * 5,
        (sequence + 1) * 5,
        "55" * 32,
        "neurocraft.eem.encoder.loom.v1",
        "66" * 32,
        tuple(
            PopulationMean(name, (i - 2) / 8, count)
            for i, (name, count) in enumerate(
                zip(display.POPULATION_NAMES, (71, 55, 1, 1, 1, 1), strict=True)
            )
        ),
    )


def test_binding_matches_real_selector_apis_and_pins_membership() -> None:
    space = TinySpace()
    binding = space.select()
    assert [len(pop.body_ids) for pop in binding.populations] == [71, 55, 1, 1, 1, 1]
    assert [pop.name for pop in binding.populations] == list(display.POPULATION_NAMES)
    assert binding.populations[2].body_ids == (102_000,)
    assert binding.populations[3].body_ids == (102_001,)
    assert binding.populations[4].body_ids == (804_642,)
    assert binding.populations[5].body_ids == (800_146,)
    assert binding.identity_sha256 == space.select().identity_sha256
    assert binding.identity_sha256 != replace(binding, annotations_hash="99" * 32).identity_sha256
    assert all(
        space.bodies[index] == body
        for pop in binding.populations
        for body, index in zip(pop.body_ids, pop.dense_indices, strict=True)
    )


@pytest.mark.parametrize("fault", ["missing", "extra", "side", "index", "duplicate", "short"])
def test_selection_refuses_dnp01_or_dense_join_drift(fault: str) -> None:
    space = TinySpace()
    args: dict[str, Any] = {}
    index = space.index[102_000]
    if fault == "missing":
        space.types[index] = None
    elif fault == "extra":
        space.types[space.index[104_000]] = "DNp01"
        space.sides[space.index[104_000]] = "L"
    elif fault == "side":
        space.sides[index] = "M"
    elif fault == "index":
        args["index_of"] = lambda body: space.index[body] + (body == 102_000)
    elif fault == "duplicate":
        bodies = list(space.bodies)
        bodies[index] = bodies[index + 1]
        args["body_ids"] = bodies
    else:
        args["type_column"] = space.types[:-1]
    with pytest.raises(BrainDisplayError):
        space.select(**args)


@pytest.mark.parametrize("fault", ["graph", "annotation", "lc4", "ttmn", "driven_dnp01"])
def test_selection_refuses_existing_binding_disagreement(fault: str) -> None:
    space = TinySpace()
    populations, ttmn = space.populations(), space.ttmn()
    if fault == "graph":
        populations = replace(populations, graph_manifest_sha256="00" * 32)
    elif fault == "annotation":
        ttmn = replace(ttmn, annotations_sha256="00" * 32)
    elif fault == "lc4":
        first = populations.populations[0]
        first = replace(first, dense_indices=tuple(reversed(first.dense_indices)))
        populations = replace(populations, populations=(first, *populations.populations[1:]))
    elif fault == "ttmn":
        ttmn = replace(
            ttmn,
            members=tuple(
                replace(m, soma_side="L" if m.soma_side == "R" else "R") for m in ttmn.members
            ),
        )
    else:
        extra = EmbodiedPopulation("other_drive", "synthetic", (102_000,), (space.index[102_000],))
        populations = replace(populations, populations=(*populations.populations, extra))
    with pytest.raises(BrainDisplayError):
        space.select(populations=populations, ttmn_pool=ttmn)


def _file_graph(space: TinySpace, directory: Path) -> tuple[Any, Path]:
    """Real feather bytes and digest-bound array bytes, without a full artifact."""
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.feather as feather  # type: ignore[import-untyped]

    path = directory / "annotations.feather"
    count = len(space.bodies)
    feather.write_feather(
        pa.table(
            {
                "bodyId": space.bodies,
                "type": space.types,
                "somaSide": space.sides,
                "class": [None] * count,
                "subclass": [None] * count,
            }
        ),
        path,
    )
    space.annotations_hash = sha256(path.read_bytes()).hexdigest()
    arrays = {
        "body_ids": np.asarray(space.bodies, dtype="<u8").tobytes(),
        "annotation_source_rows": np.arange(count, dtype="<u8").tobytes(),
    }
    manifest = SimpleNamespace(
        node_count=count,
        manifest_file_sha256=GRAPH_HASH,
        counts={"annotation_source_rows": count},
        source_artifacts=(
            SimpleNamespace(
                source_id=ANNOTATION_SOURCE_ID,
                byte_length=path.stat().st_size,
                sha256=space.annotations_hash,
            ),
        ),
        arrays=tuple(
            SimpleNamespace(
                name=name, byte_length=len(data), count=count, sha256=sha256(data).hexdigest()
            )
            for name, data in arrays.items()
        ),
    )
    graph = SimpleNamespace(
        manifest=manifest,
        index_of=space.index.__getitem__,
        **{name: SimpleNamespace(_mmap=data) for name, data in arrays.items()},
    )
    return graph, path


def test_loader_reverifies_annotation_and_array_digests_and_dense_join(tmp_path: Path) -> None:
    space = TinySpace()
    graph, path = _file_graph(space, tmp_path)
    expected = space.select()
    assert (
        load_brain_display_binding(graph, path, space.populations(), space.ttmn().identity_sha256)
        == expected
    )
    with pytest.raises(BrainDisplayError, match="controller's binding"):
        load_brain_display_binding(graph, path, space.populations(), "00" * 32)
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(EmbodiedSensesError, match="SHA-256 mismatch"):
        load_brain_display_binding(graph, path, space.populations(), space.ttmn().identity_sha256)
    path.write_bytes(original)
    original_array = graph.body_ids._mmap
    graph.body_ids._mmap = original_array[:-1] + bytes([original_array[-1] ^ 1])
    with pytest.raises(EmbodiedSensesError, match="mapping digest"):
        load_brain_display_binding(graph, path, space.populations(), space.ttmn().identity_sha256)
    # Even a newly declared array digest cannot make a wrong body-ID join valid.
    graph.manifest.arrays = tuple(
        replace_spec
        if spec.name != "body_ids"
        else SimpleNamespace(
            name=spec.name,
            byte_length=spec.byte_length,
            count=spec.count,
            sha256=sha256(graph.body_ids._mmap).hexdigest(),
        )
        for spec in graph.manifest.arrays
        for replace_spec in (spec,)
    )
    with pytest.raises(BrainDisplayError, match="bodyId join"):
        load_brain_display_binding(graph, path, space.populations(), space.ttmn().identity_sha256)


@pytest.mark.parametrize("mean", [float("nan"), float("inf"), -float("inf"), -1.001, 1.001, True])
def test_frame_refuses_non_model_values(mean: float) -> None:
    with pytest.raises(BrainDisplayError):
        PopulationMean("dnp01_left", mean, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"command_seq": True},
        {"observation_tick": -1},
        {"neural_step_start": 5},
        {"state_digest": "bad"},
        {"binding_hash": "A" * 64},
        {"encoder_schema": "bad\nlabel"},
        {"encoder_schema": "x" * 129},
        {"populations": ()},
    ],
)
def test_frame_requires_complete_causal_envelope(changes: dict[str, Any]) -> None:
    with pytest.raises(BrainDisplayError):
        replace(_frame(), **changes)


def test_frame_is_frozen_and_exact_agreed_schema() -> None:
    frame = _frame()
    with pytest.raises(FrozenInstanceError):
        frame.command_seq = 2  # type: ignore[misc]
    doc = json.loads(canonical_json_bytes(frame.to_dict()))
    assert set(doc) == {
        "schema",
        "presentation_only",
        "identity",
        "command_seq",
        "observation_tick",
        "neural_step_start",
        "neural_step_end",
        "state_digest",
        "encoder_schema",
        "binding_hash",
        "populations",
    }
    assert doc["schema"] == "neurocraft.brain_display.v1"
    assert doc["presentation_only"] is True
    assert set(doc["identity"]) == RunIdentity.FIELDS
    assert set(doc["populations"]) == set(display.POPULATION_NAMES)
    assert all(set(pop) == {"mean", "count"} for pop in doc["populations"].values())
    assert len(canonical_json_bytes(doc)) < display.MAX_FRAME_BYTES


@pytest.mark.parametrize("wrapped", [False, True])
def test_actual_completed_capture_preserves_commands_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
) -> None:
    space = TinySpace()
    core, control = space.controller(), space.controller()
    controller = ProgramSelectingController(core) if wrapped else core
    baseline = ProgramSelectingController(control) if wrapped else control
    publisher = AtomicBrainDisplayPublisher(tmp_path / "display.json")
    binding = space.select()
    producer = BrainDisplayProducer(binding, publisher, every_n_commands=1)
    # The producer and writer are independently bounded. Capture into a list here
    # so permitted lock-contention drops do not make this motor-invariance test flaky.
    captured: list[BrainDisplayFrame] = []
    original_offer = publisher.offer

    def record(frame: BrainDisplayFrame) -> bool:
        captured.append(frame)
        return True

    monkeypatch.setattr(publisher, "offer", record)
    command: Action | ProgramAction
    first_snapshot = b""
    try:
        for sequence in range(6):
            sensors = _sensors(forward=6.0 if sequence == 0 else 0.2, light=sequence)
            observation = _observation(core, sequence, sensors)
            command = controller.step(observation, command_seq=sequence)
            expected = baseline.step(observation, command_seq=sequence)
            before = canonical_json_bytes(command.to_dict())
            digest = core.engine.current_state().state_sha256
            assert producer.capture_completed_step(controller, command)
            assert canonical_json_bytes(command.to_dict()) == before
            assert before == canonical_json_bytes(expected.to_dict())
            assert core.engine.current_state().state_sha256 == digest
            if sequence == 0:
                first_snapshot = canonical_json_bytes(captured[0].to_dict())
        activity = core.engine.activity()
        assert len(captured) == 6
        assert canonical_json_bytes(captured[0].to_dict()) == first_snapshot
        assert original_offer(captured[-1])
        assert publisher.close()
        doc = json.loads(publisher.path.read_bytes())
        assert doc["command_seq"] == 5
        assert doc["state_digest"] == command.state_digest
        assert doc["identity"] == command.identity.to_dict()
        assert doc["encoder_schema"] == "neurocraft.eem.encoder.loom.v1"
        for pop in binding.populations:
            values = [float(activity[index]) for index in pop.dense_indices]
            assert doc["populations"][pop.name] == {
                "mean": pytest.approx(sum(values) / len(values)),
                "count": len(values),
            }
        assert any(pop["mean"] != 0 for pop in doc["populations"].values())
    finally:
        publisher.close()


@pytest.mark.parametrize(
    "fault", ["identity", "sequence", "tick", "interval", "digest", "advanced", "nan"]
)
def test_capture_faults_drop_only_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    space = TinySpace()
    core = space.controller()
    publisher = AtomicBrainDisplayPublisher(tmp_path / "display.json")
    producer = BrainDisplayProducer(space.select(), publisher, every_n_commands=1)
    command = core.step(_observation(core, 0, _sensors()), command_seq=0)
    supplied = command
    try:
        if fault == "identity":
            supplied = replace(command, identity=replace(command.identity, session_epoch=1))
        elif fault == "sequence":
            supplied = replace(command, command_seq=1)
        elif fault == "tick":
            supplied = replace(command, observation_tick=1)
        elif fault == "interval":
            supplied = replace(command, neural_step_start=1)
        elif fault == "digest":
            supplied = replace(command, state_digest="00" * 32)
        elif fault == "advanced":
            core.step(_observation(core, 1, _sensors()), command_seq=1)
        else:
            invalid = core.engine.activity().copy()
            invalid[space.index[102_000]] = np.nan
            invalid.flags.writeable = False
            monkeypatch.setattr(core.engine, "activity", lambda: invalid)
        before = canonical_json_bytes(supplied.to_dict())
        assert producer.capture_completed_step(core, supplied) is False
        assert producer.last_error is not None
        assert producer.dropped_count == 1
        assert canonical_json_bytes(supplied.to_dict()) == before
        assert not publisher.path.exists()
        monkeypatch.undo()
        # The next motor computation still works after the display refusal.
        core.step(_observation(core, 2, _sensors()), command_seq=2)
    finally:
        publisher.close()


def test_one_slot_coalesces_while_atomic_writer_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "display.json"
    old = canonical_json_bytes(_frame().to_dict())
    path.write_bytes(old)
    entered, release = threading.Event(), threading.Event()
    replace_file = os.replace
    replacements: list[int] = []

    def gate_replace(source: str, destination: Path) -> None:
        ready = json.loads(Path(source).read_bytes())
        if not replacements:
            assert path.read_bytes() == old
            entered.set()
            assert release.wait(2)
        replacements.append(ready["command_seq"])
        replace_file(source, destination)

    monkeypatch.setattr(os, "replace", gate_replace)
    publisher = AtomicBrainDisplayPublisher(path)
    try:
        assert publisher.offer(_frame(1))
        assert entered.wait(2)
        assert publisher.offer(_frame(2))
        assert publisher.offer(_frame(3))
        assert publisher.coalesced_count == 1
        assert path.read_bytes() == old
        release.set()
        assert publisher.close()
        assert replacements == [1, 3]
        assert publisher.published_count == 2
        assert json.loads(path.read_bytes())["command_seq"] == 3
        assert list(tmp_path.iterdir()) == [path]
    finally:
        release.set()
        publisher.close()


def test_failed_atomic_replace_preserves_previous_file_and_disables_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "display.json"
    old = canonical_json_bytes(_frame().to_dict())
    path.write_bytes(old)

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    publisher = AtomicBrainDisplayPublisher(path)
    assert publisher.offer(_frame(1))
    assert publisher.close() is False
    assert publisher.last_error == "OSError: synthetic disk failure"
    assert publisher.offer(_frame(2)) is False
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


def test_capture_cadence_and_failed_publisher_do_not_sample_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = TinySpace()
    core = space.controller()
    publisher = AtomicBrainDisplayPublisher(tmp_path / "display.json")
    producer = BrainDisplayProducer(space.select(), publisher)
    command = core.step(_observation(core, 1, _sensors()), command_seq=1)

    def forbidden_read() -> None:
        raise AssertionError("display must skip engine access")

    monkeypatch.setattr(core.engine, "activity", forbidden_read)
    try:
        assert producer.capture_completed_step(core, command) is False
        assert producer.last_error is None
        publisher.last_error = "synthetic failure"
        assert producer.capture_completed_step(core, replace(command, command_seq=4)) is False
        assert producer.last_error is None
    finally:
        publisher.close()


def test_capture_refuses_writable_state_and_post_copy_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = TinySpace()
    core = space.controller()
    publisher = AtomicBrainDisplayPublisher(tmp_path / "display.json")
    producer = BrainDisplayProducer(space.select(), publisher, every_n_commands=1)
    command = core.step(_observation(core, 0, _sensors()), command_seq=0)
    copy = core.engine.activity()
    mutable = copy.copy()
    monkeypatch.setattr(core.engine, "activity", lambda: mutable)
    try:
        assert producer.capture_completed_step(core, command) is False
        assert "immutable copy" in str(producer.last_error)
        state = core.engine.current_state()
        calls = 0

        def changed_digest() -> Any:
            nonlocal calls
            calls += 1
            return state if calls == 1 else SimpleNamespace(state_sha256="00" * 32)

        monkeypatch.setattr(core.engine, "activity", lambda: copy)
        monkeypatch.setattr(core.engine, "current_state", changed_digest)
        assert producer.capture_completed_step(core, command) is False
        assert "during display capture" in str(producer.last_error)
        assert not publisher.path.exists()
    finally:
        publisher.close()


def test_frame_does_not_retain_mutable_population_collection() -> None:
    frame = _frame()
    population_list = list(frame.populations)
    frozen = replace(frame, populations=population_list)  # type: ignore[arg-type]
    population_list.clear()
    assert frozen == frame


def test_publisher_offer_drops_immediately_when_slot_lock_is_busy(tmp_path: Path) -> None:
    publisher = AtomicBrainDisplayPublisher(tmp_path / "display.json")
    held, release = threading.Event(), threading.Event()

    def hold_lock() -> None:
        with publisher._condition:
            held.set()
            release.wait(2)

    lock_holder = threading.Thread(target=hold_lock)
    lock_holder.start()
    try:
        assert held.wait(2)
        # A nonblocking offer returns while the other thread still holds the lock.
        assert publisher.offer(_frame()) is False
        assert not release.is_set()
        assert publisher.dropped_count == 1
    finally:
        release.set()
        lock_holder.join(timeout=2)
        publisher.close()


def test_publisher_close_has_a_budget_even_when_writer_stalls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = threading.Event(), threading.Event()

    def stall(frame: BrainDisplayFrame) -> None:
        entered.set()
        release.wait(2)

    publisher = AtomicBrainDisplayPublisher(tmp_path / "display.json")
    monkeypatch.setattr(publisher, "_write", stall)
    try:
        assert publisher.offer(_frame())
        assert entered.wait(2)
        assert publisher.close(timeout=0.01) is False
        assert not release.is_set()
        assert publisher._thread.daemon
        assert "close budget" in str(publisher.last_error)
        assert publisher.offer(_frame()) is False
    finally:
        release.set()
        publisher.close()
