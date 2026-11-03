"""Exact source binding and actual five-step synthetic-engine event inputs."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
from neurocraft_controller.config import canonical_sha256
from neurocraft_controller.controller import ControllerIntegrityError
from neurocraft_controller.events_senses import (
    SPECS,
    EventEncoderBinding,
    EventInputState,
    EventPopulation,
    EventPopulationBinding,
    EventSensesError,
    select_event_populations,
)
from neurocraft_controller.full_brain_controller import (
    FullBrainController,
    FullBrainEventsConfiguration,
    FullBrainLoomV2Configuration,
    FullBrainProfileError,
)
from neurocraft_controller.loom_senses_v2 import bind_embodied_senses_loom_v2
from neurocraft_neural.full_controls import FullControlKind, build_control_view
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.spmv_kernel import NumpyReduceatKernel
from neurocraft_protocol import EventCounter, EventSample, SensorFrameV2, SensorFrameV3

from scripts.probe_loom_encoder import sensors
from tests.controller.test_embodied_senses import _observation
from tests.controller.test_steering_readout import _loom_controllers


def sample(*, food=0, brush=0, attack=0, tick=0, contact=False, near=False):
    return EventSample(
        contact,
        near or contact,
        EventCounter(food, tick if food else None),
        EventCounter(brush, tick if brush else None),
        EventCounter(attack, tick if attack else None),
    )


def frame(events=None, **changes):
    physical = replace(sensors(), **changes)
    return SensorFrameV3.from_typed_frame(
        SensorFrameV2.from_sensor_frame(physical, (0,) * 6), events or sample()
    )


def fixture_populations(core):
    # Exact production counts, explicitly synthetic identities and locations.
    # Outputs precede afferents in the fixture's directed cycle so real dynamics
    # carry pulse activity into the readouts; no fabricated readout values.
    starts = {"mn9": 500, "food": 502, "antenna": 700, "groom": 713}
    populations = []
    for name, _, _, counts in SPECS:
        start = starts[name]
        members = tuple(
            (9_000_000 + start + j, start + j, side)
            for side, first, count in (("L", 0, counts[0]), ("R", counts[0], counts[1]))
            for j in range(first, first + count)
        )
        populations.append(EventPopulation(name, members))
    return EventPopulationBinding(
        core.identity.graph_hash,
        core.embodied_senses.populations.annotations_sha256,
        core.engine.view.node_count,
        tuple(populations),
    )


def controllers(mode="faithful"):
    old, _ = _loom_controllers()
    embodied = bind_embodied_senses_loom_v2(old.embodied_senses.populations)
    escape = replace(old.escape_binding, encoder_identity_sha256=embodied.encoder.identity_sha256)
    base_config = FullBrainLoomV2Configuration(
        mode=mode, encoder_identity_sha256=embodied.encoder.identity_sha256
    )
    control = None
    view = old.engine.view
    if mode != "faithful":
        control = build_control_view(
            view,
            kind=FullControlKind(mode),
            seed=7,
            artifact_manifest_sha256=old.identity.graph_hash,
            raw_counts=np.tile([3, 5], view.node_count),
            edge_signs=np.tile([1, -1], view.node_count),
        )
        view = control.view
    common = dict(
        run_id=old.identity.run_id,
        entity_id=old.identity.entity_id,
        sensory_binding=old._sensory_binding,
        motor_binding=escape.wing,
        embodied=embodied,
        escape=escape,
        steering=old.steering_binding,
        control=control,
    )
    base = FullBrainController(
        configuration=base_config,
        engine=ArrayNativeEngine(view, kernel=NumpyReduceatKernel()),
        **common,
    )
    binding = EventEncoderBinding(
        fixture_populations(base), embodied.encoder.identity_sha256, escape.identity_sha256, ""
    )
    config = FullBrainEventsConfiguration(
        **{f.name: getattr(base_config, f.name) for f in fields(base_config) if f.name != "schema"},
        event_encoder_sha256=binding.identity_sha256,
    )
    candidate = FullBrainController(
        configuration=config,
        event_encoder=binding,
        engine=ArrayNativeEngine(view, kernel=NumpyReduceatKernel()),
        **common,
    )
    return candidate, base


def test_verified_selection_uses_root_side_for_sources_and_soma_side_for_outputs():
    core, _ = controllers()
    expected = core.event_encoder.populations
    count = expected.node_count
    bodies = list(range(1, count + 1))
    columns = {
        name: [None] * count
        for name in ("class", "subclass", "superclass", "type", "somaSide", "rootSide")
    }
    for pop, (_, terms, side_column, _) in zip(expected.populations, SPECS, strict=True):
        for body, index, side in pop.members:
            bodies[index] = body
            for column, value in terms:
                columns[column][index] = value
            columns[side_column][index] = side
            # The other side is deliberately opposite: a wrong selector fails.
            other = "somaSide" if side_column == "rootSide" else "rootSide"
            columns[other][index] = "R" if side == "L" else "L"
    args = dict(
        columns=columns,
        body_ids=bodies,
        index_of=dict(zip(bodies, range(count), strict=True)).__getitem__,
        graph_hash=expected.graph_hash,
        annotations_hash=expected.annotations_hash,
        node_count=count,
    )
    actual = select_event_populations(**args)
    assert actual == expected
    assert actual.identity_sha256 == expected.identity_sha256
    with pytest.raises(EventSensesError, match="dense order"):
        select_event_populations(**{**args, "index_of": lambda _: 0})
    columns["rootSide"][502] = "R"
    with pytest.raises(EventSensesError, match="counts"):
        select_event_populations(**args)


def test_counter_baseline_monotonicity_expiry_and_two_sources_clamped_to_one_pulse():
    core, _ = controllers()
    encoder = EventInputState(core.event_encoder)
    initial = sample(food=123, brush=99, tick=10)
    assert encoder.prepare(initial, 20).baseline_absorbed
    assert not encoder.prepare(initial, 21).food
    both = sample(food=125, brush=102, attack=100, tick=22)
    pulse = encoder.prepare(both, 30)
    assert (pulse.food, pulse.groom, pulse.food_age, pulse.groom_age) == (True, True, 8, 8)
    assert (pulse.food_delta, pulse.brush_delta, pulse.attack_delta) == (2, 3, 100)
    vector = np.zeros(core.engine.view.node_count)
    encoder.apply(vector, pulse)
    assert vector.max() == 1 and np.count_nonzero(vector) == 228
    assert not encoder.prepare(both, 31).groom
    stale = sample(food=126, brush=103, attack=101, tick=22)
    assert not encoder.prepare(stale, 32).food
    with pytest.raises(EventSensesError, match="backwards"):
        encoder.prepare(both, 33)
    with pytest.raises(EventSensesError, match="without a new count"):
        encoder.prepare(sample(food=126, brush=103, attack=101, tick=23), 33)
    encoder.reset()
    assert encoder.prepare(both, 34).baseline_absorbed


def test_real_neural_input_is_old_background_plus_exact_pulse_and_next_tick_has_no_pulse(
    monkeypatch,
):
    candidate, baseline = controllers()
    actual_vectors, baseline_vectors = [], []
    for core, vectors in ((candidate, actual_vectors), (baseline, baseline_vectors)):
        step = core.engine.step

        def capture(*, sensory, sensory_provenance_sha256, step=step, vectors=vectors):
            vectors.append(sensory.copy())
            return step(sensory=sensory, sensory_provenance_sha256=sensory_provenance_sha256)

        monkeypatch.setattr(core.engine, "step", capture)
    for tick in range(6):
        events = sample(food=int(tick >= 4), brush=int(tick >= 4), tick=4, contact=True)
        candidate_frame = frame(events, horizontal_collision=tick >= 4)
        old_frame = SensorFrameV2.from_sensor_frame(candidate_frame, (0,) * 6)
        candidate.step(_observation(candidate, tick, candidate_frame), command_seq=tick)
        baseline.step(_observation(baseline, tick, old_frame), command_seq=tick)
    increments = np.zeros(candidate.engine.view.node_count)
    for name in ("food", "groom"):
        increments[list(candidate.event_encoder.populations.population(name).indices)] = 1
    for step, (actual, background) in enumerate(zip(actual_vectors, baseline_vectors, strict=True)):
        np.testing.assert_array_equal(actual, background + (increments if 20 <= step < 25 else 0))
    assert candidate.last_event_readout.mn9_raw != baseline.engine.activity()[500:502].mean()
    assert candidate.last_event_readout.pulse.food is False
    assert candidate.identity.dynamics_hash == baseline.identity.dynamics_hash
    assert candidate.identity.decoder_hash == baseline.identity.decoder_hash
    assert candidate.identity.adapter_hash != baseline.identity.adapter_hash
    assert candidate.identity.observation_schema_hash != baseline.identity.observation_schema_hash
    descriptor = candidate.event_encoder.to_identity_material()
    assert descriptor["component_decoder_sha256"] == baseline.identity.decoder_hash
    assert "does not validate" in descriptor["calibration_scope"]


def test_reset_epoch_discards_history_and_old_events_and_reproduces_actual_state():
    core, _ = controllers()
    initial = core.identity.session_epoch
    sequence = [frame(sample(food=int(tick >= 4), tick=4, contact=True)) for tick in range(7)]
    states = []
    for pass_index in range(2):
        if pass_index:
            core.reset_session(session_epoch=initial + 9)
            assert core.last_event_readout is None
        current = []
        for tick, sensor_frame in enumerate(sequence):
            action = core.step(_observation(core, tick, sensor_frame), command_seq=tick)
            current.append(action.state_digest)
        states.append(current)
    assert states[0] == states[1]
    core.reset_session(session_epoch=initial + 10)
    core.step(_observation(core, 0, frame(sample(food=10, tick=0))), command_seq=0)
    assert core.last_event_readout.pulse.baseline_absorbed
    assert not core.last_event_readout.pulse.food


def test_actual_no_input_engine_zeros_every_old_and_event_assignment():
    core, _ = controllers("no_input")
    for tick in range(7):
        obs = _observation(
            core,
            tick,
            frame(
                sample(food=int(tick >= 4), brush=int(tick >= 4), tick=4, contact=True),
                horizontal_collision=tick >= 4,
            ),
        )
        core.step(obs, command_seq=tick)
        np.testing.assert_array_equal(core.engine.activity(), 0)
        assert core.last_event_readout.mn9_raw == core.last_event_readout.antenna_raw == 0
    assert core.last_evidence.encoder_stimulation_before_control is not None


def test_invalid_event_quality_and_missing_pair_refuse_before_neural_step():
    core, base = controllers()
    obs = _observation(core, 0, frame())
    for key in ("clamped_fields", "missing_fields"):
        with pytest.raises(ControllerIntegrityError, match=r"clamped|missing"):
            core.step(replace(obs, **{key: ("events",)}), command_seq=0)
        assert core.engine.step_index == 0
    with pytest.raises(FullBrainProfileError, match="paired"):
        FullBrainController(
            run_id="bad",
            entity_id="fly",
            engine=base.engine,
            configuration=base.configuration,
            sensory_binding=base._sensory_binding,
            motor_binding=base.escape_binding.wing,
            embodied=base.embodied_senses,
            escape=base.escape_binding,
            event_encoder=core.event_encoder,
        )
    assert (
        canonical_sha256(core.event_encoder.to_identity_material())
        == core.event_encoder.identity_sha256
    )


def test_loader_reads_real_arrow_source_root_and_superclass_and_refuses_changed_bytes(
    tmp_path, monkeypatch
):
    from hashlib import sha256
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.feather as feather
    from neurocraft_controller import events_senses

    core, _ = controllers()
    expected = core.event_encoder.populations
    count = expected.node_count
    bodies = list(range(1, count + 1))
    columns = {
        name: [None] * count
        for name in ("class", "subclass", "superclass", "type", "somaSide", "rootSide")
    }
    for pop, (_, terms, side_column, _) in zip(expected.populations, SPECS, strict=True):
        for body, index, side in pop.members:
            bodies[index] = body
            for column, value in terms:
                columns[column][index] = value
            columns[side_column][index] = side
    table = pa.table({"bodyId": bodies, **columns})
    path = tmp_path / "annotations.feather"
    feather.write_feather(table, path)
    raw = path.read_bytes()
    source = SimpleNamespace(
        source_id="annotations", byte_length=len(raw), sha256=sha256(raw).hexdigest()
    )
    graph = SimpleNamespace(
        manifest=SimpleNamespace(
            source_artifacts=[source],
            counts={"annotation_source_rows": count},
            node_count=count,
            manifest_file_sha256=expected.graph_hash,
        ),
        index_of=dict(zip(bodies, range(count), strict=True)).__getitem__,
    )
    arrays = {
        "annotation_source_rows": np.arange(count, dtype=np.uint64),
        "body_ids": np.array(bodies, dtype=np.uint64),
    }
    monkeypatch.setattr(
        events_senses, "_verified_graph_array", lambda _graph, name, _dtype: arrays[name]
    )
    loaded = events_senses.load_event_populations(graph, path)
    assert loaded.populations == expected.populations
    assert loaded.annotations_hash == source.sha256
    arrays["annotation_source_rows"][0] = 1
    with pytest.raises(EventSensesError, match="repeated"):
        events_senses.load_event_populations(graph, path)
    path.write_bytes(raw + b"corrupted")
    with pytest.raises(EventSensesError, match="digest"):
        events_senses.load_event_populations(graph, path)
