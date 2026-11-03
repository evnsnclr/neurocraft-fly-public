"""Tests for the DNa01 steering-pair selection (``steering_readout.py``).

All CI-safe: synthetic annotation frames and synthetic views only -- no artifact, no
feather tables, no calibration document. The helpers come from ``test_embodied_senses``
and ``test_escape_reflex`` so every lane shares one synthetic node space, and the wing /
eem / escape identity pins asserted there remain the regression gate proving this additive
lane moved nothing.

Nothing here measures or claims that DNa01 steers anything. The selector picks two labeled
bodies and hashes the selection; the mapping that reads them is declared engineering and
lives in ``program_policy.py``.
"""

from __future__ import annotations

import pytest
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.escape_reflex import select_ttmn_pool
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_controller.full_brain_controller import (
    EscapeMotorBindingV2,
    FullBrainController,
    FullBrainLoomV1Configuration,
    FullBrainProfileError,
    bind_escape_jump_pool,
    bind_sensor_channels,
    bind_wing_motor_pools,
)
from neurocraft_controller.loom_senses import bind_embodied_senses_loom_v1
from neurocraft_controller.steering_readout import (
    DNA01_SELECTOR,
    EXPECTED_DNA01_NODES,
    STEERING_DNA01_PAIR_SCHEMA,
    SteeringPairBinding,
    SteeringReadoutError,
    select_dna01_pair,
)
from neurocraft_neural.full_engine import ArrayNativeEngine

from tests.controller.test_embodied_senses import (
    ANNOTATIONS_SHA,
    GRAPH_SHA,
    OVERLAY_SHA,
    FakeFrame,
    _synthetic_view,
)
from tests.controller.test_escape_reflex import (
    TTMN_LEFT_BODY,
    TTMN_RIGHT_BODY,
    _ttmn_frame,
)

#: The census DNa01 bodies (``docs/research/readout-census-2026-09-02.md``): a clean
#: 1 L / 1 R pair, disjoint from every runtime population. Reused verbatim here.
DNA01_LEFT_BODY = 10_442
DNA01_RIGHT_BODY = 10_760


def _dna01_frame() -> FakeFrame:
    """The TTMn synthetic frame extended with the two census DNa01 rows."""

    frame = _ttmn_frame()
    for body, side in ((DNA01_LEFT_BODY, "L"), (DNA01_RIGHT_BODY, "R")):
        frame.types.append("DNa01")
        frame.classes.append(None)
        frame.subclasses.append(None)
        frame.sides.append(side)
        frame.body_ids.append(body)
        frame.rows += 1
    return frame


def _select(frame: FakeFrame, *, node_count: int | None = None, **overrides: object) -> object:
    count = node_count or frame.rows
    populations = frame.select(node_count=count)
    ttmn = select_ttmn_pool(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=frame.index.__getitem__,
        node_count=count,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=populations,
    )
    arguments: dict[str, object] = {
        "type_column": frame.types,
        "soma_side_column": frame.sides,
        "body_ids": frame.body_ids,
        "index_of": frame.index.__getitem__,
        "node_count": count,
        "graph_manifest_sha256": GRAPH_SHA,
        "annotations_sha256": ANNOTATIONS_SHA,
        "populations": populations,
        "ttmn_pool": ttmn,
    }
    arguments.update(overrides)
    return select_dna01_pair(**arguments)  # type: ignore[arg-type]


def _clean_ttmn_pool(frame: FakeFrame, *, node_count: int) -> object:
    """The TTMn pair selected from a clean frame, for tests that doctor another frame."""

    return select_ttmn_pool(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=frame.index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=frame.select(node_count=node_count),
    )


def _binding(frame: FakeFrame | None = None) -> SteeringPairBinding:
    binding = _select(frame or _dna01_frame())
    assert isinstance(binding, SteeringPairBinding)
    return binding


# ------------------------------------------------------------------ selection


def test_dna01_selection_matches_census_membership_and_schema() -> None:
    binding = _binding()
    assert len(binding.members) == EXPECTED_DNA01_NODES == 2
    assert tuple(member.body_id for member in binding.members) == (
        DNA01_LEFT_BODY,
        DNA01_RIGHT_BODY,
    )
    assert binding.left.body_id == DNA01_LEFT_BODY
    assert binding.right.body_id == DNA01_RIGHT_BODY
    assert {member.soma_side for member in binding.members} == {"L", "R"}
    material = binding.to_identity_material()
    assert material["schema"] == STEERING_DNA01_PAIR_SCHEMA
    assert material["selector"] == DNA01_SELECTOR == "type == 'DNa01'"
    assert len(binding.identity_sha256) == 64


def test_dna01_identity_is_stable_and_sensitive_to_membership() -> None:
    first = _binding()
    assert first.identity_sha256 == _binding().identity_sha256
    moved = SteeringPairBinding(
        graph_manifest_sha256=first.graph_manifest_sha256,
        annotations_sha256=first.annotations_sha256,
        node_count=first.node_count + 1,
        members=first.members,
    )
    assert moved.identity_sha256 != first.identity_sha256


# --------------------------------------------------------------- fail closed


def test_dna01_selection_refuses_count_drift() -> None:
    missing = _dna01_frame()
    missing.types[missing.types.index("DNa01")] = "DNa01_like"
    with pytest.raises(SteeringReadoutError, match="DNa01 count drifted"):
        _select(missing)
    extra = _dna01_frame()
    visual_row = extra.classes.index("visual")
    extra.classes[visual_row] = None
    extra.types[visual_row] = "DNa01"
    extra.sides[visual_row] = "L"
    # The doctored frame now has 6090 visual rows, so hold the population and TTMn
    # bindings to a clean frame; the DNa01 count gate is the one under test.
    clean = _dna01_frame()
    populations = clean.select(node_count=extra.rows)
    with pytest.raises(SteeringReadoutError, match="DNa01 count drifted"):
        select_dna01_pair(
            type_column=extra.types,
            soma_side_column=extra.sides,
            body_ids=extra.body_ids,
            index_of=extra.index.__getitem__,
            node_count=extra.rows,
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
            ttmn_pool=_clean_ttmn_pool(clean, node_count=extra.rows),
        )


def test_dna01_selection_refuses_side_drift() -> None:
    both_left = _dna01_frame()
    both_left.sides[both_left.body_ids.index(DNA01_RIGHT_BODY)] = "L"
    with pytest.raises(SteeringReadoutError, match="DNa01 side split drifted"):
        _select(both_left)


def test_dna01_selection_refuses_a_midline_side() -> None:
    midline = _dna01_frame()
    midline.sides[midline.body_ids.index(DNA01_RIGHT_BODY)] = "M"
    with pytest.raises(SteeringReadoutError, match="outside the declared L/R split"):
        _select(midline)


def test_dna01_selection_refuses_wing_readout_overlap() -> None:
    """The declared 52 wing bodies are the readout the projection already reads."""

    with pytest.raises(SteeringReadoutError, match="wing-motor readout"):
        _select(
            _dna01_frame(),
            wing_body_ids=frozenset(
                (DNA01_LEFT_BODY, *LEFT_WING_MOTOR_BODY_IDS, *RIGHT_WING_MOTOR_BODY_IDS)
            ),
        )


def test_dna01_selection_refuses_ttmn_pool_overlap() -> None:
    """A steering readout that is also the jump readout is one readout, not two."""

    # A duplicate body ID would break the dense-index join before the overlap gate, so
    # drive the gate directly with a synthesized TTMn membership instead.
    with pytest.raises(SteeringReadoutError, match="TTMn jump"):
        _select(
            _dna01_frame(),
            wing_body_ids=frozenset(),
            faithful_bodies=frozenset(),
            ttmn_pool=_ttmn_pool_containing(DNA01_RIGHT_BODY),
        )
    # Sanity: the real census pairs are disjoint, which is why the gate has to be forced.
    assert {TTMN_LEFT_BODY, TTMN_RIGHT_BODY}.isdisjoint({DNA01_LEFT_BODY, DNA01_RIGHT_BODY})


def _ttmn_pool_containing(body_id: int) -> object:
    """A TTMn binding whose membership deliberately captures ``body_id``."""

    from neurocraft_controller.escape_reflex import TtmnMember, TtmnPoolBinding

    members = tuple(
        sorted(
            (
                TtmnMember(body_id=body_id, dense_index=0, soma_side="L"),
                TtmnMember(body_id=TTMN_RIGHT_BODY, dense_index=1, soma_side="R"),
            ),
            key=lambda member: member.body_id,
        )
    )
    frame = _dna01_frame()
    return TtmnPoolBinding(
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        node_count=frame.rows,
        members=members,
    )


def test_dna01_selection_refuses_faithful_entry_overlap() -> None:
    with pytest.raises(SteeringReadoutError, match="faithful sensory entry"):
        _select(_dna01_frame(), faithful_bodies=frozenset((DNA01_RIGHT_BODY,)))


def test_dna01_selection_refuses_a_population_member() -> None:
    """A body that already receives an encoder drive cannot also be the steering readout.

    Driven directly through a doctored population binding, because a frame that put a
    DNa01 body inside a population would have to duplicate a body ID, and the dense-index
    join refuses that first.
    """

    from dataclasses import replace as dataclass_replace

    frame = _dna01_frame()
    populations = frame.select(node_count=frame.rows)
    lc4 = populations.population("lc4_left")
    captured = dataclass_replace(
        populations,
        populations=tuple(
            dataclass_replace(
                population,
                body_ids=(*population.body_ids, DNA01_RIGHT_BODY),
                dense_indices=(
                    *population.dense_indices,
                    frame.index[DNA01_RIGHT_BODY],
                ),
            )
            if population.name == lc4.name
            else population
            for population in populations.populations
        ),
    )
    with pytest.raises(SteeringReadoutError, match="sensory population 'lc4_left'"):
        _select(frame, populations=captured)


def test_dna01_selection_refuses_a_mismatched_binding_provenance() -> None:
    frame = _dna01_frame()
    populations = frame.select(node_count=frame.rows)
    ttmn = select_ttmn_pool(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=frame.index.__getitem__,
        node_count=frame.rows,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=populations,
    )
    with pytest.raises(SteeringReadoutError, match="different artifact"):
        select_dna01_pair(
            type_column=frame.types,
            soma_side_column=frame.sides,
            body_ids=frame.body_ids,
            index_of=frame.index.__getitem__,
            node_count=frame.rows,
            graph_manifest_sha256="ee" * 32,
            annotations_sha256=ANNOTATIONS_SHA,
            populations=populations,
            ttmn_pool=ttmn,
        )


def test_steering_pair_binding_refuses_a_one_sided_pair() -> None:
    from neurocraft_controller.steering_readout import SteeringMember

    with pytest.raises(SteeringReadoutError, match="exactly one L and one R"):
        SteeringPairBinding(
            graph_manifest_sha256=GRAPH_SHA,
            annotations_sha256=ANNOTATIONS_SHA,
            node_count=8,
            members=(
                SteeringMember(body_id=1, dense_index=0, soma_side="L"),
                SteeringMember(body_id=2, dense_index=1, soma_side="L"),
            ),
        )


# ------------------------------------------------------- controller integration


def _loom_controllers() -> tuple[FullBrainController, FullBrainController]:
    """A matched (with-steering, without-steering) pair over ONE synthetic node space.

    Identical view, sensory binding, motor binding, encoder, and escape binding: the ONLY
    difference is whether the DNa01 pair is bound. Their identities must be equal.
    """

    frame = _dna01_frame()
    index = frame.index
    cursor = frame.rows
    for body in LEFT_WING_MOTOR_BODY_IDS + RIGHT_WING_MOTOR_BODY_IDS:
        index[body] = cursor
        cursor += 1
    node_count = cursor
    populations = frame.select(node_count=node_count)
    senses = bind_embodied_senses_loom_v1(populations)
    ttmn = select_ttmn_pool(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=populations,
    )
    steering = select_dna01_pair(
        type_column=frame.types,
        soma_side_column=frame.sides,
        body_ids=frame.body_ids,
        index_of=index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
        annotations_sha256=ANNOTATIONS_SHA,
        populations=populations,
        ttmn_pool=ttmn,
    )
    sensory = bind_sensor_channels(
        channels=ConnectomeController._sensor_channels(ControllerConfiguration()),
        index_of=index.__getitem__,
        node_count=node_count,
        graph_manifest_sha256=GRAPH_SHA,
    )
    motor = bind_wing_motor_pools(
        index_of=index.__getitem__,
        gain=8.0,
        dead_zone=0.0,
        graph_manifest_sha256=GRAPH_SHA,
        sign_overlay_identity_sha256=OVERLAY_SHA,
    )
    escape = EscapeMotorBindingV2(
        wing=motor,
        jump_pool=bind_escape_jump_pool(ttmn, gain=8.0, dead_zone=0.0),
        jump_scale=1.0,
        ttmn_pool_identity_sha256=ttmn.identity_sha256,
        encoder_identity_sha256=senses.encoder.identity_sha256,
    )

    def build(bind_steering: bool) -> FullBrainController:
        return FullBrainController(
            run_id="steering-test",
            entity_id="test-fly",
            configuration=FullBrainLoomV1Configuration(
                encoder_identity_sha256=senses.encoder.identity_sha256
            ),
            engine=ArrayNativeEngine(_synthetic_view(node_count)),
            sensory_binding=sensory,
            motor_binding=motor,
            embodied=senses,
            escape=escape,
            steering=steering if bind_steering else None,
        )

    return build(True), build(False)


def test_steering_readout_does_not_rotate_the_run_identity() -> None:
    """Identity-rotation pin: binding DNa01 moves no ``RunIdentity`` field.

    The binding is read by the selection policy alone; folding it into ``decoder_hash``
    would rotate every readout calibration's cross-binding for a readout the decoder never
    sees. ``policy_hash`` carries it instead.
    """

    with_steering, without_steering = _loom_controllers()
    assert with_steering.identity == without_steering.identity
    assert with_steering.identity.decoder_hash == without_steering.identity.decoder_hash
    assert with_steering.identity.adapter_hash == without_steering.identity.adapter_hash
    assert (
        with_steering.configuration.identity_sha256
        == without_steering.configuration.identity_sha256
    )
    assert with_steering.steering_binding is not None
    assert without_steering.steering_binding is None
    # The evidence subset is unchanged too: the steering members are read straight off the
    # committed activity vector rather than copied into per-step evidence.
    assert with_steering._evidence_body_ids == without_steering._evidence_body_ids


def test_steering_readout_means_returns_left_then_right() -> None:
    with_steering, without_steering = _loom_controllers()
    left, right = with_steering.steering_readout_means()
    assert isinstance(left, float) and isinstance(right, float)
    binding = with_steering.steering_binding
    assert binding is not None
    activity = with_steering.engine.activity()
    assert left == float(activity[binding.left.dense_index])
    assert right == float(activity[binding.right.dense_index])
    with pytest.raises(FullBrainProfileError, match="steering_readout_means requires"):
        without_steering.steering_readout_means()


def test_controller_refuses_a_steering_binding_from_another_artifact() -> None:
    with_steering, _ = _loom_controllers()
    binding = with_steering.steering_binding
    assert binding is not None
    foreign = SteeringPairBinding(
        graph_manifest_sha256="ee" * 32,
        annotations_sha256=binding.annotations_sha256,
        node_count=binding.node_count,
        members=binding.members,
    )
    with pytest.raises(FullBrainProfileError, match="different artifact"):
        FullBrainController(
            run_id="steering-test",
            entity_id="test-fly",
            configuration=with_steering.configuration,
            engine=ArrayNativeEngine(_synthetic_view(binding.node_count)),
            sensory_binding=with_steering._sensory_binding,
            motor_binding=with_steering._motor_binding,
            embodied=with_steering.embodied_senses,
            escape=with_steering.escape_binding,
            steering=foreign,
        )


def test_from_artifact_refuses_steering_without_the_escape_and_embodied_bindings() -> None:
    """Argument surface only -- the refusal fires before any artifact byte is read."""

    with pytest.raises(FullBrainProfileError, match="steering readout requires"):
        FullBrainController.from_artifact(
            run_id="steering-test",
            entity_id="test-fly",
            artifact_directory=__import__("pathlib").Path("/nonexistent/artifact"),
            annotations_path=__import__("pathlib").Path("/nonexistent/annotations.feather"),
            steering_readout=True,
        )


def test_binding_the_pair_leaves_the_embodied_populations_alone() -> None:
    """The steering pair is disjoint from every driven population, by refusal above; here
    the positive half: binding it changes no population membership."""

    with_steering, without_steering = _loom_controllers()
    left = with_steering.embodied_senses
    right = without_steering.embodied_senses
    assert left is not None and right is not None
    assert left.populations.identity_sha256 == right.populations.identity_sha256
    assert left.encoder.identity_sha256 == right.encoder.identity_sha256
