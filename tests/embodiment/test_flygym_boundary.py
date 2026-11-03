from __future__ import annotations

import neurocraft_embodiment
import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController
from neurocraft_embodiment import (
    FLYGYM_JOINT_READOUT_PARAMETERS,
    FLYGYM_MOTOR_POOL_PARAMETERS,
    FLYGYM_SENSOR_CHANNEL_PARAMETERS,
    MAX_FLYGYM_MOTOR_CHANNELS,
    MAX_FLYGYM_MOTOR_FREE_PARAMETERS,
    MAX_FLYGYM_SENSOR_CHANNELS,
    MAX_FLYGYM_SENSOR_TARGETS_TOTAL,
    MAX_FLYGYM_SENSORY_FREE_PARAMETERS,
    NEUROMECHFLY_ACTUATED_DOF_COUNT,
    NEUROMECHFLY_ACTUATED_DOF_ORDER,
    NEUROMECHFLY_ADHESION_COUNT,
    NEUROMECHFLY_DOFS_PER_LEG,
    NEUROMECHFLY_LEG_ORDER,
    FlyGymBoundaryError,
    FlyGymMotorDecoder,
    FlyGymSensoryAdapter,
    JointBinding,
)
from neurocraft_neural import MotorPool, NeuralEngine, PoolMember, SignOverlay
from neurocraft_neural.adapters import MAX_MOTOR_CHANNELS, MAX_SENSOR_CHANNELS
from neurocraft_neural.model import make_neural_state

from .conftest import (
    FIXTURE_OUTPUT_BODY_IDS,
    fixture_adhesion_bindings,
    fixture_decoder_free_parameters,
    fixture_joint_bindings,
    make_fixture_decoder,
    wide_sensor_channels,
    wide_sensor_free_parameters,
)


def zero_state(controller: ConnectomeController) -> object:
    """A verified NeuralState with exactly zero activity on the bound graph."""

    live = controller.engine.current_state()
    return make_neural_state(
        step_index=live.step_index,
        graph_identity_sha256=live.graph_identity_sha256,
        dynamics_identity_sha256=live.dynamics_identity_sha256,
        sign_overlay_sha256=live.sign_overlay_sha256,
        delay_overlay_sha256=live.delay_overlay_sha256,
        weight_overlay_sha256=live.weight_overlay_sha256,
        topology_overlay_sha256=live.topology_overlay_sha256,
        activity=[0.0] * len(live.activity),
    )


# -- budgets are siblings, never edits ----------------------------------------


def test_wide_budgets_are_strictly_separate_from_the_shipped_ones() -> None:
    assert MAX_SENSOR_CHANNELS == 32
    assert MAX_MOTOR_CHANNELS == 16
    assert MAX_FLYGYM_SENSOR_CHANNELS > MAX_SENSOR_CHANNELS
    required = NEUROMECHFLY_ACTUATED_DOF_COUNT + NEUROMECHFLY_ADHESION_COUNT
    assert required <= MAX_FLYGYM_MOTOR_CHANNELS


def test_neuromechfly_dof_ordering_is_the_documented_42() -> None:
    assert len(NEUROMECHFLY_LEG_ORDER) == 6
    assert len(NEUROMECHFLY_DOFS_PER_LEG) == 7
    assert len(NEUROMECHFLY_ACTUATED_DOF_ORDER) == NEUROMECHFLY_ACTUATED_DOF_COUNT == 42
    assert len(set(NEUROMECHFLY_ACTUATED_DOF_ORDER)) == 42
    assert NEUROMECHFLY_ACTUATED_DOF_ORDER[0] == "lf_thorax_coxa_pitch"
    assert NEUROMECHFLY_ACTUATED_DOF_ORDER[-1] == "rh_tibia_tarsus1_pitch"


# -- sensory adapter -----------------------------------------------------------


def test_wide_sensory_adapter_encodes_and_self_verifies(graph: RuntimeGraph) -> None:
    adapter = FlyGymSensoryAdapter(
        graph,
        wide_sensor_channels(),
        declared_free_parameters=wide_sensor_free_parameters(),
    )
    assert adapter.free_parameters == wide_sensor_free_parameters()
    assert adapter.schema == "neurocraft.flygym.sensory-adapter.v1"
    encoded = adapter.encode({name: 0.5 for name in adapter.channel_names})
    encoded.assert_integrity()
    assert encoded.adapter_sha256 == adapter.identity_sha256
    assert len(encoded.vector) == len(graph.nodes)


def test_wide_sensory_adapter_output_is_accepted_by_the_unmodified_engine(
    graph: RuntimeGraph,
) -> None:
    """The whole point of matching the identity material byte for byte."""

    adapter = FlyGymSensoryAdapter(
        graph,
        wide_sensor_channels(),
        declared_free_parameters=wide_sensor_free_parameters(),
    )
    engine = NeuralEngine(
        graph,
        sign_overlay=SignOverlay.coarse_presynaptic_consensus(graph),
        sensory_adapter_sha256=adapter.identity_sha256,
    )
    result = engine.step(adapter.encode({name: 0.5 for name in adapter.channel_names}))
    result.state.assert_integrity()


def test_wide_sensory_adapter_declares_its_own_capacity(graph: RuntimeGraph) -> None:
    with pytest.raises(FlyGymBoundaryError, match="free-parameter count mismatch"):
        FlyGymSensoryAdapter(graph, wide_sensor_channels(), declared_free_parameters=1)


def test_wide_sensory_adapter_refuses_non_sensory_targets(graph: RuntimeGraph) -> None:
    from neurocraft_neural import SensorChannel, SensorTarget

    channel = SensorChannel(
        name="bypass",
        unit="fraction",
        targets=(SensorTarget(FIXTURE_OUTPUT_BODY_IDS[-1], 1.0),),
    )
    with pytest.raises(FlyGymBoundaryError, match="not a named sensory node"):
        FlyGymSensoryAdapter(graph, (channel,), declared_free_parameters=5)


def test_wide_sensory_adapter_rejects_an_unknown_observation_schema(
    graph: RuntimeGraph,
) -> None:
    adapter = FlyGymSensoryAdapter(
        graph,
        wide_sensor_channels(),
        declared_free_parameters=wide_sensor_free_parameters(),
    )
    with pytest.raises(FlyGymBoundaryError, match="schema mismatch"):
        adapter.encode({"not_a_channel": 1.0})


# -- motor decoder -------------------------------------------------------------


def test_wide_motor_decoder_produces_bounded_joint_targets_and_adhesion(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    decoder = make_fixture_decoder(graph)
    controller.step(
        _observation(controller),
        command_seq=0,
    )
    actuation = decoder.decode(controller.engine.current_state())
    assert len(actuation.joint_targets_rad) == 4
    assert len(actuation.adhesion) == 2
    assert all(0.0 <= value <= 1.0 for value in actuation.adhesion)
    for index, target in enumerate(actuation.joint_targets_rad):
        neutral = decoder.neutral_pose_rad[index]
        assert abs(target - neutral) <= 0.4 + 1e-12
    assert actuation.frame.decoder_sha256 == decoder.identity_sha256


def test_wide_motor_decoder_is_stateless(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """Called twice on the same state it must return the same thing.

    A pattern generator would not: a CPG carries phase between calls. This is the
    structural guard that keeps a gait out of the decoder.
    """

    decoder = make_fixture_decoder(graph)
    controller.step(_observation(controller), command_seq=0)
    state = controller.engine.current_state()
    first = decoder.decode(state)
    second = decoder.decode(state)
    assert first.joint_targets_rad == second.joint_targets_rad
    assert first.adhesion == second.adhesion
    assert first.frame.identity_sha256 == second.frame.identity_sha256


def test_zeroed_neural_readout_produces_exactly_the_neutral_pose(
    graph: RuntimeGraph,
    controller: ConnectomeController,
) -> None:
    """AB-11-style mutant: zero the connectome readout, the body must go neutral.

    If joint targets move while the graph is silent, a policy is hiding in the
    decoder. Adhesion is the declared affine midpoint of its [0, 1] range.
    """

    decoder = make_fixture_decoder(graph)
    actuation = decoder.decode(zero_state(controller))  # type: ignore[arg-type]
    assert actuation.joint_targets_rad == decoder.neutral_pose_rad
    assert actuation.adhesion == (0.5, 0.5)


def test_wide_motor_decoder_declares_its_own_capacity(graph: RuntimeGraph) -> None:
    with pytest.raises(FlyGymBoundaryError, match="free-parameter count mismatch"):
        FlyGymMotorDecoder(
            graph,
            fixture_joint_bindings(),
            fixture_adhesion_bindings(),
            declared_free_parameters=3,
        )


def test_wide_motor_decoder_refuses_a_member_shared_between_channels(
    graph: RuntimeGraph,
) -> None:
    shared = tuple(
        JointBinding(
            dof=f"dof_{index}",
            pool=MotorPool(
                name=f"pool_{index}",
                members=(PoolMember(FIXTURE_OUTPUT_BODY_IDS[0], 1.0),),
            ),
            neutral_rad=0.0,
            half_range_rad=0.2,
        )
        for index in range(2)
    )
    with pytest.raises(FlyGymBoundaryError, match="disjoint populations"):
        FlyGymMotorDecoder(graph, shared, (), declared_free_parameters=10)


def test_wide_motor_decoder_refuses_non_output_members(graph: RuntimeGraph) -> None:
    binding = JointBinding(
        dof="bad",
        pool=MotorPool(name="bad_pool", members=(PoolMember(805_722, 1.0),)),
        neutral_rad=0.0,
        half_range_rad=0.2,
    )
    with pytest.raises(FlyGymBoundaryError, match="not motor/descending"):
        FlyGymMotorDecoder(graph, (binding,), (), declared_free_parameters=5)


def test_full_42_dof_binding_is_rejected_against_the_14_node_ci_fixture(
    graph: RuntimeGraph,
) -> None:
    """Honest boundary: the real NeuroMechFly binding needs the full runtime graph.

    The CI fixture exposes six motor/descending nodes in total, so a 42-DOF decoder
    cannot be built on it. The failure must be explicit, not a silent truncation.
    """

    bindings = tuple(
        JointBinding(
            dof=dof,
            pool=MotorPool(
                name=f"pool_{dof}",
                members=(PoolMember(FIXTURE_OUTPUT_BODY_IDS[index % 6], 1.0),),
            ),
            neutral_rad=0.0,
            half_range_rad=0.3,
        )
        for index, dof in enumerate(NEUROMECHFLY_ACTUATED_DOF_ORDER)
    )
    with pytest.raises(FlyGymBoundaryError, match="disjoint populations"):
        FlyGymMotorDecoder(graph, bindings, (), declared_free_parameters=42 * 5)


def test_the_declared_capacity_must_equal_the_computed_one(graph: RuntimeGraph) -> None:
    assert fixture_decoder_free_parameters() < MAX_FLYGYM_MOTOR_FREE_PARAMETERS
    with pytest.raises(FlyGymBoundaryError, match="free-parameter count mismatch"):
        FlyGymMotorDecoder(
            graph,
            fixture_joint_bindings(),
            fixture_adhesion_bindings(),
            declared_free_parameters=MAX_FLYGYM_MOTOR_FREE_PARAMETERS + 1,
        )


def measured_sensory_charge_per_channel(graph: RuntimeGraph) -> int:
    """What the sensory constructor actually charges per channel, beyond its targets.

    Measured from a construction rather than read off the constant, so the two are
    independent facts that a test can compare.
    """

    charges = set()
    for count in (2, 3, 5):
        channels = wide_sensor_channels(count)
        targets = sum(len(channel.targets) for channel in channels)
        adapter = FlyGymSensoryAdapter(
            graph, channels, declared_free_parameters=wide_sensor_free_parameters(count)
        )
        remainder = adapter.free_parameters - targets
        assert remainder % count == 0, (count, adapter.free_parameters, targets)
        charges.add(remainder // count)
    assert len(charges) == 1, f"the per-channel charge is not constant: {sorted(charges)}"
    return charges.pop()


def measured_motor_charges(graph: RuntimeGraph) -> tuple[int, int]:
    """Solve ``fp = members + pool_charge * pools + joint_charge * joints`` from two
    decoders of different shape. Two equations, two unknowns, no constant consulted."""

    samples: list[tuple[int, int, int, int]] = []
    for joint_count in (4, 2):
        joints = fixture_joint_bindings(joint_count)
        adhesion = fixture_adhesion_bindings(2)
        pools = [*(binding.pool for binding in joints), *(binding.pool for binding in adhesion)]
        members = sum(len(pool.members) for pool in pools)
        decoder = FlyGymMotorDecoder(
            graph,
            joints,
            adhesion,
            declared_free_parameters=fixture_decoder_free_parameters(joint_count, 2),
        )
        samples.append((decoder.free_parameters, members, len(pools), joint_count))
    (fp_a, m_a, p_a, j_a), (fp_b, m_b, p_b, j_b) = samples
    # (fp - members) = pool_charge * pools + joint_charge * joints
    left_a, left_b = fp_a - m_a, fp_b - m_b
    determinant = p_a * j_b - p_b * j_a
    assert determinant != 0, "the two decoder shapes are not independent"
    pool_charge = (left_a * j_b - left_b * j_a) / determinant
    joint_charge = (p_a * left_b - p_b * left_a) / determinant
    assert pool_charge == int(pool_charge) and joint_charge == int(joint_charge)
    return int(pool_charge), int(joint_charge)


def test_the_per_unit_charges_are_the_constants_the_maxima_are_built_from(
    graph: RuntimeGraph,
) -> None:
    """LOW-6, restated correctly after the previous version was shown to be inert.

    This test used to open with two assertions of the form
    ``MAX_FLYGYM_SENSORY_FREE_PARAMETERS == MAX_FLYGYM_SENSOR_TARGETS_TOTAL +
    FLYGYM_SENSOR_CHANNEL_PARAMETERS * MAX_FLYGYM_SENSOR_CHANNELS`` -- a verbatim
    restatement of the module lines that *define* the left-hand side. They are
    tautologies: the same constant feeds both sides, so changing it moves both and
    they cannot fail. Measured against the ``4 -> 5`` tamper, only the literal pin
    below fired. Restating the definition and calling it a derivation is worse than
    omitting it, because a reader counts it as a check, so the tautologies are gone.

    What is left has power over two different mutations.

    * The measured charges are read out of real constructions. If a counting loop is
      changed without the constant it is supposed to be built from -- the "five
      literals for three rules" defect these constants were introduced to fix, in its
      returning form -- the loop and the constant disagree here.
    * The literal pin catches any change to the maxima themselves, including one made
      consistently on both sides. It is the only assertion in this file that can.
    """

    assert measured_sensory_charge_per_channel(graph) == FLYGYM_SENSOR_CHANNEL_PARAMETERS
    assert measured_motor_charges(graph) == (
        FLYGYM_MOTOR_POOL_PARAMETERS,
        FLYGYM_JOINT_READOUT_PARAMETERS,
    )
    assert (MAX_FLYGYM_SENSORY_FREE_PARAMETERS, MAX_FLYGYM_MOTOR_FREE_PARAMETERS) == (576, 288)
    assert not hasattr(neurocraft_embodiment, "MAX_FLYGYM_BOUNDARY_FREE_PARAMETERS")


def test_a_saturating_channel_is_charged_exactly_what_a_plain_one_is(
    graph: RuntimeGraph,
) -> None:
    """LOW-6, the live half: no shipped test ever built a ``saturation="tanh"`` channel.

    ``saturation`` is a per-channel declaration the constructor branches on when
    encoding, and the counting loop must *not* branch on it -- ``tanh`` is a shape of
    the mapping, not an extra scalar a designer chose. Nothing exercised that, so four
    lines charging one extra parameter for a saturating channel produced a boundary
    stamping ``max_free_parameters: 576`` while carrying 672, with the whole suite
    green. The charge is the same for both saturations, and it is asserted here on
    channels that differ in nothing else.
    """

    from neurocraft_neural import SensorChannel, SensorTarget

    from .conftest import FIXTURE_SENSORY_BODY_IDS

    def channels(saturation: str) -> tuple[SensorChannel, ...]:
        return tuple(
            SensorChannel(
                name=f"{saturation}_channel_{index}",
                unit="fraction",
                targets=(SensorTarget(FIXTURE_SENSORY_BODY_IDS[index % 2], 0.5),),
                clip_min=0.0,
                clip_max=1.0,
                saturation=saturation,
            )
            for index in range(4)
        )

    plain = FlyGymSensoryAdapter(
        graph, channels("none"), declared_free_parameters=wide_sensor_free_parameters(4)
    )
    saturating = FlyGymSensoryAdapter(
        graph, channels("tanh"), declared_free_parameters=wide_sensor_free_parameters(4)
    )
    assert saturating.free_parameters == plain.free_parameters
    # And the declaration is still hashed, so the two boundaries are not the same one.
    assert saturating.identity_sha256 != plain.identity_sha256


def test_the_free_parameter_ceiling_refuses_a_total_the_shape_budgets_admit() -> None:
    """LOW-6: the backstop that was removed for being unreachable, put back as one.

    ``_assert_declared`` is called directly, because that is the only way to reach it:
    under every counting rule shipped today the channel/pool and target/member budgets
    reject an over-wide boundary first, so no public constructor can produce a total
    above the maximum. A check no test can reach is a check nobody has read, and the
    reason it is here at all is that the reachability argument holds only while the
    per-unit charges match the rule the maxima are derived from. Add one conditional
    charge to a loop and the shape budgets keep passing while the total walks past the
    number being hashed beside it.
    """

    from neurocraft_embodiment.flygym_boundary import _assert_declared

    over = MAX_FLYGYM_SENSORY_FREE_PARAMETERS + 1
    with pytest.raises(FlyGymBoundaryError, match="free-parameter budget exceeded"):
        _assert_declared(over, over, "probe", MAX_FLYGYM_SENSORY_FREE_PARAMETERS)
    with pytest.raises(FlyGymBoundaryError, match="free-parameter budget exceeded"):
        _assert_declared(
            MAX_FLYGYM_MOTOR_FREE_PARAMETERS + 1,
            MAX_FLYGYM_MOTOR_FREE_PARAMETERS + 1,
            "probe",
            MAX_FLYGYM_MOTOR_FREE_PARAMETERS,
        )
    # Exactly at the maximum is admitted: the ceiling is inclusive, and the sensory
    # maximum is attainable (see the test below), so an off-by-one here would make a
    # legal boundary unbuildable.
    assert (
        _assert_declared(
            MAX_FLYGYM_SENSORY_FREE_PARAMETERS,
            MAX_FLYGYM_SENSORY_FREE_PARAMETERS,
            "probe",
            MAX_FLYGYM_SENSORY_FREE_PARAMETERS,
        )
        == MAX_FLYGYM_SENSORY_FREE_PARAMETERS
    )


def test_the_per_unit_counting_rules_are_the_ones_the_constructors_charge(
    graph: RuntimeGraph,
) -> None:
    """LOW-6: the maxima and the counting loops shared no constant and no test.

    ``MAX_FLYGYM_*_FREE_PARAMETERS`` appeared only in identity material and in a test
    that restated their own arithmetic, so a change to a per-channel count would have
    left numbers already inside hashed identities quietly false. The three rules are
    named constants now, and the declaration below is built from those constants
    rather than from the constructor -- ``_assert_declared`` refuses a mismatch, so
    this fails if either side of the rule moves without the other.
    """

    channels = wide_sensor_channels(3)
    declared = sum(len(channel.targets) for channel in channels) + (
        FLYGYM_SENSOR_CHANNEL_PARAMETERS * len(channels)
    )
    adapter = FlyGymSensoryAdapter(graph, channels, declared_free_parameters=declared)
    assert adapter.free_parameters == declared

    joints = fixture_joint_bindings()
    adhesion = fixture_adhesion_bindings()
    pools = [*(binding.pool for binding in joints), *(binding.pool for binding in adhesion)]
    motor_declared = (
        sum(len(pool.members) for pool in pools)
        + FLYGYM_MOTOR_POOL_PARAMETERS * len(pools)
        + FLYGYM_JOINT_READOUT_PARAMETERS * len(joints)
    )
    decoder = FlyGymMotorDecoder(graph, joints, adhesion, declared_free_parameters=motor_declared)
    assert decoder.free_parameters == motor_declared


def test_the_sensory_maximum_is_exactly_attainable(graph: RuntimeGraph) -> None:
    """LOW-6: a ceiling nothing ever reaches is a number nobody has checked.

    The fixture exposes exactly two sensory nodes, so a channel may carry at most two
    targets -- and 96 channels x 2 targets is exactly ``MAX_FLYGYM_SENSOR_TARGETS_TOTAL``.
    The widest boundary this budget admits is therefore constructible, and its
    computed free-parameter count must land exactly on the declared maximum. If the
    per-channel rule and the maximum ever disagreed, this is where it would show.

    There is no motor twin: pools must read disjoint populations and the fixture has
    six motor/descending nodes, so the 48-pool budget is unreachable here.
    """

    from neurocraft_neural import SensorChannel, SensorTarget

    from .conftest import FIXTURE_SENSORY_BODY_IDS

    channels = tuple(
        SensorChannel(
            name=f"saturating_channel_{index}",
            unit="fraction",
            targets=tuple(SensorTarget(body_id, 0.5) for body_id in FIXTURE_SENSORY_BODY_IDS),
            clip_min=0.0,
            clip_max=1.0,
        )
        for index in range(MAX_FLYGYM_SENSOR_CHANNELS)
    )
    assert sum(len(channel.targets) for channel in channels) == MAX_FLYGYM_SENSOR_TARGETS_TOTAL
    adapter = FlyGymSensoryAdapter(
        graph, channels, declared_free_parameters=MAX_FLYGYM_SENSORY_FREE_PARAMETERS
    )
    assert adapter.free_parameters == MAX_FLYGYM_SENSORY_FREE_PARAMETERS


def test_decoder_exposes_the_orders_the_body_must_cross_validate(
    graph: RuntimeGraph,
) -> None:
    """MEDIUM-6: ``FlyGymEmbodiment`` passes these positionally, so it must see them."""

    decoder = make_fixture_decoder(graph)
    assert decoder.dof_order == tuple(binding.dof for binding in fixture_joint_bindings())
    assert decoder.leg_order == tuple(binding.leg for binding in fixture_adhesion_bindings())
    assert decoder.channel_names == tuple(
        binding.pool.name for binding in (*fixture_joint_bindings(), *fixture_adhesion_bindings())
    )
    assert len(decoder.channel_names) == decoder.channel_count


def _observation(controller: ConnectomeController) -> object:
    from neurocraft_embodiment import ReferenceArena, ReferenceEmbodiment

    body = ReferenceEmbodiment(ReferenceArena())
    return body.observe(identity=controller.identity, sequence=0)
