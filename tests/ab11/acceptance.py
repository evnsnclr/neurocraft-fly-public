"""Acceptance predicates restated exactly as the named tests assert them.

Each function here is the boolean form of one existing acceptance test's assertion
block. Restating them as predicates is what lets a single AB-11 test evaluate the same
claim twice: once against the faithful baseline (it must hold) and once against the
mutant (it must fail). The wording, ordering, and constants are kept identical to the
originals so the two stay comparable under review.
"""

from __future__ import annotations

from neurocraft_controller import ConnectomeController
from neurocraft_protocol import Action, MotorVector

from tests.controller.conftest import make_observation, make_sensors

PREFIX_SENSORS = make_sensors(forward=6.0, left=1.5, right=6.0, light=15)
BLOCKED_SENSORS = make_sensors(
    forward=1.499980129301548,
    left=1.5,
    right=6.0,
    light=15,
)


def advance(
    controller: ConnectomeController,
    *,
    count: int,
    zero_input: bool = False,
) -> list[Action]:
    """Drive ``count`` ticks of constant sensory input, as the named tests do."""

    sensors = (
        make_sensors(forward=0.0, left=0.0, right=0.0, light=0) if zero_input else make_sensors()
    )
    return [
        controller.step(make_observation(controller, index, sensors=sensors), command_seq=index)
        for index in range(count)
    ]


def redecoded_motors(controller: ConnectomeController) -> MotorVector:
    """Independently recompute the primitive motors from the committed neural state.

    This is the readout half of the faithful chain executed a second time from the
    controller's own committed state, so it can only agree with an issued action whose
    motors really are that state's decoder output.
    """

    frame = controller.decoder.decode(controller.engine.current_state())
    return ConnectomeController._project_motor_frame(frame)


def real_input_acceptance_holds(
    driven: ConnectomeController,
    zero: ConnectomeController,
) -> bool:
    """Predicate of ``test_real_input_reaches_neural_state_and_primitive_motor``."""

    driven_actions = advance(driven, count=4)
    zero_actions = advance(zero, count=4, zero_input=True)
    driven_evidence = driven.last_evidence
    zero_evidence = zero.last_evidence
    if driven_evidence is None or zero_evidence is None:
        return False
    return all(
        (
            driven_evidence.feature_values_sha256 != zero_evidence.feature_values_sha256,
            driven_evidence.encoded_stimulation_sha256 != zero_evidence.encoded_stimulation_sha256,
            driven_evidence.state_after_sha256 != zero_evidence.state_after_sha256,
            driven_evidence.motor_frame.neural_state_sha256 == driven_evidence.state_after_sha256,
            driven_actions[-1].state_digest == driven_evidence.state_after_sha256,
            not driven_actions[-1].motors.is_neutral,
            zero_actions[-1].motors.is_neutral,
            driven_actions[-1].neural_step_end - driven_actions[-1].neural_step_start == 5,
            len(driven_evidence.neural_step_records) == 5,
        )
    )


def sign_ablation_acceptance_holds(
    faithful: ConnectomeController,
    zero_sign: ConnectomeController,
) -> bool:
    """Predicate of ``test_fixture_sign_ablation_removes_recurrent_sensor_to_motor_effect``."""

    faithful_action = zero_action = None
    for sequence, sensors in enumerate((PREFIX_SENSORS, PREFIX_SENSORS, BLOCKED_SENSORS)):
        faithful_action = faithful.step(
            make_observation(faithful, sequence, sensors=sensors),
            command_seq=sequence,
        )
        zero_action = zero_sign.step(
            make_observation(zero_sign, sequence, sensors=sensors),
            command_seq=sequence,
        )
    assert faithful_action is not None and zero_action is not None
    return all(
        (
            not faithful_action.motors.is_neutral,
            zero_action.motors.is_neutral,
            faithful_action.state_digest != zero_action.state_digest,
        )
    )


def ray_perturbation_acceptance_holds(
    actual: ConnectomeController,
    open_counterfactual: ConnectomeController,
) -> bool:
    """Predicate of ``test_fixture_forward_ray_perturbation_propagates_from_identical_state``."""

    for sequence in range(2):
        actual.step(
            make_observation(actual, sequence, sensors=PREFIX_SENSORS),
            command_seq=sequence,
        )
        open_counterfactual.step(
            make_observation(open_counterfactual, sequence, sensors=PREFIX_SENSORS),
            command_seq=sequence,
        )
    if actual.engine.current_state() != open_counterfactual.engine.current_state():
        return False
    actual_action = actual.step(
        make_observation(actual, 2, sensors=BLOCKED_SENSORS),
        command_seq=2,
    )
    open_action = open_counterfactual.step(
        make_observation(open_counterfactual, 2, sensors=PREFIX_SENSORS),
        command_seq=2,
    )
    actual_evidence = actual.last_evidence
    open_evidence = open_counterfactual.last_evidence
    if actual_evidence is None or open_evidence is None:
        return False
    return all(
        (
            actual_evidence.state_before_sha256 == open_evidence.state_before_sha256,
            actual_action.state_digest != open_action.state_digest,
            actual_action.motors != open_action.motors,
        )
    )
