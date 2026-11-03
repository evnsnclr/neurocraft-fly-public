"""Tests for the declared readout-to-program selection policy (``program_policy.py``).

CI-safe throughout: the policy is a pure function, and the controller tests run over the
same synthetic node space every other lane uses. The artifact-gated tests skip unless the
local-only full artifact, both annotation tables, and the committed loom-v1 calibration are
present.

Nothing asserted here is evidence about a fly. The policy is a hand-chosen engineering
mapping; these tests pin that it computes what it declares it computes, that a change to
any declared constant rotates its identity, and that it cannot select ESCAPE_JUMP without
the TTMn readout crossing its declared threshold.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest
from neurocraft_controller.program_policy import (
    DNA01_FLOOR,
    ESCAPE_HOLD_TICKS,
    ESCAPE_OFF,
    ESCAPE_ON,
    ESCAPE_REFRACTORY_TICKS,
    ESCAPE_URGENCY_SPAN,
    POLICY_FALLBACK,
    POLICY_VOCABULARY,
    PROGRAM_POLICY_SCHEMA,
    SPEED_FLOOR,
    SPEED_SPAN,
    TURN_DEADBAND,
    PolicyReadoutBinding,
    PolicyState,
    ProgramPolicyError,
    ProgramSelectingController,
    ProgramSelectionPolicy,
    policy_for_controller,
    readout_values_from_step,
)
from neurocraft_protocol import (
    Observation,
    ProgramAction,
    ProgramId,
    ReadoutValues,
    SensorFrame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ARTIFACT_DIRECTORY = REPO_ROOT / "data" / "processed" / "male-cns-v1.0-superclass-v1"
ANNOTATIONS_PATH = (
    REPO_ROOT / "data" / "cache" / "male-cns-v1.0" / "body-neurotransmitters-male-cns-v1.0.feather"
)
BODY_ANNOTATIONS_PATH = (
    REPO_ROOT
    / "data"
    / "cache"
    / "male-cns-v1.0"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
LOOM_CALIBRATION = REPO_ROOT / "data" / "calibration" / "full-readout-calibration-loom-v1.json"
ESCAPE_JUMP_PROGRAM = (
    REPO_ROOT / "mod/src/main/java/dev/neurocraft/fly/body/programs/EscapeJumpProgram.java"
)

artifact_gated = pytest.mark.skipif(
    not (FULL_ARTIFACT_DIRECTORY / "manifest.json").is_file()
    or not ANNOTATIONS_PATH.is_file()
    or not BODY_ANNOTATIONS_PATH.is_file()
    or not LOOM_CALIBRATION.is_file(),
    reason="the full MaleCNS artifact, its annotation tables, and the loom-v1 calibration"
    " are local-only",
)

BINDING = PolicyReadoutBinding(
    ttmn_pool_binding_sha256="11" * 32,
    dna01_pair_binding_sha256="22" * 32,
    wing_readout_sha256="33" * 32,
    decoder_hash="44" * 32,
)


def _policy(**overrides: float | int) -> ProgramSelectionPolicy:
    return ProgramSelectionPolicy(readouts=BINDING, **overrides)  # type: ignore[arg-type]


def _readout(*, ttmn: float = 0.0, dna01: float = 0.0, wing: float = 0.0) -> ReadoutValues:
    return ReadoutValues(ttmn_jump=ttmn, dna01_turn=dna01, wing_common=wing)


def _run(policy: ProgramSelectionPolicy, jumps: list[float]) -> tuple[list[ProgramId], PolicyState]:
    state = PolicyState()
    programs: list[ProgramId] = []
    for tick, jump in enumerate(jumps):
        decision = policy.select(_readout(ttmn=jump), state, tick)
        state = decision.state_after
        programs.append(decision.program)
    return programs, state


# ------------------------------------------------------------------- identity


def test_policy_identity_is_stable_and_sensitive() -> None:
    base = _policy()
    assert len(base.identity_sha256) == 64
    assert base.identity_sha256 == _policy().identity_sha256
    material = base.to_identity_material()
    assert material["schema"] == PROGRAM_POLICY_SCHEMA == "neurocraft.program_policy.v1"
    assert material["vocabulary"] == ["CRUISE", "ESCAPE_JUMP"]
    assert material["fallback"] == "HOVER" == POLICY_FALLBACK.value
    assert set(material["formulas"]) == {"dna01_turn", "speed", "state_machine", "urgency"}
    # The per-state output table is in the hash: before it was, a change to any value a
    # viewer actually sees left policy_hash untouched.
    assert set(material["outputs"]) == {"CRUISE", "ESCAPE_JUMP"}
    assert material["outputs"]["ESCAPE_JUMP"]["driving_readout"] == "ttmn"
    assert material["outputs"]["CRUISE"]["driving_readout"] == "wing"
    zero_hex = (0.0).hex()
    assert material["outputs"]["ESCAPE_JUMP"]["turn_bias"] == zero_hex
    assert material["outputs"]["CRUISE"]["urgency"] == zero_hex
    assert material["outputs"]["CRUISE"]["turn_bias"] == "dna01_turn"

    # One constant at a time: each must rotate the hash.
    variations: dict[str, float | int | str] = {
        "escape_on": ESCAPE_ON * 2.0,
        "escape_off": ESCAPE_OFF / 2.0,
        "escape_hold_ticks": ESCAPE_HOLD_TICKS + 1,
        "escape_refractory_ticks": ESCAPE_REFRACTORY_TICKS + 1,
        "escape_urgency_span": 0.5,
        "dna01_floor": DNA01_FLOOR * 2.0,
        "turn_deadband": TURN_DEADBAND / 2.0,
        "speed_floor": SPEED_FLOOR / 2.0,
        "speed_span": SPEED_SPAN * 2.0,
        # The per-state outputs: each is a value a HUD or a body program consumes, so each
        # must rotate the identity on its own.
        "escape_output_speed": 0.5,
        "escape_output_turn_rate": 0.5,
        "escape_output_amplitude": 0.5,
        "escape_output_turn_bias": 0.5,
        "cruise_output_turn_rate": 0.5,
        "cruise_output_amplitude": 0.5,
        "cruise_output_urgency": 0.5,
    }
    hashes = {name: _policy(**{name: value}).identity_sha256 for name, value in variations.items()}
    assert base.identity_sha256 not in hashes.values()
    assert len(set(hashes.values())) == len(variations)
    # The driving-readout labels are presentation strings, and are hashed too: they are
    # what a log or HUD prints beside the program name.
    for name in ("escape_driving_readout", "cruise_driving_readout"):
        assert _policy(**{name: "relabelled"}).identity_sha256 != base.identity_sha256

    # Every field of the policy is covered. If a field is added without entering the
    # identity material, this fails rather than quietly weakening `policy_hash`.
    from dataclasses import fields

    covered = set(material["constants"]) | {
        "ESCAPE_OUTPUT_SPEED",
        "ESCAPE_OUTPUT_TURN_RATE",
        "ESCAPE_OUTPUT_AMPLITUDE",
        "ESCAPE_OUTPUT_TURN_BIAS",
        "CRUISE_OUTPUT_TURN_RATE",
        "CRUISE_OUTPUT_AMPLITUDE",
        "CRUISE_OUTPUT_URGENCY",
        "ESCAPE_DRIVING_READOUT",
        "CRUISE_DRIVING_READOUT",
    }
    declared = {field.name.upper() for field in fields(base) if field.name != "readouts"}
    assert declared == covered

    # A different readout binding is a different policy.
    rebound = ProgramSelectionPolicy(
        readouts=PolicyReadoutBinding(
            ttmn_pool_binding_sha256="11" * 32,
            dna01_pair_binding_sha256="99" * 32,
            wing_readout_sha256="33" * 32,
            decoder_hash="44" * 32,
        )
    )
    assert rebound.identity_sha256 != base.identity_sha256


def test_declared_thresholds_are_the_measured_ramp_derivation() -> None:
    """Every escape constant is derived from ONE measured quantity, by rules declared first.

    ``ESCAPE_ON = 0.5 x peak``, ``ESCAPE_OFF = ESCAPE_ON / 2``, and
    ``ESCAPE_URGENCY_SPAN = peak - ESCAPE_ON``, where ``peak`` is the jump peak on the
    standard 20-tick approach ramp under full_mcns_loom_v1 -- 0.02277816208187006, which
    reproduces the value ``docs/encoders/eem-loom-v1.md`` records for the same stimulus.
    """

    measured_ramp_peak = 0.02277816208187006
    derived_on = 0.5 * measured_ramp_peak
    assert derived_on == ESCAPE_ON
    assert ESCAPE_OFF == ESCAPE_ON / 2.0
    derived_span = measured_ramp_peak - ESCAPE_ON
    assert derived_span == ESCAPE_URGENCY_SPAN
    assert ESCAPE_ON.hex() == "0x1.7532894d187b1p-7"
    assert ESCAPE_OFF.hex() == "0x1.7532894d187b1p-8"
    assert ESCAPE_URGENCY_SPAN.hex() == "0x1.7532894d187b1p-7"
    # The span equalling the threshold is a CONSEQUENCE of the two rules (the threshold is
    # half the same peak), not a third choice; pinned so it reads as derived, not tuned.
    assert ESCAPE_URGENCY_SPAN == ESCAPE_ON

    # The point of the span rule: the standard approach saturates urgency exactly at its
    # peak, and a transient sitting exactly at the threshold gives exactly zero.
    policy = _policy()
    assert policy.escape_urgency(measured_ramp_peak) == 1.0
    assert policy.escape_urgency(ESCAPE_ON) == 0.0


def test_policy_hash_is_a_declaration_hash_not_a_behaviour_hash() -> None:
    """Stated as a test because it is the hash's most misreadable property.

    Every numeric constant and per-state output is a field that ``select`` reads and the
    identity material hashes, so those cannot disagree. The FORMULAS enter as prose, and
    nothing machine-checks that they describe the arithmetic -- the reverted NC4 sign flip
    changed behaviour and left the hash untouched. What guards that agreement is the unit
    tests, not the hash.
    """

    material = _policy().to_identity_material()
    assert isinstance(material["formulas"]["dna01_turn"], str)
    assert "policy_hash" not in material  # the hash is OF this, never IN it
    # The module says so where a reader will find it.
    import neurocraft_controller.program_policy as module

    assert module.__doc__ is not None
    assert "DECLARATION hash" in module.__doc__


def test_policy_refuses_incoherent_constants() -> None:
    with pytest.raises(ProgramPolicyError, match="ESCAPE_OFF <= ESCAPE_ON"):
        _policy(escape_off=0.9, escape_on=0.1)
    with pytest.raises(ProgramPolicyError, match="at least one tick"):
        _policy(escape_hold_ticks=0)
    with pytest.raises(ProgramPolicyError, match="speed_span must be positive"):
        _policy(speed_span=0.0)


# ---------------------------------------------------- negative control 1 (unit)


def test_zero_ttmn_never_selects_escape() -> None:
    """NC1 as a unit test: with the jump readout at zero, ESCAPE is unreachable.

    Every other input is swept, so the only way ESCAPE could appear is if something other
    than the declared TTMn threshold selected it.
    """

    policy = _policy()
    state = PolicyState()
    for tick in range(200):
        readout = _readout(
            ttmn=0.0,
            dna01=(tick % 21 - 10) / 10.0,
            wing=(tick % 41 - 20) / 20.0,
        )
        decision = policy.select(readout, state, tick)
        state = decision.state_after
        assert decision.program is ProgramId.CRUISE
        assert decision.modulation.urgency == 0.0
        assert decision.parameters.turn_bias == readout.dna01_turn


def test_a_jump_just_below_the_threshold_never_selects_escape() -> None:
    programs, _ = _run(_policy(), [ESCAPE_ON - 1e-12] * 60)
    assert set(programs) == {ProgramId.CRUISE}


# ---------------------------------------------------------- anti-flapping rules


def test_escape_hold_and_refractory_prevent_flapping() -> None:
    """A square wave at the threshold selects ESCAPE once per hold+refractory window."""

    policy = _policy()
    period = 4
    window = ESCAPE_HOLD_TICKS + ESCAPE_REFRACTORY_TICKS
    jumps = [ESCAPE_ON if (tick // period) % 2 == 0 else 0.0 for tick in range(window * 3)]
    programs, _ = _run(policy, jumps)
    escape_ticks = [
        tick for tick, program in enumerate(programs) if program is ProgramId.ESCAPE_JUMP
    ]
    # Contiguous runs of ESCAPE_JUMP, each exactly the declared hold.
    runs: list[list[int]] = []
    for tick in escape_ticks:
        if runs and tick == runs[-1][-1] + 1:
            runs[-1].append(tick)
        else:
            runs.append([tick])
    assert runs, "the square wave must cross the threshold at least once"
    assert all(len(run) == ESCAPE_HOLD_TICKS for run in runs)
    # Successive selections are at least a full hold+refractory window apart.
    starts = [run[0] for run in runs]
    assert all(later - earlier >= window for earlier, later in pairwise(starts))


def test_a_held_supra_threshold_jump_selects_escape_exactly_once() -> None:
    """Held above ESCAPE_ON forever, the arming latch never re-arms: one selection only."""

    programs, state = _run(_policy(), [ESCAPE_ON * 4] * 400)
    assert programs.count(ProgramId.ESCAPE_JUMP) == ESCAPE_HOLD_TICKS
    assert programs[0] is ProgramId.ESCAPE_JUMP
    assert state.armed is False


def test_the_latch_rearms_only_below_escape_off() -> None:
    policy = _policy()
    # Hold high for the whole escape window, drop between ESCAPE_OFF and ESCAPE_ON (which
    # must NOT re-arm), then drop below ESCAPE_OFF and rise again.
    between = (ESCAPE_ON + ESCAPE_OFF) / 2.0
    jumps = (
        [ESCAPE_ON * 4] * (ESCAPE_HOLD_TICKS + ESCAPE_REFRACTORY_TICKS)
        + [between] * 20
        + [ESCAPE_ON * 4] * 20
    )
    programs, _ = _run(policy, jumps)
    assert programs.count(ProgramId.ESCAPE_JUMP) == ESCAPE_HOLD_TICKS

    # Below ESCAPE_OFF the latch re-arms, and by then the refractory has expired, so the
    # next rise selects a second full hold.
    rearmed = (
        [ESCAPE_ON * 4] * (ESCAPE_HOLD_TICKS + ESCAPE_REFRACTORY_TICKS)
        + [0.0] * 5
        + [ESCAPE_ON * 4] * (ESCAPE_HOLD_TICKS + 6)
    )
    programs, _ = _run(policy, rearmed)
    assert programs.count(ProgramId.ESCAPE_JUMP) == 2 * ESCAPE_HOLD_TICKS


def test_refractory_blocks_a_rearmed_crossing_until_it_expires() -> None:
    """Re-armed early, the next crossing still waits out the whole refractory window."""

    policy = _policy()
    # Two low ticks: the first is consumed by the last hold tick's own transition, the
    # second re-arms the latch. Everything after is above threshold, so only the
    # refractory counter can be what delays the second selection.
    jumps = [ESCAPE_ON * 4] * ESCAPE_HOLD_TICKS + [0.0] * 2 + [ESCAPE_ON * 4] * 100
    programs, _ = _run(policy, jumps)
    second_start = programs.index(ProgramId.ESCAPE_JUMP, ESCAPE_HOLD_TICKS)
    # The tick that ends the hold SETS the counter (it does not decrement it), so the
    # first tick that can select again is one past hold + refractory.
    assert second_start == ESCAPE_HOLD_TICKS + ESCAPE_REFRACTORY_TICKS + 1
    assert all(program is ProgramId.CRUISE for program in programs[ESCAPE_HOLD_TICKS:second_start])


def test_cruise_has_zero_minimum_dwell() -> None:
    """A threshold crossing selects on the tick it happens; dwell lives on the way back."""

    programs, _ = _run(_policy(), [0.0, ESCAPE_ON])
    assert programs == [ProgramId.CRUISE, ProgramId.ESCAPE_JUMP]


def test_policy_state_refuses_an_incoherent_hold() -> None:
    with pytest.raises(ProgramPolicyError, match="same condition"):
        PolicyState(program=ProgramId.ESCAPE_JUMP, hold_remaining=0)
    with pytest.raises(ProgramPolicyError, match="declared policy vocabulary"):
        PolicyState(program=ProgramId.HOVER)


# --------------------------------------------------------------- the formulas


def test_dna01_turn_sign_and_deadband_on_the_census_rows() -> None:
    """The declared sign convention read against the census's own sided means.

    The four literals are the escape-v2 activity table of
    ``docs/research/readout-census-2026-09-02.md``. Positive means turn toward the body's
    right (ADR 0015), and the sign was chosen so those rows read as turning AWAY from the
    blocked side. **What this test does not show:** that the convention survives loom-v1.
    The offline probe under the loom encoder finds the DNa01 difference inside the deadband
    on every held condition, so under that encoder these rows do not reproduce and the
    convention is not exercised by any held stimulus.
    """

    policy = _policy()

    def turn(left: float, right: float) -> float:
        evidence = _FakeEvidence(jump_value=0.0, left_motor=0.0, right_motor=0.0)
        return readout_values_from_step(evidence, (left, right)).dna01_turn

    # blocked_left: L -3.03e-4, R +1.32e-3 -> positive, i.e. turn right, away from the
    # obstacle on the left.
    assert turn(-3.027943e-04, +1.315155e-03) == pytest.approx(0.9417910644976543, abs=1e-12)
    # blocked_right: L +9.06e-4, R -3.16e-4 -> negative, i.e. turn left, away.
    assert turn(+9.059997e-04, -3.159365e-04) == pytest.approx(-0.9243533840740574, abs=1e-12)
    # all_open_bright baseline: |t| = 0.024, inside the 0.10 deadband -> exactly zero.
    assert turn(-1.160943e-04, -1.083040e-04) == 0.0
    # blocked_front: |t| = 0.223, outside the deadband; a front-on stimulus has no declared
    # side, so the value is recorded and not interpreted.
    assert turn(+3.760368e-04, +6.210606e-04) == pytest.approx(0.22333823779000844, abs=1e-12)
    assert policy.turn_deadband == TURN_DEADBAND == 0.10


def test_dna01_floor_bounds_a_near_zero_pair() -> None:
    """Both members at the numerical floor cannot normalize to a full-scale bias."""

    evidence = _FakeEvidence(jump_value=0.0, left_motor=0.0, right_motor=0.0)
    tiny = readout_values_from_step(evidence, (1e-12, 2e-12))
    assert tiny.dna01_turn == 0.0
    # Without the floor the same pair would normalize to +1/3; with it, it is deadbanded.
    unfloored = (2e-12 - 1e-12) / 3e-12
    assert unfloored == pytest.approx(1.0 / 3.0)


def test_speed_floor_and_span() -> None:
    policy = _policy()
    assert policy.cruise_speed(-1.0) == SPEED_FLOOR
    assert policy.cruise_speed(0.0) == SPEED_FLOOR
    assert policy.cruise_speed(SPEED_SPAN / 2.0) == pytest.approx(
        SPEED_FLOOR + (1.0 - SPEED_FLOOR) * 0.5
    )
    assert policy.cruise_speed(SPEED_SPAN) == 1.0
    assert policy.cruise_speed(1.0) == 1.0
    # Every value the formula can produce is a legal modulation attenuator.
    for step in range(-20, 21):
        assert 0.0 <= policy.cruise_speed(step / 20.0) <= 1.0


def test_escape_urgency_span() -> None:
    policy = _policy()
    assert policy.escape_urgency(0.0) == 0.0
    assert policy.escape_urgency(ESCAPE_ON) == 0.0
    assert policy.escape_urgency(ESCAPE_ON + policy.escape_urgency_span) == 1.0
    assert policy.escape_urgency(1.0) == 1.0
    # Half way up the declared span is half urgency.
    assert policy.escape_urgency(ESCAPE_ON + policy.escape_urgency_span / 2.0) == pytest.approx(0.5)


def test_escape_modulation_is_full_and_unbiased() -> None:
    decision = _policy().select(_readout(ttmn=1.0, dna01=0.9, wing=1.0), PolicyState(), 0)
    assert decision.program is ProgramId.ESCAPE_JUMP
    assert decision.modulation.speed == 1.0
    assert decision.modulation.turn_rate == 1.0
    assert decision.modulation.amplitude == 1.0
    assert decision.modulation.urgency == 1.0
    assert decision.parameters.turn_bias == 0.0
    assert decision.driving_readout == "ttmn"


def test_readout_values_refuse_a_missing_or_out_of_range_jump() -> None:
    with pytest.raises(ProgramPolicyError, match="requires the TTMn jump readout"):
        readout_values_from_step(_FakeEvidence(jump_value=None), (0.0, 0.0))
    with pytest.raises(ProgramPolicyError, match="rectified decoder's declared"):
        readout_values_from_step(_FakeEvidence(jump_value=-0.1), (0.0, 0.0))


def test_wing_common_is_the_projection_common_term() -> None:
    evidence = _FakeEvidence(jump_value=0.25, left_motor=0.4, right_motor=-0.2)
    values = readout_values_from_step(evidence, (0.0, 0.0))
    assert values.wing_common == pytest.approx(0.5 * (0.4 + -0.2))
    assert values.ttmn_jump == 0.25


# ------------------------------------------------- cross-language duration pin


def test_escape_hold_equals_the_java_program_duration() -> None:
    """``ESCAPE_HOLD_TICKS`` and the Java program's fixed duration are one number.

    If they disagree, the program's own exit and the next frame's re-selection produce a
    one-tick flap. Skips while ``EscapeJumpProgram.java`` is still in flight on its own
    lane; it must run once that file is on the branch.
    """

    if not ESCAPE_JUMP_PROGRAM.is_file():
        pytest.skip(
            "EscapeJumpProgram.java is not in this tree yet (stage-3 lane D lands it "
            "separately); re-run this test after that lane merges"
        )
    source = ESCAPE_JUMP_PROGRAM.read_text(encoding="utf-8")
    match = re.search(r"ESCAPE_JUMP_TICKS\s*=\s*(\d+)", source)
    assert match is not None, "EscapeJumpProgram.java does not declare ESCAPE_JUMP_TICKS"
    assert int(match.group(1)) == ESCAPE_HOLD_TICKS


# ------------------------------------------------------ synthetic step evidence


class _FakeEvidence:
    """The three fields ``readout_values_from_step`` reads, and nothing else."""

    def __init__(
        self,
        *,
        jump_value: float | None,
        left_motor: float = 0.0,
        right_motor: float = 0.0,
    ) -> None:
        self.jump_value = jump_value
        self.motor_frame = _FakeMotorFrame(left_motor, right_motor)


class _FakeMotorFrame:
    def __init__(self, left: float, right: float) -> None:
        self.values = (("left_motor", left), ("right_motor", right))


# ------------------------------------------- negative control 2 (synthetic stream)


def _synthetic_readout_stream(seed: str, ticks: int) -> list[ReadoutValues]:
    """A SHA-256-derived readout stream: the shape of the connectome's output, none of
    its content. The ``calibrate_full_readout.synthetic_observation`` pattern."""

    from hashlib import sha256

    stream: list[ReadoutValues] = []
    for tick in range(ticks):
        digest = sha256(f"{seed}:{tick}".encode()).digest()
        stream.append(
            ReadoutValues(
                ttmn_jump=digest[0] / 255.0 * 0.05,
                dna01_turn=(digest[1] / 255.0) * 2.0 - 1.0,
                wing_common=(digest[2] / 255.0) * 2.0 - 1.0,
            )
        )
    return stream


def test_synthetic_readout_control_changes_selection_statistics() -> None:
    """NC2 as a unit test: the same policy over a synthetic stream selects differently.

    A control lane that proves the policy is a function of its inputs -- not that the
    connectome's readouts are meaningful. The shuffle-brain control at full scale is stage
    7 work: ``FullBrainController`` has no weight or topology overlay hook, and the
    fixture-scale shuffle has no TTMn pool to read.
    """

    policy = _policy()

    def counts(stream: list[ReadoutValues]) -> dict[str, int]:
        state = PolicyState()
        selected: list[str] = []
        for tick, readout in enumerate(stream):
            decision = policy.select(readout, state, tick)
            state = decision.state_after
            selected.append(decision.program.value)
        return {name: selected.count(name) for name in ("CRUISE", "ESCAPE_JUMP")}

    # The connectome's own held-condition stream, standing in for the artifact-free case:
    # a quiet jump term with a slow wing drift, which selects CRUISE throughout.
    quiet = [ReadoutValues(ttmn_jump=0.0016, dna01_turn=0.0, wing_common=-0.01) for _ in range(120)]
    synthetic = _synthetic_readout_stream("neurocraft.program_policy.control", 120)
    assert counts(quiet) == {"CRUISE": 120, "ESCAPE_JUMP": 0}
    synthetic_counts = counts(synthetic)
    assert synthetic_counts != counts(quiet)
    assert synthetic_counts["ESCAPE_JUMP"] > 0


# ------------------------------------------------------- controller integration


def _sensors(front: float = 6.0) -> SensorFrame:
    return SensorFrame(
        position_m=(0.0, 66.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        pitch_rad=0.0,
        on_ground=False,
        horizontal_collision=False,
        vertical_collision=False,
        in_fluid=False,
        local_light=15,
        ray_max_distance_m=6.0,
        ray_distances_m=(front, 6.0, 6.0, 6.0, 6.0, 6.0),
    )


def _synthetic_program_controller() -> ProgramSelectingController:
    from tests.controller.test_steering_readout import _loom_controllers

    with_steering, _ = _loom_controllers()
    return ProgramSelectingController(with_steering)


def test_program_selecting_controller_preserves_inner_identity_and_envelope() -> None:
    """Identity-rotation pin, and the envelope copy.

    ``RunIdentity`` is the wrapped controller's, byte for byte; ``policy_hash`` rides on
    the frame and appears in no identity field.
    """

    controller = _synthetic_program_controller()
    inner = controller.controller
    assert controller.identity == inner.identity
    assert controller.run_mode is inner.run_mode
    assert controller.configuration is inner.configuration
    assert controller.policy_hash not in set(inner.identity.to_dict().values())

    frame = controller.step(
        Observation(
            identity=controller.identity,
            observation_seq=0,
            game_tick=0,
            sent_monotonic_ns=0,
            sensors=_sensors(),
        ),
        command_seq=0,
    )
    assert isinstance(frame, ProgramAction)
    assert frame.identity == inner.identity
    assert frame.policy_hash == controller.policy_hash
    evidence = controller.last_evidence
    assert evidence is not None
    inner_evidence = inner.last_evidence
    assert inner_evidence is not None
    # Every envelope field is the inner action's.
    assert frame.command_seq == inner_evidence.command_seq
    assert frame.observation_seq == 0
    assert frame.observation_tick == 0
    assert frame.target_tick == 2
    assert frame.valid_through_tick == 2
    assert frame.neural_step_start == inner_evidence.neural_step_start
    assert frame.neural_step_end == inner_evidence.neural_step_end
    assert frame.state_digest == inner_evidence.state_after_sha256
    # The raw projection is recorded and never sent: a program frame has no motors.
    assert evidence.raw_projection == inner_evidence.motors
    assert not hasattr(frame, "motors")
    assert frame.readout == evidence.readout


def test_program_controller_delegates_lifecycle_and_resets_policy_state() -> None:
    controller = _synthetic_program_controller()
    assert not controller.session_has_stepped
    for tick in range(3):
        controller.step(
            Observation(
                identity=controller.identity,
                observation_seq=tick,
                game_tick=tick,
                sent_monotonic_ns=tick * 50_000_000,
                sensors=_sensors(),
            ),
            command_seq=tick,
        )
    assert controller.warm_compute_path(passes=1) >= 0.0
    controller.reset()
    assert controller.policy_state == PolicyState()
    assert controller.last_evidence is None
    assert controller.controller.last_evidence is None
    assert controller.session_has_stepped
    controller.reset_session(session_epoch=controller.identity.session_epoch + 1)
    assert not controller.session_has_stepped


def test_program_controller_refuses_a_controller_without_the_readouts() -> None:
    from tests.controller.test_steering_readout import _loom_controllers

    _, without_steering = _loom_controllers()
    with pytest.raises(ProgramPolicyError, match="DNa01 steering readout"):
        ProgramSelectingController(without_steering)
    with pytest.raises(ProgramPolicyError, match="steering binding"):
        policy_for_controller(without_steering)


def test_program_controller_refuses_a_policy_bound_to_other_readouts() -> None:
    """L3: ``policy_hash`` must never attest bindings that are not the ones in use.

    An injected policy is allowed (the constructor takes one), but only if its readout
    binding is this controller's. Otherwise every frame would carry a hash naming a TTMn
    pool, a DNa01 pair, a wing readout or a decoder other than the one producing the
    numbers on that same frame.
    """

    controller = _synthetic_program_controller()
    bound = policy_for_controller(controller.controller)
    # The same policy re-derived is accepted, and yields the same hash.
    accepted = ProgramSelectingController(controller.controller, policy=bound)
    assert accepted.policy_hash == controller.policy_hash

    foreign = ProgramSelectionPolicy(
        readouts=PolicyReadoutBinding(
            ttmn_pool_binding_sha256=bound.readouts.ttmn_pool_binding_sha256,
            dna01_pair_binding_sha256="99" * 32,  # not this controller's pair
            wing_readout_sha256=bound.readouts.wing_readout_sha256,
            decoder_hash=bound.readouts.decoder_hash,
        )
    )
    with pytest.raises(ProgramPolicyError, match="bindings that are not this controller's"):
        ProgramSelectingController(controller.controller, policy=foreign)

    # A policy that differs only in a CONSTANT is still accepted: that is a different
    # declared mapping over the same readouts, which is exactly what the hash is for.
    retuned = ProgramSelectionPolicy(readouts=bound.readouts, turn_deadband=0.2)
    varied = ProgramSelectingController(controller.controller, policy=retuned)
    assert varied.policy_hash != controller.policy_hash


def test_policy_vocabulary_excludes_the_refusal_fallback() -> None:
    assert POLICY_VOCABULARY == (ProgramId.CRUISE, ProgramId.ESCAPE_JUMP)
    assert POLICY_FALLBACK is ProgramId.HOVER
    assert POLICY_FALLBACK not in POLICY_VOCABULARY


# --------------------------------------------------------------- artifact-gated


@artifact_gated
def test_artifact_controller_issues_program_frames_with_the_real_readouts() -> None:
    """One end-to-end tick over the real artifact; local-only."""

    from scripts.probe_program_policy import build_controller

    controller = build_controller(
        artifact_directory=FULL_ARTIFACT_DIRECTORY,
        annotations=ANNOTATIONS_PATH,
        body_annotations=BODY_ANNOTATIONS_PATH,
        calibration=LOOM_CALIBRATION,
        threads=None,
    )
    try:
        binding = controller.steering_binding
        assert binding is not None
        # The census pair, verbatim.
        assert tuple(member.body_id for member in binding.members) == (10_442, 10_760)
        assert binding.left.soma_side == "L"
        assert binding.right.soma_side == "R"
        frame = controller.step(
            Observation(
                identity=controller.identity,
                observation_seq=0,
                game_tick=0,
                sent_monotonic_ns=0,
                sensors=_sensors(),
            ),
            command_seq=0,
        )
        assert isinstance(frame, ProgramAction)
        assert frame.program is ProgramId.CRUISE
        assert frame.policy_hash == controller.policy_hash
    finally:
        controller.engine.end_run()
