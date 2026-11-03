"""AB-11 test power against the three neural-propagation mutants (M1, M2, M3).

Each test records both directions of the charter row (PLAN.md:65, anti-bypass table):

1. the faithful baseline satisfies the acceptance predicate, or the named production
   detector accepts the faithful artifact; and
2. the AB-11 mutant makes that same predicate false, or makes that same detector refuse.

The mutants live in ``tests/ab11/mutants.py`` and are falsification artifacts only,
never Faithful Organism Mode components. Controller-protocol bypasses live in
``test_ab11_controller_bypass.py`` and graph substitution in
``test_ab11_graph_substitution.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_protocol import verify_trace

from scripts.replay_minecraft_trace import iter_records, replay_minecraft_trace
from tests.controller.conftest import make_controller, make_observation

from . import mutants
from .acceptance import (
    BLOCKED_SENSORS,
    ray_perturbation_acceptance_holds,
    real_input_acceptance_holds,
    sign_ablation_acceptance_holds,
)
from .conftest import (
    FIXTURE,
    TRACE_ENTITY_ID,
    TRACE_RECORD_COUNT,
    faithful_trace_controller,
    minecraft_labelled_trace,
    mutant_trace_controller,
    zero_recurrence_divergences,
)
from .mutants import (
    MutantFrozenPropagationEngine,
    MutantIdentityPropagationEngine,
    MutantZeroPropagationEngine,
    make_mutant_engine_controller,
)

MUTANT_CLASS_COUNT = 5


# --------------------------------------------------------------------------------------
# M1 — zero propagation
# --------------------------------------------------------------------------------------


def test_ab11_zero_propagation_defeats_real_input_reaching_state_and_motor(
    controller_graph: RuntimeGraph,
) -> None:
    """AB-11(M1 zero propagation) vs the named acceptance test.

    Named test: ``tests/controller/test_causal_controller.py::
    test_real_input_reaches_neural_state_and_primitive_motor``.
    Detector: its own predicate — driven and zero-input runs must reach different
    states and the driven run must reach a non-neutral primitive motor command.
    """

    assert real_input_acceptance_holds(
        make_controller(controller_graph, run_id="ab11-m1-baseline-driven"),
        make_controller(controller_graph, run_id="ab11-m1-baseline-zero"),
    )

    mutant_driven = make_mutant_engine_controller(
        controller_graph,
        engine_type=MutantZeroPropagationEngine,
        run_id="ab11-m1-mutant-driven",
    )
    mutant_zero = make_mutant_engine_controller(
        controller_graph,
        engine_type=MutantZeroPropagationEngine,
        run_id="ab11-m1-mutant-zero",
    )
    assert not real_input_acceptance_holds(mutant_driven, mutant_zero)

    driven_evidence = mutant_driven.last_evidence
    zero_evidence = mutant_zero.last_evidence
    assert driven_evidence is not None and zero_evidence is not None
    # Real sensory encoding still differs; only the committed state is dead.
    assert driven_evidence.encoded_stimulation_sha256 != zero_evidence.encoded_stimulation_sha256
    assert driven_evidence.state_after_sha256 == zero_evidence.state_after_sha256
    assert driven_evidence.motors.is_neutral


def test_ab11_zero_propagation_trace_fails_exact_neural_replay(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """AB-11(M1 zero propagation) vs the exact-replay detector.

    Named test: ``tests/integration/test_replay_minecraft_trace.py::
    test_minecraft_labelled_trace_replays_neural_state_exactly``.
    Detector: ``scripts/replay_minecraft_trace.py`` per-record equality battery,
    ``ValueError("causal neural replay mismatch at record 0")``.
    """

    faithful_trace = tmp_path / "m1-faithful.ndjson"
    minecraft_labelled_trace(
        faithful_trace,
        faithful_trace_controller(controller_graph, run_id="ab11-m1-trace"),
    )
    receipt = replay_minecraft_trace(fixture=FIXTURE, trace=faithful_trace, stimulations={})
    assert receipt["status"] == "exact_neural_match"
    assert receipt["record_count"] == TRACE_RECORD_COUNT

    mutant_trace = tmp_path / "m1-mutant.ndjson"
    minecraft_labelled_trace(
        mutant_trace,
        mutant_trace_controller(
            controller_graph,
            engine_type=MutantZeroPropagationEngine,
            run_id="ab11-m1-trace",
        ),
    )
    # The mutant artifact is hash-authentic; only re-execution can expose it.
    assert verify_trace(mutant_trace).record_count == TRACE_RECORD_COUNT
    with pytest.raises(ValueError, match="causal neural replay mismatch at record 0"):
        replay_minecraft_trace(fixture=FIXTURE, trace=mutant_trace, stimulations={})


# --------------------------------------------------------------------------------------
# M2 — identity propagation
# --------------------------------------------------------------------------------------


def test_ab11_identity_propagation_defeats_sign_ablation_acceptance(
    controller_graph: RuntimeGraph,
) -> None:
    """AB-11(M2 identity propagation) vs the named acceptance test.

    Named test: ``tests/controller/test_causal_controller.py::
    test_fixture_sign_ablation_removes_recurrent_sensor_to_motor_effect`` (already
    docstring-tagged AB-02/AB-11).
    Detector: its own predicate — ablating the signed recurrence must remove the
    sensor-to-motor effect that the faithful controller shows.
    """

    assert sign_ablation_acceptance_holds(
        make_controller(controller_graph, run_id="ab11-m2-baseline-faithful"),
        ConnectomeController.create(
            controller_graph,
            run_id="ab11-m2-baseline-zero-sign",
            entity_id="test-fly",
            configuration=ControllerConfiguration(sign_policy="unknown_zero"),
        ),
    )

    mutant_faithful = make_mutant_engine_controller(
        controller_graph,
        engine_type=MutantIdentityPropagationEngine,
        run_id="ab11-m2-mutant-faithful",
        entity_id="test-fly",
    )
    mutant_zero_sign = make_mutant_engine_controller(
        controller_graph,
        engine_type=MutantIdentityPropagationEngine,
        run_id="ab11-m2-mutant-zero-sign",
        entity_id="test-fly",
        configuration=ControllerConfiguration(sign_policy="unknown_zero"),
    )
    assert not sign_ablation_acceptance_holds(mutant_faithful, mutant_zero_sign)

    # Mechanism, stated exactly: both arms commit the identical clipped sensory vector,
    # so the sign overlay is causally inert — identical activity, identical motors, and
    # both neutral because no motor node is sensory-encoded. The named predicate
    # therefore falsifies on its neutrality conjunct, not on its digest conjunct: the
    # committed state digests still differ, because the sign overlay hash is part of
    # state identity and changes the digest without changing anything causal.
    faithful_evidence = mutant_faithful.last_evidence
    zero_sign_evidence = mutant_zero_sign.last_evidence
    assert faithful_evidence is not None and zero_sign_evidence is not None
    assert faithful_evidence.state_after_activity == zero_sign_evidence.state_after_activity
    assert faithful_evidence.motors == zero_sign_evidence.motors
    assert faithful_evidence.motors.is_neutral
    assert zero_sign_evidence.motors.is_neutral
    assert faithful_evidence.state_after_sha256 != zero_sign_evidence.state_after_sha256


def test_ab11_identity_propagation_defeats_zero_recurrence_counterfactual(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """AB-11(M2 identity propagation) vs the EXTERNAL-BRIDGE-02 counterfactual.

    Named test: ``tests/integration/test_replay_minecraft_trace.py::
    test_minecraft_labelled_trace_replays_neural_state_exactly`` (its
    ``zero_recurrence_counterfactual`` divergence assertions).
    Detector: ``scripts/replay_minecraft_trace.py`` refusal "zero-recurrence
    counterfactual did not alter both model state and motor output", evaluated here
    through the mirror in ``conftest.zero_recurrence_divergences`` because the
    production script hardwires a faithful counterfactual controller. The mirror is
    pinned to production below: on the faithful trace its three counters must equal the
    ones the real replay receipt publishes, so the mutant half is measured against
    production behaviour rather than against a test-local reimplementation.
    """

    faithful_trace = tmp_path / "m2-faithful.ndjson"
    minecraft_labelled_trace(
        faithful_trace,
        faithful_trace_controller(controller_graph, run_id="ab11-m2-trace"),
    )
    faithful_counts = zero_recurrence_divergences(
        tuple(iter_records(faithful_trace)),
        ConnectomeController.create(
            controller_graph,
            run_id="ab11-m2-trace",
            entity_id=TRACE_ENTITY_ID,
            configuration=ControllerConfiguration(sign_policy="unknown_zero"),
        ),
    )
    assert faithful_counts.state_divergent_records >= 1
    assert faithful_counts.action_divergent_records >= 1
    assert not faithful_counts.replay_refuses

    # Pin the mirror to production: the same faithful trace, run through the real
    # replay script, must publish exactly the three counts the mirror computed. Without
    # this the mutant half would only be measured against a test-local reimplementation.
    production = replay_minecraft_trace(fixture=FIXTURE, trace=faithful_trace, stimulations={})[
        "zero_recurrence_counterfactual"
    ]
    assert production["record_count"] == TRACE_RECORD_COUNT
    assert production["state_divergent_record_count"] == faithful_counts.state_divergent_records
    assert production["action_divergent_record_count"] == faithful_counts.action_divergent_records
    assert production["neutral_action_record_count"] == faithful_counts.neutral_action_records

    mutant_trace = tmp_path / "m2-mutant.ndjson"
    minecraft_labelled_trace(
        mutant_trace,
        mutant_trace_controller(
            controller_graph,
            engine_type=MutantIdentityPropagationEngine,
            run_id="ab11-m2-trace",
        ),
    )
    mutant_counts = zero_recurrence_divergences(
        tuple(iter_records(mutant_trace)),
        make_mutant_engine_controller(
            controller_graph,
            engine_type=MutantIdentityPropagationEngine,
            run_id="ab11-m2-trace",
            entity_id=TRACE_ENTITY_ID,
            configuration=ControllerConfiguration(sign_policy="unknown_zero"),
        ),
    )
    assert mutant_counts.action_divergent_records == 0
    # Every counterfactual action is neutral: the ablated arm has nothing to decode,
    # which is why the divergence counter the production receipt publishes collapses.
    assert mutant_counts.neutral_action_records == TRACE_RECORD_COUNT
    assert mutant_counts.replay_refuses


# --------------------------------------------------------------------------------------
# M3 — frozen propagation
# --------------------------------------------------------------------------------------


def test_ab11_frozen_propagation_defeats_forward_ray_perturbation_acceptance(
    controller_graph: RuntimeGraph,
) -> None:
    """AB-11(M3 frozen propagation) vs the named acceptance test.

    Named test: ``tests/controller/test_causal_controller.py::
    test_fixture_forward_ray_perturbation_propagates_from_identical_state``.
    Detector: its own predicate — one physical sensor change, applied from an
    identical state, must alter both the state digest and the motors.

    The mutant runs live for the two prefix ticks and freezes on the perturbation tick,
    so the frozen content is a real non-zero state rather than the all-zero initial
    condition: frozen propagation is falsified here as itself, not as zero propagation.
    """

    assert ray_perturbation_acceptance_holds(
        make_controller(controller_graph, run_id="ab11-m3-baseline-actual"),
        make_controller(controller_graph, run_id="ab11-m3-baseline-counterfactual"),
    )

    mutant_actual = make_mutant_engine_controller(
        controller_graph,
        engine_type=MutantFrozenPropagationEngine,
        run_id="ab11-m3-mutant-actual",
    )
    mutant_counterfactual = make_mutant_engine_controller(
        controller_graph,
        engine_type=MutantFrozenPropagationEngine,
        run_id="ab11-m3-mutant-counterfactual",
    )
    assert not ray_perturbation_acceptance_holds(mutant_actual, mutant_counterfactual)

    # Frozen is not zero: the state that froze is the one the two live prefix ticks
    # actually produced, so this mutant is distinguishable from M1 zero propagation.
    frozen_state = mutant_actual.engine.current_state()
    assert any(value != 0.0 for value in frozen_state.activity)
    assert frozen_state.activity == mutant_counterfactual.engine.current_state().activity

    # The step-window bookkeeping the acceptance suite also checks stays intact,
    # which is precisely why the state/motor predicate has to carry the weight.
    last = mutant_actual.step(
        make_observation(mutant_actual, 3, sensors=BLOCKED_SENSORS),
        command_seq=3,
    )
    assert last.neural_step_end - last.neural_step_start == 5
    assert mutant_actual.engine.current_state().activity == frozen_state.activity


def test_ab11_frozen_propagation_trace_fails_exact_neural_replay(
    controller_graph: RuntimeGraph,
    tmp_path: Path,
) -> None:
    """AB-11(M3 frozen propagation) vs the exact-replay detector.

    Named test: ``tests/integration/test_replay_minecraft_trace.py::
    test_minecraft_labelled_trace_replays_neural_state_exactly``.
    Detector: ``scripts/replay_minecraft_trace.py`` per-record equality battery,
    ``ValueError("causal neural replay mismatch at record 2")`` — record 2 rather than
    record 0 because the mutant authors the first two ticks faithfully and only then
    freezes, which is also what makes its trace distinct from the M1 zero trace.
    """

    mutant_trace = tmp_path / "m3-mutant.ndjson"
    minecraft_labelled_trace(
        mutant_trace,
        mutant_trace_controller(
            controller_graph,
            engine_type=MutantFrozenPropagationEngine,
            run_id="ab11-m3-trace",
        ),
    )
    assert verify_trace(mutant_trace).record_count == TRACE_RECORD_COUNT
    with pytest.raises(ValueError, match="causal neural replay mismatch at record 2"):
        replay_minecraft_trace(fixture=FIXTURE, trace=mutant_trace, stimulations={})

    # Frozen propagation is not zero propagation: the same trace identity authored by
    # the M1 engine is a different artifact, with an all-zero committed state.
    zero_trace = tmp_path / "m3-zero-comparison.ndjson"
    minecraft_labelled_trace(
        zero_trace,
        mutant_trace_controller(
            controller_graph,
            engine_type=MutantZeroPropagationEngine,
            run_id="ab11-m3-trace",
        ),
    )
    assert mutant_trace.read_bytes() != zero_trace.read_bytes()
    frozen_records = tuple(iter_records(mutant_trace))
    zero_records = tuple(iter_records(zero_trace))
    assert any(value != 0.0 for value in frozen_records[-1].state_after_activity)
    assert all(value == 0.0 for value in zero_records[-1].state_after_activity)
    assert frozen_records[-1].state_after_digest != zero_records[-1].state_after_digest

    faithful_trace = tmp_path / "m3-faithful.ndjson"
    minecraft_labelled_trace(
        faithful_trace,
        faithful_trace_controller(controller_graph, run_id="ab11-m3-trace"),
    )
    assert (
        replay_minecraft_trace(fixture=FIXTURE, trace=faithful_trace, stimulations={})["status"]
        == "exact_neural_match"
    )


# --------------------------------------------------------------------------------------
# Claim discipline
# --------------------------------------------------------------------------------------


def test_ab11_mutant_artifacts_are_named_and_declared_as_falsification_only() -> None:
    """AB-11(claim discipline) — the mutants must never read as production components.

    Named requirement: CLAUDE.md "Faithful Organism Mode purity" and AGENTS.md
    "Never claim scientific fidelity without evidence".
    Detector: this test — every class defined in ``tests/ab11/mutants.py`` carries the
    ``Mutant`` prefix and an AB-11 falsification docstring.
    """

    defined = [
        name
        for name, value in vars(mutants).items()
        if isinstance(value, type) and value.__module__ == mutants.__name__
    ]
    assert len(defined) == MUTANT_CLASS_COUNT
    assert all(name.startswith("Mutant") for name in defined)
    for name in defined:
        documentation = getattr(mutants, name).__doc__ or ""
        assert documentation.startswith("AB-11 mutant:"), name
    module_documentation = mutants.__doc__ or ""
    assert "never Faithful Organism Mode components" in module_documentation
