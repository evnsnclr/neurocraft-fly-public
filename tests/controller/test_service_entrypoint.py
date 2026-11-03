from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from neurocraft_connectome import FixtureIdentityError
from neurocraft_controller import FlightConnectomeController
from neurocraft_controller.flight_controller import DEFAULT_V1_CONFIGURATION_SHA256
from neurocraft_protocol import ControlCommand, ControlOperation, RunIdentity, parse_json_frame

from scripts.run_connectome_service import (
    _refuse_full_profile_conflicts,
    build_control_handler,
    build_controller,
    build_parser,
    build_trace_recorder,
    control_catalog,
    write_identity,
)

if TYPE_CHECKING:
    from neurocraft_controller.full_brain_controller import FullBrainController

ROOT = Path(__file__).resolve().parents[2]
STEERING_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
STEERING_GRAPH_HASH = "ebc50c0db92f7ec0e0eeb737af3d6d446cd569a488564a5682a0f255af401c26"


def test_entrypoint_builds_only_explicit_connectome_controller() -> None:
    args = build_parser().parse_args(["--identity-only"])
    controller = build_controller(args)
    assert controller.configuration.mode == "faithful"
    assert controller.identity.graph_hash == controller.graph.manifest.runtime_identity_sha256
    assert len(controller.engine.current_state().activity_sha256) == 64


def test_entrypoint_writes_strict_no_clobber_identity(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--identity-only"])
    controller = build_controller(args)
    target = tmp_path / "identity.json"
    write_identity(target, controller)
    parsed = RunIdentity.from_dict(parse_json_frame(target.read_bytes()))
    assert parsed == controller.identity
    with pytest.raises(FileExistsError):
        write_identity(target, controller)


def test_default_profile_is_pair_v1_and_default_identity_is_unchanged() -> None:
    args = build_parser().parse_args(["--identity-only"])
    assert args.controller_profile == "pair_v1"
    controller = build_controller(args)
    assert not isinstance(controller, FlightConnectomeController)
    assert controller.configuration.identity_sha256 == DEFAULT_V1_CONFIGURATION_SHA256


def test_wing_motor_profile_builds_the_flight_controller() -> None:
    args = build_parser().parse_args(
        [
            "--identity-only",
            "--controller-profile",
            "wing_motor_soma_side_v1",
            "--fixture",
            str(STEERING_FIXTURE),
        ]
    )
    controller = build_controller(args)
    assert isinstance(controller, FlightConnectomeController)
    assert controller.configuration.mode == "faithful"
    assert controller.configuration.identity_sha256 != DEFAULT_V1_CONFIGURATION_SHA256
    assert controller.identity.graph_hash == STEERING_GRAPH_HASH


def test_wing_motor_profile_refuses_the_default_power_fixture() -> None:
    """The wing profile pins the steering fixture; the shipped default fixture
    must be refused at the byte gate rather than silently re-read."""

    args = build_parser().parse_args(
        ["--identity-only", "--controller-profile", "wing_motor_soma_side_v1"]
    )
    with pytest.raises(FixtureIdentityError):
        build_controller(args)


def test_trm_profile_builds_the_trained_readout_controller() -> None:
    """The trained-readout profile must build the TRM demo controller bound to
    the committed artifact's derived digest, under the TRM configuration
    schema, with a decoder identity distinct from the wing profile's."""

    import hashlib

    from neurocraft_controller.trained_readout_controller import (
        TRM_DEMO_CONFIG_SCHEMA,
        TrainedReadoutController,
        TrmDemoConfiguration,
    )

    args = build_parser().parse_args(
        [
            "--identity-only",
            "--controller-profile",
            "trm_cloned_readout_v1",
            "--fixture",
            str(STEERING_FIXTURE),
        ]
    )
    controller = build_controller(args)
    assert isinstance(controller, TrainedReadoutController)
    configuration = controller.configuration
    assert isinstance(configuration, TrmDemoConfiguration)
    assert configuration.schema == TRM_DEMO_CONFIG_SCHEMA
    assert configuration.mode == "faithful"
    artifact = ROOT / "data" / "readouts" / "trm-cloned-readout-v2.json"
    expected_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert configuration.readout_artifact_sha256 == expected_digest
    assert controller.identity.graph_hash == STEERING_GRAPH_HASH

    wing_args = build_parser().parse_args(
        [
            "--identity-only",
            "--controller-profile",
            "wing_motor_soma_side_v1",
            "--fixture",
            str(STEERING_FIXTURE),
        ]
    )
    wing = build_controller(wing_args)
    assert controller.identity.decoder_hash != wing.identity.decoder_hash
    assert controller.configuration.identity_sha256 != wing.configuration.identity_sha256


def test_trm_profile_refuses_the_default_power_fixture() -> None:
    """The trained-readout profile pins the steering fixture; the shipped
    default fixture must be refused at the byte gate."""

    args = build_parser().parse_args(
        ["--identity-only", "--controller-profile", "trm_cloned_readout_v1"]
    )
    with pytest.raises(FixtureIdentityError):
        build_controller(args)


def test_wing_motor_profile_mirrors_the_lesion_mode_contract() -> None:
    args = build_parser().parse_args(
        [
            "--identity-only",
            "--controller-profile",
            "wing_motor_soma_side_v1",
            "--fixture",
            str(STEERING_FIXTURE),
            "--mode",
            "lesion",
            "--lesion-body-id",
            "800184",
        ]
    )
    controller = build_controller(args)
    assert isinstance(controller, FlightConnectomeController)
    assert controller.configuration.mode == "lesion"


def test_lesion_requires_explicit_mode() -> None:
    args = build_parser().parse_args(["--lesion-body-id", "805722"])
    with pytest.raises(RuntimeError, match="explicit lesion"):
        build_controller(args)

    lesion_args = build_parser().parse_args(["--mode", "lesion", "--lesion-body-id", "805722"])
    controller = build_controller(lesion_args)
    assert controller.configuration.mode == "lesion"


def test_entrypoint_uses_loopback_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEUROCRAFT_HOST", "::1")
    monkeypatch.setenv("NEUROCRAFT_PORT", "9988")
    args = build_parser().parse_args(["--identity-only"])
    assert args.host == "::1"
    assert args.port == 9988


def test_entrypoint_builds_hash_bound_predeclared_intervention_registry() -> None:
    args = build_parser().parse_args(
        [
            "--identity-only",
            "--declare-stimulation",
            "test-stimulation=800241=0.5",
        ]
    )
    controller = build_controller(args)
    handler = build_control_handler(args, controller)
    assert handler is not None
    catalog = control_catalog(handler)
    entry = catalog["interventions"][0]
    assert entry["content_id"] == "test-stimulation"
    assert len(entry["content_hash"]) == 64
    handler.handle_control(
        ControlCommand(
            identity=controller.identity,
            control_seq=0,
            issued_tick=0,
            operation=ControlOperation.STIMULATION_APPLY,
            content_id=entry["content_id"],
            content_hash=entry["content_hash"],
        )
    )
    assert controller.active_stimulation_id == "test-stimulation"


def test_trace_cli_requires_manifest_and_explicit_force_document(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--identity-only", "--trace-out", str(tmp_path / "trace.ndjson")]
    )
    controller = build_controller(args)
    with pytest.raises(SystemExit, match="transition-forces"):
        build_trace_recorder(args, controller)


@pytest.mark.parametrize("profile", ["full_mcns_loom_v2", "full_mcns_loom_program_v2"])
def test_loom_v2_cheap_gate_accepts_its_calibration_and_refuses_v1(
    profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_connectome_service as service

    annotations = tmp_path / "body-annotations.feather"
    annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", annotations)
    args = service.build_parser().parse_args(["--identity-only", "--controller-profile", profile])
    service._refuse_full_profile_conflicts(args)
    monkeypatch.setattr(
        service, "DEFAULT_FULL_LOOM_V2_CALIBRATION", service.DEFAULT_FULL_LOOM_V1_CALIBRATION
    )
    with pytest.raises(SystemExit, match=r"loom-v1 calibration.*loom-v2 boundary refuses"):
        service._refuse_full_profile_conflicts(args)


def test_full_profile_is_registered() -> None:
    from scripts.run_connectome_service import (
        CONTROLLER_PROFILES,
        FULL_MCNS_BODY_ANNOTATION_PROFILES,
        FULL_MCNS_EEM_PROFILE,
        FULL_MCNS_EEM_PROFILES,
        FULL_MCNS_EEM_V2_PROFILE,
        FULL_MCNS_ESCAPE_PROFILE,
        FULL_MCNS_ESCAPE_PROFILES,
        FULL_MCNS_JUMP_READOUT_PROFILES,
        FULL_MCNS_PROFILES,
        FULL_MCNS_WING_PROFILE,
    )

    assert FULL_MCNS_WING_PROFILE == "full_mcns_wing_v1"
    assert FULL_MCNS_WING_PROFILE in CONTROLLER_PROFILES
    assert FULL_MCNS_EEM_PROFILE == "full_mcns_eem_v1"
    assert FULL_MCNS_EEM_PROFILE in CONTROLLER_PROFILES
    assert FULL_MCNS_EEM_V2_PROFILE == "full_mcns_eem_v2"
    assert FULL_MCNS_EEM_V2_PROFILE in CONTROLLER_PROFILES
    assert FULL_MCNS_ESCAPE_PROFILE == "full_mcns_escape_v1"
    assert FULL_MCNS_ESCAPE_PROFILE in CONTROLLER_PROFILES
    assert set(FULL_MCNS_EEM_PROFILES) == {FULL_MCNS_EEM_PROFILE, FULL_MCNS_EEM_V2_PROFILE}
    assert FULL_MCNS_ESCAPE_PROFILE in FULL_MCNS_ESCAPE_PROFILES
    assert set(FULL_MCNS_BODY_ANNOTATION_PROFILES) == {
        *FULL_MCNS_EEM_PROFILES,
        *FULL_MCNS_JUMP_READOUT_PROFILES,
    }
    assert set(FULL_MCNS_JUMP_READOUT_PROFILES) == {
        *FULL_MCNS_ESCAPE_PROFILES,
        "full_mcns_loom_v1",
        # The body-program lane wraps the loom-v1 controller unchanged, so it carries the
        # same TTMn jump readout and the same calibration shape.
        "full_mcns_loom_program_v1",
        "full_mcns_loom_v2",
        "full_mcns_loom_program_v2",
        "full_mcns_events_program_v1",
        "full_mcns_interactions_program_v2",
    }
    assert set(FULL_MCNS_PROFILES) == {
        FULL_MCNS_WING_PROFILE,
        *FULL_MCNS_BODY_ANNOTATION_PROFILES,
    }


def _full_profile_args(*extra: str) -> list[str]:
    """Arguments for the full profile with a guaranteed-absent artifact dir.

    Every refusal asserted below must fire BEFORE any artifact load, so these
    tests deliberately point at a directory that cannot exist: if a refusal
    were ordered after the load attempt, the error would be the absent-artifact
    message instead of the asserted one.
    """

    return [
        "--identity-only",
        "--controller-profile",
        "full_mcns_wing_v1",
        "--full-artifact-dir",
        "/nonexistent/full-artifact",
        *extra,
    ]


def test_full_profile_refuses_trace_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        _full_profile_args("--trace-out", str(tmp_path / "trace.ndjson"))
    )
    with pytest.raises(SystemExit, match="fixture-typed"):
        build_controller(args)


def test_full_profile_refuses_lesion_mode() -> None:
    args = build_parser().parse_args(_full_profile_args("--mode", "lesion"))
    with pytest.raises(SystemExit, match="faithful only"):
        build_controller(args)


def test_full_profile_refuses_interventions() -> None:
    for extra in (
        ("--lesion-body-id", "800184"),
        ("--stimulate", "77185=0.5"),
        ("--declare-lesion", "named=800184"),
        ("--declare-stimulation", "named=77185=0.5"),
    ):
        args = build_parser().parse_args(_full_profile_args(*extra))
        with pytest.raises(SystemExit, match="lesion/stimulation"):
            build_controller(args)


def test_full_profile_refuses_an_explicit_fixture() -> None:
    args = build_parser().parse_args(_full_profile_args("--fixture", str(STEERING_FIXTURE)))
    with pytest.raises(SystemExit, match="not a fixture"):
        build_controller(args)


def test_full_profile_refuses_presentation_telemetry(tmp_path: Path) -> None:
    """The writer would emit the declared boundary subset as unlabeled
    per-node activity; the flag is refused rather than silently mislabeled."""

    args = build_parser().parse_args(
        _full_profile_args("--presentation-telemetry-out", str(tmp_path / "telemetry.ndjson"))
    )
    with pytest.raises(SystemExit, match="unlabeled activity"):
        build_controller(args)


def test_full_profile_accepts_latency_only_telemetry(tmp_path: Path) -> None:
    """The refusal above is about neural content, not about telemetry. The
    latency sink copies no controller evidence -- only the command sequence and
    the three measured durations -- so the full profiles must accept it, or no
    full-artifact live run could ever record its own latency."""

    args = build_parser().parse_args(
        _full_profile_args("--action-latency-out", str(tmp_path / "latency.ndjson"))
    )
    _refuse_full_profile_conflicts(args)  # must not raise
    assert args.action_latency_out == tmp_path / "latency.ndjson"

    # Past the conflict gate the only remaining failure is the absent artifact,
    # which proves the flag was never a refusal reason.
    with pytest.raises(SystemExit, match=r"local-only profile.*not present"):
        build_controller(args)


def test_latency_telemetry_does_not_weaken_the_presentation_refusal(tmp_path: Path) -> None:
    """Asking for both still refuses the presentation writer, unchanged."""

    args = build_parser().parse_args(
        _full_profile_args(
            "--action-latency-out",
            str(tmp_path / "latency.ndjson"),
            "--presentation-telemetry-out",
            str(tmp_path / "telemetry.ndjson"),
        )
    )
    with pytest.raises(SystemExit, match="unlabeled activity"):
        build_controller(args)


def test_latency_telemetry_defaults_off_for_every_profile() -> None:
    args = build_parser().parse_args(["--identity-only"])
    assert args.action_latency_out is None


def test_full_profile_reports_the_absent_artifact(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--identity-only",
            "--controller-profile",
            "full_mcns_wing_v1",
            "--full-artifact-dir",
            str(tmp_path / "missing-artifact"),
        ]
    )
    with pytest.raises(SystemExit, match=r"local-only profile.*not present"):
        build_controller(args)


def test_full_profile_defaults_to_the_committed_calibration_path() -> None:
    """The declared calibration is committed and loaded by default; opting out
    is an explicit flag, not the absence of a file."""

    import scripts.run_connectome_service as service

    assert service.DEFAULT_FULL_CALIBRATION == (
        ROOT / "data" / "calibration" / "full-readout-calibration-v1.json"
    )
    assert service.DEFAULT_FULL_CALIBRATION.is_file()
    assert build_parser().parse_args(["--identity-only"]).full_uncalibrated is False


def test_full_profile_refuses_a_missing_calibration_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal must fire BEFORE the artifact load: the artifact directory
    here cannot exist, so an absent-artifact message would prove wrong order."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(service, "DEFAULT_FULL_CALIBRATION", tmp_path / "absent-calibration.json")
    args = build_parser().parse_args(_full_profile_args())
    with pytest.raises(SystemExit, match=r"calibrate_full_readout\.py"):
        build_controller(args)


def test_full_profile_uncalibrated_opt_out_skips_the_calibration_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the opt-out the missing calibration is not a refusal; the next
    (pre-existing) gate -- the absent artifact -- is what fires."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(service, "DEFAULT_FULL_CALIBRATION", tmp_path / "absent-calibration.json")
    args = build_parser().parse_args(_full_profile_args("--full-uncalibrated"))
    assert args.full_uncalibrated is True
    with pytest.raises(SystemExit, match=r"local-only profile.*not present"):
        build_controller(args)


def test_uncalibrated_flag_is_refused_for_other_profiles() -> None:
    args = build_parser().parse_args(["--identity-only", "--full-uncalibrated"])
    with pytest.raises(SystemExit, match="full MaleCNS profiles only"):
        build_controller(args)


def _eem_v2_args(*extra: str) -> list[str]:
    """eem-v2 arguments with a guaranteed-absent artifact dir (refusal-order proof)."""

    return [
        "--identity-only",
        "--controller-profile",
        "full_mcns_eem_v2",
        "--full-artifact-dir",
        "/nonexistent/full-artifact",
        *extra,
    ]


def test_eem_v2_missing_calibration_names_the_measurement_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Until the phase-2 calibration is measured and committed, the profile
    refuses to start and its refusal names the exact command that measures it
    (the artifact dir here cannot exist, so the refusal provably fires before
    any artifact load)."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(service, "DEFAULT_FULL_EEM_V2_CALIBRATION", tmp_path / "absent-eem-v2.json")
    args = build_parser().parse_args(_eem_v2_args())
    with pytest.raises(SystemExit, match=r"calibrate_full_readout\.py --profile eem-v2"):
        build_controller(args)


def test_eem_v2_missing_body_annotations_names_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_connectome_service as service

    monkeypatch.setattr(
        service, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    args = build_parser().parse_args(_eem_v2_args("--full-uncalibrated"))
    with pytest.raises(SystemExit, match=r"full_mcns_eem_v2 is a local-only"):
        build_controller(args)


def test_eem_v2_wing_calibration_is_refused_at_the_cheap_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wing document placed at the eem-v2 path is refused in milliseconds."""

    import scripts.run_connectome_service as service

    wing_calibration = service.DEFAULT_FULL_CALIBRATION
    if not wing_calibration.is_file():  # pragma: no cover - committed document
        pytest.skip("the committed wing calibration is absent")
    fake = tmp_path / "eem-v2-calibration.json"
    fake.write_bytes(wing_calibration.read_bytes())
    monkeypatch.setattr(service, "DEFAULT_FULL_EEM_V2_CALIBRATION", fake)
    # The body-annotations presence gate precedes the calibration sniff; fake
    # it so this test reaches the cheap gate on hosts without the local data.
    fake_annotations = tmp_path / "body-annotations.feather"
    fake_annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", fake_annotations)
    args = build_parser().parse_args(_eem_v2_args())
    with pytest.raises(SystemExit, match=r"full_mcns_eem_v2 refuses it"):
        build_controller(args)


def _escape_args(*extra: str) -> list[str]:
    """escape arguments with a guaranteed-absent artifact dir (refusal-order proof)."""

    return [
        "--identity-only",
        "--controller-profile",
        "full_mcns_escape_v1",
        "--full-artifact-dir",
        "/nonexistent/full-artifact",
        *extra,
    ]


def test_escape_missing_calibration_names_the_measurement_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Until the phase-2 escape calibration is measured and committed, the
    profile refuses to start and its refusal names the exact command that
    measures it (the artifact dir here cannot exist, so the refusal provably
    fires before any artifact load; the default path is monkeypatched to a
    tmp path so the test is independent of whatever the host has)."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(service, "DEFAULT_FULL_ESCAPE_CALIBRATION", tmp_path / "absent-escape.json")
    args = build_parser().parse_args(_escape_args())
    with pytest.raises(SystemExit, match=r"calibrate_full_readout\.py --profile escape"):
        build_controller(args)


def test_escape_missing_body_annotations_names_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--full-uncalibrated skips the (earlier) calibration gate, so the
    missing body-annotations table is the refusal, and it names the escape
    profile; every preceding gate is faked with tmp paths."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(
        service, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    args = build_parser().parse_args(_escape_args("--full-uncalibrated"))
    with pytest.raises(SystemExit, match=r"full_mcns_escape_v1 is a local-only"):
        build_controller(args)


def test_escape_refuses_v1_calibrations_at_the_cheap_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wing or eem document placed at the escape path is refused in
    milliseconds, before any artifact byte is read. The documents are
    synthesized (not read from the repo) and every preceding presence gate is
    faked, so the test runs identically on hosts without local data."""

    from neurocraft_protocol import canonical_json_bytes

    import scripts.run_connectome_service as service
    from tests.controller.test_embodied_senses import EEM_ENCODER_SHA, _calibration_document

    fake_annotations = tmp_path / "body-annotations.feather"
    fake_annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", fake_annotations)
    for label, encoder_sha256 in (("wing", None), ("eem-v1", EEM_ENCODER_SHA)):
        fake = tmp_path / f"escape-calibration-{label}.json"
        fake.write_bytes(canonical_json_bytes(_calibration_document(encoder_sha256=encoder_sha256)))
        monkeypatch.setattr(service, "DEFAULT_FULL_ESCAPE_CALIBRATION", fake)
        args = build_parser().parse_args(_escape_args())
        with pytest.raises(SystemExit, match="escape boundary refuses"):
            build_controller(args)


def _escape_v2_args(*extra: str) -> list[str]:
    """escape-v2 arguments with a guaranteed-absent artifact dir (refusal-order proof)."""

    return [
        "--identity-only",
        "--controller-profile",
        "full_mcns_escape_v2",
        "--full-artifact-dir",
        "/nonexistent/full-artifact",
        *extra,
    ]


def test_escape_v2_profile_is_registered_with_its_own_calibration() -> None:
    from scripts.run_connectome_service import (
        CONTROLLER_PROFILES,
        FULL_MCNS_BODY_ANNOTATION_PROFILES,
        FULL_MCNS_EEM_PROFILES,
        FULL_MCNS_ESCAPE_PROFILE,
        FULL_MCNS_ESCAPE_PROFILES,
        FULL_MCNS_ESCAPE_V2_PROFILE,
        FULL_MCNS_PROFILES,
        _full_profile_calibration,
    )

    assert FULL_MCNS_ESCAPE_V2_PROFILE == "full_mcns_escape_v2"
    assert FULL_MCNS_ESCAPE_V2_PROFILE in CONTROLLER_PROFILES
    assert set(FULL_MCNS_ESCAPE_PROFILES) == {
        FULL_MCNS_ESCAPE_PROFILE,
        FULL_MCNS_ESCAPE_V2_PROFILE,
    }
    assert FULL_MCNS_ESCAPE_V2_PROFILE in FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert FULL_MCNS_ESCAPE_V2_PROFILE in FULL_MCNS_PROFILES
    assert FULL_MCNS_ESCAPE_V2_PROFILE not in FULL_MCNS_EEM_PROFILES
    path, hint = _full_profile_calibration(FULL_MCNS_ESCAPE_V2_PROFILE)
    assert path.name == "full-readout-calibration-escape-v2.json"
    assert hint == "scripts/calibrate_full_readout.py --profile escape-v2"
    # The event candidate explicitly reuses a loom-v2 decoder component; its added
    # drives have their own identity and are not validated by that calibration.
    # Every earlier full profile that defines its own sensory/decoder boundary names a
    # distinct calibration document. The body-program lane is the one exception, and it is
    # an exception on purpose: it wraps the loom-v1 controller unchanged, so it must load
    # the loom-v1 document, and a document of its own would attest a boundary that does
    # not exist.
    from scripts.run_connectome_service import (
        FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
        FULL_MCNS_LOOM_V1_PROFILE,
    )

    boundary_profiles = tuple(
        profile
        for profile in FULL_MCNS_PROFILES
        if profile
        not in (
            FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
            "full_mcns_loom_program_v2",
            "full_mcns_events_program_v1",
            "full_mcns_interactions_program_v2",
        )
    )
    documents = {_full_profile_calibration(profile)[0] for profile in boundary_profiles}
    assert len(documents) == len(boundary_profiles)
    assert _full_profile_calibration(
        FULL_MCNS_LOOM_PROGRAM_V1_PROFILE
    ) == _full_profile_calibration(FULL_MCNS_LOOM_V1_PROFILE)


def test_escape_v2_missing_calibration_names_the_measurement_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Until the phase-2 escape-v2 calibration is measured and committed, the
    profile refuses to start and names the exact command that measures it."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(
        service, "DEFAULT_FULL_ESCAPE_V2_CALIBRATION", tmp_path / "absent-escape-v2.json"
    )
    args = build_parser().parse_args(_escape_v2_args())
    with pytest.raises(SystemExit, match=r"calibrate_full_readout\.py --profile escape-v2"):
        build_controller(args)


def test_escape_v2_missing_body_annotations_names_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--full-uncalibrated skips the (earlier) calibration gate, so the missing
    body-annotations table is the refusal and it names the escape-v2 profile;
    every preceding gate is faked with tmp paths."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(
        service, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    args = build_parser().parse_args(_escape_v2_args("--full-uncalibrated"))
    with pytest.raises(SystemExit, match=r"full_mcns_escape_v2 is a local-only"):
        build_controller(args)


def test_escape_v2_refuses_every_other_calibration_at_the_cheap_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wing, eem, or escape-v1 document at the escape-v2 path is refused in
    milliseconds, before any artifact byte is read. The documents are
    synthesized (not read from the repo) and every preceding presence gate is
    faked, so the test runs identically on hosts without local data."""

    from neurocraft_protocol import canonical_json_bytes

    import scripts.run_connectome_service as service
    from tests.controller.test_embodied_senses import EEM_ENCODER_SHA, _calibration_document
    from tests.controller.test_escape_reflex import _escape_calibration_document

    fake_annotations = tmp_path / "body-annotations.feather"
    fake_annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", fake_annotations)
    documents = {
        "wing": _calibration_document(encoder_sha256=None),
        "eem-v1": _calibration_document(encoder_sha256=EEM_ENCODER_SHA),
        "escape-v1": _escape_calibration_document(),
    }
    for label, document in documents.items():
        fake = tmp_path / f"escape-v2-calibration-{label}.json"
        fake.write_bytes(canonical_json_bytes(document))
        monkeypatch.setattr(service, "DEFAULT_FULL_ESCAPE_V2_CALIBRATION", fake)
        args = build_parser().parse_args(_escape_v2_args())
        with pytest.raises(SystemExit, match="escape-v2 boundary refuses"):
            build_controller(args)


def test_loom_profile_is_registered_with_its_own_calibration() -> None:
    """Argument surface only -- no artifact is loaded."""

    from scripts.run_connectome_service import (
        FULL_MCNS_LOOM_V1_PROFILE,
        _full_profile_calibration,
    )

    assert FULL_MCNS_LOOM_V1_PROFILE == "full_mcns_loom_v1"
    path, hint = _full_profile_calibration(FULL_MCNS_LOOM_V1_PROFILE)
    assert path.name == "full-readout-calibration-loom-v1.json"
    assert hint == "scripts/calibrate_full_readout.py --profile loom-v1"


def test_program_profile_is_registered_over_the_loom_controller() -> None:
    """Argument surface only -- no artifact is loaded.

    The body-program lane is the loom-v1 controller plus the declared DNa01 steering
    readout, issuing ``program_action`` frames through a hand-chosen mapping. It is a
    distinct profile name (never a flag on ``full_mcns_loom_v1``) so a session log can
    never be ambiguous about which frame kind ran.
    """

    from scripts.run_connectome_service import (
        CONTROLLER_PROFILES,
        FULL_MCNS_BODY_ANNOTATION_PROFILES,
        FULL_MCNS_EEM_PROFILES,
        FULL_MCNS_ESCAPE_PROFILES,
        FULL_MCNS_JUMP_READOUT_PROFILES,
        FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
        FULL_MCNS_LOOM_V1_PROFILE,
        FULL_MCNS_LOOM_V1_PROFILES,
        FULL_MCNS_PROFILES,
        FULL_MCNS_PROGRAM_PROFILES,
    )

    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE == "full_mcns_loom_program_v1"
    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE in CONTROLLER_PROFILES
    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE in FULL_MCNS_PROFILES
    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE in FULL_MCNS_BODY_ANNOTATION_PROFILES
    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE in FULL_MCNS_JUMP_READOUT_PROFILES
    assert FULL_MCNS_PROGRAM_PROFILES == (
        FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
        "full_mcns_loom_program_v2",
        "full_mcns_events_program_v1",
        "full_mcns_interactions_program_v2",
    )
    assert FULL_MCNS_LOOM_V1_PROFILES == (
        FULL_MCNS_LOOM_V1_PROFILE,
        FULL_MCNS_LOOM_PROGRAM_V1_PROFILE,
    )
    # The escape and eem tuples stay exactly what they were.
    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE not in FULL_MCNS_ESCAPE_PROFILES
    assert FULL_MCNS_LOOM_PROGRAM_V1_PROFILE not in FULL_MCNS_EEM_PROFILES
    # The raw loom lane is not a program lane.
    assert FULL_MCNS_LOOM_V1_PROFILE not in FULL_MCNS_PROGRAM_PROFILES


def test_every_jump_readout_profile_has_a_cheap_calibration_gate() -> None:
    """The gate that raised KeyError instead of refusing, pinned structurally.

    ``_refuse_full_profile_conflicts`` shape-checks each jump-readout profile's calibration
    before the multi-second verified load. That check was keyed on a literal per-profile
    mapping, so ``full_mcns_loom_program_v1`` -- a member of
    ``FULL_MCNS_JUMP_READOUT_PROFILES`` from the day it was added -- fell straight through
    to ``KeyError``, killing every path through ``main()`` including ``--identity-only``.
    This asserts the mapping covers the tuple, so the next profile to join cannot repeat it.
    """

    import scripts.run_connectome_service as service

    for profile in service.FULL_MCNS_JUMP_READOUT_PROFILES:
        # Reaching the gate at all is the point: it must produce a SystemExit refusal
        # (a named, actionable message), never an uncaught KeyError.
        args = build_parser().parse_args(
            [
                "--identity-only",
                "--controller-profile",
                profile,
                "--full-artifact-dir",
                "/nonexistent/full-artifact",
            ]
        )
        with pytest.raises(SystemExit):
            build_controller(args)


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[2]
        / "data/processed/male-cns-v1.0-superclass-v1"
        / "manifest.json"
    ).is_file(),
    reason="the full MaleCNS artifact is local-only",
)
def test_program_profile_identity_is_byte_identical_to_the_raw_loom_profile() -> None:
    """The body-program lane wraps the loom-v1 controller; its identity must be the same.

    This is the end-to-end form of the identity-containment claim: same eight fields, same
    bytes, through the real entrypoint rather than through a synthetic view. It also proves
    the H1 path actually runs -- before the fix, this test could not reach a controller.
    """

    import scripts.run_connectome_service as service

    identities = []
    for profile in service.FULL_MCNS_LOOM_V1_PROFILES:
        args = build_parser().parse_args(
            [
                "--identity-only",
                "--controller-profile",
                profile,
                "--run-id",
                "identity-pin",
                "--entity-id",
                "pin-fly",
            ]
        )
        controller = build_controller(args)
        identities.append(controller.identity.to_dict())
        controller.engine.end_run()  # type: ignore[union-attr]
    assert identities[0] == identities[1]
    assert len(identities[0]) == 8


def test_program_decisions_out_is_refused_for_every_other_profile() -> None:
    """The diagnostic sink is refused before any artifact byte is read."""

    for profile in ("pair_v1", "full_mcns_loom_v1", "full_mcns_escape_v2"):
        args = build_parser().parse_args(
            [
                "--identity-only",
                "--controller-profile",
                profile,
                "--full-artifact-dir",
                "/nonexistent/full-artifact",
                "--program-decisions-out",
                "/nonexistent/program-decisions.ndjson",
            ]
        )
        with pytest.raises(SystemExit, match="--program-decisions-out applies to"):
            build_controller(args)


LOOM_PROFILES = ("full_mcns_loom_v1", "full_mcns_loom_program_v1")


def _loom_args(*extra: str, profile: str = "full_mcns_loom_v1") -> list[str]:
    """Loom-lane arguments with a guaranteed-absent artifact dir (refusal-order proof).

    Parametrized over BOTH loom profiles: the body-program lane reaches every one of
    these gates through the same code path, and keying any of them on the raw lane
    alone is how the cheap calibration gate came to raise KeyError instead of the
    refusal it exists to produce.
    """

    return [
        "--identity-only",
        "--controller-profile",
        profile,
        "--full-artifact-dir",
        "/nonexistent/full-artifact",
        *extra,
    ]


def test_loom_missing_calibration_names_the_measurement_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Until the loom-v1 calibration is measured and committed, the profile
    refuses to start and names the exact command that measures it."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(
        service, "DEFAULT_FULL_LOOM_V1_CALIBRATION", tmp_path / "absent-loom-v1.json"
    )
    for profile in LOOM_PROFILES:
        args = build_parser().parse_args(_loom_args(profile=profile))
        with pytest.raises(SystemExit, match=r"calibrate_full_readout\.py --profile loom-v1"):
            build_controller(args)


def test_loom_missing_body_annotations_names_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--full-uncalibrated skips the (earlier) calibration gate, so the missing
    body-annotations table is the refusal and it names the loom profile."""

    import scripts.run_connectome_service as service

    monkeypatch.setattr(
        service, "FULL_BODY_ANNOTATIONS_PATH", tmp_path / "absent-annotations.feather"
    )
    for profile in LOOM_PROFILES:
        args = build_parser().parse_args(_loom_args("--full-uncalibrated", profile=profile))
        with pytest.raises(SystemExit, match=rf"{profile} is a local-only"):
            build_controller(args)


def test_loom_refuses_every_other_calibration_at_the_cheap_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wing, eem, escape-v1, or escape-v2 document at the loom-v1 path is
    refused in milliseconds, before any artifact byte is read. The documents
    are synthesized and every preceding presence gate is faked, so the test
    runs identically on hosts without local data."""

    from neurocraft_protocol import canonical_json_bytes

    import scripts.run_connectome_service as service
    from tests.controller.test_embodied_senses import EEM_ENCODER_SHA, _calibration_document
    from tests.controller.test_escape_reflex import _escape_calibration_document
    from tests.controller.test_escape_reflex_v2 import _escape_v2_calibration_document

    fake_annotations = tmp_path / "body-annotations.feather"
    fake_annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", fake_annotations)
    documents = {
        "wing": _calibration_document(encoder_sha256=None),
        "eem-v1": _calibration_document(encoder_sha256=EEM_ENCODER_SHA),
        "escape-v1": _escape_calibration_document(),
        "escape-v2": _escape_v2_calibration_document(),
    }
    for label, document in documents.items():
        fake = tmp_path / f"loom-calibration-{label}.json"
        fake.write_bytes(canonical_json_bytes(document))
        monkeypatch.setattr(service, "DEFAULT_FULL_LOOM_V1_CALIBRATION", fake)
        for profile in LOOM_PROFILES:
            args = build_parser().parse_args(_loom_args(profile=profile))
            with pytest.raises(SystemExit, match="loom-v1 boundary refuses"):
                build_controller(args)


def test_escape_v1_refuses_an_escape_v2_calibration_at_the_cheap_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse direction: an escape-v2 document at the escape-v1 path."""

    from neurocraft_controller.full_brain_controller import (
        FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA,
    )
    from neurocraft_protocol import canonical_json_bytes

    import scripts.run_connectome_service as service
    from tests.controller.test_escape_reflex import _escape_calibration_document

    fake_annotations = tmp_path / "body-annotations.feather"
    fake_annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", fake_annotations)
    document = _escape_calibration_document()
    document["schema"] = FULL_READOUT_CALIBRATION_ESCAPE_V2_SCHEMA
    fake = tmp_path / "escape-v1-calibration.json"
    fake.write_bytes(canonical_json_bytes(document))
    monkeypatch.setattr(service, "DEFAULT_FULL_ESCAPE_CALIBRATION", fake)
    args = build_parser().parse_args(_escape_args())
    with pytest.raises(SystemExit, match="escape-v1 boundary"):
        build_controller(args)


def test_eem_v2_refuses_an_escape_calibration_at_the_cheap_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse direction: an escape document at an eem path is refused."""

    from neurocraft_protocol import canonical_json_bytes

    import scripts.run_connectome_service as service
    from tests.controller.test_escape_reflex import _escape_calibration_document

    fake_annotations = tmp_path / "body-annotations.feather"
    fake_annotations.write_bytes(b"")
    monkeypatch.setattr(service, "FULL_BODY_ANNOTATIONS_PATH", fake_annotations)
    fake = tmp_path / "eem-v2-calibration.json"
    fake.write_bytes(canonical_json_bytes(_escape_calibration_document()))
    monkeypatch.setattr(service, "DEFAULT_FULL_EEM_V2_CALIBRATION", fake)
    args = build_parser().parse_args(_eem_v2_args())
    with pytest.raises(SystemExit, match="unsupported readout calibration schema"):
        build_controller(args)


def test_idle_rewarm_runs_until_first_step_and_cancels_cleanly() -> None:
    """The re-warm loop must fire while the engine has not stepped, stop once
    stepping begins, and cancel without error when serve finishes."""

    import asyncio

    from scripts.run_connectome_service import _serve_with_idle_rewarm

    class _StubEngine:
        step_index = 0

    class _StubController:
        def __init__(self) -> None:
            self.engine = _StubEngine()
            self.warm_calls = 0
            self.session_has_stepped = False

        def warm_compute_path(self, *, passes: int = 3) -> float:
            self.warm_calls = self.warm_calls + 1
            return 0.0

    stub = _StubController()

    async def fake_serve() -> None:
        await asyncio.sleep(0.08)
        stub.engine.step_index = 5  # stepping began; re-warm must stop
        stub.session_has_stepped = True
        await asyncio.sleep(0.08)

    asyncio.run(
        _serve_with_idle_rewarm(
            fake_serve(),
            cast("FullBrainController", stub),
            interval_seconds=0.01,
        )
    )
    assert stub.warm_calls >= 2
    calls_at_stop = stub.warm_calls
    assert calls_at_stop <= 9  # warms stopped once step_index moved


def test_idle_rewarm_logs_each_pass_cost_and_names_a_skipped_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The elapsed time is the only direct measurement of the page-eviction
    hypothesis, so it is printed rather than discarded; a pass that declined
    the contended compute lock is named as skipped instead of reported as 0 ms."""

    import asyncio

    from scripts.run_connectome_service import _serve_with_idle_rewarm

    class _StubEngine:
        step_index = 0

    class _StubController:
        def __init__(self) -> None:
            self.engine = _StubEngine()
            self.returns = [0.00312, 0.0]
            self.session_has_stepped = False

        def warm_compute_path(self, *, passes: int = 3) -> float:
            return self.returns.pop(0) if self.returns else 0.0

    stub = _StubController()

    async def fake_serve() -> None:
        while stub.returns:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        stub.engine.step_index = 5
        stub.session_has_stepped = True

    asyncio.run(
        _serve_with_idle_rewarm(
            fake_serve(),
            cast("FullBrainController", stub),
            interval_seconds=0.01,
        )
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "idle re-warm pass 1 took 3.12 ms"
    assert lines[1] == "idle re-warm pass 2 skipped (a step held the compute lock)"


@pytest.mark.parametrize("stop_reason", ["step_then_reset", "handover"])
def test_idle_rewarm_permanently_stops_on_session_progress_or_handover(stop_reason: str) -> None:
    import asyncio
    from types import SimpleNamespace

    from scripts.run_connectome_service import _serve_with_idle_rewarm

    async def exercise() -> None:
        warmed = asyncio.Event()
        calls: list[int] = []
        ownership = {"initial": True}
        stub = SimpleNamespace(engine=SimpleNamespace(step_index=0), session_has_stepped=False)

        def warm(*, passes: int = 3) -> float:
            calls.append(passes)
            warmed.set()
            return 0.001

        stub.warm_compute_path = warm

        async def serve() -> None:
            await warmed.wait()
            if stop_reason == "step_then_reset":
                # A command completed and RESET returned the engine index to zero
                # before the warm loop woke. Its session progress remains visible.
                stub.session_has_stepped = True
            else:
                ownership["initial"] = False
            await asyncio.sleep(0.02)
            assert calls == [1]
            # Even a later eligible-looking state cannot restart this completed loop.
            stub.session_has_stepped = False
            ownership["initial"] = True
            await asyncio.sleep(0.02)
            assert calls == [1]

        await _serve_with_idle_rewarm(
            serve(),
            cast("FullBrainController", stub),
            interval_seconds=0.001,
            continue_while=lambda: ownership["initial"],
        )

    asyncio.run(exercise())
