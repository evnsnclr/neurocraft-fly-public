"""Deterministic non-Minecraft physics harness for trace and replay verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_controller import (
    ConnectomeController,
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
    MotorVector,
    Observation,
    SensorFrame,
    TraceManifest,
    TraceWriter,
    ValidationResult,
    WorldState,
    canonical_json_bytes,
    parse_json_frame,
    verify_trace,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "data" / "fixtures" / "mcns-v1.0-flight-power-v1.json"
DEFAULT_TRACE = ROOT / "artifacts" / "traces" / "fixture-harness.ndjson"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_code_sha256() -> str:
    package = ROOT / "packages" / "controller-runtime" / "src" / "neurocraft_controller"
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def replay_implementation_sha256() -> str:
    """Bind every Python source file that can affect fixture execution or replay."""

    roots = (
        ROOT / "packages" / "connectome-data" / "src",
        ROOT / "packages" / "controller-runtime" / "src",
        ROOT / "packages" / "neural-runtime" / "src",
        ROOT / "packages" / "protocol" / "src",
    )
    paths = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    paths.add(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        raw = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _arena_sha256() -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "front_wall_z": HarnessWorld.FRONT_WALL_Z,
                "half_width": HarnessWorld.HALF_WIDTH,
                "schema": "neurocraft.fixture-harness.arena.v1",
            }
        )
    ).hexdigest()


@dataclass(slots=True)
class HarnessWorld:
    x: float = 0.0
    y: float = 64.0
    z: float = 0.0
    velocity_x: float = 0.0  # blocks per tick
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    on_ground: bool = True
    horizontal_collision: bool = False
    vertical_collision: bool = False

    RAY_MAX = 6.0
    HALF_WIDTH = 2.0
    FRONT_WALL_Z = 8.0
    MAX_THRUST = 0.04
    MAX_SPEED = 0.6
    GRAVITY = 0.04
    HORIZONTAL_DRAG = 0.91
    VERTICAL_DRAG = 0.98

    def observe(self, controller: ConnectomeController, sequence: int) -> Observation:
        forward = min(self.RAY_MAX, max(0.0, self.FRONT_WALL_Z - self.z))
        back = min(self.RAY_MAX, max(0.0, self.z + 2.0))
        left = min(self.RAY_MAX, max(0.0, self.x + self.HALF_WIDTH))
        right = min(self.RAY_MAX, max(0.0, self.HALF_WIDTH - self.x))
        local_light = max(0, min(15, round(15.0 * self.z / self.FRONT_WALL_Z)))
        return Observation(
            identity=controller.identity,
            observation_seq=sequence,
            game_tick=sequence,
            sent_monotonic_ns=sequence * 50_000_000,
            sensors=SensorFrame(
                position_m=(self.x, self.y, self.z),
                velocity_m_s=(
                    self.velocity_x * 20.0,
                    self.velocity_y * 20.0,
                    self.velocity_z * 20.0,
                ),
                yaw_rad=self.yaw_rad,
                pitch_rad=self.pitch_rad,
                on_ground=self.on_ground,
                horizontal_collision=self.horizontal_collision,
                vertical_collision=self.vertical_collision,
                in_fluid=False,
                local_light=local_light,
                ray_max_distance_m=self.RAY_MAX,
                ray_distances_m=(forward, back, left, right, self.RAY_MAX, 0.0),
            ),
        )

    def apply(self, motors: MotorVector) -> WorldState:
        self.yaw_rad = _wrap_radians(self.yaw_rad + math.radians(6.0) * motors.yaw_torque)
        self.pitch_rad = max(
            -math.radians(89.0),
            min(math.radians(89.0), self.pitch_rad + math.radians(4.0) * motors.pitch_torque),
        )
        self.velocity_x += -math.sin(self.yaw_rad) * motors.forward_thrust * self.MAX_THRUST
        self.velocity_z += math.cos(self.yaw_rad) * motors.forward_thrust * self.MAX_THRUST
        self.velocity_y += motors.vertical_thrust * self.MAX_THRUST - self.GRAVITY
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2)
        if speed > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
            self.velocity_z *= scale

        next_x = self.x + self.velocity_x
        next_y = self.y + self.velocity_y
        next_z = self.z + self.velocity_z
        self.horizontal_collision = not (-self.HALF_WIDTH <= next_x <= self.HALF_WIDTH) or not (
            -2.0 <= next_z <= self.FRONT_WALL_Z
        )
        self.vertical_collision = next_y < 64.0
        self.x = max(-self.HALF_WIDTH, min(self.HALF_WIDTH, next_x))
        self.z = max(-2.0, min(self.FRONT_WALL_Z, next_z))
        if next_y < 64.0:
            self.y = 64.0
            self.velocity_y = 0.0
            self.on_ground = True
        else:
            self.y = next_y
            self.on_ground = False
        self.velocity_x *= self.HORIZONTAL_DRAG
        self.velocity_y *= self.VERTICAL_DRAG
        self.velocity_z *= self.HORIZONTAL_DRAG
        return WorldState(
            position_m=(self.x, self.y, self.z),
            velocity_m_s=(
                self.velocity_x * 20.0,
                self.velocity_y * 20.0,
                self.velocity_z * 20.0,
            ),
            yaw_rad=self.yaw_rad,
            pitch_rad=self.pitch_rad,
            on_ground=self.on_ground,
            horizontal_collision=self.horizontal_collision,
            vertical_collision=self.vertical_collision,
            in_fluid=False,
        )


def _wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def build_manifest(controller: ConnectomeController, graph: RuntimeGraph) -> TraceManifest:
    lock = ROOT / "uv.lock"
    lock_hash = sha256_file(lock) if lock.is_file() else "0" * 64
    arena_hash = _arena_sha256()
    artifacts = ArtifactIdentities(
        source_hash=graph.manifest.source_manifest_sha256,
        canonical_graph_hash=graph.manifest.canonical_graph_sha256,
        runtime_graph_hash=graph.manifest.runtime_identity_sha256,
        fixture_hash=graph.manifest.fixture_file_sha256,
        code_hash=replay_implementation_sha256(),
        dependency_lock_hash=lock_hash,
        adapter_hash=controller.identity.adapter_hash,
        dynamics_hash=controller.identity.dynamics_hash,
        sign_hash=controller.sign_overlay.identity_sha256,
        decoder_hash=controller.identity.decoder_hash,
        arena_hash=arena_hash,
        configuration_hash=controller.configuration.identity_sha256,
        observation_schema_hash=controller.identity.observation_schema_hash,
    )
    return TraceManifest(
        run_id=controller.identity.run_id,
        initial_session_epoch=controller.identity.session_epoch,
        mode=controller.run_mode,
        data_tier=DataTier.MALECNS_FIXTURE,
        claim_scope=ClaimScope.G2,
        artifacts=artifacts,
        game_tick_hz=20.0,
        neural_step_ms=controller.dynamics.neural_step_ms,
        precision=FloatPrecision.FLOAT64,
        backend=controller.dynamics.backend,
        thread_count=1,
        seed=0,
        determinism_level=DeterminismLevel.SAME_BUILD_EXACT,
        machine=platform.machine() or "unknown",
        os=platform.system() or "unknown",
        python=platform.python_version(),
        java="not_used_in_fixture_harness",
        minecraft="not_used_in_fixture_harness",
        fabric="not_used_in_fixture_harness",
        mod="not_used_in_fixture_harness",
        privacy_policy="local_trace_no_secrets_v1",
        expected_chunks=1,
        checkpoint_interval_ticks=100,
    )


def run_harness(*, fixture: Path, output: Path, ticks: int) -> dict[str, Any]:
    if ticks < 1:
        raise ValueError("ticks must be positive")
    graph = load_malecns_fixture(fixture)
    controller = ConnectomeController.create(
        graph,
        run_id="fixture-harness",
        entity_id="harness-fly",
    )
    world = HarnessWorld()
    with TraceWriter(output, build_manifest(controller, graph)) as writer:
        for tick in range(ticks):
            observation = world.observe(controller, tick)
            start = time.perf_counter_ns()
            action = controller.step(observation, command_seq=tick)
            compute_ns = time.perf_counter_ns() - start
            world_after = world.apply(action.motors)
            writer.append(
                causal_record_from_applied_action(
                    controller,
                    record_seq=tick,
                    observation=observation,
                    world_after=world_after,
                    world_after_observation_seq=observation.observation_seq + 1,
                    world_after_game_tick=observation.game_tick + 1,
                    passive_forces=(0.0, -HarnessWorld.GRAVITY, 0.0),
                    compute_ns=compute_ns,
                )
            )
    verified = verify_trace(output)
    return {
        "evidence_boundary": (
            "Deterministic local fixture/physics harness; this is not a Minecraft run or "
            "behavioral-validation result."
        ),
        "file_sha256": verified.file_sha256,
        "final_hash": verified.final_hash,
        "first_tick": verified.first_tick,
        "last_tick": verified.last_tick,
        "path": str(output),
        "record_count": verified.record_count,
        "schema": "neurocraft.fixture-harness.receipt.v1",
    }


def iter_records(path: Path) -> Iterator[CausalRecord]:
    with path.open("rb") as stream:
        for raw in stream:
            envelope = parse_json_frame(raw.rstrip(b"\n"))
            if envelope.get("kind") == "record":
                yield CausalRecord.from_dict(envelope["payload"])


def replay_trace(*, fixture: Path, trace: Path) -> dict[str, Any]:
    verification = verify_trace(trace)
    manifest = verification.manifest
    records = tuple(iter_records(trace))
    if len(records) != verification.record_count or not records:
        raise ValueError("trace record count is empty or inconsistent")
    first = records[0]
    graph = load_malecns_fixture(fixture)
    controller = ConnectomeController.create(
        graph,
        run_id=first.identity.run_id,
        entity_id=first.identity.entity_id,
        session_epoch=first.identity.session_epoch,
    )
    if controller.identity != first.identity:
        raise ValueError("current controller identities do not match trace")
    lock = ROOT / "uv.lock"
    expected_lock = sha256_file(lock) if lock.is_file() else "0" * 64
    artifacts = manifest.artifacts
    if (
        manifest.claim_scope is not ClaimScope.G2
        or manifest.data_tier is not DataTier.MALECNS_FIXTURE
        or manifest.determinism_level is not DeterminismLevel.SAME_BUILD_EXACT
        or artifacts.source_hash != graph.manifest.source_manifest_sha256
        or artifacts.canonical_graph_hash != graph.manifest.canonical_graph_sha256
        or artifacts.runtime_graph_hash != graph.manifest.runtime_identity_sha256
        or artifacts.fixture_hash != sha256_file(fixture)
        or artifacts.code_hash != replay_implementation_sha256()
        or artifacts.dependency_lock_hash != expected_lock
        or artifacts.adapter_hash != controller.identity.adapter_hash
        or artifacts.dynamics_hash != controller.identity.dynamics_hash
        or artifacts.sign_hash != controller.sign_overlay.identity_sha256
        or artifacts.decoder_hash != controller.identity.decoder_hash
        or artifacts.arena_hash != _arena_sha256()
        or artifacts.configuration_hash != controller.configuration.identity_sha256
        or artifacts.observation_schema_hash != controller.identity.observation_schema_hash
    ):
        raise ValueError("fixture trace manifest does not match the current replay environment")
    indexed_observations = {
        (record.observation.observation_seq, record.observation.game_tick): (index, record)
        for index, record in enumerate(records)
    }
    if len(indexed_observations) != len(records):
        raise ValueError("fixture trace contains duplicate observation lineage")
    final_observation = records[-1].observation
    world_transitions_cross_linked = 0
    terminal_world_transitions_not_cross_linked = 0
    for record_index, record in enumerate(records):
        if record.validation is not ValidationResult.APPLIED or record.command_seq is None:
            raise ValueError("fixture harness replay requires applied commands")
        if world_state_from_observation(record.observation) != record.world_before:
            raise ValueError(f"world-before mismatch at record {record.record_seq}")
        lineage = (record.world_after_observation_seq, record.world_after_game_tick)
        target = indexed_observations.get(lineage)
        if target is not None:
            target_index, target_record = target
            if target_index <= record_index or record.world_after != target_record.world_before:
                raise ValueError(f"world-after lineage mismatch at record {record.record_seq}")
            world_transitions_cross_linked += 1
        elif (
            record.world_after_observation_seq > final_observation.observation_seq
            and record.world_after_game_tick > final_observation.game_tick
        ):
            terminal_world_transitions_not_cross_linked += 1
        else:
            raise ValueError(
                f"world-after lineage is missing within trace at record {record.record_seq}"
            )
        if controller.engine.current_state().activity != record.state_before_activity:
            raise ValueError(f"full state-before mismatch at record {record.record_seq}")
        action = controller.step(record.observation, command_seq=record.command_seq)
        evidence = controller.last_evidence
        if evidence is None:
            raise ValueError(f"controller evidence missing at record {record.record_seq}")
        if (
            action.state_digest != record.state_after_digest
            or action.motors != record.decoder_output
            or action.motors != record.applied_action
            or action.neural_step_start != record.neural_step_start
            or action.neural_step_end != record.neural_step_end
            or dict(evidence.encoded_stimulation) != dict(record.encoded_stimulation)
            or evidence.state_after_activity != record.state_after_activity
        ):
            raise ValueError(f"causal replay mismatch at record {record.record_seq}")
    return {
        "evidence_boundary": (
            "Same-build neural replay of a non-Minecraft fixture harness with exact "
            "in-trace world-state lineage cross-links; terminal lineage is reported separately."
        ),
        "file_sha256": verification.file_sha256,
        "final_state_sha256": records[-1].state_after_digest,
        "record_count": len(records),
        "world_transitions_cross_linked": world_transitions_cross_linked,
        "terminal_world_transitions_not_cross_linked": (
            terminal_world_transitions_not_cross_linked
        ),
        "schema": "neurocraft.fixture-harness.replay.v1",
        "status": "exact_match",
        "claim_scope": ClaimScope.G2.value,
        "replay_implementation_sha256": replay_implementation_sha256(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="write a new no-clobber harness trace")
    run.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    run.add_argument("--output", type=Path, default=DEFAULT_TRACE)
    run.add_argument("--ticks", type=int, default=200)
    replay = subparsers.add_parser("replay", help="verify and re-execute a harness trace")
    replay.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    replay.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        receipt = run_harness(fixture=args.fixture, output=args.output, ticks=args.ticks)
    else:
        receipt = replay_trace(fixture=args.fixture, trace=args.trace)
    json.dump(receipt, sys.stdout, allow_nan=False, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
