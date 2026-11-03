#!/usr/bin/env python3
"""Run or exactly replay the first-execution FlyGym embodiment configuration.

This is the entry point behind ``EMBODIMENT-02``: the first real execution of the
FlyGym/NeuroMechFly body through the seam's unchanged recording loop
(`run_embodiment`) and pinned `ConnectomeController`, and its brain-and-body
exact replay. It requires the optional flygym lane
(``UV_CACHE_DIR=.uv-cache uv pip install "flygym==2.1.0"``); the core environment
deliberately does not install it.

Construction evidence only. The decoder below assigns steering-fixture wing-motor
bodies to NeuroMechFly leg DOFs one-to-one in sorted body-ID order, purely so the
real body can be constructed and stepped: 48 disjoint single-member pools over the
first 48 of the fixture's 52 wing-motor bodies, bound to the skeleton's actuated
DOF order exactly as the skeleton reports it. No behavioral or biological meaning
attaches to that assignment, and nothing here is a Faithful Organism Mode claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from neurocraft_connectome import RuntimeGraph, load_malecns_fixture
from neurocraft_connectome.flight_pin import MALECNS_FLIGHT_FIXTURE_V1_PIN
from neurocraft_controller import ConnectomeController, ControllerConfiguration
from neurocraft_controller.flight_pools import (
    LEFT_WING_MOTOR_BODY_IDS,
    RIGHT_WING_MOTOR_BODY_IDS,
)
from neurocraft_embodiment import AdhesionBinding, FlyGymMotorDecoder, JointBinding
from neurocraft_embodiment.flygym_body import (
    FlyGymEmbodiment,
    _actuated_dof_names,
    require_flygym,
)
from neurocraft_embodiment.loop import replay_embodiment_trace, run_embodiment
from neurocraft_neural import MotorPool, PoolMember

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "mcns-v1.0-flight-steering-v1.json"
FLIGHT_FIXTURE_BYTES = 284_949
DEFAULT_TRACE = REPOSITORY_ROOT / "artifacts" / "traces" / "flygym-first-execution.ndjson"
RUN_ID = "flygym-first-execution"
ENTITY_ID = "flygym-fly"
ADHESION_LEG_ORDER = ("lf", "lm", "lh", "rf", "rm", "rh")


def build_graph() -> RuntimeGraph:
    return load_malecns_fixture(
        FLIGHT_FIXTURE,
        expected_pin=MALECNS_FLIGHT_FIXTURE_V1_PIN,
        expected_file_bytes=FLIGHT_FIXTURE_BYTES,
    )


def build_configuration() -> ControllerConfiguration:
    """The steering-fixture controller boundary used by the recorded run."""

    return ControllerConfiguration(
        left_motor_body_id=LEFT_WING_MOTOR_BODY_IDS[0],
        right_motor_body_id=RIGHT_WING_MOTOR_BODY_IDS[0],
    )


def build_embodiment(graph: RuntimeGraph) -> FlyGymEmbodiment:
    """The real body, bound to the skeleton's own reported actuated DOF order."""

    modules = require_flygym()
    anatomy = modules.anatomy
    skeleton = anatomy.Skeleton(
        axis_order=anatomy.AxisOrder.YAW_PITCH_ROLL,
        joint_preset=anatomy.JointPreset.LEGS_ONLY,
    )
    real_dof_order = _actuated_dof_names(
        skeleton.get_actuated_dofs_from_preset(anatomy.ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    )
    bodies = tuple(sorted(set(LEFT_WING_MOTOR_BODY_IDS) | set(RIGHT_WING_MOTOR_BODY_IDS)))
    required = len(real_dof_order) + len(ADHESION_LEG_ORDER)
    if len(bodies) < required:
        raise SystemExit(
            f"the steering fixture exposes {len(bodies)} wing-motor bodies but the "
            f"body needs {required} disjoint single-member pools"
        )
    joints = tuple(
        JointBinding(
            dof=dof,
            pool=MotorPool(
                name=f"exec_joint_{index:02d}",
                members=(PoolMember(bodies[index], 1.0),),
                gain=1.0,
            ),
            neutral_rad=0.0,
            half_range_rad=0.2,
        )
        for index, dof in enumerate(real_dof_order)
    )
    adhesion = tuple(
        AdhesionBinding(
            leg=leg,
            pool=MotorPool(
                name=f"exec_adhesion_{index}",
                members=(PoolMember(bodies[len(real_dof_order) + index], 1.0),),
                gain=1.0,
            ),
        )
        for index, leg in enumerate(ADHESION_LEG_ORDER)
    )
    free_parameters = (len(real_dof_order) + len(ADHESION_LEG_ORDER)) * 3 + len(real_dof_order) * 2
    decoder = FlyGymMotorDecoder(graph, joints, adhesion, declared_free_parameters=free_parameters)
    return FlyGymEmbodiment(decoder)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="write a new no-clobber FlyGym embodiment trace")
    run.add_argument("--output", type=Path, default=DEFAULT_TRACE)
    run.add_argument("--ticks", type=int, default=3)
    replay = subparsers.add_parser(
        "replay", help="re-execute brain and body from a recorded trace, requiring exactness"
    )
    replay.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graph = build_graph()
    if args.command == "run":
        controller = ConnectomeController.create(
            graph,
            run_id=RUN_ID,
            entity_id=ENTITY_ID,
            configuration=build_configuration(),
        )
        receipt: dict[str, Any] = run_embodiment(
            controller=controller,
            graph=graph,
            embodiment=build_embodiment(graph),
            output=args.output,
            ticks=args.ticks,
        )
    else:
        receipt = replay_embodiment_trace(
            trace=args.trace,
            graph_factory=build_graph,
            embodiment_factory=lambda: build_embodiment(build_graph()),
            configuration=build_configuration(),
        )
    json.dump(receipt, sys.stdout, allow_nan=False, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
