"""Authored NeuroCraft Fly lab inside the launcher's existing glass arena.

The floor surface is Y=65. The central +/-12 flight area stays flat and clear;
all raised decoration is peripheral. These are ordinary world blocks, so their
collision and light effects are real environmental changes, not neural inputs
invented by this script. No mobs, dropped items, or interaction events are added.

``showcase_commands()`` returns commands for the launcher to apply explicitly.
Running this module only prints that list; it never connects or writes files.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

FLOOR_Y = 64
MAX_COMMANDS = 384
MAX_PLAYGROUND_COMMANDS = 448

# Original three-by-five block lettering. Runs of filled pixels become short
# /fill commands, keeping the authored labels cheap to construct and inspect.
_LETTERS = {
    " ": ("000", "000", "000", "000", "000"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "Y": ("101", "101", "010", "010", "010"),
}


def _fill(low: tuple[int, int, int], high: tuple[int, int, int], block: str) -> str:
    bounds = (*low, *high)
    return "fill " + " ".join(map(str, bounds)) + f" minecraft:{block}"


def _lettering(
    text: str,
    origin: tuple[int, int, int],
    block: str,
    *,
    plane: str = "floor",
) -> list[str]:
    """Floor text faces south; rotated floor text faces west; wall text faces south."""

    def point(u: int, v: int) -> tuple[int, int, int]:
        x, y, z = origin
        if plane == "wall":
            return x + u, y - v, z
        if plane == "rotated_floor":
            return x - v, y, z + u
        return x + u, y, z + v

    commands = []
    for index, letter in enumerate(text):
        for row, pixels in enumerate(_LETTERS[letter]):
            column = 0
            while column < 3:
                if pixels[column] == "0":
                    column += 1
                    continue
                end = column
                while end + 1 < 3 and pixels[end + 1] == "1":
                    end += 1
                a, b = point(index * 4 + column, row), point(index * 4 + end, row)
                commands.append(_fill(a, b, block))
                column = end + 1
    return commands


def showcase_commands() -> tuple[str, ...]:
    """Fixed lab layout; apply after the empty arena is built, before spawning the fly."""

    commands = [
        # Continuous solid floor; inlays replace full blocks and never form steps.
        _fill((-24, FLOOR_Y, -24), (24, FLOOR_Y, 24), "light_gray_concrete"),
        _fill((-12, FLOOR_Y, -12), (12, FLOOR_Y, 12), "smooth_quartz"),
        # A slim perimeter walk and a restrained central registration cross.
        _fill((-24, FLOOR_Y, -24), (24, FLOOR_Y, -23), "polished_andesite"),
        _fill((-24, FLOOR_Y, 23), (24, FLOOR_Y, 24), "polished_andesite"),
        _fill((-24, FLOOR_Y, -22), (-23, FLOOR_Y, 22), "polished_andesite"),
        _fill((23, FLOOR_Y, -22), (24, FLOOR_Y, 22), "polished_andesite"),
        _fill((-2, FLOOR_Y, 0), (2, FLOOR_Y, 0), "light_gray_concrete"),
        _fill((0, FLOOR_Y, -2), (0, FLOOR_Y, 2), "light_gray_concrete"),
        # North: approach lane with orange edge rails and an inward arrow.
        _fill((-16, FLOOR_Y, -22), (16, FLOOR_Y, -18), "smooth_quartz"),
        _fill((-5, FLOOR_Y, -17), (-5, FLOOR_Y, -13), "orange_concrete"),
        _fill((5, FLOOR_Y, -17), (5, FLOOR_Y, -13), "orange_concrete"),
        _fill((0, FLOOR_Y, -17), (0, FLOOR_Y, -13), "orange_concrete"),
        _fill((-1, FLOOR_Y, -14), (1, FLOOR_Y, -14), "orange_concrete"),
        _fill((-2, FLOOR_Y, -15), (2, FLOOR_Y, -15), "orange_concrete"),
        # East: teal landing target, empty until the operator supplies an item.
        _fill((13, FLOOR_Y, -10), (23, FLOOR_Y, 10), "cyan_concrete"),
        _fill((14, FLOOR_Y, -9), (22, FLOOR_Y, 9), "smooth_quartz"),
        _fill((14, FLOOR_Y, -3), (18, FLOOR_Y, 3), "cyan_concrete"),
        _fill((19, FLOOR_Y, -9), (23, FLOOR_Y, 9), "smooth_quartz"),
        # West: warm wooden grooming pad, with an orange gesture marker.
        _fill((-23, FLOOR_Y, -11), (-13, FLOOR_Y, 11), "orange_concrete"),
        _fill((-22, FLOOR_Y, -10), (-14, FLOOR_Y, 10), "birch_planks"),
        _fill((-18, FLOOR_Y, -3), (-14, FLOOR_Y, 3), "smooth_quartz"),
        # South: labeled light apron, bright open bay, shaded bay to its east.
        _fill((-11, FLOOR_Y, 13), (11, FLOOR_Y, 23), "cyan_concrete"),
        _fill((-10, FLOOR_Y, 14), (10, FLOOR_Y, 22), "smooth_quartz"),
        _fill((2, FLOOR_Y, 18), (10, FLOOR_Y, 23), "polished_deepslate"),
        _fill((1, 70, 17), (11, 70, 24), "smooth_stone"),
        _fill((1, 71, 17), (11, 71, 17), "stripped_spruce_log[axis=x]"),
        _fill((10, 65, 18), (10, 69, 24), "stripped_spruce_log[axis=y]"),
        _fill((2, 65, 24), (9, 69, 24), "stripped_spruce_log[axis=y]"),
        _fill((2, 65, 18), (2, 69, 18), "stripped_spruce_log[axis=y]"),
        # North backdrop, built inside the existing glass rather than replacing it.
        _fill((-22, 71, -24), (22, 77, -24), "cyan_concrete"),
        _fill((-14, 65, -24), (14, 69, -24), "polished_andesite"),
    ]
    commands.extend(_lettering("APPROACH", (-15, FLOOR_Y, -22), "orange_concrete"))
    commands.extend(_lettering("FOOD", (23, FLOOR_Y, -7), "cyan_concrete", plane="rotated_floor"))
    commands.extend(_lettering("H", (15, FLOOR_Y, -2), "smooth_quartz"))
    commands.extend(_lettering("GROOM", (-19, FLOOR_Y, -9), "gray_concrete", plane="rotated_floor"))
    commands.extend(_lettering("G", (-17, FLOOR_Y, -2), "orange_concrete"))
    commands.extend(_lettering("LIGHT", (-9, FLOOR_Y, 13), "gray_concrete"))
    commands.extend(_lettering("NEUROCRAFT", (-19, 76, -23), "smooth_quartz", plane="wall"))
    commands.extend(_lettering("FLY LAB", (-13, 69, -23), "smooth_quartz", plane="wall"))

    # Four permanent planted beds. Leaves are persistent; no fruit, fluid, mobs,
    # item entities, random trees, or growth-dependent decoration is introduced.
    for x in (-20, 20):
        for z in (-20, 20):
            commands.extend(
                [
                    _fill((x - 2, 65, z - 3), (x + 2, 65, z + 3), "smooth_quartz"),
                    _fill((x - 1, 65, z - 2), (x + 1, 65, z + 2), "moss_block"),
                    _fill(
                        (x - 1, 66, z - 2),
                        (x + 1, 66, z + 2),
                        "azalea_leaves[persistent=true]",
                    ),
                    _fill(
                        (x, 67, z - 1),
                        (x, 67, z + 1),
                        "flowering_azalea_leaves[persistent=true]",
                    ),
                ]
            )
    # Architectural corner columns and high perimeter rails leave the glass shell
    # and the Y=87 interior ceiling untouched. No overhead beam crosses the centre.
    for x in (-24, 24):
        for z in (-24, 24):
            commands.append(_fill((x, 65, z), (x, 78, z), "quartz_pillar[axis=y]"))
    for edge in (-24, 24):
        commands.append(_fill((-24, 78, edge), (24, 78, edge), "smooth_quartz"))
        commands.append(_fill((edge, 78, -24), (edge, 78, 24), "smooth_quartz"))
    # These wall lights are far from the shaded bay; its light remains world-derived.
    for x in (-24, 24):
        for z in (-12, 0):
            commands.append(_fill((x, 71, z), (x, 72, z), "sea_lantern"))
    if len(commands) > MAX_COMMANDS or any(len(command) > 512 for command in commands):
        raise ValueError("authored showcase command budget exceeded")
    return tuple(commands)


def playground_commands() -> tuple[str, ...]:
    """Opt-in close-up pads; native inputs and model control remain the player's job.

    All additions are flat, full-block floor inlays. No item, actor, event,
    command block or model selection is generated. The existing shaded light bay
    remains available; the central pads are staging marks, not input buttons.
    """

    commands = list(showcase_commands())
    commands.extend(
        [
            _fill((-11, FLOOR_Y, -7), (-3, FLOOR_Y, 7), "orange_concrete"),
            _fill((-10, FLOOR_Y, -6), (-4, FLOOR_Y, 6), "birch_planks"),
            _fill((3, FLOOR_Y, -7), (11, FLOOR_Y, 7), "cyan_concrete"),
            _fill((4, FLOOR_Y, -6), (10, FLOOR_Y, 6), "smooth_quartz"),
            # Spawn yaw is zero (+Z), so this lane begins in front of the fly.
            _fill((-1, FLOOR_Y, 3), (1, FLOOR_Y, 11), "orange_concrete"),
            _fill((0, FLOOR_Y, 3), (0, FLOOR_Y, 10), "smooth_quartz"),
            _fill((-2, FLOOR_Y, -1), (2, FLOOR_Y, 2), "smooth_quartz"),
            _fill((-1, FLOOR_Y, 0), (1, FLOOR_Y, 1), "cyan_concrete"),
        ]
    )
    commands.extend(
        _lettering("BRUSH", (-4, FLOOR_Y, -7), "orange_concrete", plane="rotated_floor")
    )
    commands.extend(_lettering("FOOD", (10, FLOOR_Y, -5), "cyan_concrete", plane="rotated_floor"))
    if len(commands) > MAX_PLAYGROUND_COMMANDS:
        raise ValueError("playground command budget exceeded")
    return tuple(commands)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="print the ordered command list as JSON"
    )
    parser.add_argument(
        "--playground", action="store_true", help="include close-up native-interaction pads"
    )
    args = parser.parse_args(argv)
    commands = playground_commands() if args.playground else showcase_commands()
    print(json.dumps(commands) if args.json else "\n".join(commands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
