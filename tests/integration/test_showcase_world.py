from __future__ import annotations

from collections import Counter

from scripts.showcase_world import (
    FLOOR_Y,
    MAX_COMMANDS,
    MAX_PLAYGROUND_COMMANDS,
    playground_commands,
    showcase_commands,
)


def test_authored_world_preserves_containment_and_clear_flight_with_four_stations() -> None:
    commands = showcase_commands()
    assert commands == showcase_commands()
    assert 0 < len(commands) <= MAX_COMMANDS
    blocks: dict[tuple[int, int, int], str] = {}
    for command in commands:
        assert len(command) <= 512
        verb, *coordinates, block = command.split()
        # A command list consisting only of block fills cannot spawn stimuli,
        # run model/clock commands, or carry hidden command-block instructions.
        assert verb == "fill" and len(coordinates) == 6
        assert block.startswith("minecraft:") and "{" not in block
        assert not any(name in block for name in ("command_block", "barrier", "water", "lava"))
        x1, y1, z1, x2, y2, z2 = map(int, coordinates)
        assert -24 <= x1 <= x2 <= 24 and -24 <= z1 <= z2 <= 24
        assert FLOOR_Y <= y1 <= y2 <= 78
        assert (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1) <= 32768
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    blocks[x, y, z] = block
    assert all((x, 64, z) in blocks for x in range(-24, 25) for z in range(-24, 25))
    assert all(y == FLOOR_Y or abs(x) > 12 or abs(z) > 12 for x, y, z in blocks)
    # Inspect the final composed map, not just the presence of overwritten commands.
    assert blocks[0, 64, -13] == "minecraft:orange_concrete"  # approach arrow
    assert blocks[16, 64, 0] == "minecraft:smooth_quartz"  # landing H
    assert blocks[-17, 64, 0] == "minecraft:orange_concrete"  # grooming G
    assert blocks[5, 64, 21] == "minecraft:polished_deepslate"  # shaded bay floor
    assert blocks[5, 70, 21] == "minecraft:smooth_stone"  # actual solid shade roof
    assert not any(x == -5 and z == 21 and y > FLOOR_Y for x, y, z in blocks)  # open light bay
    greenery = Counter(block for block in blocks.values() if "leaves" in block)
    assert greenery["minecraft:azalea_leaves[persistent=true]"] == 60
    assert greenery["minecraft:flowering_azalea_leaves[persistent=true]"] == 12


def test_playground_adds_only_flat_staging_marks_without_creating_stimuli() -> None:
    original = showcase_commands()
    commands = playground_commands()
    assert commands[: len(original)] == original
    assert len(original) < len(commands) <= MAX_PLAYGROUND_COMMANDS
    additions = commands[len(original) :]
    for command in additions:
        verb, *coordinates, block = command.split()
        assert verb == "fill" and len(coordinates) == 6
        x1, y1, z1, x2, y2, z2 = map(int, coordinates)
        assert y1 == y2 == FLOOR_Y
        assert -12 <= x1 <= x2 <= 12 and -12 <= z1 <= z2 <= 12
        assert "{" not in block
        assert block in {
            "minecraft:orange_concrete",
            "minecraft:birch_planks",
            "minecraft:cyan_concrete",
            "minecraft:smooth_quartz",
        }
