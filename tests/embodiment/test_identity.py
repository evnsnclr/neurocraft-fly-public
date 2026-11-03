from __future__ import annotations

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_embodiment import (
    CHANNEL_SOURCE_CONFIG_CONSTANT,
    CHANNEL_SOURCE_DYNAMIC,
    CHANNEL_SOURCE_LITERAL_CONSTANT,
    CONSUMES_NEURAL_STATE,
    CONSUMES_PRIMITIVE_MOTOR_VECTOR,
    BoundaryIdentity,
    ChannelDeclaration,
    EmbodimentIdentity,
    EmbodimentIdentityError,
    sorted_provenance,
)
from neurocraft_protocol import DeterminismLevel


def make_boundary(role: str = "sensory", **overrides: object) -> BoundaryIdentity:
    values: dict[str, object] = {
        "role": role,
        "schema": f"neurocraft.test.{role}.v1",
        "identity_sha256": "a" * 64,
        "channel_count": 4,
        "channels": tuple(
            ChannelDeclaration(name, CHANNEL_SOURCE_DYNAMIC, "measured")
            for name in ("alpha", "beta", "gamma", "delta")
        ),
        "declared_dynamic_channels": 4,
        "free_parameters": 12,
    }
    values.update(overrides)
    return BoundaryIdentity(**values)  # type: ignore[arg-type]


def make_identity(**overrides: object) -> EmbodimentIdentity:
    values: dict[str, object] = {
        "embodiment_id": "test-body",
        "embodiment_version": "v1",
        "configuration_sha256": "b" * 64,
        "sensory_boundary": make_boundary("sensory"),
        "motor_boundary": make_boundary("motor", identity_sha256="c" * 64),
        "motor_currency": CONSUMES_PRIMITIVE_MOTOR_VECTOR,
        "seed": 0,
        "control_hz": 20.0,
        "physics_steps_per_control_tick": 25,
        "physics_timestep_s": 0.002,
        "determinism_level": DeterminismLevel.SAME_BUILD_EXACT,
    }
    values.update(overrides)
    return EmbodimentIdentity(**values)  # type: ignore[arg-type]


def test_identity_hash_is_stable_and_covers_every_field() -> None:
    identity = make_identity()
    assert len(identity.identity_sha256) == 64
    assert identity.identity_sha256 == make_identity().identity_sha256

    baseline = identity.identity_sha256
    variants = (
        make_identity(embodiment_id="other-body"),
        make_identity(embodiment_version="v2"),
        make_identity(configuration_sha256="d" * 64),
        make_identity(seed=1),
        make_identity(motor_currency=CONSUMES_NEURAL_STATE),
        make_identity(determinism_level=DeterminismLevel.NUMERICAL_TOLERANCE),
        make_identity(sensory_boundary=make_boundary("sensory", free_parameters=13)),
        make_identity(motor_boundary=make_boundary("motor", identity_sha256="e" * 64)),
        make_identity(provenance=sorted_provenance({"library_version": "1.2.3"})),
        make_identity(physics_steps_per_control_tick=50, physics_timestep_s=0.001),
    )
    hashes = {variant.identity_sha256 for variant in variants}
    assert baseline not in hashes
    assert len(hashes) == len(variants)


def test_identity_rejects_a_clock_that_does_not_close() -> None:
    with pytest.raises(EmbodimentIdentityError, match="must equal 1 / control_hz"):
        make_identity(physics_steps_per_control_tick=25, physics_timestep_s=0.003)


def test_identity_rejects_swapped_boundary_roles() -> None:
    with pytest.raises(EmbodimentIdentityError, match="wrong roles"):
        make_identity(sensory_boundary=make_boundary("motor"))


def test_identity_rejects_an_undeclared_motor_currency() -> None:
    with pytest.raises(EmbodimentIdentityError, match="motor_currency"):
        make_identity(motor_currency="whatever_the_body_feels_like")


def test_identity_requires_sorted_unique_provenance() -> None:
    with pytest.raises(EmbodimentIdentityError, match="sorted"):
        make_identity(provenance=(("zeta", "1"), ("alpha", "2")))
    with pytest.raises(EmbodimentIdentityError, match="unique"):
        make_identity(provenance=(("alpha", "1"), ("alpha", "2")))


def test_boundary_identity_rejects_zero_capacity_declarations() -> None:
    with pytest.raises(EmbodimentIdentityError, match="channel_count"):
        make_boundary(channel_count=0)
    with pytest.raises(EmbodimentIdentityError, match="free_parameters"):
        make_boundary(free_parameters=0)


def test_boundary_identity_refuses_a_channel_count_its_channel_list_contradicts() -> None:
    """MEDIUM-4: the count was decorative and was wrong in both shipped bodies.

    ``__post_init__`` only asserted ``channel_count >= 1``, so the reference body
    declared 11 next to a ten-entry channel list and the FlyGym body declared 11 next
    to a comment deriving 20. The count is now asserted against the names, the way
    ``flygym_boundary._assert_declared`` asserts the free-parameter count.
    """

    three = tuple(
        ChannelDeclaration(name, CHANNEL_SOURCE_DYNAMIC, "measured")
        for name in ("only", "three", "names")
    )
    with pytest.raises(EmbodimentIdentityError, match="channel-count mismatch"):
        make_boundary(channel_count=11, channels=three, declared_dynamic_channels=3)
    with pytest.raises(EmbodimentIdentityError, match="channel names must be unique"):
        make_boundary(
            channel_count=2,
            channels=tuple(
                ChannelDeclaration(name, CHANNEL_SOURCE_DYNAMIC, "measured")
                for name in ("same", "same")
            ),
            declared_dynamic_channels=2,
        )
    with pytest.raises(EmbodimentIdentityError, match=r"boundary\.channels must be a tuple"):
        make_boundary(channels=[*three, three[0]])


def test_identity_hash_covers_the_channel_names() -> None:
    baseline = make_identity().identity_sha256
    renamed = make_identity(
        sensory_boundary=make_boundary(
            "sensory",
            channels=tuple(
                ChannelDeclaration(name, CHANNEL_SOURCE_DYNAMIC, "measured")
                for name in ("alpha", "beta", "gamma", "epsilon")
            ),
        )
    )
    assert renamed.identity_sha256 != baseline


def test_boundary_identity_refuses_a_dynamic_count_its_channel_records_contradict() -> None:
    """MEDIUM-3: the count equality could not fire, so a second declaration was added.

    Every call site passed ``len(names)`` as ``channel_count``, so the equality
    ``_assert_channels`` enforced was structurally unfireable for a real body. The
    number that *cannot* be derived from anywhere else in the repository is how many
    of a boundary's channels actually vary, so that is now declared and asserted.
    """

    with pytest.raises(EmbodimentIdentityError, match="dynamic-channel mismatch"):
        make_boundary(declared_dynamic_channels=3)
    with pytest.raises(EmbodimentIdentityError, match="dynamic-channel mismatch"):
        make_boundary(
            channels=(
                ChannelDeclaration("alpha", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("beta", CHANNEL_SOURCE_LITERAL_CONSTANT, "false_always"),
                ChannelDeclaration("gamma", CHANNEL_SOURCE_CONFIG_CONSTANT, "from_config"),
                ChannelDeclaration("delta", CHANNEL_SOURCE_DYNAMIC, "measured"),
            ),
            declared_dynamic_channels=4,
        )


def test_channel_declaration_rejects_an_undeclared_source() -> None:
    with pytest.raises(EmbodimentIdentityError, match="channel source"):
        ChannelDeclaration("alpha", "body_sourced", "measured")
    with pytest.raises(EmbodimentIdentityError, match="derivation"):
        ChannelDeclaration("alpha", CHANNEL_SOURCE_DYNAMIC, "")


def test_the_derived_channel_tallies_cannot_rot() -> None:
    """The discriminating numbers are read off the records, never declared twice."""

    boundary = make_boundary(
        channels=(
            ChannelDeclaration("alpha", CHANNEL_SOURCE_DYNAMIC, "measured"),
            ChannelDeclaration("beta", CHANNEL_SOURCE_LITERAL_CONSTANT, "false_always"),
            ChannelDeclaration("gamma", CHANNEL_SOURCE_CONFIG_CONSTANT, "from_config"),
            ChannelDeclaration("delta", CHANNEL_SOURCE_DYNAMIC, "measured"),
        ),
        declared_dynamic_channels=2,
    )
    assert boundary.channel_names == ("alpha", "beta", "gamma", "delta")
    assert boundary.dynamic_channel_count == 2
    assert boundary.constant_channel_count == 2
    material = boundary.to_identity_material()
    assert "dynamic_channel_count" not in material
    assert "declared_dynamic_channels" not in material
    assert material["channels"][1] == {
        "derivation": "false_always",
        "name": "beta",
        "source": CHANNEL_SOURCE_LITERAL_CONSTANT,
    }


def test_identity_hash_covers_the_per_channel_source_and_derivation() -> None:
    """Two boundaries carrying the same names but sourcing them differently differ."""

    baseline = make_identity().identity_sha256
    retagged = make_identity(
        sensory_boundary=make_boundary(
            "sensory",
            channels=(
                ChannelDeclaration("alpha", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("beta", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("gamma", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("delta", CHANNEL_SOURCE_LITERAL_CONSTANT, "false_always"),
            ),
            declared_dynamic_channels=3,
        )
    )
    assert retagged.identity_sha256 != baseline

    rederived = make_identity(
        sensory_boundary=make_boundary(
            "sensory",
            channels=(
                ChannelDeclaration("alpha", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("beta", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("gamma", CHANNEL_SOURCE_DYNAMIC, "measured"),
                ChannelDeclaration("delta", CHANNEL_SOURCE_DYNAMIC, "a_different_derivation"),
            ),
        )
    )
    assert rederived.identity_sha256 != baseline


# -- the shipped digests, pinned ------------------------------------------------
#
# R-F: this suite pinned exactly two 64-hex digests -- the reference arena's
# configuration hash and the fixture graph's -- and not one ``BoundaryIdentity`` or
# ``EmbodimentIdentity`` value anywhere. The consequence was measured, not argued:
# changing the *value* of ``DeterminismLevel.SAME_BUILD_EXACT`` in
# ``packages/protocol/.../trace.py`` rotates every embodiment identity in the
# repository, because ``EmbodimentIdentity.to_identity_material`` folds
# ``determinism_level.value`` in -- and the whole suite stayed green, because nothing
# knew what the old value had been.
#
# The reassurance offered at the time was that no receipt pinned any rotated value.
# That is the vulnerability stated as comfort: nothing pinned the new values either,
# so an identity could rotate as often as anyone liked and no artifact would notice.
# The six digests below are the ones a trace, a receipt or a comparison would be
# recorded under, so they are pinned the same way the twenty channel names are.
#
# What this is, precisely: a suite-run tripwire, not a fail-closed guard. A rotation
# is detected when the suite runs, by comparison against a recorded number. It cannot
# prevent one. In particular a *reorder* of two ``SensorFrame`` fields -- swapping
# ``yaw_rad`` and ``pitch_rad`` in the protocol dataclass -- imports cleanly, changes
# no name and no count, and rotates these digests through the channel ordering; it is
# caught here and by the ordered name pin in ``test_seam_channels``, and by nothing
# at runtime. Rotating a digest deliberately means editing this block, which is the
# point: the edit is the decision, recorded in the diff.

REFERENCE_EMBODIMENT_SHA256 = "3103af43a99572b75e5ef220c592fb77a5c18568cd22411b4d5ce3e58a8b19ed"
REFERENCE_SENSORY_BOUNDARY_SHA256 = (
    "4efc3e44ca79fbf3c34cb4cc2d4c7480a53a0644f5737fa4711d823694849efe"
)
REFERENCE_MOTOR_BOUNDARY_SHA256 = "aca18951e4ee859fdd97415998ff78c7a12533ee318cfada88c992b2ef9148e3"
FLYGYM_STUB_EMBODIMENT_SHA256 = "d48dbd7f88fdf813a181e195ed0e00acede6d84356a2c30f4efc2a9edb464a73"
FLYGYM_STUB_SENSORY_BOUNDARY_SHA256 = (
    "35009fa97e03b558a1a21d8aca0fb2169d9572f595522d73c376a5ac1a5dba66"
)
FLYGYM_STUB_MOTOR_BOUNDARY_SHA256 = (
    "289e789b496f5d4cfc330e53b9c82e1ee364fae5c56a33ae6ef9c5bf1977798c"
)


def test_the_reference_body_identities_are_the_pinned_ones() -> None:
    """The body whose trace and replay receipts every other test is written against."""

    from neurocraft_embodiment import ReferenceArena, ReferenceEmbodiment

    identity = ReferenceEmbodiment(ReferenceArena()).identity
    assert identity.identity_sha256 == REFERENCE_EMBODIMENT_SHA256
    assert identity.sensory_boundary.identity_sha256 == REFERENCE_SENSORY_BOUNDARY_SHA256
    assert identity.motor_boundary.identity_sha256 == REFERENCE_MOTOR_BOUNDARY_SHA256


def test_the_flygym_stub_body_identities_are_the_pinned_ones(graph: RuntimeGraph) -> None:
    """The stub-backed FlyGym body, pinned for the same reason and with a caveat.

    This identity is the one a *stub-backed* construction produces. flygym is not
    installed, has never been executed, and not one MuJoCo step has run; the pin says
    that the identity this repository can build has not moved, and says nothing about
    any identity a real install would produce. Its motor half folds in the fixture
    graph's runtime identity, so it pins that too.
    """

    from neurocraft_embodiment import FlyGymEmbodiment

    from .conftest import make_fixture_decoder
    from .test_flygym_body import _stub_modules

    identity = FlyGymEmbodiment(
        make_fixture_decoder(graph),
        modules=_stub_modules(),
    ).identity
    assert identity.identity_sha256 == FLYGYM_STUB_EMBODIMENT_SHA256
    assert identity.sensory_boundary.identity_sha256 == FLYGYM_STUB_SENSORY_BOUNDARY_SHA256
    assert identity.motor_boundary.identity_sha256 == FLYGYM_STUB_MOTOR_BOUNDARY_SHA256


def test_the_two_bodies_have_no_digest_in_common() -> None:
    """Six distinct pins, so a copy-paste error cannot make one of them decorative."""

    pinned = {
        REFERENCE_EMBODIMENT_SHA256,
        REFERENCE_SENSORY_BOUNDARY_SHA256,
        REFERENCE_MOTOR_BOUNDARY_SHA256,
        FLYGYM_STUB_EMBODIMENT_SHA256,
        FLYGYM_STUB_SENSORY_BOUNDARY_SHA256,
        FLYGYM_STUB_MOTOR_BOUNDARY_SHA256,
    }
    assert len(pinned) == 6
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in pinned)


def test_the_boundary_digests_are_inside_the_embodiment_digest() -> None:
    """Pinning a boundary is not independent decoration; it is a component of the body.

    If a boundary hash were not folded into the embodiment material, pinning it would
    guard a number nothing downstream carries. This asserts the containment the pins
    above rely on, over the material rather than over the docstring.
    """

    material = make_identity().to_identity_material()
    assert material["sensory_boundary"]["identity_sha256"] == "a" * 64
    assert material["motor_boundary"]["identity_sha256"] == "c" * 64
    assert material["determinism_level"] == DeterminismLevel.SAME_BUILD_EXACT.value
