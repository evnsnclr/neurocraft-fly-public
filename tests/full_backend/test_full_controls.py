"""Full-scale control views: identity, construction, and the faithful pin.

Everything here runs on a small synthetic CSR so it executes on CI, where the
25.6 M-edge artifact is absent. The one artifact-shaped check at the bottom
skips when the bytes are not present, exactly like
``tests/full_backend/test_real_artifact.py``.

``test_historical_faithful_dynamics_identity_math_is_unchanged`` reconstructs
the historical ``5da651e7...`` identity from captured historical build metadata,
independently of later calibrations. This checks identity calculation without claiming
to execute the historical binary. A fresh compile can have different bytes, even
with identical source and compiler flags: Mach-O embeds the absolute install name.
A separate live-kernel test verifies that those current bytes receive their own
dynamics identity, unchanged across thread counts. Calibration provenance binds
its own measured kernel; local compilation does not reproduce its measurements.

Two source pins sit beside it, and they are pins on two different things:

``IDENTITY_HASHED_SOURCES``
    Files whose **bytes are digested** into an identity. ``full_view.py``
    hashes itself through ``view_construction_sha256()`` into
    ``ArrayDynamicsConfig.identity_material``; ``_spmv.c`` reaches the same
    material through the kernel identity's ``kernel_source_sha256``;
    ``neurocraft_connectome/{loader,model}.py`` are hashed together by
    ``_loader_implementation_sha256`` into every fixture graph identity. Edit
    one byte of any of them and a published hash is wrong.

``BEHAVIOUR_BOUND_SOURCES``
    Files with **no digest of their own anywhere**, whose behaviour
    nonetheless determines those identities: the dynamics material's shape
    (``full_records.py``), which kernel loads and how its identity is built
    (``spmv_kernel.py``), how the engine assembles state material
    (``full_engine.py``), and how overlays derive their identities
    (``neurocraft_neural/model.py``). They are pinned here too, but labelled
    honestly: an edit there can move a published hash while no digest of the
    file moves, which is the harder failure to notice.

``_spmv.c`` is in the first list for a specific reason. Appending one newline
to it does not fail the suite on its own: the ``_spmv.build.json`` sidecar
stops matching, ``load_c_kernel`` refuses, and every C-kernel test **skips**
green. Only a byte pin catches that.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neurocraft_neural.full_controls import (
    CONTROL_IDENTITY_SCHEMA,
    CONTROL_RNG_DESCRIPTION,
    FullControlError,
    FullControlIdentity,
    FullControlKind,
    FullControlSignPolicy,
    FullControlView,
    assert_is_permutation,
    build_control_view,
    control_permutation,
    load_permutation,
    permutation_path,
    permutation_sha256,
    rebuild_control_view,
    write_permutation,
)
from neurocraft_neural.full_engine import ArrayNativeEngine
from neurocraft_neural.full_records import ArrayDynamicsConfig
from neurocraft_neural.full_view import (
    FullRuntimeView,
    ViewProvenance,
    build_view,
    view_construction_sha256,
)
from neurocraft_neural.model import ZERO_SHA256
from neurocraft_neural.spmv_kernel import (
    KernelIdentity,
    NumpyReduceatKernel,
    SpmvKernelError,
    c_kernel_identity_material,
    library_sha256,
    load_c_kernel,
)
from numpy.typing import NDArray

from tests.full_backend.helpers import (
    FULL_ARTIFACT_DIRECTORY,
    FsumReference,
    SyntheticCsr,
    synthetic_csr,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The calibration belongs to the exact measured binary, not every rebuild.
LOOM_CALIBRATION = REPO_ROOT / "data/calibration/full-readout-calibration-loom-v1.json"
#: Kernel identity measured for the native Windows-port source's macOS
#: calibration (relative-install-name library e16576a5..., 2026-10-24). Other binaries
#: keep their own identities and require their own runtime calibrations.
CALIBRATION_KERNEL_SHA256 = "4fa592bf7ab324d16ecc3e5826ab6eac5d9fbd025858828aad0eeb0cba560610"

#: Captured historical _spmv.build.json material. FULL-ENGINE-02 records this
#: library digest, source digest, kernel identity ac916a89..., and dynamics
#: identity 5da651e7.... These literals never follow the current local sidecar.
#: No shared library is loaded to reconstruct the historical identity below.
HISTORICAL_KERNEL_BUILD: dict[str, Any] = {
    "abi_version": 2,
    "compiler_id": "clang",
    "compiler_version": "Apple clang version 17.0.0 (clang-1700.6.4.2)",
    "flags": [
        "-O3",
        "-std=c99",
        "-march=native",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-pthread",
        "-Wall",
        "-Wextra",
    ],
    "library_filename": "_spmv.so",
    "library_sha256": "ce0eb6631925f5792940a3c6383fe2825e5f8df82e809c99aa4253ce961ba337",
    "schema": "neurocraft.spmv.build.v2",
    "source_filename": "_spmv.c",
    "source_sha256": "903852b5528c25b9c75eae5defab71388594fce4b2df028e5412af3a4b257e26",
    "target_triple": "arm64-apple-darwin24.6.0",
}
HISTORICAL_KERNEL_SHA256 = "ac916a89fe49ca91a2d6a533e4c72a518e209f3e74eb56b93bf58f0a78a75939"

#: Files whose BYTES are digested into a published identity. These are the
#: ``main``-at-``e1edc27`` bytes except the explicit native-Windows kernel port
#: below. Historical kernel/build/calibration metadata above is unchanged.
#: A future lane that legitimately edits one
#: must update the constant here **in the same change** as it regenerates the
#: calibrations and readout bindings the rotation invalidates (CLAUDE.md,
#: "Identity rotation is a practical hazard"). ``full_view.py``'s entry is
#: cross-checked below against the digest the runtime itself computes, so the
#: constant cannot drift from the value the identity actually uses.
IDENTITY_HASHED_SOURCES: dict[str, str] = {
    # self-hashed by view_construction_sha256() into the dynamics material
    "packages/neural-runtime/src/neurocraft_neural/full_view.py": (
        "56071d645a3d82329d2adac88a7f47eda8c90ec3e77115b240cac84d0c3d01fb"
    ),
    # digested into the kernel identity as kernel_source_sha256, which the
    # dynamics material folds; a newline here otherwise only makes the C kernel
    # refuse and every C-kernel test skip green
    # Superseded for the ABI-preserving Win32/pthread native port (2026-10-24).
    # New binaries and runtime calibrations receive their own identities.
    "packages/neural-runtime/src/neurocraft_neural/_spmv.c": (
        "b92dc17c593450d861b2673a86505d35002d06f8788f00662f7b10a60ecbf9a6"
    ),
    # hashed together by _loader_implementation_sha256 into every fixture graph
    # identity (loader.py:43-53)
    "packages/connectome-data/src/neurocraft_connectome/loader.py": (
        "91f4283f1ff2011714ba93b8804d87d660a3e43bbddf1a93ba260f02e675b18c"
    ),
    "packages/connectome-data/src/neurocraft_connectome/model.py": (
        "f912a93a547e43cdebafbf92e11d5df930b86fcc77785dfcce108999426d526f"
    ),
}

#: Files with NO digest of their own in any identity, whose behaviour
#: nonetheless determines published identities. Pinned so an edit is visible in
#: review, but the label is the honest one: an edit here can rotate a hash
#: without any digest of the file changing, so this pin is a tripwire, not a
#: derivation.
BEHAVIOUR_BOUND_SOURCES: dict[str, str] = {
    # 2026-10-25 transaction/snapshot hardening (ported to the trunk on
    # 2026-11-03) was reviewed separately. The ArrayDynamicsConfig identity
    # recipe and valid-step/state digest recipes are unchanged; the
    # historical-kernel reconstruction test below re-derives the committed
    # dynamics hash, and transaction regressions preserve valid replay while
    # rejecting invalid mutation. Snapshots gain their own new content digest
    # rather than changing any historical identity material.
    # ArrayDynamicsConfig.identity_material -- the dynamics material's shape
    "packages/neural-runtime/src/neurocraft_neural/full_records.py": (
        "f7fd2220387db68150835305c7b914f38c1139341703d2f23df8ea612b212998"
    ),
    # which kernel loads, and how KernelIdentity is assembled; superseded for
    # native DLL selection and platform-specific build flags (2026-10-24)
    "packages/neural-runtime/src/neurocraft_neural/spmv_kernel.py": (
        "bbb2555b78a1d579932944a03c253113d2c558c851b7d5927e3a7bcea3626a8b"
    ),
    # how the engine assembles state material and pins its buffers
    "packages/neural-runtime/src/neurocraft_neural/full_engine.py": (
        "12d4827d0eefcc37f4e45cfe128c6e2281ba27253dffcd8d326772b1b301e87f"
    ),
    # how overlay identities are derived (the fixture-scale controls)
    "packages/neural-runtime/src/neurocraft_neural/model.py": (
        "a82b425447ca65a359a2e5cca485664e2c7f95a6e1821e0ac193006c223f2c66"
    ),
}

MANIFEST_A = "a" * 64
MANIFEST_B = "b" * 64
VIEW_A = "e" * 64


# ---------------------------------------------------------------------------
# fixtures


class Faithful:
    """A small faithful view plus the edge arrays it was built from."""

    def __init__(self, *, node_count: int, edge_count: int, seed: int, count_bump: int = 0) -> None:
        csr = synthetic_csr(
            node_count=node_count,
            edge_count=edge_count,
            seed=seed,
            empty_rows=(0, node_count - 1, node_count // 2),
        )
        rng = np.random.default_rng(seed + 1)
        node_signs = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=node_count)
        node_signs.setflags(write=False)
        self.node_signs: NDArray[np.int8] = node_signs
        # ``count_bump`` gives a SECOND faithful view over the SAME topology and
        # the same edge count: different weights, therefore a different
        # ``view_sha256``, but an identical drawn permutation. That is what
        # isolates ``faithful_view_sha256`` in the identity.
        self.raw_counts: NDArray[np.int64] = np.ascontiguousarray(
            csr.counts + count_bump, dtype=np.int64
        )
        self.edge_signs: NDArray[np.int64] = node_signs[csr.indices].astype(np.int64)
        self.view: FullRuntimeView = build_view(
            indptr=csr.indptr,
            pre_indices=csr.indices,
            raw_counts=self.raw_counts,
            signs=self.edge_signs,
            provenance=ViewProvenance(
                origin="synthetic_faithful",
                graph_identity_sha256="c" * 64,
                sign_overlay_sha256="d" * 64,
                source_manifest_sha256=MANIFEST_A,
            ),
        )

    def control(self, kind: FullControlKind, **overrides: Any) -> FullControlView:
        arguments: dict[str, Any] = {
            "kind": kind,
            "seed": 7,
            "artifact_manifest_sha256": MANIFEST_A,
            "raw_counts": self.raw_counts,
            "edge_signs": self.edge_signs,
            "node_signs": self.node_signs,
        }
        arguments.update(overrides)
        return build_control_view(self.view, **arguments)

    def rebuild(
        self, identity: FullControlIdentity, permutation: NDArray[Any] | None = None
    ) -> FullControlView:
        return rebuild_control_view(
            self.view,
            identity,
            permutation,
            raw_counts=self.raw_counts,
            edge_signs=self.edge_signs,
            node_signs=self.node_signs,
        )


@pytest.fixture(scope="module")
def faithful() -> Faithful:
    return Faithful(node_count=512, edge_count=6_000, seed=11)


# ---------------------------------------------------------------------------
# identity recipe


def test_the_schema_string_names_the_eight_key_recipe() -> None:
    """A schema string must describe the material it labels.

    ``v1`` is the design's §4.2 seven-key recipe. It was never published: no
    ``v1`` identity exists in the tree, in a receipt, or on a wire, so the
    eighth key took a new schema rather than silently widening the old one.
    """

    assert CONTROL_IDENTITY_SCHEMA == "neurocraft.full_control.v2"
    assert CONTROL_IDENTITY_SCHEMA != "neurocraft.full_control.v1"


def test_the_schema_string_is_inside_the_identity(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    downgraded = replace(control.identity, schema="neurocraft.full_control.v1")
    assert downgraded.identity_sha256 != control.identity.identity_sha256


def test_identity_material_is_exactly_the_declared_recipe(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    material = control.identity.material
    assert set(material) == {
        "artifact_manifest_sha256",
        "edge_count",
        "faithful_view_sha256",
        "kind",
        "permutation_sha256",
        "rng",
        "schema",
        "seed",
    }
    assert material["schema"] == CONTROL_IDENTITY_SCHEMA == "neurocraft.full_control.v2"
    assert material["rng"] == CONTROL_RNG_DESCRIPTION
    assert material["kind"] == "shuffled_weight"
    assert material["seed"] == 7
    assert material["edge_count"] == faithful.view.edge_count
    assert material["artifact_manifest_sha256"] == MANIFEST_A
    assert control.permutation is not None
    assert material["permutation_sha256"] == permutation_sha256(control.permutation)
    # The eighth key: the design's seven-key §4.2 recipe is v1 and was never
    # published, so adding faithful_view_sha256 makes the schema v2 rather than
    # leaving a v1 string over a recipe it no longer describes.
    assert material["faithful_view_sha256"] == faithful.view.view_sha256


def test_the_rewire_identity_carries_the_sign_policy_key(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    assert control.identity.material["sign_policy"] == "rederived_from_presynaptic"


def test_identity_rotates_with_the_seed(faithful: Faithful) -> None:
    first = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=7)
    second = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=8)
    assert first.identity.identity_sha256 != second.identity.identity_sha256


def test_identity_rotates_with_the_kind(faithful: Faithful) -> None:
    shuffled = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=7)
    rewired = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM, seed=7)
    assert shuffled.permutation is not None and rewired.permutation is not None
    # Same seed, therefore the same permutation bytes: only the kind (and the
    # sign-policy key it brings) separates the two identities.
    assert permutation_sha256(shuffled.permutation) == permutation_sha256(rewired.permutation)
    assert shuffled.identity.identity_sha256 != rewired.identity.identity_sha256


def test_identity_rotates_with_the_sign_policy(faithful: Faithful) -> None:
    rederived = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    slot_bound = faithful.control(
        FullControlKind.DEGREE_PRESERVING_RANDOM,
        sign_policy=FullControlSignPolicy.SLOT,
    )
    assert rederived.identity.identity_sha256 != slot_bound.identity.identity_sha256


def test_identity_rotates_with_the_permutation_bytes(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    tampered = FullControlIdentity(
        kind=control.identity.kind,
        seed=control.identity.seed,
        artifact_manifest_sha256=control.identity.artifact_manifest_sha256,
        edge_count=control.identity.edge_count,
        permutation_sha256="f" * 64,
        faithful_view_sha256=control.identity.faithful_view_sha256,
        sign_policy=control.identity.sign_policy,
    )
    assert tampered.identity_sha256 != control.identity.identity_sha256


def test_identity_rotates_with_the_manifest_and_the_edge_count(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    other_manifest = faithful.control(
        FullControlKind.SHUFFLED_WEIGHT, artifact_manifest_sha256=MANIFEST_B
    )
    assert other_manifest.identity.identity_sha256 != control.identity.identity_sha256
    shorter = FullControlIdentity(
        kind=control.identity.kind,
        seed=control.identity.seed,
        artifact_manifest_sha256=control.identity.artifact_manifest_sha256,
        edge_count=control.identity.edge_count - 1,
        permutation_sha256=control.identity.permutation_sha256,
        faithful_view_sha256=control.identity.faithful_view_sha256,
    )
    assert shorter.identity_sha256 != control.identity.identity_sha256


def test_the_identity_refuses_material_that_contradicts_the_kind() -> None:
    with pytest.raises(FullControlError, match="sign policy"):
        FullControlIdentity(
            kind=FullControlKind.DEGREE_PRESERVING_RANDOM,
            seed=1,
            artifact_manifest_sha256=MANIFEST_A,
            edge_count=4,
            permutation_sha256="0" * 63 + "1",
            faithful_view_sha256=VIEW_A,
        )
    with pytest.raises(FullControlError, match="does not move signs"):
        FullControlIdentity(
            kind=FullControlKind.SHUFFLED_WEIGHT,
            seed=1,
            artifact_manifest_sha256=MANIFEST_A,
            edge_count=4,
            permutation_sha256="0" * 63 + "1",
            faithful_view_sha256=VIEW_A,
            sign_policy=FullControlSignPolicy.SLOT,
        )
    with pytest.raises(FullControlError, match="zero sentinel"):
        FullControlIdentity(
            kind=FullControlKind.NO_INPUT,
            seed=1,
            artifact_manifest_sha256=MANIFEST_A,
            edge_count=4,
            permutation_sha256="0" * 63 + "1",
            faithful_view_sha256=VIEW_A,
        )
    with pytest.raises(FullControlError, match="uint64"):
        FullControlIdentity(
            kind=FullControlKind.NO_INPUT,
            seed=-1,
            artifact_manifest_sha256=MANIFEST_A,
            edge_count=4,
            permutation_sha256=ZERO_SHA256,
            faithful_view_sha256=VIEW_A,
        )


# ---------------------------------------------------------------------------
# H1: the supplied edge arrays are verified, not trusted


def test_negated_edge_signs_are_refused(faithful: Faithful) -> None:
    """The review's own attack: same identity, different weights.

    Before the fix this built silently and minted a byte-identical
    ``identity_sha256`` and ``ViewProvenance.identity_sha256`` over a view whose
    every weight had the opposite sign.
    """

    with pytest.raises(FullControlError, match="do not rebuild the faithful view"):
        faithful.control(FullControlKind.SHUFFLED_WEIGHT, edge_signs=-faithful.edge_signs)


def test_rolled_raw_counts_are_refused(faithful: Faithful) -> None:
    with pytest.raises(FullControlError, match="do not rebuild the faithful view"):
        faithful.control(
            FullControlKind.SHUFFLED_WEIGHT, raw_counts=np.roll(faithful.raw_counts, 1)
        )


def test_the_rewire_refuses_node_signs_that_are_not_the_faithful_ones(
    faithful: Faithful,
) -> None:
    """Under the re-deriving policy the faithful signs come from ``node_signs``.

    So the same rebuild check binds ``node_signs`` itself, which is the array
    the executed control signs are gathered through.
    """

    with pytest.raises(FullControlError, match="do not rebuild the faithful view"):
        faithful.control(
            FullControlKind.DEGREE_PRESERVING_RANDOM,
            node_signs=(-faithful.node_signs).astype(np.int8),
        )


def test_the_slot_policy_rewire_also_verifies_its_edge_signs(faithful: Faithful) -> None:
    with pytest.raises(FullControlError, match="do not rebuild the faithful view"):
        faithful.control(
            FullControlKind.DEGREE_PRESERVING_RANDOM,
            sign_policy=FullControlSignPolicy.SLOT,
            edge_signs=-faithful.edge_signs,
        )


def test_the_identity_changes_when_the_faithful_view_changes(faithful: Faithful) -> None:
    """H1(b): ``faithful_view_sha256`` is inside the material.

    Two faithful views, one seed, one kind, one manifest: the permutation bytes
    are identical, and the identities must not be.
    """

    other = Faithful(node_count=512, edge_count=6_000, seed=11, count_bump=1)
    assert other.view.view_sha256 != faithful.view.view_sha256
    assert other.view.edge_count == faithful.view.edge_count
    here = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=7)
    there = other.control(FullControlKind.SHUFFLED_WEIGHT, seed=7)
    assert here.permutation is not None and there.permutation is not None
    assert permutation_sha256(here.permutation) == permutation_sha256(there.permutation)
    assert here.identity.identity_sha256 != there.identity.identity_sha256
    assert here.identity.faithful_view_sha256 == faithful.view.view_sha256
    assert there.identity.faithful_view_sha256 == other.view.view_sha256


def test_no_input_is_bound_to_its_faithful_view_too(faithful: Faithful) -> None:
    other = Faithful(node_count=512, edge_count=6_000, seed=11, count_bump=1)
    here = faithful.control(FullControlKind.NO_INPUT, seed=7)
    there = other.control(FullControlKind.NO_INPUT, seed=7)
    assert here.identity.identity_sha256 != there.identity.identity_sha256


def test_the_correct_arrays_still_build(faithful: Faithful) -> None:
    # The verification must not be a false-positive machine: the arrays the
    # faithful view was actually built from pass, for every kind and policy.
    for kind in FullControlKind:
        assert faithful.control(kind).identity.identity_sha256
    assert faithful.control(
        FullControlKind.DEGREE_PRESERVING_RANDOM, sign_policy=FullControlSignPolicy.SLOT
    ).identity.identity_sha256


# ---------------------------------------------------------------------------
# determinism


@pytest.mark.parametrize(
    "kind",
    [FullControlKind.SHUFFLED_WEIGHT, FullControlKind.DEGREE_PRESERVING_RANDOM],
)
def test_one_seed_builds_byte_identical_permutations_and_views(
    faithful: Faithful, kind: FullControlKind
) -> None:
    first = faithful.control(kind, seed=4242)
    second = faithful.control(kind, seed=4242)
    assert first.permutation is not None and second.permutation is not None
    assert first.permutation.tobytes() == second.permutation.tobytes()
    assert first.identity.identity_sha256 == second.identity.identity_sha256
    assert first.view.view_sha256 == second.view.view_sha256
    assert first.view.provenance.identity_sha256 == second.view.provenance.identity_sha256


def test_different_seeds_draw_different_permutations() -> None:
    first = control_permutation(edge_count=10_000, seed=1)
    second = control_permutation(edge_count=10_000, seed=2)
    assert first.tobytes() != second.tobytes()
    assert_is_permutation(first, edge_count=10_000)
    assert_is_permutation(second, edge_count=10_000)


def test_the_permutation_is_a_bijection_over_the_edge_slots() -> None:
    permutation = control_permutation(edge_count=5_000, seed=7)
    assert permutation.dtype == np.uint32
    assert np.array_equal(np.sort(permutation), np.arange(5_000, dtype=np.uint32))


# ---------------------------------------------------------------------------
# shuffled_weight


def test_shuffled_weight_shares_the_topology_and_keeps_the_signs(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    view = control.view
    # indptr and indices are the faithful view's frozen arrays, by reference.
    assert view.indptr is faithful.view.indptr
    assert view.indices is faithful.view.indices
    # empty_rows is *recomputed* by the unchanged build_view -- equal in value,
    # a new (small) array. build_view derives it and takes no override, so the
    # design's "shared by reference" holds for indptr and indices only.
    assert view.empty_rows is not faithful.view.empty_rows
    assert np.array_equal(view.empty_rows, faithful.view.empty_rows)
    assert view.empty_rows.flags.writeable is False
    # Every edge keeps its sign.
    assert np.array_equal(np.sign(view.weights), np.sign(faithful.view.weights))


def test_shuffled_weight_preserves_the_count_multiset(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    assert control.permutation is not None
    permuted = faithful.raw_counts[control.permutation.astype(np.intp)]
    assert np.array_equal(np.sort(permuted), np.sort(faithful.raw_counts))
    # And the view's weights are exactly those permuted counts, row-normalised
    # with the faithful signs, bit for bit.
    expected = build_view(
        indptr=faithful.view.indptr,
        pre_indices=faithful.view.indices,
        raw_counts=permuted,
        signs=faithful.edge_signs,
        provenance=control.view.provenance,
    )
    assert np.array_equal(control.view.weights, expected.weights)
    assert control.view.view_sha256 == expected.view_sha256


def test_shuffled_weight_actually_changes_the_weights(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    assert control.view.view_sha256 != faithful.view.view_sha256
    assert not np.array_equal(control.view.weights, faithful.view.weights)


def test_shuffled_weight_carries_the_identity_in_the_weight_overlay_slot(
    faithful: Faithful,
) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    provenance = control.view.provenance
    assert provenance.weight_overlay_sha256 == control.identity.identity_sha256
    assert provenance.topology_overlay_sha256 == faithful.view.provenance.topology_overlay_sha256
    assert provenance.identity_sha256 != faithful.view.provenance.identity_sha256
    assert control.identity.kind.value in provenance.origin


# ---------------------------------------------------------------------------
# degree_preserving_random


def test_the_rewire_preserves_every_in_and_out_degree(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    node_count = faithful.view.node_count
    assert np.array_equal(np.diff(control.view.indptr), np.diff(faithful.view.indptr))
    assert np.array_equal(
        np.bincount(control.view.indices, minlength=node_count),
        np.bincount(faithful.view.indices, minlength=node_count),
    )
    assert control.view.edge_count == faithful.view.edge_count


def test_the_rewire_rederives_signs_from_the_new_presynaptic_neuron(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    expected = faithful.node_signs[control.view.indices].astype(np.int64)
    # counts are >= 1 and denominators >= 1, so sign(weight) == the edge sign.
    assert np.array_equal(np.sign(control.view.weights).astype(np.int64), expected)
    assert control.identity.sign_policy is FullControlSignPolicy.REDERIVED_FROM_PRESYNAPTIC


def test_the_slot_sign_policy_keeps_the_faithful_edge_signs(faithful: Faithful) -> None:
    control = faithful.control(
        FullControlKind.DEGREE_PRESERVING_RANDOM,
        sign_policy=FullControlSignPolicy.SLOT,
    )
    assert np.array_equal(
        np.sign(control.view.weights).astype(np.int64), np.sign(faithful.edge_signs)
    )
    assert control.identity.sign_policy is FullControlSignPolicy.SLOT


def test_the_rewire_keeps_the_counts_and_moves_the_indices(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    assert control.permutation is not None
    expected_indices = faithful.view.indices[control.permutation.astype(np.intp)]
    assert np.array_equal(control.view.indices, expected_indices)
    assert control.view.indices is not faithful.view.indices
    assert control.view.indices.flags.writeable is False
    assert control.view.indptr is faithful.view.indptr


def _independent_pair_census(indptr: NDArray[np.int64], indices: NDArray[Any]) -> tuple[int, int]:
    """Duplicates and self-edges by lexsort + adjacent comparison.

    Deliberately not the module's ``row * node_count + pre`` key and
    ``np.unique``: a test that recomputes the implementation's own formula
    pins the formula against itself and would agree with it while both were
    wrong.
    """

    node_count = int(indptr.shape[0]) - 1
    rows = np.repeat(np.arange(node_count, dtype=np.int64), np.diff(indptr))
    columns = indices.astype(np.int64, copy=False)
    order = np.lexsort((columns, rows))
    sorted_rows = rows[order]
    sorted_columns = columns[order]
    repeated = (sorted_rows[1:] == sorted_rows[:-1]) & (sorted_columns[1:] == sorted_columns[:-1])
    return int(np.count_nonzero(repeated)), int(np.count_nonzero(rows == columns))


#: The exact census of the module-scoped synthetic fixture under seed 7. Fixed
#: numbers, not ">= 0": the point of counting duplicates and self-edges is that
#: the count is reported, and a test that accepts any count reports nothing.
EXPECTED_SYNTHETIC_DUPLICATES = 186
EXPECTED_SYNTHETIC_SELF_EDGES = 13


def test_the_rewire_counts_duplicates_and_self_edges(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    statistics = control.rewire_statistics
    assert statistics is not None
    duplicates, self_edges = _independent_pair_census(faithful.view.indptr, control.view.indices)
    source_self_edges = _independent_pair_census(faithful.view.indptr, faithful.view.indices)[1]
    assert statistics.edge_count == faithful.view.edge_count
    assert statistics.duplicate_edge_count == duplicates == EXPECTED_SYNTHETIC_DUPLICATES
    assert statistics.self_edge_count == self_edges == EXPECTED_SYNTHETIC_SELF_EDGES
    assert statistics.source_self_edge_count == source_self_edges
    assert statistics.distinct_pair_count == statistics.edge_count - duplicates
    # The declared difference from the fixture rewire, which refused both:
    # duplicates and self-edges are permitted here and counted.
    assert statistics.duplicate_edge_count > 0
    assert statistics.self_edge_count > 0


def test_the_rewire_needs_node_signs_to_rederive(faithful: Faithful) -> None:
    with pytest.raises(FullControlError, match="node_signs"):
        build_control_view(
            faithful.view,
            kind=FullControlKind.DEGREE_PRESERVING_RANDOM,
            seed=7,
            artifact_manifest_sha256=MANIFEST_A,
            raw_counts=faithful.raw_counts,
            edge_signs=faithful.edge_signs,
            node_signs=None,
        )


def test_the_rewire_carries_the_identity_in_the_topology_overlay_slot(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    provenance = control.view.provenance
    assert provenance.topology_overlay_sha256 == control.identity.identity_sha256
    assert provenance.weight_overlay_sha256 == faithful.view.provenance.weight_overlay_sha256
    assert provenance.identity_sha256 != faithful.view.provenance.identity_sha256
    assert control.view.view_sha256 != faithful.view.view_sha256


# ---------------------------------------------------------------------------
# no_input


def test_no_input_returns_the_faithful_view_and_a_marker(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.NO_INPUT)
    assert control.view is faithful.view
    assert control.permutation is None
    assert control.rewire_statistics is None
    assert control.identity.permutation_sha256 == ZERO_SHA256
    assert control.identity.sign_policy is None
    assert control.zeroes_sensory_input is True
    for kind in (FullControlKind.SHUFFLED_WEIGHT, FullControlKind.DEGREE_PRESERVING_RANDOM):
        assert faithful.control(kind).zeroes_sensory_input is False


def test_no_input_records_a_seed_that_draws_nothing(faithful: Faithful) -> None:
    # Stated in the module docstring and pinned here so nobody reads two
    # no_input identities as two different executed dynamics: the wiring and
    # the view are identical; only the recorded seed differs.
    first = faithful.control(FullControlKind.NO_INPUT, seed=7)
    second = faithful.control(FullControlKind.NO_INPUT, seed=8)
    assert first.identity.identity_sha256 != second.identity.identity_sha256
    assert first.view is second.view


# ---------------------------------------------------------------------------
# the faithful lane, byte-unchanged


def _committed_dynamics_hash() -> str:
    document = json.loads(LOOM_CALIBRATION.read_text(encoding="utf-8"))
    return str(document["measured_under"]["dynamics_hash"])


def _historical_kernel_identity() -> KernelIdentity:
    return KernelIdentity(
        kernel="c99_rowblock_sequential",
        thread_count=1,
        numeric_reproducibility_class="ieee754_binary64_scalar_sequential",
        load_path="historical_metadata_only",
        load_note="identity reconstruction only; no historical binary loaded or executed",
        material=c_kernel_identity_material(HISTORICAL_KERNEL_BUILD),
    )


def test_historical_faithful_dynamics_identity_math_is_unchanged() -> None:
    """Historical identity math remains pinned on every host, with no C build needed."""
    historical = _historical_kernel_identity()
    assert historical.identity_sha256 == HISTORICAL_KERNEL_SHA256
    dynamics = ArrayDynamicsConfig().identity_sha256(historical)
    assert dynamics == "5da651e72477a186c82a6091afa0f415b7e4a182fcd8f014870922143d526d8e"


def test_current_compiled_kernel_binds_actual_bytes_and_is_thread_independent(
    faithful: Faithful,
) -> None:
    """Local binaries publish their own identity; no historical execution is inferred."""

    try:
        kernel = load_c_kernel(thread_count=1)
    except SpmvKernelError as exc:  # pragma: no cover - depends on the local build
        pytest.skip(f"compiled spmv kernel is not available: {exc}")
    actual = library_sha256(kernel.library_path)
    assert kernel.build_document["library_sha256"] == actual
    assert kernel.identity.material["library_sha256"] == actual
    current_dynamics = ArrayDynamicsConfig().identity_sha256(kernel.identity)
    assert (current_dynamics == _committed_dynamics_hash()) == (
        kernel.identity.identity_sha256 == CALIBRATION_KERNEL_SHA256
    )
    for threads in (1, 2, 8):
        threaded = kernel.with_thread_count(threads)
        engine = ArrayNativeEngine(faithful.view, kernel=threaded)
        assert threaded.thread_count == threads
        assert engine.kernel_identity.identity_sha256 == kernel.identity.identity_sha256
        assert engine.dynamics_identity_sha256 == current_dynamics
        engine.end_run()

    # Metadata-only mutation: another library cannot borrow the current dynamics
    # identity even when compiler, source and flags are unchanged. No foreign
    # binary is loaded or executed by this identity-calculation regression.
    changed_build = dict(kernel.build_document)
    changed_build["library_sha256"] = "0" * 64 if actual != "0" * 64 else "1" * 64
    changed = replace(kernel.identity, material=c_kernel_identity_material(changed_build))
    assert changed.identity_sha256 != kernel.identity.identity_sha256
    assert ArrayDynamicsConfig().identity_sha256(changed) != current_dynamics


def test_building_controls_leaves_the_faithful_view_byte_identical(faithful: Faithful) -> None:
    before = {
        "view_sha256": faithful.view.view_sha256,
        "provenance": faithful.view.provenance.identity_sha256,
        "validation": faithful.view.validation.identity_sha256,
        "buffers": [array.tobytes() for _, array in faithful.view.named_buffers()],
    }
    controls = [
        faithful.control(FullControlKind.SHUFFLED_WEIGHT),
        faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM),
        faithful.control(FullControlKind.NO_INPUT),
    ]
    assert len({control.identity.identity_sha256 for control in controls}) == 3
    assert faithful.view.view_sha256 == before["view_sha256"]
    assert faithful.view.provenance.identity_sha256 == before["provenance"]
    assert faithful.view.validation.identity_sha256 == before["validation"]
    assert [array.tobytes() for _, array in faithful.view.named_buffers()] == before["buffers"]
    for _, array in faithful.view.named_buffers():
        assert array.flags.writeable is False


def test_the_identity_hashed_source_files_are_unedited() -> None:
    """No byte of a file whose digest is inside a published identity moved.

    The digests are recorded constants (documented above), not a live ``git
    show``: the pin must hold in a released checkout with no git history and
    must not silently follow whatever ``main`` happens to be.
    """

    for relative, expected in IDENTITY_HASHED_SOURCES.items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, f"{relative} changed; a digest inside an identity moved"
    assert (
        view_construction_sha256()
        == IDENTITY_HASHED_SOURCES["packages/neural-runtime/src/neurocraft_neural/full_view.py"]
    )


def test_the_behaviour_bound_source_files_are_unedited() -> None:
    """Tripwire, not a derivation: these files carry no digest of their own.

    An edit to any of them can rotate ``dynamics_hash`` or an overlay identity
    while no digest of the file itself changes anywhere -- which is why they
    are pinned separately and labelled differently from
    :data:`IDENTITY_HASHED_SOURCES`.
    """

    for relative, expected in BEHAVIOUR_BOUND_SOURCES.items():
        actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, (
            f"{relative} changed; it carries no digest of its own, so re-derive every "
            "identity its behaviour feeds before updating this constant"
        )


def test_the_two_source_pins_do_not_overlap() -> None:
    assert not set(IDENTITY_HASHED_SOURCES) & set(BEHAVIOUR_BOUND_SOURCES)


# ---------------------------------------------------------------------------
# both kernels step a control view like the reference arithmetic

#: Acceptance bound for fast-vs-oracle divergence, the same constant and the
#: same rationale as ``tests/full_backend/test_full_engine.py``: roughly 150x
#: the measured plateau and about 1e12 below the [-1, 1] state range. Every
#: test that uses it also reports the measured maximum.
TOLERANCE_LINF = 1e-13


def _oracle_csr(view: FullRuntimeView, counts: NDArray[Any], signs: NDArray[Any]) -> SyntheticCsr:
    return SyntheticCsr(
        indptr=np.ascontiguousarray(view.indptr, dtype=np.int64),
        indices=np.ascontiguousarray(view.indices, dtype=np.uint32),
        counts=np.ascontiguousarray(counts, dtype=np.int64),
        signs=np.ascontiguousarray(signs, dtype=np.int64),
    )


def _worst_divergence(view: FullRuntimeView, csr: SyntheticCsr, kernel: Any, seed: int) -> float:
    oracle = FsumReference(csr)
    engine = ArrayNativeEngine(view, kernel=kernel)
    rng = np.random.default_rng(seed)
    direct = rng.uniform(-1.0, 1.0, size=view.node_count)
    worst = 0.0
    for _ in range(40):
        expected = oracle.step(direct.tolist())
        engine.step(direct=direct)
        worst = max(worst, float(np.max(np.abs(np.asarray(expected) - engine.activity()))))
    return worst


def _kernels() -> list[tuple[str, Any]]:
    kernels: list[tuple[str, Any]] = [("numpy", NumpyReduceatKernel())]
    with contextlib.suppress(SpmvKernelError):  # the C kernel is optional
        kernels.append(("c", load_c_kernel(thread_count=4)))
    return kernels


def test_both_kernels_step_a_rewired_view_like_the_reference_arithmetic(
    faithful: Faithful,
) -> None:
    """The rewired view's rows are unsorted and carry duplicates; both kernels
    must still reproduce the ``math.fsum`` oracle to the same bound the
    faithful view meets.

    This is the property the within-row column order note in
    ``full_controls`` rests on: the kernels read the order only as a summation
    order, never as an index invariant.
    """

    control = faithful.control(FullControlKind.DEGREE_PRESERVING_RANDOM)
    signs = faithful.node_signs[control.view.indices].astype(np.int64)
    # The control's rows really are unsorted and really do repeat pairs, or
    # this test would be measuring the faithful case twice.
    rows = np.repeat(
        np.arange(faithful.view.node_count, dtype=np.int64), np.diff(faithful.view.indptr)
    )
    columns = control.view.indices.astype(np.int64)
    descents = int(np.count_nonzero((rows[1:] == rows[:-1]) & (columns[1:] > columns[:-1])))
    ascents = int(np.count_nonzero((rows[1:] == rows[:-1]) & (columns[1:] < columns[:-1])))
    assert ascents > 0 and descents > 0, "the control view's rows are not actually unsorted"
    assert _independent_pair_census(faithful.view.indptr, control.view.indices)[0] > 0

    control_csr = _oracle_csr(control.view, faithful.raw_counts, signs)
    faithful_csr = _oracle_csr(faithful.view, faithful.raw_counts, faithful.edge_signs)
    for label, kernel in _kernels():
        control_worst = _worst_divergence(control.view, control_csr, kernel, seed=5)
        faithful_worst = _worst_divergence(faithful.view, faithful_csr, kernel, seed=5)
        assert control_worst <= TOLERANCE_LINF, f"{label} kernel, control view: {control_worst}"
        assert faithful_worst <= TOLERANCE_LINF, f"{label} kernel, faithful view: {faithful_worst}"


def test_both_kernels_step_a_shuffled_view_like_the_reference_arithmetic(
    faithful: Faithful,
) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    assert control.permutation is not None
    counts = faithful.raw_counts[control.permutation.astype(np.intp)]
    csr = _oracle_csr(control.view, counts, faithful.edge_signs)
    for label, kernel in _kernels():
        worst = _worst_divergence(control.view, csr, kernel, seed=6)
        assert worst <= TOLERANCE_LINF, f"{label} kernel, shuffled view: {worst}"


# ---------------------------------------------------------------------------
# persistence


def test_the_permutation_round_trips_through_the_artifact_file(
    faithful: Faithful, tmp_path: Path
) -> None:
    control = faithful.control(
        FullControlKind.SHUFFLED_WEIGHT, persist_directory=tmp_path / "controls"
    )
    assert control.permutation is not None
    assert control.permutation_path == permutation_path(control.identity, tmp_path / "controls")
    assert control.permutation_path is not None
    assert control.permutation_path.name == f"{control.identity.identity_sha256}.u4"
    assert control.permutation_path.stat().st_size == control.view.edge_count * 4
    reloaded = load_permutation(control.identity, tmp_path / "controls")
    assert reloaded.tobytes() == control.permutation.tobytes()
    # And the reloaded permutation rebuilds a byte-identical view.
    rebuilt = faithful.rebuild(control.identity, reloaded)
    assert rebuilt.view.view_sha256 == control.view.view_sha256
    assert rebuilt.identity == control.identity


def test_a_corrupted_permutation_file_is_refused(faithful: Faithful, tmp_path: Path) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT, persist_directory=tmp_path)
    target = permutation_path(control.identity, tmp_path)
    raw = bytearray(target.read_bytes())
    raw[0] ^= 0xFF
    target.write_bytes(bytes(raw))
    with pytest.raises(FullControlError, match="does not match its identity digest"):
        load_permutation(control.identity, tmp_path)


def test_a_truncated_permutation_file_is_refused(faithful: Faithful, tmp_path: Path) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT, persist_directory=tmp_path)
    target = permutation_path(control.identity, tmp_path)
    target.write_bytes(target.read_bytes()[:-4])
    with pytest.raises(FullControlError, match="bytes, expected"):
        load_permutation(control.identity, tmp_path)


def test_a_missing_permutation_file_is_refused(faithful: Faithful, tmp_path: Path) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    with pytest.raises(FullControlError, match="could not read"):
        load_permutation(control.identity, tmp_path)


def test_persisting_a_permutation_under_a_foreign_identity_is_refused(
    faithful: Faithful, tmp_path: Path
) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=7)
    other = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=8)
    assert control.permutation is not None
    with pytest.raises(FullControlError, match="refusing to persist"):
        write_permutation(control.permutation, other.identity, tmp_path)


def test_a_file_whose_bytes_are_not_a_permutation_is_refused(tmp_path: Path) -> None:
    raw = np.zeros(16, dtype="<u4").tobytes()
    identity = FullControlIdentity(
        kind=FullControlKind.SHUFFLED_WEIGHT,
        seed=1,
        artifact_manifest_sha256=MANIFEST_A,
        edge_count=16,
        permutation_sha256=hashlib.sha256(raw).hexdigest(),
        faithful_view_sha256=VIEW_A,
    )
    target = permutation_path(identity, tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    with pytest.raises(FullControlError, match="bijection"):
        load_permutation(identity, tmp_path)


# ---------------------------------------------------------------------------
# refusals


def test_a_permutation_that_is_not_the_identitys_is_refused(faithful: Faithful) -> None:
    """M1: a seed and unrelated bytes can no longer be paired in one call."""

    seven = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=7)
    eight = faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=8)
    assert eight.permutation is not None
    with pytest.raises(FullControlError, match="not the one this identity was minted from"):
        faithful.rebuild(seven.identity, eight.permutation)
    bad = np.zeros(faithful.view.edge_count, dtype=np.uint32)
    with pytest.raises(FullControlError, match="not the one this identity was minted from"):
        faithful.rebuild(seven.identity, bad)


def test_the_fresh_build_path_takes_no_permutation(faithful: Faithful) -> None:
    with pytest.raises(TypeError):
        build_control_view(  # type: ignore[call-arg]
            faithful.view,
            kind=FullControlKind.SHUFFLED_WEIGHT,
            seed=7,
            artifact_manifest_sha256=MANIFEST_A,
            raw_counts=faithful.raw_counts,
            edge_signs=faithful.edge_signs,
            permutation=np.arange(faithful.view.edge_count, dtype=np.uint32),
        )


def test_a_rebuild_against_a_foreign_faithful_view_is_refused(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.SHUFFLED_WEIGHT)
    assert control.permutation is not None
    other = Faithful(node_count=512, edge_count=6_000, seed=11, count_bump=1)
    assert other.view.view_sha256 != faithful.view.view_sha256
    with pytest.raises(FullControlError, match="different faithful view"):
        rebuild_control_view(
            other.view,
            control.identity,
            control.permutation,
            raw_counts=other.raw_counts,
            edge_signs=other.edge_signs,
        )


def test_a_rebuild_refuses_a_permutation_for_no_input(faithful: Faithful) -> None:
    control = faithful.control(FullControlKind.NO_INPUT)
    with pytest.raises(FullControlError, match="permutes nothing"):
        rebuild_control_view(
            faithful.view,
            control.identity,
            np.arange(faithful.view.edge_count, dtype=np.uint32),
        )
    rebuilt = rebuild_control_view(faithful.view, control.identity)
    assert rebuilt.identity == control.identity
    assert rebuilt.view is faithful.view


def test_edge_arrays_of_the_wrong_length_are_refused(faithful: Faithful) -> None:
    with pytest.raises(FullControlError, match="one entry per edge"):
        faithful.control(FullControlKind.SHUFFLED_WEIGHT, raw_counts=faithful.raw_counts[:-1])


def test_missing_raw_counts_are_refused(faithful: Faithful) -> None:
    with pytest.raises(FullControlError, match="raw_counts is required"):
        faithful.control(FullControlKind.SHUFFLED_WEIGHT, raw_counts=None)


def test_an_out_of_range_seed_is_refused(faithful: Faithful) -> None:
    with pytest.raises(FullControlError, match="uint64"):
        faithful.control(FullControlKind.SHUFFLED_WEIGHT, seed=1 << 64)


# ---------------------------------------------------------------------------
# the real artifact (skipped when the bulk bytes are absent)


@pytest.mark.skipif(
    not (FULL_ARTIFACT_DIRECTORY / "manifest.json").is_file(),
    reason="full MaleCNS runtime artifact is not present (bulk data is not tracked)",
)
def test_the_permutation_shape_matches_the_real_edge_count() -> None:
    """Only the cheap artifact-shaped fact; the full build is the S1 probe.

    ``scripts/probe_full_controls.py`` is where both control views are built
    against the real artifact and timed. This test asserts the one property
    that would break the ``<u4`` persistence contract at full scale.
    """

    from neurocraft_connectome.full_runtime import load_full_runtime_artifact

    manifest = FULL_ARTIFACT_DIRECTORY / "manifest.json"
    graph = load_full_runtime_artifact(
        FULL_ARTIFACT_DIRECTORY,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    try:
        edge_count = int(graph.manifest.edge_count)
        node_count = int(graph.manifest.node_count)
    finally:
        graph.close()
    assert edge_count == 25_582_938
    assert node_count == 166_700
    assert edge_count < (1 << 32)
    # The pair key the rewire statistics build is int64-safe at this scale.
    assert (node_count - 1) * node_count + (node_count - 1) < (1 << 63)
