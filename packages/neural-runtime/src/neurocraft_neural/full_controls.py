"""Full-scale control views over the MaleCNS runtime CSR arrays.

A *control* here is an engineering construct: a declared, content-addressed
transform of the executed wiring, built so that a run under it can be compared
against the faithful run with the same encoder, the same bindings and the same
decoder scales. Nothing in this module measures, shows, or establishes anything
about a fly. It changes which numbers the kernel multiplies, and it names the
change:

``shuffled_weight``
    Every edge keeps its endpoints and its sign; the raw synaptic counts are
    permuted globally across edge slots. The anatomy is untouched, the
    magnitudes are not.

``degree_preserving_random``
    Every edge keeps its postsynaptic row (``indptr`` is unchanged, so each
    node's in-degree is exact) and the presynaptic index array is permuted
    globally (a permutation preserves value multiplicities, so each node's
    out-degree is exact too). Who talks to whom is destroyed; the degree
    sequence is not.

``no_input``
    The wiring is the faithful wiring, byte for byte. Only the identity and a
    marker are produced here; zeroing the encoded sensory vector is the
    controller's job, not this module's.

Declared differences from the fixture-scale controls
----------------------------------------------------
The *semantics* are reused from :mod:`neurocraft_neural.model`; the
*algorithms* are not, and the differences are declared rather than implied:

1. **RNG.** ``WeightOverlay.shuffled`` sorts ``range(E)`` by a per-edge Python
   ``sha256`` key (``model.py:176-183``). At 25,582,938 edges that is 25.6 M
   Python hash calls and a 25.6 M-tuple list. Here the permutation is
   ``numpy.random.Generator(PCG64(seed)).permutation(E)``. NumPy does not
   promise bit-stream stability across major versions, so the identity hashes
   the **resulting permutation bytes**: a changed stream produces a different
   identity and a different lane label, never a silent drift.
2. **Duplicates and self-edges.** ``TopologyOverlay.degree_preserving_random``
   is a double-edge-swap loop that refuses to create a duplicate directed pair
   and pins the self-edge count (``model.py:319-370``). A global presynaptic
   permutation cannot make those guarantees and does not try: duplicate
   ``(post, pre)`` pairs and self-edges are **allowed**, and
   :class:`RewireStatistics` counts them so the count is reported rather than
   assumed.
3. **Signs.** The fixture rewire leaves each edge's sign bound to its slot.
   Here the default is ``rederived_from_presynaptic``: the executed sign is
   ``node_signs[permuted_indices]`` through the existing
   :func:`~neurocraft_neural.full_signs.edge_signs_from_node_signs`, which
   keeps the coarse-consensus rule "an edge's sign is its presynaptic
   neuron's transmitter" true under the control. The slot-bound alternative
   the reservoir harness documents stays available as
   :attr:`FullControlSignPolicy.SLOT`; the choice is inside the identity, so
   the two are different, separately named controls, never the same one.
4. **Within-row column order.** The faithful artifact stores each row's
   presynaptic indices in ascending order. A global permutation of
   ``indices`` does not preserve that, so a control view's rows are
   target-major (``indptr`` untouched) but not column-sorted. The kernels do
   not require sorted columns -- neither the numpy fallback nor the C
   row-block kernel reads the order for anything but the sequential summation
   -- but the row summation order therefore differs from the faithful view's.
   It is a different graph; this is one of the ways in which it is different.

What binds a control identity to what actually executed
-------------------------------------------------------
An identity that names a transform but not the thing it transformed is not an
identity. Two rules close that:

* The ``raw_counts`` and ``edge_signs`` a caller supplies are **verified, not
  trusted**: :func:`build_control_view` rebuilds the faithful view from them
  through the unchanged ``build_view`` and refuses unless the result's
  ``view_sha256`` is the faithful view's. Negated signs, rolled counts, or a
  ``node_signs`` array that is not the one the faithful view was built from all
  fail here rather than minting a correct-looking identity over different
  weights. Under the re-deriving sign policy the faithful signs are themselves
  derived from ``node_signs``, so that array is bound by the same check. The
  cost is one extra ``build_view`` at load (about a second and 205 MB transient
  at full scale) and it is paid every time.
* ``faithful_view_sha256`` is **inside the identity material**. Without it the
  same ``identity_sha256`` could name a control over two different faithful
  views -- and a later lane folds ``identity_sha256`` into ``dynamics_hash``,
  where that ambiguity would become a false claim on every frame. The design's
  §4.2 recipe lists seven keys and is named ``neurocraft.full_control.v1``;
  adding an eighth key makes it a different recipe, so the implemented schema
  is ``neurocraft.full_control.v2``. Nothing was ever published under ``v1``:
  the seven-key form is a design document, not a released identity, and no
  ``v1`` identity exists anywhere in the tree, in a receipt, or on a wire.

A permutation that is loaded rather than drawn goes through
:func:`rebuild_control_view`, which takes the identity as the authority and
refuses unless the permutation's digest is the identity's. There is deliberately
no way to pass a seed and an unrelated permutation to the same call: that
combination would mint an identity whose ``seed`` and ``rng`` fields describe
bytes that never executed.

What this module must not do
----------------------------
Several files feed the identities this lane must not move, in two different
ways, and the distinction matters when reasoning about what an edit breaks:

* **Byte-hashed.** :mod:`neurocraft_neural.full_view` hashes its own bytes into
  ``view_construction_sha256()``, which ``ArrayDynamicsConfig.identity_material``
  folds into ``dynamics_hash``. ``_spmv.c`` and the compiled shared library
  enter the same digest through the kernel identity's ``kernel_source_sha256``.
  ``neurocraft_connectome/loader.py`` and ``neurocraft_connectome/model.py`` are
  hashed together by ``_loader_implementation_sha256`` into every fixture graph
  identity.
* **Behaviour-bound.** :mod:`neurocraft_neural.full_engine`,
  :mod:`neurocraft_neural.full_records`, :mod:`neurocraft_neural.spmv_kernel`
  and :mod:`neurocraft_neural.model` have no digest of their own in any
  identity; what they *compute* -- the dynamics material's shape, which kernel
  loads, how overlay identities are derived -- determines those identities
  instead. An edit there can move a hash without any digest of the file moving.

Control views are therefore built by calling the **unchanged**
:func:`~neurocraft_neural.full_view.build_view` with a
:class:`~neurocraft_neural.full_view.ViewProvenance` carrying the control
identity, and this module imports from those files without editing one byte of
them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
from neurocraft_connectome.model import sha256_json
from numpy.typing import NDArray

from neurocraft_neural.full_signs import edge_signs_from_node_signs
from neurocraft_neural.full_view import FullRuntimeView, ViewProvenance, build_view
from neurocraft_neural.model import ZERO_SHA256

#: Schema of the control identity material. Bumping it rotates every control
#: identity, which is the point: an identity recipe change is a new control.
#: ``v1`` is the design's seven-key §4.2 recipe and was never published; ``v2``
#: is that recipe plus ``faithful_view_sha256``.
CONTROL_IDENTITY_SCHEMA: Final = "neurocraft.full_control.v2"

#: The exact RNG contract recorded in every control identity. It names the
#: generator and the call, not the resulting bytes; the bytes are pinned
#: separately by ``permutation_sha256``.
CONTROL_RNG_DESCRIPTION: Final = "numpy.random.Generator(PCG64(seed)).permutation(E)"

#: Permutations are persisted as little-endian uint32, so the edge count must
#: fit in uint32. The real artifact has 25,582,938 edges.
PERMUTATION_DTYPE: Final = "<u4"
UINT32_LIMIT: Final = 1 << 32
UINT64_LIMIT: Final = 1 << 64

#: Default directory for persisted permutations. ``artifacts/`` is gitignored
#: in its entirety, so nothing here is ever committed.
DEFAULT_PERMUTATION_DIRECTORY: Final = Path("artifacts/controls")

_HEX64 = frozenset("0123456789abcdef")


class FullControlError(RuntimeError):
    """Raised when a control view cannot be built or verified exactly."""


class FullControlKind(StrEnum):
    """The three control identities this module builds.

    The values are the :class:`neurocraft_experiments.identity.RunMode` values
    for the same controls, spelled identically so that one vocabulary crosses
    the wire, the HUD and the trace. They are duplicated rather than imported
    because ``neurocraft-neural`` does not depend on the experiment SDK; a
    cross-package vocabulary test is the place to pin the equality.
    """

    SHUFFLED_WEIGHT = "shuffled_weight"
    DEGREE_PRESERVING_RANDOM = "degree_preserving_random"
    NO_INPUT = "no_input"


class FullControlSignPolicy(StrEnum):
    """What happens to an edge's sign when its presynaptic index moves.

    ``SLOT``
        The sign stays with the edge slot, as the fixture reservoir harness
        declares (``reservoir.py:14-21``). The executed sign then names a
        neuron that no longer sits at that slot.
    ``REDERIVED_FROM_PRESYNAPTIC``
        The sign is re-gathered from the new presynaptic neuron, keeping the
        coarse-consensus rule true under the control. This is the default.
    """

    SLOT = "slot"
    REDERIVED_FROM_PRESYNAPTIC = "rederived_from_presynaptic"


@dataclass(frozen=True, slots=True)
class FullControlIdentity:
    """Content identity of one control.

    The material is ``neurocraft.full_control.v2``: the design's seven-key §4.2
    recipe (which is ``v1``, and was never published) plus
    ``faithful_view_sha256``, the ``view_sha256`` of the faithful view the
    control transforms. Without that key one ``identity_sha256`` could name a
    control over two different faithful views, and a later lane folds
    ``identity_sha256`` into ``dynamics_hash``.

    ``permutation_sha256`` is the SHA-256 of ``permutation.astype("<u4")
    .tobytes()``. For :attr:`FullControlKind.NO_INPUT`, which permutes
    nothing, it is the repo's :data:`~neurocraft_neural.model.ZERO_SHA256`
    sentinel -- 64 zero characters -- and **not** the SHA-256 of an empty
    buffer, which would be indistinguishable from a legitimate permutation of
    a zero-edge graph.

    ``sign_policy`` is a **conditional key**: it is present in the hashed
    material only for :attr:`FullControlKind.DEGREE_PRESERVING_RANDOM`, the
    only kind under which the choice has an effect. This is the conditional-key
    rule the full backend already uses for state material
    (``full_records.py:184-209``). No collision is possible across kinds
    because ``kind`` is always present.

    ``seed`` is recorded for every kind, including ``no_input``, because the
    recipe declares one identity shape for all three. Under ``no_input`` the
    seed draws nothing: two ``no_input`` controls with different seeds carry
    different identities and byte-identical dynamics. That is stated here
    rather than hidden.
    """

    kind: FullControlKind
    seed: int
    artifact_manifest_sha256: str
    edge_count: int
    permutation_sha256: str
    faithful_view_sha256: str
    sign_policy: FullControlSignPolicy | None = None
    schema: str = CONTROL_IDENTITY_SCHEMA
    rng: str = CONTROL_RNG_DESCRIPTION

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FullControlKind):
            raise FullControlError("kind must be a FullControlKind, not a bare string")
        if self.sign_policy is not None and not isinstance(self.sign_policy, FullControlSignPolicy):
            raise FullControlError("sign_policy must be a FullControlSignPolicy or None")
        _require_uint64("seed", self.seed)
        _require_sha256("artifact_manifest_sha256", self.artifact_manifest_sha256)
        _require_sha256("permutation_sha256", self.permutation_sha256)
        _require_sha256("faithful_view_sha256", self.faithful_view_sha256)
        if isinstance(self.edge_count, bool) or not isinstance(self.edge_count, int):
            raise FullControlError("edge_count must be an integer")
        if self.edge_count < 0:
            raise FullControlError("edge_count must not be negative")
        rederives = self.kind is FullControlKind.DEGREE_PRESERVING_RANDOM
        if rederives and self.sign_policy is None:
            raise FullControlError("a presynaptic permutation must declare a sign policy")
        if not rederives and self.sign_policy is not None:
            raise FullControlError(
                f"{self.kind.value} does not move signs; sign_policy must be None"
            )
        if self.kind is FullControlKind.NO_INPUT and self.permutation_sha256 != ZERO_SHA256:
            raise FullControlError("no_input permutes nothing and carries the zero sentinel")

    @property
    def material(self) -> dict[str, Any]:
        """The hashed ``neurocraft.full_control.v2`` material, in one shape."""

        material: dict[str, Any] = {
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "edge_count": self.edge_count,
            "faithful_view_sha256": self.faithful_view_sha256,
            "kind": self.kind.value,
            "permutation_sha256": self.permutation_sha256,
            "rng": self.rng,
            "schema": self.schema,
            "seed": self.seed,
        }
        if self.sign_policy is not None:
            material["sign_policy"] = self.sign_policy.value
        return material

    @property
    def identity_sha256(self) -> str:
        """SHA-256 over the canonical JSON encoding of :attr:`material`.

        ``sha256_json`` is the repo's canonical hashing helper -- the same one
        ``full_view.py`` and ``full_records.py`` hash their identities with.
        """

        return sha256_json(self.material)

    @property
    def label(self) -> str:
        """Short human label. Engineering, not anatomy; never a fly claim."""

        return f"{self.kind.value}:seed={self.seed}:{self.identity_sha256[:12]}"


@dataclass(frozen=True, slots=True)
class RewireStatistics:
    """What the presynaptic permutation actually produced, counted not assumed.

    ``duplicate_edge_count`` is ``edge_count`` minus the number of distinct
    ``(post, pre)`` pairs: the number of edge slots that repeat a pair some
    other slot already carries. ``self_edge_count`` is the number of slots
    whose presynaptic index equals its postsynaptic row.
    """

    edge_count: int
    distinct_pair_count: int
    duplicate_edge_count: int
    self_edge_count: int
    source_self_edge_count: int


@dataclass(frozen=True, slots=True)
class FullControlView:
    """One built control lane: its identity, its view, and what it changed.

    ``view`` is the executable view. For :attr:`FullControlKind.NO_INPUT` it is
    the faithful view object itself, unchanged and unwrapped -- the control is
    on the input, and this module does not apply it.
    """

    identity: FullControlIdentity
    view: FullRuntimeView
    permutation: NDArray[np.uint32] | None
    rewire_statistics: RewireStatistics | None = None
    permutation_path: Path | None = None

    @property
    def kind(self) -> FullControlKind:
        return self.identity.kind

    @property
    def zeroes_sensory_input(self) -> bool:
        """Whether the controller must zero the encoded sensory vector.

        True for :attr:`FullControlKind.NO_INPUT` only. The zeroing itself is
        the controller's, not this module's; this is the marker it reads.
        """

        return self.identity.kind is FullControlKind.NO_INPUT


# ---------------------------------------------------------------------------
# permutation


def control_permutation(*, edge_count: int, seed: int) -> NDArray[np.uint32]:
    """Draw the declared permutation of ``range(edge_count)`` for ``seed``."""

    _require_uint64("seed", seed)
    if isinstance(edge_count, bool) or not isinstance(edge_count, int) or edge_count < 0:
        raise FullControlError("edge_count must be a non-negative integer")
    if edge_count >= UINT32_LIMIT:
        raise FullControlError("control permutations are persisted as uint32 edge indices")
    generator = np.random.Generator(np.random.PCG64(seed))
    permutation: NDArray[np.uint32] = generator.permutation(edge_count).astype(
        np.uint32, copy=False
    )
    return np.ascontiguousarray(permutation, dtype=np.uint32)


def permutation_sha256(permutation: NDArray[Any]) -> str:
    """SHA-256 of ``permutation.astype("<u4").tobytes()``, streamed."""

    contiguous = np.ascontiguousarray(permutation, dtype=PERMUTATION_DTYPE)
    digest = hashlib.sha256()
    with memoryview(contiguous).cast("B") as block:
        digest.update(block)
    return digest.hexdigest()


def permutation_path(identity: FullControlIdentity, directory: Path | str) -> Path:
    """Where one control's permutation is persisted."""

    return Path(directory) / f"{identity.identity_sha256}.u4"


def write_permutation(
    permutation: NDArray[np.uint32],
    identity: FullControlIdentity,
    directory: Path | str = DEFAULT_PERMUTATION_DIRECTORY,
) -> Path:
    """Persist the permutation under its identity, refusing a digest mismatch.

    The file is ``artifacts/controls/<identity>.u4`` and ``artifacts/`` is
    gitignored, so this never puts bulk data in the repository. Re-writing an
    existing file is allowed and idempotent: the digest is checked first, so
    the bytes cannot change under a fixed identity.
    """

    digest = permutation_sha256(permutation)
    if digest != identity.permutation_sha256:
        raise FullControlError(
            "refusing to persist a permutation whose digest is not the identity's"
        )
    target = permutation_path(identity, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(permutation, dtype=PERMUTATION_DTYPE)
    target.write_bytes(contiguous.tobytes())
    return target


def load_permutation(
    identity: FullControlIdentity,
    directory: Path | str = DEFAULT_PERMUTATION_DIRECTORY,
) -> NDArray[np.uint32]:
    """Re-load a persisted permutation, verified against the identity.

    Three checks, all fail-closed: the byte length must match the identity's
    edge count, the SHA-256 must match ``permutation_sha256``, and the values
    must actually be a permutation of ``range(edge_count)``. The last is not
    redundant -- the digest binds the bytes to the identity, not the bytes to
    the *meaning* of a permutation, and a control view built from a
    non-permutation would silently change the count multiset.
    """

    source = permutation_path(identity, directory)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise FullControlError(
            f"could not read the control permutation at {source}: {exc}"
        ) from exc
    expected_bytes = identity.edge_count * 4
    if len(raw) != expected_bytes:
        raise FullControlError(
            f"control permutation at {source} is {len(raw)} bytes, expected {expected_bytes}"
        )
    if hashlib.sha256(raw).hexdigest() != identity.permutation_sha256:
        raise FullControlError(
            f"control permutation at {source} does not match its identity digest"
        )
    permutation = np.frombuffer(raw, dtype=PERMUTATION_DTYPE).astype(np.uint32, copy=True)
    assert_is_permutation(permutation, edge_count=identity.edge_count)
    return permutation


def assert_is_permutation(permutation: NDArray[Any], *, edge_count: int) -> None:
    """Refuse anything that is not a permutation of ``range(edge_count)``."""

    if permutation.ndim != 1 or int(permutation.shape[0]) != edge_count:
        raise FullControlError("control permutation has the wrong shape")
    if not np.issubdtype(permutation.dtype, np.integer):
        raise FullControlError("control permutation must be an integer array")
    if edge_count == 0:
        return
    seen = np.zeros(edge_count, dtype=bool)
    values = permutation.astype(np.int64, copy=False)
    if int(values.min()) < 0 or int(values.max()) >= edge_count:
        raise FullControlError("control permutation contains an out-of-range edge index")
    seen[values] = True
    if not bool(seen.all()):
        raise FullControlError("control permutation is not a bijection over the edge slots")


# ---------------------------------------------------------------------------
# building


def build_control_identity(
    *,
    kind: FullControlKind,
    seed: int,
    artifact_manifest_sha256: str,
    edge_count: int,
    faithful_view_sha256: str,
    permutation: NDArray[Any] | None,
    sign_policy: FullControlSignPolicy | None,
) -> FullControlIdentity:
    """Assemble the declared identity from the material that produced it."""

    if kind is FullControlKind.NO_INPUT:
        if permutation is not None:
            raise FullControlError("no_input draws no permutation")
        digest = ZERO_SHA256
    else:
        if permutation is None:
            raise FullControlError(f"{kind.value} requires the permutation it was built from")
        digest = permutation_sha256(permutation)
    return FullControlIdentity(
        kind=kind,
        seed=seed,
        artifact_manifest_sha256=artifact_manifest_sha256,
        edge_count=edge_count,
        permutation_sha256=digest,
        faithful_view_sha256=faithful_view_sha256,
        sign_policy=sign_policy,
    )


def build_control_view(
    faithful: FullRuntimeView,
    *,
    kind: FullControlKind,
    seed: int,
    artifact_manifest_sha256: str,
    raw_counts: NDArray[Any] | None = None,
    edge_signs: NDArray[Any] | None = None,
    node_signs: NDArray[np.int8] | None = None,
    sign_policy: FullControlSignPolicy = FullControlSignPolicy.REDERIVED_FROM_PRESYNAPTIC,
    collect_rewire_statistics: bool = True,
    persist_directory: Path | str | None = None,
) -> FullControlView:
    """Draw a fresh permutation from ``seed`` and build one control lane.

    ``faithful`` supplies ``indptr`` and ``indices``; ``raw_counts`` and
    ``edge_signs`` are the same two edge arrays the faithful view was built
    from (the view keeps only the derived ``weights``, so they cannot be
    recovered from it and must be supplied). ``node_signs`` is required for
    :attr:`FullControlKind.DEGREE_PRESERVING_RANDOM` under the re-deriving sign
    policy, and supplies the faithful signs for the verification below.

    **The supplied arrays are verified, never trusted.** Whatever reaches the
    faithful-signs slot is used to rebuild the faithful view through the
    unchanged ``build_view``; unless that rebuild's ``view_sha256`` is the
    faithful view's, this refuses. Otherwise a caller could hand over negated
    signs or rolled counts and mint an identity that names a control over
    weights it does not describe.

    There is no ``permutation`` argument: a permutation loaded from disk goes
    through :func:`rebuild_control_view`, which takes the identity as the
    authority. Passing a seed and an unrelated permutation to one call would
    produce an identity whose ``seed`` and ``rng`` fields are false.

    The faithful view is never mutated and never rebuilt into the result: for
    ``shuffled_weight`` its frozen ``indptr`` and ``indices`` are handed
    straight to ``build_view``, which returns them unchanged because
    ``np.ascontiguousarray`` on a contiguous input of the right dtype is the
    identity and ``_freeze`` is idempotent. ``empty_rows`` is **recomputed**
    from ``indptr`` -- equal in value, a new (small) array -- because
    ``build_view`` derives it and takes no override.
    """

    permutation = (
        None
        if kind is FullControlKind.NO_INPUT
        else control_permutation(edge_count=faithful.edge_count, seed=seed)
    )
    return _assemble_control_view(
        faithful,
        kind=kind,
        seed=seed,
        artifact_manifest_sha256=artifact_manifest_sha256,
        permutation=permutation,
        raw_counts=raw_counts,
        edge_signs=edge_signs,
        node_signs=node_signs,
        sign_policy=sign_policy,
        collect_rewire_statistics=collect_rewire_statistics,
        persist_directory=persist_directory,
    )


def rebuild_control_view(
    faithful: FullRuntimeView,
    identity: FullControlIdentity,
    permutation: NDArray[Any] | None = None,
    *,
    raw_counts: NDArray[Any] | None = None,
    edge_signs: NDArray[Any] | None = None,
    node_signs: NDArray[np.int8] | None = None,
    collect_rewire_statistics: bool = True,
) -> FullControlView:
    """Rebuild a control view from a persisted permutation, identity-first.

    The identity is the authority: ``kind``, ``seed``, ``sign_policy``,
    ``artifact_manifest_sha256`` and ``edge_count`` all come from it, never
    from the caller, so there is no way to pair a seed with bytes it did not
    draw. The permutation is refused unless its digest is the identity's, the
    faithful view is refused unless its ``view_sha256`` is the identity's, and
    the identity the rebuild produces is compared against the one supplied --
    a total check, not a field-by-field one.
    """

    if faithful.view_sha256 != identity.faithful_view_sha256:
        raise FullControlError(
            "this identity was minted over a different faithful view "
            f"({identity.faithful_view_sha256[:12]}..., not {faithful.view_sha256[:12]}...)"
        )
    if faithful.edge_count != identity.edge_count:
        raise FullControlError("the faithful view's edge count is not the identity's")
    if identity.kind is FullControlKind.NO_INPUT:
        if permutation is not None:
            raise FullControlError("no_input permutes nothing; do not supply a permutation")
    else:
        if permutation is None:
            raise FullControlError(f"{identity.kind.value} must be rebuilt from its permutation")
        if permutation_sha256(permutation) != identity.permutation_sha256:
            raise FullControlError(
                "the supplied permutation is not the one this identity was minted from"
            )
    rebuilt = _assemble_control_view(
        faithful,
        kind=identity.kind,
        seed=identity.seed,
        artifact_manifest_sha256=identity.artifact_manifest_sha256,
        permutation=permutation,
        raw_counts=raw_counts,
        edge_signs=edge_signs,
        node_signs=node_signs,
        sign_policy=(
            identity.sign_policy
            if identity.sign_policy is not None
            else FullControlSignPolicy.REDERIVED_FROM_PRESYNAPTIC
        ),
        collect_rewire_statistics=collect_rewire_statistics,
        persist_directory=None,
    )
    if rebuilt.identity != identity:
        raise FullControlError(  # pragma: no cover - the checks above are exhaustive
            "the rebuilt control identity is not the one supplied"
        )
    return rebuilt


def _assemble_control_view(
    faithful: FullRuntimeView,
    *,
    kind: FullControlKind,
    seed: int,
    artifact_manifest_sha256: str,
    permutation: NDArray[Any] | None,
    raw_counts: NDArray[Any] | None,
    edge_signs: NDArray[Any] | None,
    node_signs: NDArray[np.int8] | None,
    sign_policy: FullControlSignPolicy,
    collect_rewire_statistics: bool,
    persist_directory: Path | str | None,
) -> FullControlView:
    """The one construction path, shared by the fresh build and the rebuild."""

    if kind is FullControlKind.NO_INPUT:
        identity = build_control_identity(
            kind=kind,
            seed=seed,
            artifact_manifest_sha256=artifact_manifest_sha256,
            edge_count=faithful.edge_count,
            faithful_view_sha256=faithful.view_sha256,
            permutation=None,
            sign_policy=None,
        )
        return FullControlView(identity=identity, view=faithful, permutation=None)

    assert permutation is not None
    counts = _require_edge_array("raw_counts", raw_counts, faithful.edge_count, np.int64)
    drawn = np.ascontiguousarray(permutation, dtype=np.uint32)
    assert_is_permutation(drawn, edge_count=faithful.edge_count)
    slots = drawn.astype(np.intp, copy=False)

    rederives = (
        kind is FullControlKind.DEGREE_PRESERVING_RANDOM
        and sign_policy is FullControlSignPolicy.REDERIVED_FROM_PRESYNAPTIC
    )
    if rederives:
        if node_signs is None:
            raise FullControlError(
                "re-deriving signs from the new presynaptic neuron requires node_signs"
            )
        faithful_signs = edge_signs_from_node_signs(node_signs, faithful.indices).astype(
            np.int64, copy=False
        )
    else:
        faithful_signs = _require_edge_array(
            "edge_signs", edge_signs, faithful.edge_count, np.int64
        )
    _assert_rebuilds_the_faithful_view(faithful, counts=counts, signs=faithful_signs)

    statistics: RewireStatistics | None = None
    if kind is FullControlKind.SHUFFLED_WEIGHT:
        resolved_policy: FullControlSignPolicy | None = None
        control_counts = counts[slots]
        control_indices: NDArray[Any] = faithful.indices
        control_signs = faithful_signs
        provenance_field = "weight_overlay_sha256"
    elif kind is FullControlKind.DEGREE_PRESERVING_RANDOM:
        resolved_policy = sign_policy
        control_counts = counts
        control_indices = np.ascontiguousarray(faithful.indices[slots], dtype=np.uint32)
        if rederives:
            assert node_signs is not None
            control_signs = edge_signs_from_node_signs(node_signs, control_indices).astype(
                np.int64, copy=False
            )
        else:
            control_signs = faithful_signs
        provenance_field = "topology_overlay_sha256"
        if collect_rewire_statistics:
            statistics = _rewire_statistics(
                indptr=faithful.indptr,
                source_indices=faithful.indices,
                control_indices=control_indices,
            )
    else:  # pragma: no cover - StrEnum is exhaustive above
        raise FullControlError(f"unknown control kind {kind!r}")

    identity = build_control_identity(
        kind=kind,
        seed=seed,
        artifact_manifest_sha256=artifact_manifest_sha256,
        edge_count=faithful.edge_count,
        faithful_view_sha256=faithful.view_sha256,
        permutation=drawn,
        sign_policy=resolved_policy,
    )
    provenance = _control_provenance(faithful.provenance, field=provenance_field, identity=identity)
    view = build_view(
        indptr=faithful.indptr,
        pre_indices=control_indices,
        raw_counts=control_counts,
        signs=control_signs,
        provenance=provenance,
    )
    persisted: Path | None = None
    if persist_directory is not None:
        persisted = write_permutation(drawn, identity, persist_directory)
    return FullControlView(
        identity=identity,
        view=view,
        permutation=drawn,
        rewire_statistics=statistics,
        permutation_path=persisted,
    )


def _assert_rebuilds_the_faithful_view(
    faithful: FullRuntimeView,
    *,
    counts: NDArray[np.int64],
    signs: NDArray[np.int64],
) -> None:
    """Refuse edge arrays that are not the ones the faithful view was built from.

    ``view_sha256`` digests ``indptr``, ``indices``, ``weights`` and
    ``empty_rows``; the first, second and fourth are the faithful view's own
    here, so this is exactly a check on the derived weights, and therefore on
    the counts and signs that derived them. It costs one extra ``build_view``
    -- about a second and 205 MB of transient float64 at full scale -- which is
    the price of an identity that cannot name weights it does not describe.
    """

    candidate = build_view(
        indptr=faithful.indptr,
        pre_indices=faithful.indices,
        raw_counts=counts,
        signs=signs,
        provenance=faithful.provenance,
    )
    if candidate.view_sha256 != faithful.view_sha256:
        raise FullControlError(
            "the supplied raw_counts/edge_signs do not rebuild the faithful view "
            f"({candidate.view_sha256[:12]}..., not {faithful.view_sha256[:12]}...); "
            "a control identity may not be minted over arrays it does not describe"
        )


def _control_provenance(
    faithful: ViewProvenance, *, field: str, identity: FullControlIdentity
) -> ViewProvenance:
    """The faithful provenance with the control identity in its overlay slot.

    ``ViewProvenance.identity_sha256`` hashes every field, and the view's own
    ``provenance`` is what the engine folds into each committed state digest,
    so putting the control identity here is what makes a control lane's state
    material differ from the faithful lane's in a recomputable way.
    """

    if field == "weight_overlay_sha256":
        return ViewProvenance(
            origin=f"{faithful.origin}:control:{identity.kind.value}",
            graph_identity_sha256=faithful.graph_identity_sha256,
            sign_overlay_sha256=faithful.sign_overlay_sha256,
            delay_overlay_sha256=faithful.delay_overlay_sha256,
            weight_overlay_sha256=identity.identity_sha256,
            topology_overlay_sha256=faithful.topology_overlay_sha256,
            source_manifest_sha256=faithful.source_manifest_sha256,
        )
    return ViewProvenance(
        origin=f"{faithful.origin}:control:{identity.kind.value}",
        graph_identity_sha256=faithful.graph_identity_sha256,
        sign_overlay_sha256=faithful.sign_overlay_sha256,
        delay_overlay_sha256=faithful.delay_overlay_sha256,
        weight_overlay_sha256=faithful.weight_overlay_sha256,
        topology_overlay_sha256=identity.identity_sha256,
        source_manifest_sha256=faithful.source_manifest_sha256,
    )


def _rewire_statistics(
    *,
    indptr: NDArray[np.int64],
    source_indices: NDArray[Any],
    control_indices: NDArray[Any],
) -> RewireStatistics:
    """Count duplicate pairs and self-edges, before and after the permutation.

    The pair key is ``row * node_count + pre`` in int64. For the real artifact
    that peaks at ``166_699 * 166_700 + 166_699`` -- comfortably inside int64
    -- and the distinct-pair count is one ``np.unique`` sort over 25.6 M int64
    keys. The intermediates are released as soon as they are consumed because
    each is 205 MB at full scale.
    """

    node_count = int(indptr.shape[0]) - 1
    edge_count = int(control_indices.shape[0])
    if edge_count == 0:
        return RewireStatistics(
            edge_count=0,
            distinct_pair_count=0,
            duplicate_edge_count=0,
            self_edge_count=0,
            source_self_edge_count=0,
        )
    degrees = np.diff(indptr)
    rows = np.repeat(np.arange(node_count, dtype=np.int64), degrees)
    self_edge_count = int(np.count_nonzero(rows == control_indices))
    source_self_edge_count = int(np.count_nonzero(rows == source_indices))
    keys = rows * np.int64(node_count) + control_indices.astype(np.int64, copy=False)
    del rows
    distinct = int(np.unique(keys).shape[0])
    del keys
    return RewireStatistics(
        edge_count=edge_count,
        distinct_pair_count=distinct,
        duplicate_edge_count=edge_count - distinct,
        self_edge_count=self_edge_count,
        source_self_edge_count=source_self_edge_count,
    )


# ---------------------------------------------------------------------------
# validation helpers


def _require_uint64(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullControlError(f"{name} must be an integer")
    if not 0 <= value < UINT64_LIMIT:
        raise FullControlError(f"{name} must be a uint64")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX64:
        raise FullControlError(f"{name} must be a lowercase 64-character hex digest")


def _require_edge_array(
    name: str, value: NDArray[Any] | None, edge_count: int, dtype: Any
) -> NDArray[Any]:
    if value is None:
        raise FullControlError(f"{name} is required to build this control view")
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.ndim != 1 or int(array.shape[0]) != edge_count:
        raise FullControlError(f"{name} must have exactly one entry per edge")
    return array
