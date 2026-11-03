"""MEDIUM-6: the shipped adapters must not hash their own capacity budgets.

``flygym_boundary``'s module docstring justifies shipping *sibling* wide boundaries
rather than raising ``neurocraft_neural.adapters``' budgets in place, and part of that
justification is a factual claim: raising those budgets would leave every recorded
identity byte-identical, because the constants are nowhere in any hashed material. If
that claim were false, the sibling argument would be arguing against a cost that does
not exist -- and, far worse, a budget that *is* hashed becomes a lever for rotating
every recorded digest in the repository by editing one number.

The claim is checked two ways here: by resolving what the identity expression actually
depends on, and by raising every budget and watching the digests not move.

Why this file is not the previous version of this check
-------------------------------------------------------
Five independent evasions walked through the previous version with the whole suite
green, and they had three distinct root causes. Each is closed by a different piece of
what follows, and each is exercised by :data:`EVASIONS`.

1. **The closure stopped at ``__init__``.** ``bindings`` was built from
   ``ast.walk(init)`` alone, so a name bound at module scope, in another function, in
   another method, or in an argument default was a dead end. The docstring's
   "transitive closure of the bindings that feed it" was true only within one
   function. ``_collect_bindings`` now also runs over module-level statements, over
   the parameter defaults of every function it enters, and over the body of any
   function or method the closure reaches.

2. **The forbidden set was a hand-written blacklist of four names.** The module
   declares six ``MAX_*`` constants; ``MAX_ABSOLUTE_ADAPTER_WEIGHT`` and
   ``MAX_ABSOLUTE_ADAPTER_PARAMETER`` were simply not on the list, so writing one of
   them straight into the identity material -- no aliasing, no obfuscation -- passed.
   The check is now a **whitelist**: every identifier the material depends on must be
   one of :data:`PERMITTED_IDENTITY_MATERIAL_NAMES`, so an unrecognised name fails
   closed whether or not anybody thought of it. The budget set is derived from the
   module rather than typed out, and is asserted disjoint from the whitelist.

3. **The runtime experiment could not see the defect it was testing for.** It patched
   two of the six constants, only ever built a ``SensoryAdapter``, and used
   ``monkeypatch.setattr``, which cannot reach a value that was copied into another
   object at import time -- so an alias or an argument default was invisible to it by
   construction. It now rewrites every ``MAX_*`` in the source and loads that source as
   a fresh module, and it builds both classes.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
from neurocraft_connectome import RuntimeGraph
from neurocraft_embodiment import (
    MAX_FLYGYM_SENSOR_CHANNELS,
    FlyGymSensoryAdapter,
)
from neurocraft_neural import MotorPool, PoolMember
from neurocraft_neural import adapters as shipped

from .conftest import (
    FIXTURE_OUTPUT_BODY_IDS,
    REPO_ROOT,
    wide_sensor_channels,
    wide_sensor_free_parameters,
)

ADAPTERS = REPO_ROOT / "packages/neural-runtime/src/neurocraft_neural/adapters.py"


def adapters_source() -> str:
    return ADAPTERS.read_text(encoding="utf-8")


# -- the resolver --------------------------------------------------------------

#: Calls that add to a container already bound to a name, e.g. ``xs.append(...)``.
_MUTATORS = frozenset({"append", "extend", "insert", "update", "add", "setdefault"})


def _binding_keys(target: ast.AST) -> set[str]:
    """The name(s) an assignment target binds: ``x``, ``self.x``, tuples of those."""

    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name) and target.value.id == "self":
            return {f"self.{target.attr}"}
        return _binding_keys(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return {key for element in target.elts for key in _binding_keys(element)}
    if isinstance(target, (ast.Subscript, ast.Starred)):
        return _binding_keys(target.value)
    return set()


def _collect_bindings(scope: ast.AST, bindings: dict[str, list[ast.AST]]) -> None:
    """Record every expression that can flow into a name inside ``scope``.

    Parameter defaults are collected first and separately, because they are the one
    binding form that is not a statement: ``def __init__(self, graph, channels,
    _capacity: int = MAX_SENSOR_CHANNELS)`` binds a budget to a name with no
    assignment anywhere for a statement walk to find.
    """

    def bind(target: ast.AST, value: ast.AST) -> None:
        for key in _binding_keys(target):
            bindings.setdefault(key, []).append(value)

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = scope.args
        positional = [*arguments.posonlyargs, *arguments.args]
        offset = len(positional) - len(arguments.defaults)
        for argument, default in zip(positional[offset:], arguments.defaults, strict=True):
            bindings.setdefault(argument.arg, []).append(default)
        for argument, keyword_default in zip(
            arguments.kwonlyargs, arguments.kw_defaults, strict=True
        ):
            if keyword_default is not None:
                bindings.setdefault(argument.arg, []).append(keyword_default)

    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind(node.target, node.iter)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
        ):
            for argument in node.args:
                bind(node.func.value, argument)


def identity_material_names(source: str, class_name: str) -> set[str]:
    """Every identifier the class's hashed identity expression actually depends on.

    Starts from the argument of the ``sha256_json`` call assigned to
    ``self._identity`` and walks to a fixpoint over three kinds of edge: names bound
    by statements or parameter defaults, functions and methods reached by name (whose
    bodies are then scanned and whose ``return`` expressions are enqueued), and
    module-level bindings. Slicing the source text between ``sha256_json(`` and the
    first ``)`` -- what this used to do -- sees only literals, because every
    interesting operand is assembled somewhere else.
    """

    tree = ast.parse(source)
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    init = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
            functions[f"self.{node.name}"] = node

    bindings: dict[str, list[ast.AST]] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect_bindings(statement, bindings)
    _collect_bindings(init, bindings)

    seed = next(
        node.value.args[0]
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any("self._identity" in _binding_keys(target) for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "sha256_json"
        and node.value.args
    )

    resolved: set[str] = set()
    pending: list[ast.AST] = [seed]
    seen_expressions: set[int] = set()
    seen_functions: set[int] = set()
    while True:
        while pending:
            expression = pending.pop()
            if id(expression) in seen_expressions:
                continue
            seen_expressions.add(id(expression))
            for node in ast.walk(expression):
                if isinstance(node, ast.Name):
                    resolved.add(node.id)
                elif isinstance(node, ast.Attribute):
                    resolved |= _binding_keys(node)
                    resolved.add(node.attr)
        grew = False
        for key in sorted(resolved):
            function = functions.get(key)
            if function is not None and id(function) not in seen_functions:
                seen_functions.add(id(function))
                _collect_bindings(function, bindings)
                pending.extend(
                    node.value
                    for node in ast.walk(function)
                    if isinstance(node, ast.Return) and node.value is not None
                )
                grew = True
            for value in bindings.get(key, ()):
                if id(value) not in seen_expressions:
                    pending.append(value)
                    grew = True
        if not grew:
            return resolved


# -- what the shipped material is allowed to depend on --------------------------

#: Every identifier the two shipped identity expressions may resolve to.
#:
#: A whitelist rather than a list of forbidden names, because the forbidden list can
#: only ever contain what somebody thought of, and the thing being defended against is
#: precisely a name nobody thought of. Asserted in both directions below: nothing
#: outside it may appear, and nothing in it may be unused.
PERMITTED_IDENTITY_MATERIAL_NAMES = frozenset(
    {
        # The compiled per-channel / per-pool records and the loop variables that
        # build them.
        "channel",
        "channels",
        "channel_values",
        "identity_channels",
        "identity_pools",
        "member",
        "members",
        "name",
        "names",
        "pool",
        "pools",
        "pool_values",
        "target",
        "targets",
        # Declared per-channel and per-pool scalars.
        "clip_max",
        "clip_min",
        "dead_zone",
        "gain",
        "offset",
        "saturation",
        "unit",
        "weight",
        # The graph the boundary is bound to, and the node identity it records.
        "body_id",
        "graph",
        "index",
        "index_of",
        "manifest",
        "nodes",
        "runtime_identity_sha256",
        "self",
        "self._graph_identity",
        "_graph_identity",
        # Encoders shared with the engine's integrity checks.
        "_finite_float",
        "_float_hex",
        # Builtins.
        "tuple",
    }
)


def budget_constants_from_the_module() -> frozenset[str]:
    """Every ``MAX_*`` the adapters module declares, read off the imported module."""

    return frozenset(name for name in vars(shipped) if name.startswith("MAX_"))


def budget_constants_from_the_source() -> frozenset[str]:
    """The same set, read off the source's module-level assignments."""

    tree = ast.parse(adapters_source())
    return frozenset(
        key
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        for key in _binding_keys(target)
        if key.startswith("MAX_")
    )


def test_the_budget_set_is_derived_from_the_module_not_typed_out() -> None:
    """The previous list named four of six by hand, and the two it missed were real.

    ``MAX_ABSOLUTE_ADAPTER_WEIGHT`` and ``MAX_ABSOLUTE_ADAPTER_PARAMETER`` are budgets
    -- the module's own error text calls one "declared parameter magnitude budget" --
    and neither was on the list, so writing either straight into the hashed material
    passed. Two independent derivations of the set, so neither can drift.
    """

    from_module = budget_constants_from_the_module()
    assert from_module == budget_constants_from_the_source()
    assert {"MAX_ABSOLUTE_ADAPTER_WEIGHT", "MAX_ABSOLUTE_ADAPTER_PARAMETER"} <= from_module
    assert len(from_module) > 4
    # A budget can never also be permitted material; the whitelist is what enforces
    # that, and this states the relationship the two sets are supposed to have.
    assert from_module.isdisjoint(PERMITTED_IDENTITY_MATERIAL_NAMES)


def test_the_resolver_reaches_material_assembled_through_a_variable() -> None:
    """A resolver that resolves nothing would pass every assertion below vacuously."""

    source = adapters_source()
    sensory = identity_material_names(source, "SensoryAdapter")
    # ``identity_channels`` is a bare local; these live only inside the dicts appended
    # to it, so seeing them proves the closure followed the variable into its contents.
    assert {"identity_channels", "_float_hex", "clip_max", "saturation"} <= sensory
    motor = identity_material_names(source, "MotorDecoder")
    assert {"identity_pools", "_float_hex", "dead_zone"} <= motor


def test_the_permitted_name_list_is_exactly_what_the_shipped_material_uses() -> None:
    """Both directions, so the whitelist can be neither too narrow nor decorative.

    Too narrow and the shipped code fails; too wide and a permission sits there
    covering a name nothing uses, which is how a whitelist rots into a rubber stamp.
    """

    source = adapters_source()
    used = identity_material_names(source, "SensoryAdapter") | identity_material_names(
        source, "MotorDecoder"
    )
    assert used - PERMITTED_IDENTITY_MATERIAL_NAMES == set()
    assert PERMITTED_IDENTITY_MATERIAL_NAMES - used == set()


def test_the_shipped_adapters_do_not_hash_their_budget_constants() -> None:
    """MEDIUM-6: the sibling classes' justification, verified rather than grepped.

    Stated twice on purpose. The whitelist above is the general guarantee; this is the
    specific claim ``flygym_boundary``'s docstring makes, named in the terms the
    docstring uses, so a reader chasing that claim finds a test that says it.
    """

    source = adapters_source()
    budgets = budget_constants_from_the_module()
    for class_name in ("SensoryAdapter", "MotorDecoder"):
        resolved = identity_material_names(source, class_name)
        smuggled = sorted(budgets & resolved)
        assert smuggled == [], (
            f"{class_name} hashes {smuggled}; raising a budget would then rotate "
            "every recorded identity, and the sibling justification would be wrong"
        )


# -- the five demonstrated evasions, applied to the real source in memory -------

_SENSORY_IDENTITY = """        self._identity = sha256_json(
            {
                "channels": identity_channels,"""

_MOTOR_IDENTITY = """        self._identity = sha256_json(
            {
                "graph_identity_sha256": self._graph_identity,"""

_SENSORY_INIT = (
    "    def __init__(self, graph: RuntimeGraph, channels: Sequence[SensorChannel]) -> None:"
)

_CONSTANTS = "MAX_ABSOLUTE_ADAPTER_PARAMETER = 8.0\n"

#: Each evasion is a list of ``(anchor, replacement)`` edits applied to the *real*
#: ``adapters.py`` text in memory. Nothing is written to the repository; the point is
#: to run the check against source that really contains the defect, rather than
#: against a hand-written miniature that might be easier than the real thing.
#:
#: An anchor that stops matching fails these tests loudly. That is deliberate:
#: ``adapters.py`` is promoted anti-bypass evidence and does not change, so a
#: no-longer-matching anchor means something moved that these tests must be re-read
#: against rather than silently skipped.
EVASIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "E1_module_level_alias": (
        (_CONSTANTS, _CONSTANTS + "_ADAPTER_CAPACITY_TAG = MAX_SENSOR_CHANNELS\n"),
        (
            _SENSORY_IDENTITY,
            _SENSORY_IDENTITY + '\n                "budget": _ADAPTER_CAPACITY_TAG,',
        ),
    ),
    "E3_keyword_argument_default": (
        (
            _SENSORY_INIT,
            "    def __init__(\n"
            "        self,\n"
            "        graph: RuntimeGraph,\n"
            "        channels: Sequence[SensorChannel],\n"
            "        _capacity: int = MAX_SENSOR_CHANNELS,\n"
            "    ) -> None:",
        ),
        (_SENSORY_IDENTITY, _SENSORY_IDENTITY + '\n                "budget": _capacity,'),
    ),
    "E4_an_unlisted_real_budget_written_plainly": (
        (
            _SENSORY_IDENTITY,
            _SENSORY_IDENTITY + '\n                "budget": MAX_ABSOLUTE_ADAPTER_WEIGHT,',
        ),
    ),
    "E6_module_level_mapping_spread": (
        (_CONSTANTS, _CONSTANTS + '_CAPACITY_MATERIAL = {"budget": MAX_SENSOR_CHANNELS}\n'),
        (_SENSORY_IDENTITY, _SENSORY_IDENTITY + "\n                **_CAPACITY_MATERIAL,"),
    ),
    "E7_module_level_alias_on_the_motor_decoder": (
        (_CONSTANTS, _CONSTANTS + "_POOL_CAPACITY_TAG = MAX_MOTOR_CHANNELS\n"),
        (_MOTOR_IDENTITY, _MOTOR_IDENTITY + '\n                "budget": _POOL_CAPACITY_TAG,'),
    ),
}


def evaded_source(name: str) -> str:
    text = adapters_source()
    for anchor, replacement in EVASIONS[name]:
        assert text.count(anchor) == 1, f"{name}: anchor matched {text.count(anchor)} times"
        text = text.replace(anchor, replacement)
    return text


@pytest.mark.parametrize("evasion", sorted(EVASIONS))
def test_every_demonstrated_evasion_is_refused(evasion: str) -> None:
    """Each of these passed the whole suite with the budget really inside the hash.

    Two assertions, in this order, because the second is worthless without the first.
    An evasion has to *be* a defect -- the smuggled budget must reach the identity
    material, so that changing that budget's value would rotate a recorded digest --
    and only then is it interesting that the check refuses it. A probe that smuggled
    nothing would make the refusal below a test of an inert edit.

    The two mechanisms are layered rather than redundant. The whitelist is the general
    guard: it refuses an unrecognised name whether or not it is a budget. The budget
    resolution is what keeps that guard from being the only one -- if the whitelist
    ever has to be widened for a legitimate new name, a budget reached *through* that
    name is still caught, and the closure over module scope and argument defaults is
    what makes that reachable at all.
    """

    source = evaded_source(evasion)
    ast.parse(source)  # the evasion must be real code, not a text smear
    budgets = budget_constants_from_the_module()
    smuggled: list[str] = []
    unrecognised: list[str] = []
    for class_name in ("SensoryAdapter", "MotorDecoder"):
        resolved = identity_material_names(source, class_name)
        smuggled.extend(sorted(budgets & resolved))
        unrecognised.extend(sorted(resolved - PERMITTED_IDENTITY_MATERIAL_NAMES))
    assert smuggled, f"{evasion} does not put a budget constant in the hash preimage"
    assert unrecognised, f"{evasion} was not seen by the whitelist"


# -- the same claim as an experiment rather than an inspection ------------------


def raise_every_budget(source: str) -> str:
    """Rewrite every ``MAX_*`` constant in the source to a larger value."""

    lines = source.splitlines(keepends=True)
    rewritten: list[str] = []
    changed = 0
    for line in lines:
        name, separator, value = line.partition(" = ")
        if separator and name.startswith("MAX_") and "\n" in value:
            number = float(value.strip())
            raised = f"{number * 2.0}" if "." in value else f"{int(number) + 64}"
            rewritten.append(f"{name} = {raised}\n")
            changed += 1
            continue
        rewritten.append(line)
    assert changed == len(budget_constants_from_the_module()), changed
    return "".join(rewritten)


def load_probe_module(source: str, tmp_path: Path) -> object:
    """Import ``source`` as a fresh module, so import-time copies are made from it.

    ``monkeypatch.setattr`` cannot do this job: it rebinds a module attribute after
    import, so any value already copied into a default, an alias or another module's
    namespace keeps the old number and the experiment sees nothing. Recompiling the
    source is the only way the experiment can observe the evasions it is checking for.
    """

    path = tmp_path / "adapters_budget_probe.py"
    path.write_text(source, encoding="utf-8")
    name = "neurocraft_neural_adapters_budget_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def fixture_pools() -> tuple[MotorPool, ...]:
    return tuple(
        MotorPool(name=f"pool_{index}", members=(PoolMember(body_id, 1.0),), gain=1.0)
        for index, body_id in enumerate(FIXTURE_OUTPUT_BODY_IDS)
    )


def test_raising_every_budget_leaves_both_shipped_digests_untouched(
    graph: RuntimeGraph,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both classes, every budget, from a recompiled source.

    The previous version raised two of six constants with ``monkeypatch``, built only a
    ``SensoryAdapter``, and therefore could not have observed a motor-side alias or an
    argument default at all.

    The contrast is the other half of the experiment: the flygym sibling *does* hash
    its budgets, so its digest must move under the same treatment. One claim, two
    mechanisms, and a control that proves the measurement can detect a change.
    """

    channels = wide_sensor_channels(2)
    pools = fixture_pools()
    sensory_before = shipped.SensoryAdapter(graph, channels).identity_sha256
    motor_before = shipped.MotorDecoder(graph, pools).identity_sha256

    probe = load_probe_module(raise_every_budget(adapters_source()), tmp_path)
    try:
        assert probe.MAX_SENSOR_CHANNELS != shipped.MAX_SENSOR_CHANNELS
        assert probe.MAX_ABSOLUTE_ADAPTER_WEIGHT != shipped.MAX_ABSOLUTE_ADAPTER_WEIGHT
        assert probe.SensoryAdapter(graph, channels).identity_sha256 == sensory_before
        assert probe.MotorDecoder(graph, pools).identity_sha256 == motor_before
    finally:
        sys.modules.pop("neurocraft_neural_adapters_budget_probe", None)

    import neurocraft_embodiment.flygym_boundary as sibling

    wide_before = FlyGymSensoryAdapter(
        graph, channels, declared_free_parameters=wide_sensor_free_parameters(2)
    ).identity_sha256
    monkeypatch.setattr(sibling, "MAX_FLYGYM_SENSOR_CHANNELS", MAX_FLYGYM_SENSOR_CHANNELS + 64)
    wide_after = FlyGymSensoryAdapter(
        graph, channels, declared_free_parameters=wide_sensor_free_parameters(2)
    ).identity_sha256
    assert wide_after != wide_before


def test_the_sibling_docstring_states_the_reason_the_code_supports() -> None:
    """The docstring must claim schema separation, which is the reason that holds."""

    boundary = (
        REPO_ROOT / "packages/embodiment-runtime/src/neurocraft_embodiment/flygym_boundary.py"
    ).read_text(encoding="utf-8")
    docstring = boundary.split('"""')[1]
    assert "schema separation" in docstring
