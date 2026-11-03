"""MEDIUM-4/MEDIUM-5: which arena constants each code path turns into behaviour.

``BoundaryIdentity.free_parameters`` is the count of scalars a designer could have
chosen, and it is hashed. It is only meaningful if the list behind it is derived from
what the code dereferences rather than maintained by hand, so these tests re-derive
the per-path sets from ``reference.py``'s own AST and assert the declarations in
``reference_sensing`` against them.

The scan resolves the *object*, not the identifier. The previous version matched two
hard-coded binding forms -- a local literally named ``arena``, and ``self._arena`` --
while its docstring claimed "a new helper cannot hide a constant by not being named".
One rename walked past it: ``cfg = self._arena`` inside ``_substep``, or a helper
written ``def _bias(cfg: ReferenceArena)``, and the constant it read was in no
classification group with the whole suite green.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap

import pytest


def _expression_key(node: ast.AST) -> str | None:
    """The name an expression is reachable under: ``x`` or ``self.x``, else ``None``."""

    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return f"self.{node.attr}"
    return None


def _is_arena_valued(node: ast.AST, aliases: set[str]) -> bool:
    """Whether an expression evaluates to a :class:`ReferenceArena`."""

    if _expression_key(node) in aliases:
        return True
    if isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Name) and function.id == "ReferenceArena":
            return True
        # ``replace(arena, ...)`` returns an arena; ``replace(state, ...)`` does not.
        named_replace = (isinstance(function, ast.Name) and function.id == "replace") or (
            isinstance(function, ast.Attribute) and function.attr == "replace"
        )
        if named_replace and node.args:
            return _is_arena_valued(node.args[0], aliases)
    if isinstance(node, ast.BoolOp):
        return any(_is_arena_valued(value, aliases) for value in node.values)
    if isinstance(node, ast.IfExp):
        return _is_arena_valued(node.body, aliases) or _is_arena_valued(node.orelse, aliases)
    return False


def _arena_aliases(tree: ast.AST) -> set[str]:
    """Every name in a source tree that holds a ``ReferenceArena``.

    MEDIUM-4's guard used to hard-code two binding forms: a local literally named
    ``arena``, and ``self._arena``. The docstring claimed "a new helper cannot hide a
    constant by not being named", but the *binding form* list was still curated, so
    one rename defeated it -- ``cfg = self._arena`` inside ``_substep``, or a helper
    written ``def _bias(cfg: ReferenceArena)``, and the whole suite stayed green while
    the constant it read was in no classification group.

    The roots are ``self._arena`` and any parameter *annotated* ``ReferenceArena``
    (including ``ReferenceArena | None``), so the name is irrelevant. Assignments are
    then followed to a fixpoint, so a chain of aliases resolves too. What is matched
    is the object, not the identifier.
    """

    aliases = {"self._arena"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            annotation = argument.annotation
            if annotation is not None and any(
                isinstance(inner, ast.Name) and inner.id == "ReferenceArena"
                for inner in ast.walk(annotation)
            ):
                aliases.add(argument.arg)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not _is_arena_valued(value, aliases):
                continue
            for target in targets:
                key = _expression_key(target)
                if key is not None and key not in aliases:
                    aliases.add(key)
                    changed = True
    return aliases


def _arena_attribute_reads(source: str, fields: set[str]) -> tuple[str, ...]:
    """Every ``ReferenceArena`` field dereferenced off an arena-valued expression.

    Restricted to real fields so derived attributes such as ``identity_sha256`` are not
    mistaken for parameters. Reads inside ``ReferenceArena``'s own methods use bare
    ``self`` and are deliberately not matched: ``__post_init__`` validates, and
    ``to_identity_material`` hashes, and neither is a path that turns a constant into
    behaviour. ``_build_identity`` is the same kind of path and is classified as such
    rather than excluded -- see ``REFERENCE_IDENTITY_PARAMETERS``.
    """

    tree = ast.parse(textwrap.dedent(source))
    aliases = _arena_aliases(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in fields:
            continue
        if _expression_key(node.value) in aliases:
            names.add(node.attr)
    return tuple(sorted(names))


def _arena_fields() -> set[str]:
    from neurocraft_embodiment import reference as module

    return {field.name for field in dataclasses.fields(module.ReferenceArena)}


def _arena_attributes_read_by(*functions: object) -> tuple[str, ...]:
    """Every arena field one code path dereferences, however the arena is bound."""

    fields = _arena_fields()
    names: set[str] = set()
    for function in functions:
        names.update(_arena_attribute_reads(inspect.getsource(function), fields))  # type: ignore[arg-type]
    return tuple(sorted(names))


def _arena_attributes_read_anywhere_in_the_module() -> tuple[str, ...]:
    """Every arena field read anywhere in ``reference.py``, full stop.

    No function list. The previous helper took four hand-picked functions and could
    therefore only see what someone had already thought of; it was blind to
    ``_initial_state``, which ``__init__`` and ``reset`` both call and which reads a
    constant neither boundary classified.
    """

    from neurocraft_embodiment import reference as module

    return _arena_attribute_reads(inspect.getsource(module), _arena_fields())


#: Sources in which a constant named ``stealth_gain`` is read off an arena under a
#: name the previous two-form matcher did not recognise. Each is a rewrite of the
#: demonstrated evasion: adding the field to ``ReferenceArena`` and reading it as
#: ``cfg = self._arena; ... cfg.stealth_gain`` left the scan reporting nothing.
_ALIASED_READS = {
    "a local alias of self._arena": """
        class ReferenceEmbodiment:
            def _substep(self, state, motors):
                cfg = self._arena
                return state.velocity_x_m_s * cfg.stealth_gain
    """,
    "a chain of aliases": """
        class ReferenceEmbodiment:
            def _substep(self, state, motors):
                first = self._arena
                second = first
                return second.stealth_gain
    """,
    "a helper whose arena parameter is not named 'arena'": """
        def _bias(cfg: ReferenceArena) -> float:
            return cfg.stealth_gain
    """,
    "an optional-annotated parameter": """
        def _bias(config: ReferenceArena | None) -> float:
            return config.stealth_gain
    """,
    "a freshly constructed arena": """
        def _bias() -> float:
            local = ReferenceArena()
            return local.stealth_gain
    """,
}


@pytest.mark.parametrize("label", sorted(_ALIASED_READS))
def test_the_scan_sees_an_arena_constant_read_under_any_name(label: str) -> None:
    """MEDIUM-4: the binding-form list was curated, so one rename walked past it."""

    assert _arena_attribute_reads(_ALIASED_READS[label], {"stealth_gain"}) == ("stealth_gain",)


def test_the_scan_does_not_match_a_same_named_attribute_on_something_else() -> None:
    """The scan must resolve the object, not pattern-match the attribute name.

    Otherwise it would be a grep with extra steps, and it would report a constant that
    is not an arena constant at all -- which would make the classification lists wrong
    in the other direction.
    """

    body_state_read = """
        class ReferenceEmbodiment:
            def _substep(self, state, motors):
                return state.stealth_gain
    """
    assert _arena_attribute_reads(body_state_read, {"stealth_gain"}) == ()


def test_every_arena_constant_any_code_path_reads_is_classified() -> None:
    """MEDIUM-4: the AST test was real but its function list was hand-picked.

    ``_initial_state`` reads ``spawn_height_m`` and is called from ``__init__`` and
    from ``reset``, so the constant parameterises the body's entire trajectory --
    every observation and every actuation correction downstream of where it starts.
    It appeared in neither boundary's parameter list, so both ``free_parameters``
    counts undercounted by one, and the test that was supposed to catch exactly this
    could not see the function.

    The list is not curated any more: this asserts over every dereference in the
    module, so a new helper cannot hide a constant by not being named.
    """

    from neurocraft_embodiment import reference_sensing as declarations

    classified = set(declarations.REFERENCE_SENSORY_PARAMETERS) | set(
        declarations.REFERENCE_MOTOR_PARAMETERS
    )
    dereferenced = set(_arena_attributes_read_anywhere_in_the_module())
    assert dereferenced - classified == set()
    assert classified - dereferenced == set()
    assert "spawn_height_m" in classified

    # Every named path, not just the two boundary unions. The union is what the
    # identity records, but a union can stay right while one of the paths feeding it
    # is not named at all -- which is exactly the state ``_build_identity`` was in.
    grouped = set().union(
        declarations.REFERENCE_GEOMETRY_PARAMETERS,
        declarations.REFERENCE_INITIAL_STATE_PARAMETERS,
        declarations.REFERENCE_OBSERVATION_PARAMETERS,
        declarations.REFERENCE_ACTUATION_PARAMETERS,
        declarations.REFERENCE_IDENTITY_PARAMETERS,
    )
    assert grouped == dereferenced


def test_parameter_classification_matches_what_each_path_dereferences() -> None:
    """MEDIUM-5: assert the CONTENTS of both lists, not just their lengths.

    The previous test asserted only the counts (11 and 4), so it could not see that
    ``contact_epsilon_m`` was classified sensory while only the collision tie-break
    reads it, nor that seven constants ``_substep`` reads every step were missing
    from the motor list entirely. These sets are re-derived here from the functions'
    own ASTs, so the declaration cannot drift from the code again.
    """

    from neurocraft_embodiment import reference as module
    from neurocraft_embodiment import reference_sensing as declarations

    geometry = _arena_attributes_read_by(module._generate_pillars)
    observation = _arena_attributes_read_by(
        module.ReferenceEmbodiment.observe, module._ray_distance
    )
    actuation = _arena_attributes_read_by(
        module.ReferenceEmbodiment.apply, module.ReferenceEmbodiment._substep
    )
    initial = _arena_attributes_read_by(module.ReferenceEmbodiment._initial_state)
    identity = _arena_attributes_read_by(module.ReferenceEmbodiment._build_identity)
    assert geometry == declarations.REFERENCE_GEOMETRY_PARAMETERS
    assert observation == declarations.REFERENCE_OBSERVATION_PARAMETERS
    assert actuation == declarations.REFERENCE_ACTUATION_PARAMETERS
    assert initial == declarations.REFERENCE_INITIAL_STATE_PARAMETERS
    assert identity == declarations.REFERENCE_IDENTITY_PARAMETERS

    # ``_build_identity`` hashes rather than acts, so it is *not* unioned into either
    # boundary's capacity -- charging a boundary for a scalar it does not act on would
    # be as wrong as omitting one it does. It was previously covered by nothing but
    # arithmetic luck: every constant it reads happens to be in another group. That
    # containment is now the assertion, so if it ever reaches for a constant no
    # behaviour path reads, this fails instead of the counts quietly staying right.
    assert set(identity) <= {*geometry, *initial, *observation, *actuation}
    assert identity, "the identity path reads no arena constant at all"

    # The classification the identity records is the union of a path with the shared
    # geometry it intersects and the state the body starts in.
    sensory = tuple(sorted({*geometry, *initial, *observation}))
    motor = tuple(sorted({*geometry, *initial, *actuation}))
    assert sensory == declarations.REFERENCE_SENSORY_PARAMETERS
    assert motor == declarations.REFERENCE_MOTOR_PARAMETERS

    # The specific misclassifications the reviewer demonstrated.
    assert "contact_epsilon_m" not in declarations.REFERENCE_SENSORY_PARAMETERS
    assert "contact_epsilon_m" in declarations.REFERENCE_MOTOR_PARAMETERS
    for name in (
        "gravity_m_s2",
        "linear_drag_per_s",
        "max_speed_m_s",
        "max_pitch_rad",
        "pillar_radius_m",
        "half_extent_x_m",
        "half_extent_z_m",
    ):
        assert name in declarations.REFERENCE_MOTOR_PARAMETERS
