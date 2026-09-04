"""L0 over every name pandas has, generated from the committed inventory.

The other case modules are written by hand, because what `groupby(...).agg(...)`
should return is a judgement. This question is not a judgement. "Does the name
resolve" has a mechanical answer for all 1413 names, and the mechanical answer is the
only one worth having: a hand written L0 list measures the names somebody thought to
check, and this one measures the names pandas has.

So the cases are generated, one per public name, from the same committed inventory the
scoreboard counts against. That makes the oracle run a real check on the generator
rather than a formality, since pandas has to agree with a file that was written by
walking pandas. The L1 half is next door in `signature.py` and it shares the machinery
in here.

Two decisions worth stating.

**A missing name is unimplemented and not a failure.** The runner separates those two
by traceback depth, so every expression here raises `AttributeError` from its own
frame when the name is absent, including when what is absent is the whole accessor. A
firepanda with no `.str` reads as 57 unimplemented names, which is a schedule, rather
than as 57 failures, which is a bug list. Those two numbers are read by different
people for different reasons and collapsing them would make both useless.

**It compares the kind and not just the presence.** A name that pandas exposes as a
property and firepanda exposes as a method is not the same name, because `frame.shape`
and `frame.shape()` are different programs. Comparing the kind costs nothing and
catches it.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from fpcompat import surface
from fpcompat.cases import case, section

section("resolution")

# One frame, the smallest one, and no case here looks at it. These cases are about the
# API and not about data, and running each of them on five frames would be five copies
# of the same answer.
FRAME = ("empty",)

INVENTORY = json.loads(surface.path_for(pd.__version__).read_text())

# The sample objects, built per module the first time an expression on that module
# needs one. `Series.str` is a descriptor, so the only way to ask what is in the string
# namespace is to build a string series and look, and doing that 2447 times per engine
# would be most of the run. Keyed by the module object rather than by its id, because a
# worker holds two modules for its whole life and an id is only unique while the thing
# it names is alive.
_SAMPLES: dict[Any, dict[str, Any]] = {}


def _namespace(module: Any, space: str) -> Any:
    """The sample object for one namespace, built out of the given module.

    Args:
        module: The engine module.
        space: The namespace name as the inventory spells it.

    Returns:
        The object whose members are that namespace.

    Raises:
        AttributeError: When this engine cannot build that namespace at all. The
            caller re-raises it from its own frame so that the runner sees the right
            traceback depth and calls it unimplemented.
    """
    built = _SAMPLES.get(module)
    if built is None:
        built = surface.namespaces(module)
        _SAMPLES[module] = built
    obj = built[space]
    if isinstance(obj, Exception):
        raise AttributeError(f"the {space} namespace could not be built: {obj}") from obj
    return obj


def _kind(obj: Any, member: str) -> str:
    """What a member is, in the inventory's vocabulary.

    Args:
        obj: The namespace sample object.
        member: The member name, which is known to exist.

    Returns:
        `callable`, `property` or `attribute`.
    """
    owner = obj if isinstance(obj, type) else type(obj)
    if isinstance(getattr(owner, member, None), property):
        return "property"
    return "callable" if callable(getattr(obj, member)) else "attribute"


def _resolution(space: str, member: str):
    """Builds the L0 expression for one name.

    Args:
        space: The namespace.
        member: The member name.

    Returns:
        An expression returning the kind of thing the name is.
    """

    def expr(module: Any, df: Any) -> str:
        try:
            obj = _namespace(module, space)
            if not hasattr(obj, member):
                raise AttributeError(f"{space}.{member} does not exist")
        except AttributeError as error:
            # Re-raised from this frame on purpose. The runner reads traceback depth to
            # tell a name that is not there yet from a method that got three frames in
            # and gave up, and an absent name has to land on the first of those.
            raise AttributeError(str(error)) from error
        return _kind(obj, member)

    return expr


def case_id(prefix: str, space: str, member: str) -> str:
    """The stable id for a generated case.

    Args:
        prefix: `resolution` or `signature`.
        space: The namespace.
        member: The member name.

    Returns:
        The id, lower cased, since case ids are lower case and two pandas names never
        differ only by case within one namespace.
    """
    return f"{prefix}/{space.lower()}.{member.lower()}"


def members() -> list[tuple[str, str, dict[str, Any]]]:
    """Every name in the inventory, in a stable order.

    Returns:
        Namespace, member and the inventory entry, sorted, so that both generated
        modules walk the surface the same way and a result file reads the same twice.
    """
    return [
        (space, member, info)
        for space, entry in sorted(INVENTORY["namespaces"].items())
        for member, info in sorted(entry["members"].items())
    ]


def _generate() -> int:
    """Registers one L0 case per name.

    Returns:
        How many, which a test compares against the inventory totals so that a pandas
        release adding names adds cases.
    """
    for space, member, _ in members():
        case(
            case_id("resolution", space, member),
            f"{space}.{member}",
            level="L0",
            frames=FRAME,
            expr=_resolution(space, member),
        )
    return len(members())


RESOLUTION_CASES = _generate()
