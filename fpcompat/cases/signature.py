"""L1 over every callable whose signature pandas can report.

The other half of the surface sweep. `resolution.py` asks whether the name is there
and this asks whether it takes what pandas takes, for the 1034 callables that
`inspect.signature` can read.

**It compares parameter names and kinds and not defaults.** A wrong default is a real
break, but the repr of `lib.no_default` is not portable between two libraries and
comparing defaults would produce a wall of failures that are all about sentinel
identity rather than about behaviour. A wrong default shows up as an L2 failure, which
is where a user would notice it, because what a user sees is the answer and not the
repr.

**The 91 callables pandas itself cannot introspect get no case.** Both engines failing
to read a signature is not evidence of anything, and scoring it as agreement would be
the flattering kind of wrong. Those names are listed by the coverage report, where an
unmeasured thing belongs.
"""

from __future__ import annotations

from typing import Any

from fpcompat import surface
from fpcompat.cases import case, section
from fpcompat.cases.resolution import FRAME, case_id, members
from fpcompat.cases.resolution import _namespace as namespace

section("signature")


def _expression(space: str, member: str):
    """Builds the L1 expression for one callable.

    Args:
        space: The namespace.
        member: The member name.

    Returns:
        An expression returning the parameter names and kinds, in order.
    """

    def expr(module: Any, df: Any) -> Any:
        try:
            obj = namespace(module, space)
            if not hasattr(obj, member):
                raise AttributeError(f"{space}.{member} does not exist")
        except AttributeError as error:
            raise AttributeError(str(error)) from error
        params = surface.signature_of(getattr(obj, member))
        if params is None:
            # pandas could read this one, so an engine that cannot is not matching it.
            # Returning the sentinel rather than None keeps the failure readable.
            return "the signature could not be read"
        return [f"{param['name']}:{param['kind']}" for param in params]

    return expr


def _generate() -> int:
    """Registers one L1 case per readable callable.

    Returns:
        How many.
    """
    count = 0
    for space, member, info in members():
        if info["kind"] != "callable" or info.get("signature") is None:
            continue
        case(
            case_id("signature", space, member),
            f"{space}.{member}",
            level="L1",
            frames=FRAME,
            expr=_expression(space, member),
        )
        count += 1
    return count


SIGNATURE_CASES = _generate()
